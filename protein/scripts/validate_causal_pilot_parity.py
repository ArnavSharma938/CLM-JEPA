#!/usr/bin/env python
"""Real-checkpoint parity certificate for optimized pilot paths."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.fitness import official_rita_fitness
from src.modeling import load_rita
from src.nextlat import FaithfulNextLatPredictor
from src.pilot import (
    attach_rank32_attention_lora, causal_pilot_loss, load_protocol,
    lora_named_parameters, reference_two_stage_backward,
)
from src.pilot_eval import (
    bidirectional_summary, cached_generate_rita, directional_lm_rows,
    sample_top_p, sample_top_p_uniform, sequence_representations,
)
from src.tokenization import collate_encoded, encode_canonical


def maximum_difference(first, second):
    values = []
    for left, right in zip(first, second):
        if left is None and right is None:
            continue
        values.append(float((left.float() - right.float()).abs().max()))
    return max(values, default=0.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/causal_pilot.json")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    protocol = load_protocol(args.protocol)
    seed = protocol["replicates"]["seeds"][0]
    torch.manual_seed(seed)
    loaded = load_rita(device=args.device, dtype=torch.float32, freeze=True)
    model = attach_rank32_attention_lora(loaded.model, protocol).to(args.device).eval()
    torch.manual_seed(seed + 100_000)
    predictor = FaithfulNextLatPredictor(1024, 1.6).to(args.device)
    sequences = ["ACDEFGHIKLMNPQRSTVWY", "MSTNPKPQRITV"]
    batch = collate_encoded([encode_canonical(loaded.tokenizer, value) for value in sequences])
    batch = {name: value.to(args.device) for name, value in batch.items()}
    model_parameters = [value for _, value in lora_named_parameters(model)]
    predictor_parameters = list(predictor.parameters())

    direct = causal_pilot_loss(model, predictor, batch, "nextlat")
    direct.total.backward()
    direct_model = [None if p.grad is None else p.grad.detach().clone() for p in model_parameters]
    direct_predictor = [None if p.grad is None else p.grad.detach().clone()
                        for p in predictor_parameters]
    model.zero_grad(set_to_none=True)
    predictor.zero_grad(set_to_none=True)
    reference = reference_two_stage_backward(model, predictor, batch)
    reference_model = [None if p.grad is None else p.grad.detach().clone() for p in model_parameters]
    reference_predictor = [None if p.grad is None else p.grad.detach().clone()
                           for p in predictor_parameters]

    output = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                   use_cache=False)
    live_logits = torch.nn.functional.linear(
        output.hidden_states, model.get_output_embeddings().weight
    )
    logit_difference = float((live_logits - output.logits).abs().max())

    rows = [{"id": str(index), "sequence": sequence}
            for index, sequence in enumerate(sequences)]
    batched = bidirectional_summary(
        directional_lm_rows(model, loaded.tokenizer, rows, args.device, batch_size=2)
    )
    fitness_difference = max(abs(
        -(row["forward_ntp_loss"] + row["reverse_ntp_loss"])
        - official_rita_fitness(model, loaded.tokenizer, sequences[int(row["id"])], args.device)
    ) for row in batched)
    features_batch = sequence_representations(
        model, loaded.tokenizer, rows, args.device, batch_size=2
    )
    features_scalar = sequence_representations(
        model, loaded.tokenizer, rows, args.device, batch_size=1
    )
    representation_difference = max(
        float(np.max(np.abs(features_batch[row["id"]]["tokens"]
                            - features_scalar[row["id"]]["tokens"])))
        for row in rows
    )

    prefix = torch.tensor([
        encode_canonical(loaded.tokenizer, sequences[0][:8]).input_ids[:-1]
    ], device=args.device)
    optimized = cached_generate_rita(
        model, prefix, do_sample=False, max_new_tokens=5, eos_token_id=2,
    )
    reference_ids = prefix.clone()
    for _ in range(5):
        next_id = model(input_ids=reference_ids, use_cache=False).logits[:, -1].argmax(-1, keepdim=True)
        reference_ids = torch.cat([reference_ids, next_id], dim=1)
        if int(next_id) == 2:
            break
    generation_equal = bool(torch.equal(optimized, reference_ids))
    cached_generator = torch.Generator(device=args.device).manual_seed(20260915)
    reference_generator = torch.Generator(device=args.device).manual_seed(20260915)
    sampled_cached = cached_generate_rita(
        model, prefix, do_sample=True, max_new_tokens=5, eos_token_id=2,
        temperature=1.0, top_p=.95, generator=cached_generator,
    )
    sampled_reference = prefix.clone()
    for _ in range(5):
        next_logits = model(input_ids=sampled_reference, use_cache=False).logits[:, -1]
        next_id = sample_top_p(next_logits, 1.0, .95, reference_generator)
        sampled_reference = torch.cat((sampled_reference, next_id), dim=1)
        if int(next_id) == 2:
            break
    sampled_generation_equal = bool(torch.equal(sampled_cached, sampled_reference))
    batched_prefix = prefix.expand(2, -1).clone()
    uniforms = torch.tensor([[.13, .31, .57, .79, .91],
                             [.07, .23, .43, .67, .83]], device=args.device)
    batched_sampled = cached_generate_rita(
        model, batched_prefix, do_sample=True, max_new_tokens=5, eos_token_id=2,
        temperature=1.0, top_p=.95, uniforms=uniforms,
    )
    scalar_sampled = []
    for row_index in range(2):
        value = prefix.clone()
        for step in range(5):
            logits = model(input_ids=value, use_cache=False).logits[:, -1]
            next_id = sample_top_p_uniform(logits, uniforms[row_index, step:step + 1], 1.0, .95)
            value = torch.cat((value, next_id), dim=1)
            if int(next_id) == 2:
                break
        if value.shape[1] < batched_sampled.shape[1]:
            value = torch.cat((value, torch.full(
                (1, batched_sampled.shape[1] - value.shape[1]), 2,
                device=args.device, dtype=value.dtype,
            )), dim=1)
        scalar_sampled.append(value)
    batched_sampled_equal = all(
        torch.equal(batched_sampled[index, :value.shape[1]], value[0])
        for index, value in enumerate(scalar_sampled)
    )
    certificate = {
        "real_checkpoint": protocol["model"],
        "optimized_vs_upstream_two_stage": {
            "total_loss_abs": abs(float(direct.total) - float(reference.total)),
            "ntp_abs": abs(float(direct.ntp) - float(reference.ntp)),
            "smooth_l1_abs": abs(float(direct.latent) - float(reference.latent)),
            "kl_abs": abs(float(direct.kl) - float(reference.kl)),
            "lora_gradient_max_abs": maximum_difference(direct_model, reference_model),
            "predictor_gradient_max_abs": maximum_difference(
                direct_predictor, reference_predictor
            ),
        },
        "hidden_to_live_head_logits_max_abs": logit_difference,
        "batched_vs_official_forward_reverse_fitness_max_abs": fitness_difference,
        "batched_vs_scalar_probe_input_max_abs": representation_difference,
        "cached_vs_reference_greedy_generation_exact": generation_equal,
        "cached_vs_reference_sampled_generation_exact": sampled_generation_equal,
        "batched_uniform_cached_vs_scalar_full_prefix_generation_exact": batched_sampled_equal,
        "tolerances": {
            "fp32_loss_abs": 2e-6, "fp32_gradient_abs": 2e-5,
            "fp32_logits_abs": 2e-6, "fitness_abs": 2e-6,
            "probe_input_abs": 2e-6,
        },
    }
    checks = [
        certificate["optimized_vs_upstream_two_stage"][key] <= (
            2e-5 if "gradient" in key else 2e-6
        )
        for key in certificate["optimized_vs_upstream_two_stage"]
    ] + [
        logit_difference <= 2e-6,
        fitness_difference <= 2e-6,
        representation_difference <= 2e-6,
        generation_equal,
        sampled_generation_equal,
        batched_sampled_equal,
    ]
    certificate["passed"] = all(checks)
    if not certificate["passed"]:
        raise AssertionError(certificate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(certificate, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
