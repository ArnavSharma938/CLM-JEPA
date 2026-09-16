#!/usr/bin/env python
"""Transition, LoRA-gradient, and broad-depth mechanism audit for one pair."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.modeling import capture_layer, load_rita
from src.nextlat import FaithfulNextLatPredictor, faithful_losses, transition_diagnostics
from src.pilot import (
    attach_rank32_attention_lora, labels_with_padding_ignored, load_protocol,
    lora_named_parameters, tensor_state_sha256,
)
from src.tokenization import collate_encoded, encode_canonical, next_residue_transition_mask


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def stable_decoder_transition_relative_js(diagnostics, eps=1e-8):
    """Aggregate before dividing so near-zero individual transitions stay stable."""
    return diagnostics["decoder_js"].sum() / (
        diagnostics["decoder_transition_js"].sum() + eps
    )


def depth(name):
    index = int(name.split("transformer.layers.", 1)[1].split(".", 1)[0])
    return "early" if index < 8 else "middle" if index < 16 else "late"


def gradient_geometry(names, ntp, auxiliary):
    cells = {"global": [0.0, 0.0, 0.0]}
    for name, first, second in zip(names, ntp, auxiliary):
        if first is None or second is None:
            continue
        first, second = first.float(), second.float()
        for key in ("global", depth(name)):
            cell = cells.setdefault(key, [0.0, 0.0, 0.0])
            cell[0] += float((first * first).sum())
            cell[1] += float((second * second).sum())
            cell[2] += float((first * second).sum())
    result = {}
    for key, (n2, a2, dot) in cells.items():
        n, a = math.sqrt(n2), math.sqrt(a2)
        result[key] = {
            "norm_ratio": a / max(n, 1e-30),
            "cosine": dot / max(n * a, 1e-30),
            "ntp_direction_retention": 1 + dot / max(n2, 1e-30),
        }
    return result


def bootstrap_mean_ci(values, seed, draws=10_000):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    sampled = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(1)
    return [float(value) for value in np.quantile(sampled, [.025, .975])]


def load_pair(protocol, replicate_root, fraction, replicate, device):
    seed = protocol["replicates"]["seeds"][replicate]
    models = {}
    initial_hashes = {}
    for arm in ("native", "nextlat"):
        torch.manual_seed(seed)
        loaded = load_rita(device=device, dtype=torch.bfloat16, freeze=True)
        model = attach_rank32_attention_lora(loaded.model, protocol).to(device)
        initial_hashes[arm] = tensor_state_sha256(lora_named_parameters(model))
        if fraction:
            adapter = replicate_root / arm / f"checkpoint_{fraction:03d}" / "adapter"
            del model
            loaded = load_rita(device=device, dtype=torch.bfloat16, freeze=True)
            model = PeftModel.from_pretrained(
                loaded.model, adapter, is_trainable=True,
                autocast_adapter_dtype=False,
            ).to(device)
        model.eval()
        models[arm] = model
        tokenizer = loaded.tokenizer
    if initial_hashes["native"] != initial_hashes["nextlat"]:
        raise AssertionError("paired LoRA initialization differs")
    torch.manual_seed(seed + 100_000)
    predictor = FaithfulNextLatPredictor(1024, 1.6).to(device)
    if fraction:
        payload = torch.load(
            replicate_root / "nextlat" / f"checkpoint_{fraction:03d}" / "predictor.pt",
            map_location=device, weights_only=True,
        )
        predictor.load_state_dict(payload["state_dict"])
    predictor.eval()
    for parameter in predictor.parameters():
        parameter.requires_grad_(False)
    return tokenizer, models, predictor, initial_hashes["native"]


def states_and_layers(model, batch):
    captured = {}
    with torch.autocast("cuda", dtype=torch.bfloat16,
                        enabled=batch["input_ids"].device.type == "cuda"):
        with capture_layer(model, 7) as early, capture_layer(model, 15) as middle, capture_layer(model, 23) as late:
            output = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                           use_cache=False)
    for name, values in (("early", early), ("middle", middle), ("late", late)):
        value = values[0]
        captured[name] = value[0] if isinstance(value, tuple) else value
    captured["final"] = output.hidden_states
    return output, captured


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("replicate_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--replicate", type=int, choices=range(8), required=True)
    parser.add_argument("--fraction", type=int, choices=(0, 25, 50, 100), required=True)
    parser.add_argument("--data", type=Path, default=ROOT / "data/causal_pilot")
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/causal_pilot.json")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    protocol = load_protocol(args.protocol)
    tokenizer, models, predictor, init_hash = load_pair(
        protocol, args.replicate_root, args.fraction, args.replicate, args.device
    )
    rows = read_jsonl(args.data / "mechanism_set.jsonl")
    loaded_base = load_rita(device=args.device, dtype=torch.bfloat16, freeze=True)
    base = loaded_base.model.eval()
    records = []
    for row in rows:
        encoded = encode_canonical(tokenizer, row["sequence"])
        batch = collate_encoded([encoded])
        batch = {key: value.to(args.device) for key, value in batch.items()}
        _, base_layers = states_and_layers(base, batch)
        arm_values = {}
        for arm, model in models.items():
            output, layers = states_and_layers(model, batch)
            mask = next_residue_transition_mask(batch["input_ids"], batch["attention_mask"])
            current, future = layers["final"][:, :-1], layers["final"][:, 1:]
            embedding = model.get_input_embeddings()(batch["input_ids"][:, 1:])
            with torch.autocast("cuda", dtype=torch.bfloat16,
                                enabled=batch["input_ids"].device.type == "cuda"):
                latent, kl, predicted = faithful_losses(
                    predictor, current, future, embedding, mask,
                    model.get_output_embeddings().weight,
                )
            center = future[mask].float().mean(0)
            diagnostics = transition_diagnostics(
                current[mask], future[mask], predicted[mask], center,
                model.get_output_embeddings().weight,
            )
            arm_values[arm] = {
                "smooth_l1": float(latent), "kl": float(kl),
                "faithful_total": float(latent + kl),
                "decoder_transition_relative_js": float(
                    stable_decoder_transition_relative_js(diagnostics)
                ),
                "decoder_top1_agreement": float(
                    F.linear(predicted[mask].float(), model.get_output_embeddings().weight.float()).argmax(-1)
                    .eq(F.linear(future[mask].float(), model.get_output_embeddings().weight.float()).argmax(-1))
                    .float().mean()
                ),
                "drift": {
                    name: float((layers[name][:, :len(row["sequence"])].float()
                                 - base_layers[name][:, :len(row["sequence"])].float())
                                .square().mean().sqrt())
                    for name in ("early", "middle", "late")
                },
            }
        model = models["nextlat"]
        names, parameters = zip(*lora_named_parameters(model))
        with torch.autocast("cuda", dtype=torch.bfloat16,
                            enabled=batch["input_ids"].device.type == "cuda"):
            output = model(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                labels=labels_with_padding_ignored(batch), use_cache=False,
            )
        ntp_grad = torch.autograd.grad(output.loss, parameters)
        with torch.autocast("cuda", dtype=torch.bfloat16,
                            enabled=batch["input_ids"].device.type == "cuda"):
            output = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                           use_cache=False)
            mask = next_residue_transition_mask(batch["input_ids"], batch["attention_mask"])
            latent, kl, _ = faithful_losses(
                predictor, output.hidden_states[:, :-1], output.hidden_states[:, 1:],
                model.get_input_embeddings()(batch["input_ids"][:, 1:]), mask,
                model.get_output_embeddings().weight,
            )
        auxiliary_grad = torch.autograd.grad(latent + kl, parameters)
        geometry = gradient_geometry(names, ntp_grad, auxiliary_grad)
        records.append({
            "id": row["id"], "cluster_id": row["cluster_id"],
            "arms": arm_values, "nextlat_lora_gradient": geometry,
        })
    result = {
        "replicate": args.replicate, "fraction": args.fraction,
        "initial_lora_sha256": init_hash, "proteins": len(records),
        "records": records,
        "summary": {
            arm: {
                metric: float(np.mean([row["arms"][arm][metric] for row in records]))
                for metric in ("smooth_l1", "kl", "faithful_total",
                               "decoder_transition_relative_js", "decoder_top1_agreement")
            } for arm in ("native", "nextlat")
        },
        "gradient_summary": {
            part: {
                metric: {
                    "mean": float(np.mean([
                        row["nextlat_lora_gradient"][part][metric] for row in records
                    ])),
                    "protein_bootstrap_ci95": bootstrap_mean_ci(
                        [row["nextlat_lora_gradient"][part][metric] for row in records],
                        20260915 + args.replicate * 100 + args.fraction +
                        ("global", "early", "middle", "late").index(part) * 10 +
                        ("norm_ratio", "cosine", "ntp_direction_retention").index(metric),
                    ),
                }
                for metric in ("norm_ratio", "cosine", "ntp_direction_retention")
            } | {
                "negative_cosine_fraction": {
                    "mean": float(np.mean([
                        row["nextlat_lora_gradient"][part]["cosine"] < 0 for row in records
                    ])),
                    "protein_bootstrap_ci95": bootstrap_mean_ci(
                        [row["nextlat_lora_gradient"][part]["cosine"] < 0
                         for row in records],
                        20261915 + args.replicate * 100 + args.fraction +
                        ("global", "early", "middle", "late").index(part),
                    ),
                }
            }
            for part in ("global", "early", "middle", "late")
        },
        "parameters_updated_during_audit": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")


if __name__ == "__main__":
    main()
