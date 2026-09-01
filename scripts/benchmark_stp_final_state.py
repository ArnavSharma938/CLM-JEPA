"""Benchmark exact final-state capture against all-layer hidden-state output."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chemfm import (  # noqa: E402
    MODEL_DIR, TOKENIZER_DIR, ReactionCollator, load_lora_model,
    load_reaction_tokenizer,
)
from jepa import add_predictor_tokens  # noqa: E402
from stp import SemanticTubePrediction  # noqa: E402


TRAIN = ROOT / "data/clm_jepa_uspto_mit_pilot_1280/uspto_mit_train.csv"


def reference_call(model, batch, method, weight):
    outputs = model(**batch, output_hidden_states=True)
    hidden = outputs.hidden_states[-1]
    user, assistant = method.content_boundaries(batch)
    left = torch.zeros((hidden.shape[0], hidden.shape[-1]), device=hidden.device)
    right = torch.zeros_like(left)
    spans = []
    for index in range(hidden.shape[0]):
        full = int(
            user[index, 1] - user[index, 0]
            + assistant[index, 1] - assistant[index, 0]
        )
        start, end = method.get_s_t(full, device=hidden.device)
        before, patch, after = method.get_embeddings(
            hidden[index], user[index], assistant[index], start, end
        )
        left[index] = before + after
        right[index] = patch
        spans.append((int(start), int(end), full))
    stp_loss = 1.0 - F.cosine_similarity(left, right, dim=-1).mean()
    return outputs.loss + weight * stp_loss, outputs.loss, stp_loss, tuple(spans)


def make_method(tokenizer, seed):
    return SemanticTubePrediction(
        seed=seed,
        reactant_start_token_id=tokenizer.convert_tokens_to_ids("<rstart>"),
        product_start_token_id=tokenizer.convert_tokens_to_ids("<prostart>"),
        eos_token_id=tokenizer.eos_token_id,
    )


def run_mode(model, batch, tokenizer, *, mode, seed, iterations, warmup):
    times = []
    peaks = []
    for iteration in range(warmup + iterations):
        model.zero_grad(set_to_none=True)
        torch.manual_seed(seed + iteration)
        method = make_method(tokenizer, seed + iteration)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        if mode == "reference":
            loss, _, _, _ = reference_call(model, batch, method, 0.02)
        else:
            loss = method(model, batch, stp_weight=0.02).loss
        loss.backward()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        if iteration >= warmup:
            times.append(elapsed)
            peaks.append(torch.cuda.max_memory_allocated())
    return {
        "times_seconds": times,
        "mean_seconds": statistics.fmean(times),
        "median_seconds": statistics.median(times),
        "peak_cuda_bytes": max(peaks),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank", type=int, choices=(8, 128), default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument(
        "--gradient-checkpointing", action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    torch.manual_seed(533)
    tokenizer = load_reaction_tokenizer(TOKENIZER_DIR)
    chemfm_vocab_size = len(tokenizer)
    add_predictor_tokens(tokenizer)
    with TRAIN.open(newline="", encoding="utf-8") as handle:
        csv_rows = list(csv.DictReader(handle))[:args.batch_size]
    batch = ReactionCollator(tokenizer)([
        {"src": row["source"], "tgt": row["target"]} for row in csv_rows
    ])
    batch = {
        key: value.cuda() for key, value in batch.items() if torch.is_tensor(value)
    }
    model = load_lora_model(
        MODEL_DIR, tokenizer, chemfm_vocab_size=chemfm_vocab_size,
        attn_implementation="sdpa", lora_rank=args.rank, lora_alpha=args.rank,
    ).cuda().train()
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.enable_input_require_grads()

    # One strict trajectory-equivalence check with identical model/dropout and
    # dedicated-sampler RNG states.
    model.zero_grad(set_to_none=True)
    torch.manual_seed(911)
    reference_method = make_method(tokenizer, 1907)
    reference = reference_call(model, batch, reference_method, 0.02)
    reference[0].backward()
    reference_gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is not None
    }
    model.zero_grad(set_to_none=True)
    torch.manual_seed(911)
    optimized_method = make_method(tokenizer, 1907)
    optimized = optimized_method(model, batch, stp_weight=0.02)
    optimized.loss.backward()
    maximum_gradient_difference = 0.0
    for name, parameter in model.named_parameters():
        if name in reference_gradients:
            maximum_gradient_difference = max(
                maximum_gradient_difference,
                float((parameter.grad - reference_gradients[name]).abs().max()),
            )
    equivalence = {
        "spans_identical": reference[3] == optimized.sampled_spans,
        "total_loss_absolute_difference": float((reference[0] - optimized.loss).abs()),
        "native_loss_absolute_difference": float((reference[1] - optimized.native_loss).abs()),
        "stp_loss_absolute_difference": float((reference[2] - optimized.jepa_loss).abs()),
        "maximum_trainable_gradient_absolute_difference": maximum_gradient_difference,
    }
    del reference_gradients
    model.zero_grad(set_to_none=True)

    reference_timing = run_mode(
        model, batch, tokenizer, mode="reference", seed=3001,
        iterations=args.iterations, warmup=args.warmup,
    )
    capture_timing = run_mode(
        model, batch, tokenizer, mode="capture", seed=3001,
        iterations=args.iterations, warmup=args.warmup,
    )
    payload = {
        "rank": args.rank, "alpha": args.rank, "batch_size": args.batch_size,
        "tokens": int(batch["attention_mask"].sum()),
        "gradient_checkpointing": args.gradient_checkpointing,
        "iterations": args.iterations, "warmup": args.warmup,
        "equivalence": equivalence,
        "reference_all_hidden_states": reference_timing,
        "optimized_final_norm_capture": capture_timing,
        "speedup": reference_timing["mean_seconds"] / capture_timing["mean_seconds"],
        "peak_cuda_bytes_saved": (
            reference_timing["peak_cuda_bytes"] - capture_timing["peak_cuda_bytes"]
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
