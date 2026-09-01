"""Frozen-batch released-vs-paper STP gradient and span diagnostics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chemfm import (  # noqa: E402
    MODEL_DIR, TOKENIZER_DIR, ReactionCollator, load_lora_model,
    load_reaction_tokenizer,
)
from jepa import add_predictor_tokens  # noqa: E402
from stp import PaperSemanticTubePrediction, SemanticTubePrediction  # noqa: E402


TRAIN = ROOT / "data/clm_jepa_uspto_mit_pilot_1280/uspto_mit_train.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def gradients(loss, parameters, *, retain_graph: bool):
    return torch.autograd.grad(
        loss, parameters, retain_graph=retain_graph, allow_unused=True
    )


def gradient_metrics(native, auxiliary) -> dict[str, float]:
    native_sq = torch.zeros((), device=native[0].device)
    auxiliary_sq = torch.zeros_like(native_sq)
    dot = torch.zeros_like(native_sq)
    for left, right in zip(native, auxiliary):
        if left is not None:
            native_sq += left.float().square().sum()
        if right is not None:
            auxiliary_sq += right.float().square().sum()
        if left is not None and right is not None:
            dot += (left.float() * right.float()).sum()
    native_norm = native_sq.sqrt()
    auxiliary_norm = auxiliary_sq.sqrt()
    cosine = dot / native_norm.clamp_min(1e-30) / auxiliary_norm.clamp_min(1e-30)
    return {
        "native_gradient_norm": float(native_norm),
        "stp_gradient_norm": float(auxiliary_norm),
        "stp_to_native_norm_ratio": float(auxiliary_norm / native_norm.clamp_min(1e-30)),
        "stp_native_gradient_cosine": float(cosine),
    }


def span_fraction(span: tuple[int, ...]) -> float:
    if len(span) == 3:
        return (span[1] - span[0]) / span[2]
    return (span[2] - span[0]) / span[3]


def summarize(rows: list[dict]) -> dict:
    output = {}
    for key in (
        "native_loss", "stp_loss", "native_gradient_norm", "stp_gradient_norm",
        "stp_to_native_norm_ratio", "stp_native_gradient_cosine", "mean_span_fraction",
    ):
        values = [row[key] for row in rows]
        output[key] = {
            "mean": float(np.mean(values)), "sd": float(np.std(values, ddof=1)),
            "min": float(np.min(values)), "max": float(np.max(values)),
        }
    output["span_fraction_vs_stp_loss_pearson"] = float(np.corrcoef(
        [row["mean_span_fraction"] for row in rows],
        [row["stp_loss"] for row in rows],
    )[0, 1])
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank", type=int, choices=(8, 128), required=True)
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=533)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    tokenizer = load_reaction_tokenizer(TOKENIZER_DIR)
    chemfm_vocab_size = len(tokenizer)
    add_predictor_tokens(tokenizer)
    collator = ReactionCollator(tokenizer)
    with TRAIN.open(newline="", encoding="utf-8") as handle:
        source_rows = list(csv.DictReader(handle))[:args.rows]
    rows = [{"src": row["source"], "tgt": row["target"]} for row in source_rows]
    loader = DataLoader(rows, batch_size=args.batch_size, shuffle=False, collate_fn=collator)
    model = load_lora_model(
        MODEL_DIR, tokenizer, chemfm_vocab_size=chemfm_vocab_size,
        attn_implementation="sdpa", lora_rank=args.rank, lora_alpha=args.rank,
    ).cuda().eval()
    parameters = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    methods = {
        "released": SemanticTubePrediction(
            seed=args.seed,
            reactant_start_token_id=tokenizer.convert_tokens_to_ids("<rstart>"),
            product_start_token_id=tokenizer.convert_tokens_to_ids("<prostart>"),
            eos_token_id=tokenizer.eos_token_id,
        ),
        "paper": PaperSemanticTubePrediction(
            seed=args.seed,
            reactant_start_token_id=tokenizer.convert_tokens_to_ids("<rstart>"),
            product_start_token_id=tokenizer.convert_tokens_to_ids("<prostart>"),
            eos_token_id=tokenizer.eos_token_id,
        ),
    }
    records = {name: [] for name in methods}
    torch.cuda.reset_peak_memory_stats()
    for batch_index, raw in enumerate(loader):
        batch = {
            key: value.cuda(non_blocking=True)
            for key, value in raw.items() if torch.is_tensor(value)
        }
        for name, method in methods.items():
            output = method(model, batch, stp_weight=0.02)
            native_gradients = gradients(output.native_loss, parameters, retain_graph=True)
            auxiliary_gradients = gradients(output.jepa_loss, parameters, retain_graph=False)
            row = {
                "batch_index": batch_index,
                "native_loss": float(output.native_loss.detach()),
                "stp_loss": float(output.jepa_loss.detach()),
                "mean_span_fraction": float(np.mean([
                    span_fraction(span) for span in output.sampled_spans
                ])),
                "sampled_spans": output.sampled_spans,
                **gradient_metrics(native_gradients, auxiliary_gradients),
            }
            records[name].append(row)
            model.zero_grad(set_to_none=True)

    payload = {
        "type": "frozen_stp_objective_comparison",
        "model": str(MODEL_DIR), "rank": args.rank, "alpha": args.rank,
        "seed": args.seed, "rows": args.rows, "batch_size": args.batch_size,
        "train_manifest": str(TRAIN), "train_manifest_sha256": sha256(TRAIN),
        "model_training": False, "model_mode": "eval", "stp_coefficient": 0.02,
        "gradient_scope": "all trainable LoRA plus modules_to_save parameters",
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()),
        "conditions": {
            name: {"summary": summarize(values), "batches": values}
            for name, values in records.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({name: value["summary"] for name, value in payload["conditions"].items()}))


if __name__ == "__main__":
    main()
