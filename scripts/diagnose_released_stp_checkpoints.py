"""Frozen fixed-span released-STP diagnostics for trained ChemFM checkpoints."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chemfm import ReactionCollator  # noqa: E402
from eval_uspto_mit_five_view_a6000 import load_endpoint  # noqa: E402
from stp import SemanticTubePrediction  # noqa: E402


PANEL = ROOT / "data/clm_jepa_uspto_mit_official_endpoint/prespecified_stage1_512.jsonl"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


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
    return {
        "native_gradient_norm": float(native_norm),
        "stp_gradient_norm": float(auxiliary_norm),
        "stp_to_native_norm_ratio": float(auxiliary_norm / native_norm.clamp_min(1e-30)),
        "stp_native_gradient_cosine": float(
            dot / native_norm.clamp_min(1e-30) / auxiliary_norm.clamp_min(1e-30)
        ),
    }


def summary(rows: list[dict]) -> dict:
    output = {}
    for key in (
        "native_loss", "released_stp_loss", "span_fraction",
        "native_gradient_norm", "stp_gradient_norm",
        "stp_to_native_norm_ratio", "stp_native_gradient_cosine",
    ):
        values = [row[key] for row in rows]
        output[key] = {
            "mean": statistics.fmean(values),
            "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values), "max": max(values),
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reactions", type=int, default=16)
    parser.add_argument("--sampler-seed", type=int, default=4242)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    model, tokenizer = load_endpoint(args.checkpoint)
    collator = ReactionCollator(tokenizer)
    panel = read_jsonl(PANEL)[:args.reactions]
    method = SemanticTubePrediction(
        seed=args.sampler_seed,
        reactant_start_token_id=tokenizer.convert_tokens_to_ids("<rstart>"),
        product_start_token_id=tokenizer.convert_tokens_to_ids("<prostart>"),
        eos_token_id=tokenizer.eos_token_id,
    )
    parameters = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    rows = []
    torch.cuda.reset_peak_memory_stats()
    for panel_row in panel:
        raw = collator([{
            "src": panel_row["sources"][0], "tgt": panel_row["targets"][0]
        }])
        batch = {
            key: value.cuda() for key, value in raw.items() if torch.is_tensor(value)
        }
        output = method(model, batch, stp_weight=0.02)
        native = torch.autograd.grad(
            output.native_loss, parameters, retain_graph=True, allow_unused=True
        )
        auxiliary = torch.autograd.grad(
            output.jepa_loss, parameters, retain_graph=False, allow_unused=True
        )
        start, end, full = output.sampled_spans[0]
        rows.append({
            "reaction_identity": panel_row["reaction_identity"],
            "native_loss": float(output.native_loss.detach()),
            "released_stp_loss": float(output.jepa_loss.detach()),
            "released_transition_cosine": 1.0 - float(output.jepa_loss.detach()),
            "sampled_span": output.sampled_spans[0],
            "span_fraction": (end - start) / full,
            **gradient_metrics(native, auxiliary),
        })
        model.zero_grad(set_to_none=True)

    payload = {
        "type": "fixed_span_released_stp_checkpoint_diagnostic",
        "checkpoint": str(args.checkpoint.resolve()),
        "adapter": model._chemfm_adapter_metadata,
        "reactions": args.reactions, "sampler_seed": args.sampler_seed,
        "model_training": False, "view": 0,
        "gradient_scope": "all trainable LoRA plus modules_to_save parameters",
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()),
        "summary": summary(rows), "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"]))


if __name__ == "__main__":
    main()
