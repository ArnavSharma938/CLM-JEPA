"""Localize paired ChemFM teacher-forced token-decision changes."""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chemfm import IGNORE_INDEX, ReactionCollator  # noqa: E402
from eval_uspto_mit_five_view_a6000 import (  # noqa: E402
    file_sha256, load_endpoint, read_jsonl,
)


RANK_THRESHOLDS = (3, 5, 10)
MATERIAL_RANK_DELTA = 5
MATERIAL_MARGIN_DELTA = 0.5


def token_metrics(logits: torch.Tensor, labels: torch.Tensor) -> list[dict]:
    shifted_logits = logits[:, :-1].float()
    shifted_labels = labels[:, 1:]
    rows = []
    for row_logits, row_labels in zip(shifted_logits, shifted_labels):
        active = row_labels.ne(IGNORE_INDEX)
        selected_logits = row_logits[active]
        selected_labels = row_labels[active]
        correct = selected_logits.gather(1, selected_labels[:, None]).squeeze(1)
        ranks = 1 + selected_logits.gt(correct[:, None]).sum(dim=1)
        top_values, top_indices = selected_logits.topk(2, dim=-1)
        best_other = torch.where(
            top_indices[:, 0].eq(selected_labels), top_values[:, 1], top_values[:, 0]
        )
        rows.append({
            "labels": selected_labels.cpu().tolist(),
            "correct_ranks": ranks.cpu().tolist(),
            "correct_margins": (correct - best_other).cpu().tolist(),
        })
    return rows


def collect(checkpoint: Path, manifest: list[dict], batch_size: int) -> tuple[list[dict], dict]:
    model, tokenizer = load_endpoint(checkpoint, predictor_tokens=True)
    collator = ReactionCollator(tokenizer, task="forward")
    flat = []
    identities = []
    for reaction in manifest:
        for view, (source, target) in enumerate(zip(
            reaction["sources"], reaction["targets"]
        )):
            flat.append({"src": source, "tgt": target})
            identities.append((reaction["reaction_identity"], view))
    records = []
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for start in range(0, len(flat), batch_size):
            raw = collator(flat[start:start + batch_size])
            batch = {
                key: value.to(model.device)
                for key, value in raw.items() if torch.is_tensor(value)
            }
            output = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            values = token_metrics(output.logits, batch["labels"])
            for offset, value in enumerate(values):
                identity, view = identities[start + offset]
                records.append({
                    "reaction_identity": identity, "view_index": view, **value
                })
    metadata = {
        "checkpoint": str(checkpoint.resolve()),
        "adapter_sha256": file_sha256(
            checkpoint / "USPTO-MIT-Synthesis/adapter_model.safetensors"
        ),
        "adapter": model._chemfm_adapter_metadata,
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()),
        "gpu": torch.cuda.get_device_name(0),
    }
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return records, metadata


def first_index(flags: list[bool]) -> int | None:
    return next((index for index, value in enumerate(flags) if value), None)


def paired_row(native: dict, treatment: dict) -> dict:
    if native["labels"] != treatment["labels"]:
        raise ValueError("paired teacher-forced target tokens differ")
    left_rank = native["correct_ranks"]
    right_rank = treatment["correct_ranks"]
    left_margin = native["correct_margins"]
    right_margin = treatment["correct_margins"]
    top1 = first_index([
        (left == 1) != (right == 1) for left, right in zip(left_rank, right_rank)
    ])
    threshold = {
        str(cutoff): first_index([
            (left <= cutoff) != (right <= cutoff)
            for left, right in zip(left_rank, right_rank)
        ])
        for cutoff in RANK_THRESHOLDS
    }
    rank_delta = first_index([
        abs(right - left) >= MATERIAL_RANK_DELTA
        for left, right in zip(left_rank, right_rank)
    ])
    margin_delta = first_index([
        abs(right - left) >= MATERIAL_MARGIN_DELTA
        for left, right in zip(left_margin, right_margin)
    ])
    candidates = [top1, rank_delta, margin_delta, *threshold.values()]
    material = min(value for value in candidates if value is not None) if any(
        value is not None for value in candidates
    ) else None
    return {
        "reaction_identity": native["reaction_identity"],
        "view_index": native["view_index"],
        "target_tokens": len(left_rank),
        "first_top1_change": top1,
        "first_rank_threshold_crossing": threshold,
        "first_rank_delta_ge_5": rank_delta,
        "first_abs_margin_delta_ge_0.5": margin_delta,
        "first_material_change": material,
        "mean_correct_rank_delta": statistics.fmean(
            right - left for left, right in zip(left_rank, right_rank)
        ),
        "mean_correct_margin_delta": statistics.fmean(
            right - left for left, right in zip(left_margin, right_margin)
        ),
        "native_correct_tokens": sum(rank == 1 for rank in left_rank),
        "treatment_correct_tokens": sum(rank == 1 for rank in right_rank),
        "native_to_treatment_top1_losses": sum(
            left == 1 and right != 1 for left, right in zip(left_rank, right_rank)
        ),
        "native_to_treatment_top1_gains": sum(
            left != 1 and right == 1 for left, right in zip(left_rank, right_rank)
        ),
    }


def position_summary(rows: list[dict], key: str) -> dict:
    values = [row[key] for row in rows if row[key] is not None]
    return {
        "rows_with_event": len(values),
        "fraction_with_event": len(values) / len(rows) if rows else None,
        "median_zero_based_position": statistics.median(values) if values else None,
        "mean_zero_based_position": statistics.fmean(values) if values else None,
    }


def summarize(rows: list[dict]) -> dict:
    tokens = sum(row["target_tokens"] for row in rows)
    return {
        "rows": len(rows), "target_tokens": tokens,
        "correct_token_rate_difference": sum(
            row["treatment_correct_tokens"] - row["native_correct_tokens"]
            for row in rows
        ) / tokens,
        "top1_token_gains": sum(
            row["native_to_treatment_top1_gains"] for row in rows
        ),
        "top1_token_losses": sum(
            row["native_to_treatment_top1_losses"] for row in rows
        ),
        "mean_correct_rank_delta": statistics.fmean(
            row["mean_correct_rank_delta"] for row in rows
        ),
        "mean_correct_margin_delta": statistics.fmean(
            row["mean_correct_margin_delta"] for row in rows
        ),
        "first_top1_change": position_summary(rows, "first_top1_change"),
        "first_rank_delta_ge_5": position_summary(rows, "first_rank_delta_ge_5"),
        "first_abs_margin_delta_ge_0.5": position_summary(
            rows, "first_abs_margin_delta_ge_0.5"
        ),
        "first_material_change": position_summary(rows, "first_material_change"),
    }


def generation_categories(native_predictions: Path, treatment_predictions: Path) -> dict:
    native = read_jsonl(native_predictions)
    treatment = read_jsonl(treatment_predictions)
    if [row["reaction_identity"] for row in native] != [
        row["reaction_identity"] for row in treatment
    ]:
        raise ValueError("paired generation identities differ")
    categories = {}
    for left, right in zip(native, treatment):
        before, after = bool(left["exact"][0]), bool(right["exact"][0])
        category = (
            "both" if before and after else
            "native_only" if before else
            "treatment_only" if after else "neither"
        )
        categories[left["reaction_identity"]] = category
    return categories


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-checkpoint", type=Path, required=True)
    parser.add_argument("--treatment-checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--native-predictions", type=Path, required=True)
    parser.add_argument("--treatment-predictions", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise EnvironmentError("CUDA is required")
    manifest = read_jsonl(args.manifest)
    native, native_metadata = collect(
        args.native_checkpoint, manifest, args.batch_size
    )
    treatment, treatment_metadata = collect(
        args.treatment_checkpoint, manifest, args.batch_size
    )
    if [(row["reaction_identity"], row["view_index"]) for row in native] != [
        (row["reaction_identity"], row["view_index"]) for row in treatment
    ]:
        raise ValueError("paired teacher-forced rows differ")
    rows = [paired_row(left, right) for left, right in zip(native, treatment)]
    categories = generation_categories(
        args.native_predictions, args.treatment_predictions
    )
    for row in rows:
        row["generation_category"] = categories[row["reaction_identity"]]
    by_category = {
        category: summarize([
            row for row in rows if row["generation_category"] == category
        ])
        for category in ("native_only", "treatment_only", "both", "neither")
        if any(row["generation_category"] == category for row in rows)
    }
    payload = {
        "type": "paired_teacher_forced_token_localization",
        "protocol": {
            "manifest": str(args.manifest.resolve()),
            "manifest_sha256": file_sha256(args.manifest),
            "reactions": len(manifest), "views_per_reaction": 5,
            "rank_thresholds": RANK_THRESHOLDS,
            "material_rank_delta": MATERIAL_RANK_DELTA,
            "material_margin_delta": MATERIAL_MARGIN_DELTA,
            "native": native_metadata, "treatment": treatment_metadata,
        },
        "overall": summarize(rows),
        "by_view": {
            str(view): summarize([row for row in rows if row["view_index"] == view])
            for view in range(5)
        },
        "by_generation_category": by_category,
        "generation_category_counts": dict(Counter(categories.values())),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output.resolve()),
        "overall": payload["overall"],
        "by_generation_category": by_category,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
