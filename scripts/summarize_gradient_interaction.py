"""Aggregate the frozen gradient-interaction training and evaluation artifacts."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


LABELS = (
    "lambda_025", "lambda_05", "lambda_10", "lambda_20",
    "pcgrad", "cagrad", "aux_similarity",
)


def read(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def numeric_summary(values) -> dict[str, float | int]:
    values = [float(value) for value in values]
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "minimum": ordered[0],
        "maximum": ordered[-1],
    }


def training_summary(payload: dict) -> dict:
    active = [row for row in payload["curves"] if row["jepa_active"]]
    interactions = [row["gradient_interaction"] for row in active]
    by_epoch = {}
    for epoch in (1, 2, 3, 4):
        rows = [row for row in active if row["epoch"] == epoch]
        by_epoch[str(epoch)] = {
            "active_updates": len(rows),
            "cosine": numeric_summary(
                row["gradient_interaction"]["cosine"] for row in rows
            ),
            "conflict_fraction": statistics.fmean(
                float(row["gradient_interaction"]["conflict"]) for row in rows
            ),
            "raw_auxiliary_to_main_norm_ratio": numeric_summary(
                row["gradient_interaction"]["raw_auxiliary_to_main_norm_ratio"]
                for row in rows
            ),
            "applied_auxiliary_to_main_norm_ratio": numeric_summary(
                row["gradient_interaction"]["auxiliary_to_main_norm_ratio"]
                for row in rows
            ),
            "modification_relative_to_raw_sum": numeric_summary(
                row["gradient_interaction"]["modification_relative_to_raw_sum"]
                for row in rows
            ),
        }
    return {
        "gradient_interaction": payload["config"]["gradient_interaction"],
        "lambda_eff": payload["config"]["lambda_eff"],
        "actual_lambda": payload["config"]["actual_lambda"],
        "selected_epoch": payload["selected_epoch"],
        "internal_validation_native_loss": payload["validation_native_loss"],
        "internal_validation_metrics": payload["validation_metrics"],
        "compute": payload["compute"],
        "active_updates": len(active),
        "gradient_diagnostics": {
            "cosine": numeric_summary(row["cosine"] for row in interactions),
            "conflict_fraction": statistics.fmean(
                float(row["conflict"]) for row in interactions
            ),
            "raw_auxiliary_to_main_norm_ratio": numeric_summary(
                row["raw_auxiliary_to_main_norm_ratio"] for row in interactions
            ),
            "applied_auxiliary_to_main_norm_ratio": numeric_summary(
                row["auxiliary_to_main_norm_ratio"] for row in interactions
            ),
            "modification_relative_to_raw_sum": numeric_summary(
                row["modification_relative_to_raw_sum"] for row in interactions
            ),
            "auxiliary_gate": numeric_summary(
                row["auxiliary_gate"] for row in interactions
                if row["auxiliary_gate"] is not None
            ),
        },
        "gradient_diagnostics_by_epoch": by_epoch,
    }


def representation_summary(payload: dict) -> dict:
    condition = next(
        value for name, value in payload["conditions"].items()
        if name != "pretrained"
    )
    metrics = condition["metrics"]
    keys = (
        "source_variance", "target_variance", "pair_center_spread",
        "source_effective_rank", "target_effective_rank",
        "source_mean_direction_energy", "target_mean_direction_energy",
        "correct_cosine", "matched_shuffle_cosine", "correct_minus_matched",
        "retrieval_top1", "retrieval_mrr", "ridge_explained_variance",
        "pca_structure", "residual_pc2",
    )
    return {key: metrics[key] for key in keys if key in metrics}


def decoder_summary(payload: dict, label: str) -> dict:
    generation = payload["generation"]
    comparison = generation[label]
    cross_entropy = payload["cross_entropy"]
    comparison_ce_key = next(
        key for key in cross_entropy
        if key.endswith("_aggregate_target_token_ce") and not key.startswith("native_")
    )
    return {
        "generation": comparison,
        "paired": generation["paired"],
        "native_aggregate_target_token_ce": (
            cross_entropy["native_aggregate_target_token_ce"]
        ),
        "comparison_aggregate_target_token_ce": cross_entropy[comparison_ce_key],
        "relative_aggregate_improvement": (
            cross_entropy["relative_aggregate_improvement"]
        ),
        "mean_reaction_ce_improvement": (
            cross_entropy["mean_reaction_ce_improvement"]
        ),
        "mean_reaction_ce_improvement_bootstrap_95_ci": (
            cross_entropy["mean_reaction_ce_improvement_bootstrap_95_ci"]
        ),
        "fraction_reactions_improved": cross_entropy["fraction_reactions_improved"],
        "wilcoxon_two_sided_p": cross_entropy["wilcoxon_two_sided_p"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path("runs/gradient_interaction/a6000"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    gradient_checkpoints = {}
    for path in sorted(args.root.glob("gradient_checkpoints*.json")):
        gradient_checkpoints.update(read(path)["conditions"])
    output = {"schema_version": 1, "root": str(args.root), "conditions": {}}
    for label in LABELS:
        condition = {}
        training = read(args.root / "training" / label / "result.json")
        if training:
            condition["training"] = training_summary(training)
        representation = read(args.root / "representation" / f"{label}.json")
        if representation:
            condition["representation"] = representation_summary(representation)
        decoder = read(args.root / "decoder" / f"{label}_summary.json")
        if decoder:
            condition["decoder"] = decoder_summary(decoder, label)
        official = read(args.root / "official" / f"{label}_paired.json")
        if official:
            condition["official_five_view"] = {
                "primary_top1": official["primary_top1"],
                "secondary": official["secondary"],
                "validity": official["validity"],
            }
        if label in gradient_checkpoints:
            condition["heldout_gradient_checkpoints"] = gradient_checkpoints[label]
        output["conditions"][label] = condition
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "conditions": len(output["conditions"])}))


if __name__ == "__main__":
    main()
