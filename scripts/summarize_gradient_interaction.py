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

BEHAVIORAL_LABELS = (
    "native", "direct_mse_sigreg", "lambda_025",
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


def jsonl_identities(path: Path) -> tuple[list[str], list[int]]:
    identities = []
    panel_indices = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            identities.append(row["reaction_identity"])
            panel_indices.append(int(row["panel_index"]))
    return identities, panel_indices


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path("runs/gradient_interaction/a6000"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "data/clm_jepa_uspto_mit_official_endpoint/"
            "prespecified_stage1_256.jsonl"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    gradient_checkpoints = {}
    for path in sorted(args.root.glob("gradient_checkpoints*.json")):
        gradient_checkpoints.update(read(path)["conditions"])
    output = {"schema_version": 2, "root": str(args.root), "conditions": {}}
    for label in LABELS:
        condition = {}
        training = read(args.root / "training" / label / "result.json")
        if training:
            condition["training"] = training_summary(training)
        representation = read(args.root / "representation" / f"{label}.json")
        if representation:
            condition["representation"] = representation_summary(representation)
        decoder = read(
            args.root / "endpoint_256" / "decoder" / f"{label}_summary.json"
        ) or read(args.root / "decoder" / f"{label}_summary.json")
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

    endpoint_root = args.root / "endpoint_256"
    if endpoint_root.exists():
        manifest_identities, manifest_indices = jsonl_identities(args.manifest)
        if manifest_indices != list(range(256)):
            raise ValueError("endpoint manifest is not ordered panel_index 0..255")
        if len(set(manifest_identities)) != 256:
            raise ValueError("endpoint manifest does not contain 256 unique reactions")

        prediction_paths = {
            "native": endpoint_root / "references" / "native_predictions.jsonl",
            "direct_mse_sigreg": (
                endpoint_root / "references" / "direct_mse_sigreg_predictions.jsonl"
            ),
            "lambda_025": (
                endpoint_root / "references" / "lambda_025_predictions.jsonl"
            ),
            "pcgrad": endpoint_root / "official" / "pcgrad" / "predictions.jsonl",
            "cagrad": endpoint_root / "official" / "cagrad" / "predictions.jsonl",
            "aux_similarity": (
                endpoint_root / "official" / "aux_similarity" / "predictions.jsonl"
            ),
        }
        identity_validation = {}
        for label, path in prediction_paths.items():
            identities, panel_indices = jsonl_identities(path)
            exact_order = (
                identities == manifest_identities and panel_indices == manifest_indices
            )
            if not exact_order:
                raise ValueError(f"{label} predictions do not match endpoint manifest")
            identity_validation[label] = {
                "rows": len(identities),
                "unique_reaction_identities": len(set(identities)),
                "exact_manifest_order": exact_order,
            }

        behavioral = {}
        paired_names = {
            "direct_mse_sigreg": "direct",
            "lambda_025": "lambda_025",
            "pcgrad": "pcgrad",
            "cagrad": "cagrad",
            "aux_similarity": "aux_similarity",
        }
        for label, filename_label in paired_names.items():
            row = {}
            vs_native = read(
                endpoint_root / "paired" / f"{filename_label}_vs_native.json"
            )
            if vs_native:
                row["official_five_view_vs_native"] = {
                    "primary_top1": vs_native["primary_top1"],
                    "secondary": vs_native["secondary"],
                    "validity": vs_native["validity"],
                }
            if label not in ("direct_mse_sigreg",):
                vs_direct = read(
                    endpoint_root / "paired" / f"{filename_label}_vs_direct.json"
                )
                if vs_direct:
                    row["official_five_view_vs_direct"] = {
                        "primary_top1": vs_direct["primary_top1"],
                        "secondary": vs_direct["secondary"],
                        "validity": vs_direct["validity"],
                    }
            official_runtime = read(
                endpoint_root / "official" / filename_label / "summary.json"
            )
            if official_runtime:
                row["official_runtime"] = official_runtime
            behavioral[label] = row

        # Native metrics are the baseline side of every paired file.  Keep one
        # explicit native row so downstream reporting does not need to infer it.
        direct_pair = read(endpoint_root / "paired" / "direct_vs_native.json")
        behavioral["native"] = {
            "official_five_view": {
                "top1_accuracy": direct_pair["primary_top1"]["native_accuracy"],
                "top3_accuracy": direct_pair["secondary"]["top3"]["native_accuracy"],
                "top5_accuracy": direct_pair["secondary"]["top5"]["native_accuracy"],
                "top10_accuracy": direct_pair["secondary"]["top10"]["native_accuracy"],
                "validity": direct_pair["validity"]["native"],
            }
        }
        output["endpoint_256"] = {
            "manifest": str(args.manifest),
            "reactions": 256,
            "behavioral_labels": list(BEHAVIORAL_LABELS),
            "identity_validation": identity_validation,
            "conditions": behavioral,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "conditions": len(output["conditions"])}))


if __name__ == "__main__":
    main()
