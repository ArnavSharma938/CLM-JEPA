from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, norm


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(path)


def prepare(args) -> None:
    rows = read_jsonl(args.source_manifest)
    if len(rows) != args.maximum_n:
        raise ValueError(f"expected {args.maximum_n} frozen reactions, found {len(rows)}")
    if len({row["reaction_identity"] for row in rows}) != len(rows):
        raise ValueError("source manifest contains duplicate reaction identities")
    source_hash = sha256(args.source_manifest)
    random.Random(args.order_seed).shuffle(rows)
    for sequential_index, row in enumerate(rows):
        row["frozen_panel_index"] = row["panel_index"]
        row["panel_index"] = sequential_index
        row["sequential_index"] = sequential_index
    write_jsonl(args.ordered_manifest, rows)
    write_jsonl(args.stage1_manifest, rows[: args.stage1_n])
    metadata = {
        "created_before_primary_endpoint_inference": True,
        "source_manifest": str(args.source_manifest.resolve()),
        "source_manifest_sha256": source_hash,
        "ordered_manifest": str(args.ordered_manifest.resolve()),
        "ordered_manifest_sha256": sha256(args.ordered_manifest),
        "stage1_manifest": str(args.stage1_manifest.resolve()),
        "stage1_manifest_sha256": sha256(args.stage1_manifest),
        "order": "deterministic random permutation of the already frozen 3,300-reaction sample",
        "order_seed": args.order_seed,
        "primary_endpoint": "paired exact top-1 under official five-view beam-10 ChemFM evaluation",
        "minimum_effect_of_interest": 0.01,
        "maximum_n": args.maximum_n,
        "stage1_n": args.stage1_n,
        "stage1_futility_rule": (
            "stop if the two-sided 99% Wald upper confidence bound for paired "
            "JEPA-minus-native top-1 difference is strictly below +0.01"
        ),
        "otherwise": "continue the unchanged ordered manifest to maximum_n",
        "final_test": "two-sided exact McNemar alpha=0.05",
        "power_calibration": {
            "pilot_discordant_pairs": 4,
            "pilot_n": 256,
            "conservative_discordance": 0.039520756533374946,
            "conservative_bound": "upper endpoint of the pre-existing two-sided 95% Clopper-Pearson interval",
            "true_difference": 0.01,
            "simulation_seed": 533,
            "simulation_repetitions": 300000,
            "unadjusted_maximum_n_power": 0.8055933333333334,
            "sequential_power_with_futility": 0.80457,
            "probability_of_stage1_stop_under_boundary_alternative": 0.006196666666666666,
            "monte_carlo_standard_error_at_power_0_805": 0.000723,
        },
        "scientific_invariance": (
            "stopping design changes neither checkpoint, identities, five views, "
            "beam settings, generation, ranking, nor final paired test"
        ),
    }
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, sort_keys=True))


def top1(row: dict) -> bool:
    return bool(row["ranked_candidates"]) and row["ranked_candidates"][0] == row["target"]


def interim(args) -> None:
    manifest = read_jsonl(args.manifest)
    native = read_jsonl(args.native_predictions)
    clm = read_jsonl(args.clm_predictions)
    expected = [row["reaction_identity"] for row in manifest]
    if [row["reaction_identity"] for row in native] != expected:
        raise ValueError("native predictions do not exactly follow the stage manifest")
    if [row["reaction_identity"] for row in clm] != expected:
        raise ValueError("cLM-JEPA predictions do not exactly follow the stage manifest")
    native_correct = np.asarray([top1(row) for row in native], dtype=np.int8)
    clm_correct = np.asarray([top1(row) for row in clm], dtype=np.int8)
    differences = clm_correct - native_correct
    estimate = float(differences.mean())
    variance = float(differences.var(ddof=1))
    standard_error = math.sqrt(variance / len(differences))
    critical = float(norm.ppf(0.995))
    interval = [estimate - critical * standard_error, estimate + critical * standard_error]
    native_only = int(np.sum((native_correct == 1) & (clm_correct == 0)))
    clm_only = int(np.sum((native_correct == 0) & (clm_correct == 1)))
    discordant = native_only + clm_only
    p_value = 1.0 if discordant == 0 else float(
        binomtest(clm_only, discordant, 0.5, alternative="two-sided").pvalue
    )
    result = {
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256(args.manifest),
        "n": len(manifest),
        "native_top1": float(native_correct.mean()),
        "clm_jepa_top1": float(clm_correct.mean()),
        "paired_difference": estimate,
        "paired_wald_99_ci": interval,
        "both_correct": int(np.sum((native_correct == 1) & (clm_correct == 1))),
        "native_only_correct": native_only,
        "clm_jepa_only_correct": clm_only,
        "neither_correct": int(np.sum((native_correct == 0) & (clm_correct == 0))),
        "exact_mcnemar_two_sided_p_descriptive_at_interim": p_value,
        "futility_threshold": 0.01,
        "decision": "STOP_FOR_FUTILITY" if interval[1] < 0.01 else "CONTINUE_TO_3300",
        "decision_interpretation": (
            "The prespecified 99% upper bound excludes the +1 pp effect of interest."
            if interval[1] < 0.01 else
            "The prespecified 99% upper bound does not exclude the +1 pp effect of interest."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prep = subparsers.add_parser("prepare")
    prep.add_argument("--source-manifest", type=Path, required=True)
    prep.add_argument("--ordered-manifest", type=Path, required=True)
    prep.add_argument("--stage1-manifest", type=Path, required=True)
    prep.add_argument("--metadata", type=Path, required=True)
    prep.add_argument("--maximum-n", type=int, default=3300)
    prep.add_argument("--stage1-n", type=int, default=1280)
    prep.add_argument("--order-seed", type=int, default=533)
    prep.set_defaults(function=prepare)
    check = subparsers.add_parser("interim")
    check.add_argument("--manifest", type=Path, required=True)
    check.add_argument("--native-predictions", type=Path, required=True)
    check.add_argument("--clm-predictions", type=Path, required=True)
    check.add_argument("--output", type=Path, required=True)
    check.set_defaults(function=interim)
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    parsed.function(parsed)
