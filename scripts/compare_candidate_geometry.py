#!/usr/bin/env python
"""Compare overlapping candidate-geometry rows before/after batching."""

from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import json
import math
from pathlib import Path

import numpy as np


IDENTITY = (
    "checkpoint", "panel_index", "reaction_identity", "view", "beam_rank",
    "aggregate_rank", "candidate", "canonical_candidate", "role", "layer",
)


def rows(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


def numeric_leaves(value, prefix=""):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from numeric_leaves(child, f"{prefix}.{key}" if prefix else key)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from numeric_leaves(child, f"{prefix}[{index}]")
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        yield prefix, float(value)


def run(reference: Path, optimized: Path, output: Path):
    baseline = {
        tuple(row[key] for key in IDENTITY): row
        for row in rows(reference) if row["checkpoint"] == "native_r8_s533"
    }
    compared = 0
    leaves = 0
    maximum = 0.0
    squared = 0.0
    above = {"1e-7": 0, "1e-6": 0, "1e-5": 0, "1e-4": 0}
    missing = []
    by_key = defaultdict(lambda: {"count": 0, "maximum": 0.0, "squared": 0.0, "above_1e4": 0})
    paired_final = []
    for row in rows(optimized):
        identity = tuple(row[key] for key in IDENTITY)
        old = baseline.get(identity)
        if old is None:
            missing.append(identity)
            continue
        old_values = dict(numeric_leaves(old))
        new_values = dict(numeric_leaves(row))
        if row["layer"] == "final_post_norm" and row["role"] in {"gold", "highest_wrong"}:
            def summary_metrics(value):
                rms = value["tube_scale"]["rms"]
                return {
                    "tube_rms_integral": float(np.mean(rms)) if rms else math.nan,
                    "euclidean_inefficiency": 1.0 - float(value["euclidean_path_efficiency"]),
                    "fisher_inefficiency": 1.0 - float(value["fisher_path_efficiency"]),
                    "fisher_local_curvature": float(value["fisher_local_curvature"]),
                    "normal_acceleration": float(value["normalized_normal_acceleration"]),
                }
            paired_final.append((identity, summary_metrics(old), summary_metrics(row)))
        for key in old_values.keys() & new_values.keys():
            # This diagnostic was intentionally corrected from a product-only
            # proxy to the executable released source+product semantic path.
            if key in {"released_loss", "paper_loss"}:
                continue
            first, second = old_values[key], new_values[key]
            if not (math.isfinite(first) and math.isfinite(second)):
                continue
            difference = abs(first - second)
            maximum = max(maximum, difference)
            squared += difference * difference
            leaves += 1
            item = by_key[key]
            item["count"] += 1
            item["maximum"] = max(item["maximum"], difference)
            item["squared"] += difference * difference
            item["above_1e4"] += difference > 1e-4
            for threshold in above:
                above[threshold] += difference > float(threshold)
        compared += 1
    aggregate_stability = {}
    for version_index, version in ((1, "reference"), (2, "optimized")):
        role_values = {}
        for identity, old_summary, new_summary in paired_final:
            summary = old_summary if version_index == 1 else new_summary
            role_values[(identity[1], identity[3], identity[8])] = summary
        contrasts = defaultdict(list)
        for (panel, view, role), wrong in role_values.items():
            if role != "highest_wrong":
                continue
            gold = role_values.get((panel, view, "gold"))
            if gold is None:
                continue
            for metric in wrong:
                if math.isfinite(wrong[metric]) and math.isfinite(gold[metric]):
                    contrasts[metric].append(wrong[metric] - gold[metric])
        aggregate_stability[version] = {
            metric: {"n": len(values), "mean_wrong_minus_gold": float(np.mean(values))}
            for metric, values in contrasts.items()
        }

    payload = {
        "reference": str(reference), "optimized": str(optimized),
        "overlapping_rows": compared, "optimized_rows_missing_reference": len(missing),
        "numeric_leaves": leaves, "maximum_absolute_difference": maximum,
        "rms_absolute_difference": math.sqrt(squared / max(1, leaves)),
        "counts_above_tolerance": above,
        "field_differences": {
            key: {
                "count": value["count"],
                "maximum_absolute_difference": value["maximum"],
                "rms_absolute_difference": math.sqrt(value["squared"] / max(1, value["count"])),
                "count_above_1e4": value["above_1e4"],
            }
            for key, value in sorted(
                by_key.items(), key=lambda pair: pair[1]["maximum"], reverse=True
            )
        },
        "aggregate_wrong_minus_gold_stability": aggregate_stability,
        "intentional_exclusions": [
            "released_loss (corrected semantic path and bounded fixed spans)",
            "paper_loss (bounded fixed spans)",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("optimized", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.reference, args.optimized, args.output)
