#!/usr/bin/env python
"""Reaction-clustered uncertainty for the STP signal/noise audit."""

from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def span_bin(length: int) -> str:
    if length == 2:
        return "2"
    if length <= 8:
        return "3_8"
    if length <= 24:
        return "9_24"
    return "25_plus"


def accumulator() -> dict[str, float]:
    return defaultdict(float)


def bootstrap_mean(values: np.ndarray, seed: int = 260222617) -> list[float]:
    if len(values) < 2:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    estimates = []
    for start in range(0, 10_000, 250):
        indices = rng.integers(0, len(values), size=(min(250, 10_000 - start), len(values)))
        estimates.extend(values[indices].mean(axis=1))
    return np.quantile(estimates, [.025, .975]).tolist()


def bootstrap_ratio(numerator: np.ndarray, denominator: np.ndarray, seed: int = 260222617) -> list[float]:
    if len(numerator) < 2:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    estimates = []
    for start in range(0, 10_000, 250):
        indices = rng.integers(0, len(numerator), size=(min(250, 10_000 - start), len(numerator)))
        estimates.extend(numerator[indices].mean(axis=1) / denominator[indices].mean(axis=1))
    return np.quantile(estimates, [.025, .975]).tolist()


def run(source: Path, output: Path) -> None:
    sensitivity = defaultdict(accumulator)
    interventions = defaultdict(accumulator)
    with gzip.open(source, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            common = (row["checkpoint"], row["segment"], span_bin(int(row["span_length"])), int(row["panel_index"]))
            group = sensitivity[common]
            perpendicular = abs(float(row["perpendicular_signed_sensitivity"]))
            parallel = abs(float(row["parallel_signed_sensitivity"]))
            cosine = abs(float(row["perpendicular_cosine_sensitivity"]))
            group["n"] += 1
            group["perpendicular"] += perpendicular
            group["parallel"] += parallel
            group["cosine"] += cosine
            group["cosine_gt_.1"] += cosine > .1
            for item in row["interventions"]:
                ikey = (
                    row["checkpoint"], row["segment"], span_bin(int(row["span_length"])),
                    float(item["gamma"]), bool(item["norm_restored"]), int(row["panel_index"]),
                )
                target = interventions[ikey]
                target["n"] += 1
                target["logp"] += float(item["delta_gold_log_probability"])
                target["margin"] += float(item["delta_gold_margin"])
                target["logp_harmed"] += float(item["delta_gold_log_probability"]) < 0
                target["margin_harmed"] += float(item["delta_gold_margin"]) < 0

    sensitivity_rows = defaultdict(list)
    for (checkpoint, segment, scale, panel), values in sensitivity.items():
        n = values["n"]
        sensitivity_rows[(checkpoint, segment, scale)].append({
            "panel_index": panel,
            "perpendicular": values["perpendicular"] / n,
            "parallel": values["parallel"] / n,
            "cosine": values["cosine"] / n,
            "cosine_gt_.1": values["cosine_gt_.1"] / n,
        })
    intervention_rows = defaultdict(list)
    for (checkpoint, segment, scale, gamma, restored, panel), values in interventions.items():
        n = values["n"]
        intervention_rows[(checkpoint, segment, scale, gamma, restored)].append({
            "panel_index": panel,
            "logp": values["logp"] / n,
            "margin": values["margin"] / n,
            "logp_harmed": values["logp_harmed"] / n,
            "margin_harmed": values["margin_harmed"] / n,
        })

    result = {"sensitivity": [], "interventions": []}
    for key, rows in sorted(sensitivity_rows.items()):
        perpendicular = np.asarray([row["perpendicular"] for row in rows])
        parallel = np.asarray([row["parallel"] for row in rows])
        record = {
            "checkpoint": key[0], "segment": key[1], "span_bin": key[2], "reactions": len(rows),
            "absolute_perpendicular_to_parallel_ratio": float(perpendicular.mean() / parallel.mean()),
            "absolute_perpendicular_to_parallel_ratio_ci95": bootstrap_ratio(perpendicular, parallel),
        }
        for metric in ("cosine", "cosine_gt_.1"):
            values = np.asarray([row[metric] for row in rows])
            record[metric] = float(values.mean())
            record[f"{metric}_ci95"] = bootstrap_mean(values)
        result["sensitivity"].append(record)

    for key, rows in sorted(intervention_rows.items()):
        record = {
            "checkpoint": key[0], "segment": key[1], "span_bin": key[2],
            "gamma": key[3], "norm_restored": key[4], "reactions": len(rows),
        }
        for metric in ("logp", "margin", "logp_harmed", "margin_harmed"):
            values = np.asarray([row[metric] for row in rows])
            record[metric] = float(values.mean())
            record[f"{metric}_ci95"] = bootstrap_mean(values)
        result["interventions"].append(record)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"sensitivity_rows": len(result["sensitivity"]), "intervention_rows": len(result["interventions"])}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    run(arguments.source, arguments.output)
