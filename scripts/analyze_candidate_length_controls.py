#!/usr/bin/env python
"""Length-controlled gold-versus-wrong trajectory geometry summaries."""

from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


METRICS = (
    "tube_rms_integral",
    "paper_loss",
    "released_loss",
    "euclidean_inefficiency",
    "fisher_inefficiency",
    "fisher_local_curvature",
    "normal_acceleration",
)


def bootstrap_mean(values: np.ndarray, seed: int = 260222617) -> list[float]:
    if len(values) < 2:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    estimates = []
    for start in range(0, 10_000, 250):
        sample = rng.choice(values, size=(min(250, 10_000 - start), len(values)), replace=True)
        estimates.extend(sample.mean(axis=1))
    return np.quantile(estimates, [.025, .975]).tolist()


def bootstrap_intercept(x: np.ndarray, y: np.ndarray, seed: int = 260222617) -> list[float]:
    if len(x) < 3 or np.ptp(x) == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(10_000):
        chosen = rng.integers(0, len(x), len(x))
        xb, yb = x[chosen], y[chosen]
        if np.ptp(xb) == 0:
            continue
        estimates.append(float(np.polyfit(xb, yb, 1)[1]))
    return np.quantile(estimates, [.025, .975]).tolist()


def run(source: Path, output: Path) -> None:
    states: dict[tuple, dict] = {}
    with gzip.open(source, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["layer"] != "final_post_norm" or row["role"] not in {"gold", "highest_wrong"}:
                continue
            rms = row["tube_scale"]["rms"]
            states[(row["checkpoint"], row["panel_index"], row["view"], row["role"])] = {
                "tokens": int(row["tokens"]),
                "tube_rms_integral": float(np.mean(rms)) if rms else float("nan"),
                "paper_loss": float(row["paper_loss"]),
                "released_loss": float(row["released_loss"]),
                "euclidean_inefficiency": 1.0 - float(row["euclidean_path_efficiency"]),
                "fisher_inefficiency": 1.0 - float(row["fisher_path_efficiency"]),
                "fisher_local_curvature": float(row["fisher_local_curvature"]),
                "normal_acceleration": float(row["normalized_normal_acceleration"]),
            }

    views = []
    for (checkpoint, panel, view, role), wrong in states.items():
        if role != "highest_wrong":
            continue
        gold = states.get((checkpoint, panel, view, "gold"))
        if gold is None:
            continue
        views.append({
            "checkpoint": checkpoint,
            "panel_index": int(panel),
            "token_difference": wrong["tokens"] - gold["tokens"],
            **{metric: wrong[metric] - gold[metric] for metric in METRICS},
        })

    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in views:
        grouped[(row["checkpoint"], row["panel_index"])].append(row)
    reactions = []
    for (checkpoint, panel), rows in grouped.items():
        reactions.append({
            "checkpoint": checkpoint,
            "panel_index": panel,
            "token_difference": float(np.mean([row["token_difference"] for row in rows])),
            **{metric: float(np.mean([row[metric] for row in rows])) for metric in METRICS},
        })

    result = []
    checkpoints = sorted({row["checkpoint"] for row in reactions})
    for checkpoint in checkpoints:
        checkpoint_rows = [row for row in reactions if row["checkpoint"] == checkpoint]
        strata = {
            "all": checkpoint_rows,
            "abs_token_delta_le_2": [row for row in checkpoint_rows if abs(row["token_difference"]) <= 2],
            "abs_token_delta_le_5": [row for row in checkpoint_rows if abs(row["token_difference"]) <= 5],
            "wrong_not_shorter": [row for row in checkpoint_rows if row["token_difference"] >= 0],
        }
        for name, selected in strata.items():
            if not selected:
                continue
            record = {"checkpoint": checkpoint, "stratum": name, "reactions": len(selected)}
            for metric in METRICS:
                values = np.asarray([row[metric] for row in selected], dtype=float)
                values = values[np.isfinite(values)]
                record[metric] = float(values.mean())
                record[f"{metric}_ci95"] = bootstrap_mean(values)
            result.append(record)

        x = np.asarray([row["token_difference"] for row in checkpoint_rows], dtype=float)
        adjusted = {"checkpoint": checkpoint, "stratum": "linear_length_adjusted", "reactions": len(x)}
        for metric in METRICS:
            y = np.asarray([row[metric] for row in checkpoint_rows], dtype=float)
            finite = np.isfinite(x) & np.isfinite(y)
            slope, intercept = np.polyfit(x[finite], y[finite], 1)
            adjusted[f"{metric}_intercept_at_equal_length"] = float(intercept)
            adjusted[f"{metric}_token_slope"] = float(slope)
            adjusted[f"{metric}_intercept_ci95"] = bootstrap_intercept(x[finite], y[finite])
        result.append(adjusted)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(result), "output": str(output)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    run(arguments.source, arguments.output)
