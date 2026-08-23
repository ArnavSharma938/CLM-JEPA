"""Summarize instrumented training-step time from a saved checkpoint."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch


TIMING_KEYS = (
    "data_seconds",
    "auxiliary_statistics_vjp_seconds",
    "gradient_forward_backward_seconds",
    "optimizer_seconds",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    records = state["curves"]
    summary: dict[str, object] = {
        "checkpoint": str(args.checkpoint),
        "records": len(records),
        "active_records": sum(bool(row["jepa_active"]) for row in records),
        "elapsed_wall_time_seconds": state.get("elapsed_wall_time_seconds"),
        "timings": {},
    }
    timings = summary["timings"]
    assert isinstance(timings, dict)
    for key in TIMING_KEYS:
        values = [float(row[key]) for row in records]
        timings[key] = {
            "sum_seconds": sum(values),
            "mean_seconds": statistics.mean(values),
            "median_seconds": statistics.median(values),
            "maximum_seconds": max(values),
        }
    summary["instrumented_total_seconds"] = sum(
        sum(float(row[key]) for key in TIMING_KEYS) for row in records
    )
    rendered = json.dumps(summary, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
