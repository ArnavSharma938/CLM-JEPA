from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import HPO_EQUIVALENCE_NLL


def select_learning_rate(histories: dict[float, list[dict]]) -> dict:
    summaries = []
    for learning_rate, history in histories.items():
        values = [float(event["validation_nll"]) for event in history]
        if not values:
            raise ValueError(f"LR {learning_rate} has no validation observations")
        summaries.append({
            "learning_rate": learning_rate,
            "minimum_validation_nll": min(values),
            "terminal_validation_nll": values[-1],
            "trajectory": values,
            "deterioration_from_minimum": values[-1] - min(values),
        })
    best = min(item["minimum_validation_nll"] for item in summaries)
    equivalent = [
        item for item in summaries
        if item["minimum_validation_nll"] <= best + HPO_EQUIVALENCE_NLL
    ]
    # The lower LR is the registered stability tie-break, not a post-hoc minimum chase.
    selected = min(equivalent, key=lambda item: item["learning_rate"])
    return {
        "equivalence_tolerance_nll": HPO_EQUIVALENCE_NLL,
        "selection_rule": "minimum validation NLL; within tolerance choose lower LR",
        "selected_learning_rate": selected["learning_rate"],
        "candidates": sorted(summaries, key=lambda item: item["learning_rate"]),
    }


def select_from_root(root: Path, learning_rates: list[float]) -> dict:
    histories = {}
    for learning_rate in learning_rates:
        path = root / f"lr_{learning_rate:g}" / "history.json"
        histories[learning_rate] = json.loads(path.read_text(encoding="utf-8"))
    return select_learning_rate(histories)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--learning-rates", required=True, nargs="+", type=float)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = select_from_root(args.root, args.learning_rates)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
