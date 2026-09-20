from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .queue_pilot import COMPACT_EVIDENCE_SEEDS
from .gate import coverage_gate

MODES = ("ft", "dtft", "lora8", "lora64")
CONVERGENCE_TOLERANCE = 1e-4


def classify_trajectory(history: list[dict]) -> tuple[str, list[float]]:
    tail = history[-4:]
    deltas = [
        float(tail[index]["validation_nll"] - tail[index - 1]["validation_nll"])
        for index in range(1, len(tail))
    ]
    if len(deltas) < 3:
        return "insufficient", deltas
    if all(delta < -CONVERGENCE_TOLERANCE for delta in deltas):
        return "improving", deltas
    if (
        all(delta > CONVERGENCE_TOLERANCE for delta in deltas)
        and history[-1]["validation_nll"] > min(event["validation_nll"] for event in history) + CONVERGENCE_TOLERANCE
    ):
        return "deteriorating", deltas
    if all(abs(delta) <= CONVERGENCE_TOLERANCE for delta in deltas):
        return "plateaued", deltas
    return "mixed", deltas


def summarize(root: Path) -> dict:
    metrics, convergence = {}, {}
    base = json.loads((root / "evaluation" / "base" / "summary.json").read_text(encoding="utf-8"))
    metrics["base"] = {key: base[key] for key in ("heldout_nll", "domain_spearman_mean", "domain_auroc_mean", "sequence_recovery")}
    for mode in MODES:
        runs, curves = [], []
        for seed in COMPACT_EVIDENCE_SEEDS:
            summary = json.loads((root / "evaluation" / mode / f"seed_{seed}" / "summary.json").read_text(encoding="utf-8"))
            history = json.loads((root / "evidence" / mode / f"seed_{seed}" / "history.json").read_text(encoding="utf-8"))
            result = json.loads((root / "evidence" / mode / f"seed_{seed}" / "result.json").read_text(encoding="utf-8"))
            runs.append(summary)
            status, deltas = classify_trajectory(history)
            curves.append({
                "seed": seed, "best_step": result["best_step"], "epochs_completed": result["epochs_completed"],
                "stopped_early": result["stopped_early"], "history": history,
                "last_validation_deltas": deltas,
                "optimization_status": status,
            })
        keys = ("heldout_nll", "domain_spearman_mean", "domain_auroc_mean", "sequence_recovery")
        metrics[mode] = {
            key: {"mean": float(np.mean([run[key] for run in runs])), "values": [run[key] for run in runs]}
            for key in keys
        }
        convergence[mode] = curves
    compact = lambda mode: {key: metrics[mode][key]["mean"] for key in metrics[mode]}
    trajectories = {
        mode: {
            "all_runs_plateaued": all(run["optimization_status"] == "plateaued" for run in convergence[mode]),
            "status_counts": {
                status: sum(run["optimization_status"] == status for run in convergence[mode])
                for status in ("improving", "plateaued", "deteriorating", "mixed", "insufficient")
            },
            "runs": convergence[mode],
        }
        for mode in MODES
    }
    hpo = json.loads((root / "hpo_selection.json").read_text(encoding="utf-8"))
    return {
        "metrics": metrics,
        "convergence": trajectories,
        "hpo": hpo,
        "lora_gap_interpretable": {
            mode: trajectories[mode]["all_runs_plateaued"] for mode in ("lora8", "lora64")
        },
        "coverage_gates": {
            mode: coverage_gate(compact("ft"), compact("dtft"), compact(mode))
            for mode in ("lora8", "lora64")
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    payload = summarize(args.root)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
