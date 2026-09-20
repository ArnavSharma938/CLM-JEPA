from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .queue_pilot import COMPACT_EVIDENCE_SEEDS
from .gate import coverage_gate

MODES = ("ft", "dtft", "lora8", "lora64")


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
            tail = history[-4:]
            deltas = [tail[index]["validation_nll"] - tail[index - 1]["validation_nll"] for index in range(1, len(tail))]
            curves.append({
                "seed": seed, "best_step": result["best_step"], "epochs_completed": result["epochs_completed"],
                "stopped_early": result["stopped_early"], "history": history,
                "last_validation_deltas": deltas,
                "still_clearly_improving": len(deltas) == 3 and all(delta < 0 for delta in deltas),
            })
        keys = ("heldout_nll", "domain_spearman_mean", "domain_auroc_mean", "sequence_recovery")
        metrics[mode] = {
            key: {"mean": float(np.mean([run[key] for run in runs])), "values": [run[key] for run in runs]}
            for key in keys
        }
        convergence[mode] = curves
    compact = lambda mode: {key: metrics[mode][key]["mean"] for key in metrics[mode]}
    plateau = {
        mode: {"all_runs_near_plateau": not any(run["still_clearly_improving"] for run in convergence[mode]), "runs": convergence[mode]}
        for mode in MODES
    }
    return {
        "metrics": metrics,
        "convergence": plateau,
        "lora_gap_interpretable": {
            mode: plateau[mode]["all_runs_near_plateau"] for mode in ("lora8", "lora64")
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
