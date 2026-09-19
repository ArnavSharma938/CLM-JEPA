from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .config import EVIDENCE_SEEDS
from .gate import coverage_gate


PRIMARY = ("heldout_nll", "domain_spearman_mean")


def _read_mode(root: Path, mode: str) -> dict[int, dict]:
    return {
        seed: json.loads((root / "evaluation" / mode / f"seed_{seed}" / "summary.json").read_text(encoding="utf-8"))
        for seed in EVIDENCE_SEEDS
    }


def _summary(runs: dict[int, dict]) -> dict:
    return {
        metric: {
            "by_seed": {str(seed): values[metric] for seed, values in runs.items()},
            "mean": float(np.mean([values[metric] for values in runs.values()])),
            "sample_sd": float(np.std([values[metric] for values in runs.values()], ddof=1)),
        }
        for metric in PRIMARY
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    modes = {mode: _read_mode(args.root, mode) for mode in ("ft", "dtft", "lora8", "lora64")}
    aggregate = {mode: _summary(runs) for mode, runs in modes.items()}
    mean_metrics = {
        mode: {metric: aggregate[mode][metric]["mean"] for metric in PRIMARY} for mode in aggregate
    }
    gates = {
        lora: coverage_gate(mean_metrics["ft"], mean_metrics["dtft"], mean_metrics[lora])
        for lora in ("lora8", "lora64")
    }
    paired = {}
    for left, right in (("ft", "dtft"), ("dtft", "lora8"), ("dtft", "lora64")):
        paired[f"{left}_minus_{right}"] = {
            metric: {
                str(seed): modes[left][seed][metric] - modes[right][seed][metric]
                for seed in EVIDENCE_SEEDS
            }
            for metric in PRIMARY
        }
    result = {
        "independent_replication_unit": "training_seed",
        "evidence_seeds": list(EVIDENCE_SEEDS),
        "metrics": aggregate,
        "paired_differences": paired,
        "coverage_gates": gates,
        "equivalence_note": "No automatic LoRA≈DTFT verdict: no equivalence margin was preregistered.",
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
