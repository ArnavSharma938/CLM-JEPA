#!/usr/bin/env python
"""Outcome-blind exact A6000 worker-scaling gate for confirmation evaluation."""

from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "data/clm_jepa_uspto_mit_official_endpoint/stp_confirmation_eval_benchmark_64.jsonl"
OUTPUT = ROOT / "runs/stp_confirmation/a6000/performance"
CHECKPOINTS = {
    "native": ROOT / "runs/pair_residual/a6000/results/seed_533/native/training/checkpoints/epoch_4",
    "released": ROOT / "runs/stp/a6000/results/seed_533/stp/training/checkpoints/epoch_4",
    "paper": ROOT / "runs/stp_matrix/a6000/stage_c/paper_r8_l0.02/seed_533/training/checkpoints/epoch_4",
}
FIELDS = (
    "reaction_identity", "raw_candidates_by_view", "canonical_candidates_by_view",
    "ranked_candidates", "rank_scores", "invalid_by_beam", "unique_valid_per_view", "exact",
)


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def environment() -> dict[str, str]:
    env = os.environ.copy()
    env.update({
        "PYTHONPATH": "src", "TOKENIZERS_PARALLELISM": "false",
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1", "RAYON_NUM_THREADS": "1",
        "CHEMFM_EXACT_PREALLOCATED_CACHE": "1", "CHEMFM_EXACT_BEAM_SCORER": "1",
        "CHEMFM_DECODE_BATCH_SIZE": "10", "CHEMFM_EXACT_LORA_FASTPATH": "1",
        "CHEMFM_EXACT_LORA_CUDAGRAPH": "1", "CHEMFM_EXACT_LAYER_CUDAGRAPH": "1",
        "CHEMFM_EXACT_RMSNORM_CUDAGRAPH": "1", "CHEMFM_EXACT_ROPE_CUDAGRAPH": "1",
    })
    return env


def run(label: str, workers: int) -> dict:
    destination = OUTPUT / "worker_scaling" / label / f"w{workers}"
    subprocess.run([
        sys.executable, "-u", "src/eval_uspto_mit_five_view_a6000.py", "run",
        "--checkpoint", str(CHECKPOINTS[label]), "--manifest", str(PANEL),
        "--workers", str(workers), "--threads-per-worker", "1",
        "--prompt-batch-size", "1", "--batch-mode", "left-pad",
        "--assignment-mode", "round-robin", "--output-dir", str(destination),
    ], cwd=ROOT, env=environment(), check=True)
    return json.loads((destination / "summary.json").read_text(encoding="utf-8"))


def compare(label: str, workers: int, summary: dict) -> dict:
    reference_dir = OUTPUT / label / "round-robin"
    reference_summary = json.loads((reference_dir / "summary.json").read_text(encoding="utf-8"))
    reference = rows(reference_dir / "predictions.jsonl")
    candidate = rows(Path(summary["predictions"]))
    equality = {field: [row[field] for row in reference] == [row[field] for row in candidate] for field in FIELDS}
    return {
        "workers": workers, "reference_wall_seconds": reference_summary["wall_seconds_including_model_load"],
        "candidate_wall_seconds": summary["wall_seconds_including_model_load"],
        "wall_speedup": reference_summary["wall_seconds_including_model_load"] / summary["wall_seconds_including_model_load"],
        "reference_active_seconds": reference_summary["active_evaluation_seconds"],
        "candidate_active_seconds": summary["active_evaluation_seconds"],
        "candidate_gpu": summary["gpu"], "ordered_equivalence": equality,
    }


def main() -> None:
    native = [compare("native", workers, run("native", workers)) for workers in (5, 6)]
    viable = [row for row in native if all(row["ordered_equivalence"].values())]
    best = max(viable, key=lambda row: row["wall_speedup"])
    results = {"native": {f'w{row["workers"]}': row for row in native}}
    chosen = int(best["workers"])
    if best["wall_speedup"] >= 1.05:
        for label in ("released", "paper"):
            results[label] = compare(label, chosen, run(label, chosen))
        validation = [best, results["released"], results["paper"]]
        retain = statistics.median(row["wall_speedup"] for row in validation) >= 1.05 and all(
            all(row["ordered_equivalence"].values()) for row in validation
        )
    else:
        validation, retain = [best], False
    decision = {
        "confirmation_panel_used": False, "benchmark_panel": str(PANEL),
        "results": results, "retention_rule": "median wall speedup >=1.05 across Native/Released/Paper with exact equality of all ordered and derived fields",
        "selected_workers": chosen if retain else 4,
        "median_validated_speedup": statistics.median(row["wall_speedup"] for row in validation),
    }
    path = OUTPUT / "worker_scaling/decision.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(decision, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
