#!/usr/bin/env python3
"""Benchmark the only new exact evaluator candidate before confirmation."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eval_uspto_mit_five_view_a6000 import (  # noqa: E402
    _generation_cost_proxy,
    file_sha256,
    read_jsonl,
    write_jsonl,
)


SOURCE = ROOT / "data/clm_jepa_uspto_mit_official_endpoint/prespecified_stage1_512.jsonl"
BENCHMARK = ROOT / "data/clm_jepa_uspto_mit_official_endpoint/stp_confirmation_eval_benchmark_64.jsonl"
CHECKPOINTS = {
    "native": ROOT / "runs/pair_residual/a6000/results/seed_533/native/training/checkpoints/epoch_4",
    "released": ROOT / "runs/stp/a6000/results/seed_533/stp/training/checkpoints/epoch_4",
    "paper": ROOT / "runs/stp_matrix/a6000/stage_c/paper_r8_l0.02/seed_533/training/checkpoints/epoch_4",
}


def prepare_panel() -> None:
    rows = read_jsonl(SOURCE)
    ordered = sorted(rows, key=lambda row: (_generation_cost_proxy(row), row["panel_index"]))
    positions = np.linspace(0, len(ordered) - 1, 64, dtype=int)
    selected = []
    for index, position in enumerate(positions):
        row = dict(ordered[int(position)])
        row["benchmark_source_panel_index"] = row["panel_index"]
        row["panel_index"] = index
        selected.append(row)
    if len({row["reaction_identity"] for row in selected}) != 64:
        raise ValueError("benchmark panel is not unique")
    if BENCHMARK.exists() and read_jsonl(BENCHMARK) != selected:
        raise ValueError("existing evaluator benchmark panel changed")
    write_jsonl(BENCHMARK, selected)


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


def run_one(checkpoint: Path, output: Path, mode: str) -> dict:
    subprocess.run([
        sys.executable, "-u", "src/eval_uspto_mit_five_view_a6000.py", "run",
        "--checkpoint", str(checkpoint), "--manifest", str(BENCHMARK),
        "--workers", "4", "--threads-per-worker", "1", "--prompt-batch-size", "1",
        "--batch-mode", "left-pad", "--assignment-mode", mode,
        "--output-dir", str(output),
    ], cwd=ROOT, env=environment(), check=True)
    return json.loads((output / "summary.json").read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=ROOT / "runs/stp_confirmation/a6000/performance")
    args = parser.parse_args()
    prepare_panel()
    results = {}
    fields = (
        "reaction_identity", "raw_candidates_by_view", "canonical_candidates_by_view",
        "ranked_candidates", "rank_scores", "invalid_by_beam", "unique_valid_per_view", "exact",
    )
    for label, checkpoint in CHECKPOINTS.items():
        results[label] = {}
        for mode in ("round-robin", "length-balanced"):
            output = args.output_root / label / mode
            summary = run_one(checkpoint, output, mode)
            results[label][mode] = {
                "wall_seconds": summary["wall_seconds_including_model_load"],
                "active_seconds": summary["active_evaluation_seconds"],
                "mean_gpu_utilization_percent": summary["gpu"]["mean_utilization_percent"],
                "peak_memory_mib": summary["gpu"]["peak_memory_mib"],
                "predictions_sha256": summary["predictions_sha256"],
                "worker_seconds": [row["evaluation_seconds"] for row in summary["worker_statistics"]],
            }
        reference = read_jsonl(args.output_root / label / "round-robin/predictions.jsonl")
        candidate = read_jsonl(args.output_root / label / "length-balanced/predictions.jsonl")
        equality = {
            field: [row[field] for row in reference] == [row[field] for row in candidate]
            for field in fields
        }
        if not all(equality.values()):
            raise ValueError(f"ordered-beam equivalence failed for {label}: {equality}")
        results[label]["ordered_equivalence"] = equality
        results[label]["wall_speedup"] = (
            results[label]["round-robin"]["wall_seconds"]
            / results[label]["length-balanced"]["wall_seconds"]
        )
    speedups = [results[label]["wall_speedup"] for label in CHECKPOINTS]
    retain = statistics.median(speedups) >= 1.03 and all(
        all(results[label]["ordered_equivalence"].values()) for label in CHECKPOINTS
    )
    decision = {
        "benchmark_manifest": str(BENCHMARK),
        "benchmark_manifest_sha256": file_sha256(BENCHMARK),
        "confirmation_panel_used": False,
        "checkpoints": results,
        "retention_rule": "median wall speedup >=1.03 across Native/Released/Paper and exact ordered output equality",
        "median_wall_speedup": statistics.median(speedups),
        "retained_assignment_mode": "length-balanced" if retain else "round-robin",
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(decision, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
