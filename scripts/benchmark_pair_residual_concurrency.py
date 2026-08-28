"""Parity-gate concurrent independent residual trajectories on one GPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import load_file


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "data/clm_jepa_uspto_mit_pilot_1280/uspto_mit_train.csv"
VALIDATION = ROOT / "data/clm_jepa_uspto_mit_validation_256/uspto_mit_validation_length_stratified_256.csv"


def adapter_file(checkpoint: Path) -> Path:
    return checkpoint / "USPTO-MIT-Synthesis" / "adapter_model.safetensors"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compare_adapters(reference: Path, candidate: Path) -> dict:
    left = load_file(str(reference), device="cpu")
    right = load_file(str(candidate), device="cpu")
    if left.keys() != right.keys():
        raise ValueError("adapter tensor names differ")
    exact_tensors = 0
    maximum_absolute_error = 0.0
    squared_error = 0.0
    squared_reference = 0.0
    for name in left:
        reference_tensor = left[name].float()
        candidate_tensor = right[name].float()
        exact_tensors += int(torch.equal(left[name], right[name]))
        difference = candidate_tensor - reference_tensor
        maximum_absolute_error = max(
            maximum_absolute_error, float(difference.abs().max())
        )
        squared_error += float(difference.double().square().sum())
        squared_reference += float(reference_tensor.double().square().sum())
    return {
        "tensor_count": len(left),
        "exact_tensors": exact_tensors,
        "all_tensors_bit_exact": exact_tensors == len(left),
        "maximum_absolute_error": maximum_absolute_error,
        "relative_l2_error": (
            (squared_error / squared_reference) ** 0.5
            if squared_reference else 0.0
        ),
        "reference_sha256": sha256(reference),
        "candidate_sha256": sha256(candidate),
    }


def command(output: Path) -> list[str]:
    return [
        sys.executable, "-u", "src/train.py",
        "--gate", "5",
        "--dataset", "uspto_mit_synthesis",
        "--condition", "clm_jepa_pair_residual",
        "--seed", "533",
        "--learning-rate", "1e-4",
        "--k", "0",
        "--lambda-eff", "1.0",
        "--dropout", "0.5",
        "--epochs", "1",
        "--stop-after-epoch", "1",
        "--evaluation-epochs", "1",
        "--batch-size", "4",
        "--gradient-accumulation-steps", "4",
        "--sigreg-batch-size", "16",
        "--no-gradient-checkpointing",
        "--fused-adamw",
        "--attention-implementation", "sdpa",
        "--pin-memory",
        "--dataloader-workers", "0",
        "--eval-generation-batch-size", "1",
        "--train-manifest", str(TRAIN),
        "--validation-manifest", str(VALIDATION),
        "--max-train-rows", "256",
        "--max-validation-rows", "2",
        "--checkpoint-dir", str(output / "checkpoints"),
        "--no-wandb",
        "--output", str(output / "result.json"),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-result", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--processes", type=int, default=3)
    args = parser.parse_args()
    if args.processes < 2:
        parser.error("concurrency benchmark requires at least two processes")
    reference_result = json.loads(args.reference_result.read_text())
    reference_checkpoint = Path(reference_result["selected_checkpoint"])
    reference_adapter = adapter_file(reference_checkpoint)
    args.output_root.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update({
        "TOKENIZERS_PARALLELISM": "false",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    })
    processes = []
    logs = []
    started = time.perf_counter()
    for index in range(args.processes):
        output = args.output_root / f"worker_{index}"
        output.mkdir(parents=True, exist_ok=True)
        log = (output / "train.log").open("w", encoding="utf-8")
        logs.append(log)
        processes.append(subprocess.Popen(
            command(output), cwd=ROOT, env=environment,
            stdout=log, stderr=subprocess.STDOUT,
        ))
    exit_codes = [process.wait() for process in processes]
    wall_seconds = time.perf_counter() - started
    for log in logs:
        log.close()
    if any(exit_codes):
        raise RuntimeError(f"concurrent training failed: {exit_codes}")

    comparisons = []
    result_payloads = []
    for index in range(args.processes):
        output = args.output_root / f"worker_{index}"
        result = json.loads((output / "result.json").read_text())
        result_payloads.append(result)
        comparisons.append(compare_adapters(
            reference_adapter, adapter_file(Path(result["selected_checkpoint"])),
        ))
    bit_exact = all(row["all_tensors_bit_exact"] for row in comparisons)
    payload = {
        "schema_version": 1,
        "processes": args.processes,
        "reference_result": str(args.reference_result.resolve()),
        "reference_wall_seconds": reference_result["compute"]["wall_time_seconds"],
        "concurrent_wall_seconds": wall_seconds,
        "sequential_equivalent_wall_seconds": (
            args.processes * reference_result["compute"]["wall_time_seconds"]
        ),
        "aggregate_speedup": (
            args.processes * reference_result["compute"]["wall_time_seconds"]
            / wall_seconds
        ),
        "all_adapters_bit_exact": bit_exact,
        "comparisons": comparisons,
        "initial_trainable_sha256": [
            row["config"]["initial_trainable_sha256"] for row in result_payloads
        ],
        "optimizer_steps": [row["compute"]["optimizer_steps"] for row in result_payloads],
        "selection_rule": (
            "adopt concurrency only if every saved adapter tensor is bit exact "
            "to the sequential reference"
        ),
    }
    (args.output_root / "benchmark.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "all_adapters_bit_exact": bit_exact,
        "aggregate_speedup": payload["aggregate_speedup"],
        "wall_seconds": wall_seconds,
    }, sort_keys=True))
    if not bit_exact:
        raise SystemExit("concurrent training failed exact adapter parity")


if __name__ == "__main__":
    main()
