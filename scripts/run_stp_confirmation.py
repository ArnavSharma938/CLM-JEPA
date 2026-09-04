#!/usr/bin/env python3
"""Execute the preregistered untouched-panel three-arm STP confirmation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, t

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from confirmation_design import file_sha256, read_jsonl, validate_panel  # noqa: E402


PRIMARY_SEEDS = (2027, 3163)
CONTINGENT_SEEDS = (4211, 5393)
ARMS = {
    "native": "native",
    "released": "stp_released",
    "paper": "stp_paper",
}
PANEL_ROOT = ROOT / "data/clm_jepa_uspto_mit_stp_confirmation"
PANEL = PANEL_ROOT / "untouched_640.jsonl"
TRAIN = ROOT / "data/clm_jepa_uspto_mit_pilot_1280/uspto_mit_train.csv"
VALIDATION = ROOT / "data/clm_jepa_uspto_mit_validation_256/uspto_mit_validation_length_stratified_256.csv"
DEFAULT_OUTPUT = ROOT / "runs/stp_confirmation/a6000"


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def locked_json(path: Path, payload) -> None:
    normalized = json.loads(json.dumps(payload, sort_keys=True))
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != normalized:
        raise ValueError(f"locked record changed: {path}")
    write_json(path, normalized)


def checkpoint(root: Path, seed: int, arm: str) -> Path:
    return root / f"trajectories/seed_{seed}/{arm}/training/checkpoints/epoch_4"


def evaluation(root: Path, seed: int, arm: str) -> Path:
    return root / f"trajectories/seed_{seed}/{arm}/evaluation"


def run(
    command: list[str], *, env: dict[str, str] | None = None, log: Path | None = None
) -> None:
    print(json.dumps({"event": "launch", "command": command}), flush=True)
    if log is None:
        subprocess.run(command, cwd=ROOT, env=env, check=True)
        return
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        subprocess.run(
            command, cwd=ROOT, env=env, check=True,
            stdout=handle, stderr=subprocess.STDOUT,
        )


def train_one(root: Path, seed: int, arm: str) -> Path:
    training = root / f"trajectories/seed_{seed}/{arm}/training"
    result = training / "result.json"
    if not result.exists():
        run([
            sys.executable, "-u", "src/train.py", "--gate", "5",
            "--dataset", "uspto_mit_synthesis", "--condition", ARMS[arm],
            "--seed", str(seed), "--learning-rate", "1e-4",
            "--stp-lambda", "0.02", "--lora-rank", "8", "--lora-alpha", "8",
            "--k", "0", "--lambda-eff", "1.0", "--dropout", "0.5",
            "--epochs", "4", "--stop-after-epoch", "4", "--evaluation-epochs", "4",
            "--final-checkpoint-only", "--batch-size", "4",
            "--gradient-accumulation-steps", "4", "--no-gradient-checkpointing",
            "--fused-adamw", "--attention-implementation", "sdpa", "--pin-memory",
            "--eval-generation-batch-size", "1", "--train-manifest", str(TRAIN),
            "--validation-manifest", str(VALIDATION), "--max-validation-rows", "2",
            "--checkpoint-dir", str(training / "checkpoints"), "--no-wandb",
            "--output", str(result),
        ], log=training / "train.log")
    payload = json.loads(result.read_text(encoding="utf-8"))
    config = payload["config"]
    if (
        payload["seed"] != seed
        or payload["condition"] != ARMS[arm]
        or payload["compute"]["optimizer_steps"] != 320
        or payload["selected_epoch"] != 4
        or config["lora_rank"] != 8
        or config["lora_alpha"] != 8
        or (
            arm != "native"
            and float(config["stp_lambda"]) != 0.02
        )
    ):
        raise ValueError(f"trajectory identity/config mismatch: {result}")
    return checkpoint(root, seed, arm)


def monitor_resources(stop: threading.Event, samples: list[dict]) -> None:
    while not stop.wait(10):
        sample = {"unix_time": time.time()}
        try:
            values = subprocess.check_output([
                "nvidia-smi", "--query-gpu=utilization.gpu,memory.used,power.draw",
                "--format=csv,noheader,nounits",
            ], text=True).strip().split(",")
            sample.update({
                "gpu_utilization_percent": float(values[0]),
                "gpu_memory_mib": float(values[1]),
                "gpu_power_watts": float(values[2]),
            })
        except Exception:
            pass
        samples.append(sample)


def train_seeds(root: Path, seeds: tuple[int, ...]) -> None:
    tasks = [(seed, arm) for seed in seeds for arm in ARMS]
    missing = [task for task in tasks if not (root / f"trajectories/seed_{task[0]}/{task[1]}/training/result.json").exists()]
    if not missing:
        return
    samples: list[dict] = []
    stop = threading.Event()
    monitor = threading.Thread(target=monitor_resources, args=(stop, samples), daemon=True)
    monitor.start()
    started = time.perf_counter()
    try:
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = {pool.submit(train_one, root, seed, arm): (seed, arm) for seed, arm in missing}
            for future in as_completed(futures):
                seed, arm = futures[future]
                future.result()
                print(json.dumps({"event": "training_complete", "seed": seed, "arm": arm}), flush=True)
    finally:
        stop.set()
        monitor.join(timeout=2)
    numeric = [row for row in samples if "gpu_utilization_percent" in row]
    write_json(root / f"resources/training_{'_'.join(map(str, seeds))}.json", {
        "seeds": seeds,
        "concurrency": 3,
        "wall_seconds": time.perf_counter() - started,
        "mean_gpu_utilization_percent": statistics.fmean(row["gpu_utilization_percent"] for row in numeric) if numeric else None,
        "peak_gpu_memory_mib": max((row["gpu_memory_mib"] for row in numeric), default=None),
        "samples": samples,
    })
    for seed in seeds:
        hashes = {}
        for arm in ARMS:
            payload = json.loads((root / f"trajectories/seed_{seed}/{arm}/training/result.json").read_text())
            hashes[arm] = payload.get("initial_trainable_sha256") or payload.get("diagnostics", {}).get("initial_trainable_sha256")
        if len({value for value in hashes.values() if value is not None}) > 1:
            raise ValueError(f"same-seed arms did not start identically: {seed}: {hashes}")
        write_json(root / f"trajectories/seed_{seed}/initialization_audit.json", hashes)


def evaluation_env() -> dict[str, str]:
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


def assignment_mode(root: Path) -> str:
    decision = json.loads((root / "performance/decision.json").read_text(encoding="utf-8"))
    mode = decision["retained_assignment_mode"]
    if mode not in ("round-robin", "length-balanced"):
        raise ValueError("invalid locked evaluator assignment decision")
    return mode


def evaluation_workers(root: Path) -> int:
    decision = root / "performance/worker_scaling/decision.json"
    if not decision.exists():
        return 4
    workers = int(json.loads(decision.read_text(encoding="utf-8"))["selected_workers"])
    if workers not in (4, 5, 6):
        raise ValueError("invalid locked evaluator worker decision")
    return workers


def evaluate_one(root: Path, seed: int, arm: str) -> None:
    output = evaluation(root, seed, arm)
    predictions = output / "predictions.jsonl"
    expected = len(read_jsonl(PANEL))
    if predictions.exists() and len(read_jsonl(predictions)) == expected:
        return
    workers = evaluation_workers(root)
    run([
        sys.executable, "-u", "src/eval_uspto_mit_five_view_a6000.py", "run",
        "--checkpoint", str(checkpoint(root, seed, arm)), "--manifest", str(PANEL),
        "--workers", str(workers), "--threads-per-worker", "1", "--prompt-batch-size", "1",
        "--batch-mode", "left-pad", "--assignment-mode", assignment_mode(root),
        "--output-dir", str(output),
    ], env=evaluation_env())
    if len(read_jsonl(predictions)) != expected:
        raise ValueError(f"incomplete endpoint: {predictions}")
    print(json.dumps({"event": "evaluation_complete", "seed": seed, "arm": arm}), flush=True)


def evaluate_seeds(root: Path, seeds: tuple[int, ...]) -> None:
    for seed in seeds:
        for arm in ARMS:
            evaluate_one(root, seed, arm)


def exact_vector(root: Path, seed: int, arm: str, cutoff: int = 1) -> np.ndarray:
    rows = read_jsonl(evaluation(root, seed, arm) / "predictions.jsonl")
    return np.asarray([any(row["exact"][:cutoff]) for row in rows], dtype=np.int8)


def bootstrap_ci(values: np.ndarray, seed: int, repetitions: int = 20000) -> list[float]:
    generator = np.random.default_rng(seed)
    means = np.empty(repetitions)
    for start in range(0, repetitions, 500):
        count = min(500, repetitions - start)
        indices = generator.integers(0, len(values), size=(count, len(values)))
        means[start:start + count] = values[indices].mean(axis=1)
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def arm_metrics(rows: list[dict]) -> dict:
    result = {
        f"top_{cutoff}": float(np.mean([any(row["exact"][:cutoff]) for row in rows]))
        for cutoff in (1, 3, 5, 10)
    }
    result["individual_view"] = []
    for view in range(5):
        result["individual_view"].append({
            f"top_{cutoff}": float(np.mean([
                row["target"] in row["canonical_candidates_by_view"][view][:cutoff]
                for row in rows
            ]))
            for cutoff in (1, 3, 5, 10)
        })
    canonical = [candidate for row in rows for view in row["canonical_candidates_by_view"] for candidate in view]
    result["candidate_validity"] = float(np.mean([bool(value) for value in canonical]))
    result["view_top1_validity"] = float(np.mean([
        bool(view[0]) for row in rows for view in row["canonical_candidates_by_view"]
    ]))
    return result


def contrast(root: Path, seeds: tuple[int, ...], arm: str) -> dict:
    seed_results = []
    matrices = []
    for seed in seeds:
        native = exact_vector(root, seed, "native")
        treatment = exact_vector(root, seed, arm)
        delta = treatment - native
        wins = int(np.sum(delta == 1))
        losses = int(np.sum(delta == -1))
        discordant = wins + losses
        seed_results.append({
            "seed": seed,
            "native_top1": float(native.mean()),
            "treatment_top1": float(treatment.mean()),
            "difference": float(delta.mean()),
            "reaction_bootstrap_95_ci": bootstrap_ci(delta, seed + 91001),
            "wins": wins,
            "losses": losses,
            "ties": int(len(delta) - discordant),
            "mcnemar_exact_p": float(binomtest(wins, discordant, 0.5).pvalue) if discordant else 1.0,
        })
        matrices.append(delta)
    effects = np.asarray([row["difference"] for row in seed_results])
    if len(effects) > 1:
        half = float(t.ppf(0.975, len(effects) - 1) * effects.std(ddof=1) / np.sqrt(len(effects)))
        seed_ci = [float(effects.mean() - half), float(effects.mean() + half)]
    else:
        seed_ci = [float("nan"), float("nan")]
    matrix = np.stack(matrices)
    generator = np.random.default_rng(78001 + sum(seeds) + len(arm))
    crossed = np.empty(20000)
    for index in range(len(crossed)):
        seed_indices = generator.integers(0, len(seeds), len(seeds))
        reaction_indices = generator.integers(0, matrix.shape[1], matrix.shape[1])
        crossed[index] = matrix[seed_indices][:, reaction_indices].mean()
    reaction_cluster_delta = matrix.mean(axis=0)
    permutation = np.random.default_rng(99173 + sum(seeds) + len(arm))
    observed = abs(float(reaction_cluster_delta.mean()))
    extreme = 0
    for _ in range(20000):
        randomized = reaction_cluster_delta * permutation.choice((-1.0, 1.0), len(reaction_cluster_delta))
        extreme += abs(float(randomized.mean())) >= observed - 1e-15
    return {
        "arm": arm,
        "seed_results": seed_results,
        "mean_difference": float(effects.mean()),
        "seed_t_95_ci_fragile": seed_ci,
        "crossed_seed_reaction_bootstrap_95_ci": [float(value) for value in np.quantile(crossed, [0.025, 0.975])],
        "reaction_cluster_sign_flip_p": (extreme + 1) / 20001,
    }


def analyze(root: Path, seeds: tuple[int, ...]) -> dict:
    panel_rows = read_jsonl(PANEL)
    prediction_rows = {
        (seed, arm): read_jsonl(evaluation(root, seed, arm) / "predictions.jsonl")
        for seed in seeds for arm in ARMS
    }
    exact = {
        key: np.asarray([row["exact"][0] for row in rows], dtype=np.int8)
        for key, rows in prediction_rows.items()
    }
    arms = {}
    for seed in seeds:
        arms[str(seed)] = {}
        for arm in ARMS:
            arms[str(seed)][arm] = arm_metrics(prediction_rows[(seed, arm)])
    contrasts = {arm: contrast(root, seeds, arm) for arm in ("released", "paper")}
    raw_p = {arm: contrasts[arm]["reaction_cluster_sign_flip_p"] for arm in contrasts}
    ordered_p = sorted(raw_p, key=raw_p.get)
    adjusted = {}
    running = 0.0
    for index, arm in enumerate(ordered_p):
        running = max(running, min(1.0, raw_p[arm] * (len(ordered_p) - index)))
        adjusted[arm] = running
    for arm in contrasts:
        contrasts[arm]["holm_adjusted_reaction_cluster_p"] = adjusted[arm]
    reaction_rows = []
    for panel_index, panel_row in enumerate(panel_rows):
        row = {
            "panel_index": panel_index,
            "reaction_identity": panel_row["reaction_identity"],
            "chemical_pair_id": panel_row["chemical_pair_id"],
        }
        for seed in seeds:
            for arm in ARMS:
                row[f"seed_{seed}_{arm}_top1"] = int(exact[(seed, arm)][panel_index])
        reaction_rows.append(row)
    reaction_path = root / "analysis/reaction_level_top1.jsonl"
    reaction_path.parent.mkdir(parents=True, exist_ok=True)
    with reaction_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in reaction_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    payload = {
        "schema_version": 1,
        "panel_sha256": file_sha256(PANEL),
        "seeds": seeds,
        "arms": arms,
        "contrasts": contrasts,
        "reaction_level": str(reaction_path),
        "reaction_level_sha256": file_sha256(reaction_path),
        "development_results_excluded": True,
    }
    write_json(root / "analysis/confirmation.json", payload)
    return payload


def futility(root: Path) -> bool:
    effects = {
        arm: [float((exact_vector(root, seed, arm) - exact_vector(root, seed, "native")).mean()) for seed in PRIMARY_SEEDS]
        for arm in ("released", "paper")
    }
    stop = all(value <= 0 for values in effects.values() for value in values)
    locked_json(root / "decision/futility.json", {
        "primary_seed_effects": effects,
        "stop_for_futility": stop,
        "rule": "stop iff Released and Paper are each nonpositive versus Native in both primary seeds",
    })
    return stop


def prereg_guard(root: Path) -> None:
    validate_panel(
        PANEL, PANEL_ROOT / "untouched_640.metadata.json",
        PANEL_ROOT / "exclusion_ledger.json", expected_reactions=640,
    )
    prereg = ROOT / "docs/preregistrations/STP_UNTOUCHED_CONFIRMATION_PROTOCOL.md"
    expected = json.loads((PANEL_ROOT / "preregistration.json").read_text(encoding="utf-8"))
    if expected["panel_sha256"] != file_sha256(PANEL) or not prereg.exists():
        raise ValueError("committed preregistration/panel guard failed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("train-primary", "evaluate-primary", "decide", "train-contingent", "evaluate-contingent", "analyze", "all"))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    root = args.output_root
    prereg_guard(root)
    phase = args.phase
    if phase in ("train-primary", "all"):
        train_seeds(root, PRIMARY_SEEDS)
    if phase in ("evaluate-primary", "all"):
        evaluate_seeds(root, PRIMARY_SEEDS)
    if phase in ("decide", "all"):
        stop = futility(root)
        print(json.dumps({"event": "futility_decision", "stop": stop}), flush=True)
    else:
        stop = (root / "decision/futility.json").exists() and json.loads((root / "decision/futility.json").read_text())["stop_for_futility"]
    if phase in ("train-contingent", "all") and not stop:
        train_seeds(root, CONTINGENT_SEEDS)
    if phase in ("evaluate-contingent", "all") and not stop:
        evaluate_seeds(root, CONTINGENT_SEEDS)
    if phase in ("analyze", "all"):
        seeds = PRIMARY_SEEDS if stop else PRIMARY_SEEDS + CONTINGENT_SEEDS
        result = analyze(root, seeds)
        print(json.dumps({"event": "confirmation_complete", "seeds": result["seeds"]}), flush=True)


if __name__ == "__main__":
    main()
