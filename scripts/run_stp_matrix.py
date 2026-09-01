"""Run the preregistered STP capacity/formulation/lambda matrix on one A6000."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import psutil


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (533, 917)
STAGE_A_SEEDS = (533, 917, 1301)
TRAIN = ROOT / "data/clm_jepa_uspto_mit_pilot_1280/uspto_mit_train.csv"
VALIDATION = ROOT / "data/clm_jepa_uspto_mit_validation_256/uspto_mit_validation_length_stratified_256.csv"
ENDPOINT_ROOT = ROOT / "data/clm_jepa_uspto_mit_official_endpoint"
SOURCE_PANEL = ENDPOINT_ROOT / "prespecified_stage1_1280.jsonl"
PANEL = ENDPOINT_ROOT / "prespecified_stage1_512.jsonl"
EQUIVALENCE_PANEL = ENDPOINT_ROOT / "stp_rank128_equivalence_24.jsonl"
OLD_NATIVE = ROOT / "runs/pair_residual/a6000/results"
OLD_RELEASED = ROOT / "runs/stp/a6000/results"
DEFAULT_OUTPUT = ROOT / "runs/stp_matrix/a6000"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def freeze_panel() -> None:
    source_lines = SOURCE_PANEL.read_text(encoding="utf-8").splitlines(keepends=True)
    if len(source_lines) < 512:
        raise ValueError("frozen source endpoint has fewer than 512 reactions")
    expected = "".join(source_lines[:512])
    if PANEL.exists() and PANEL.read_text(encoding="utf-8") != expected:
        raise ValueError("existing 512 panel does not match frozen 1280-panel prefix")
    PANEL.write_text(expected, encoding="utf-8", newline="")
    rows = read_jsonl(PANEL)
    if len(rows) != 512 or [row["panel_index"] for row in rows] != list(range(512)):
        raise ValueError("invalid 512-reaction endpoint")
    equivalence = "".join(source_lines[:24])
    if EQUIVALENCE_PANEL.exists() and EQUIVALENCE_PANEL.read_text(encoding="utf-8") != equivalence:
        raise ValueError("existing rank-128 equivalence panel changed")
    EQUIVALENCE_PANEL.write_text(equivalence, encoding="utf-8", newline="")


def run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    print(json.dumps({"launch": command}), flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def locked_record(path: Path, payload: dict) -> None:
    if path.exists() and read_json(path) != payload:
        raise ValueError(f"locked stage record changed: {path}")
    write_json(path, payload)


def evaluation_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update({
        "PYTHONPATH": "src", "TOKENIZERS_PARALLELISM": "false",
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
        "RAYON_NUM_THREADS": "1", "CHEMFM_EXACT_PREALLOCATED_CACHE": "1",
        "CHEMFM_EXACT_BEAM_SCORER": "1", "CHEMFM_DECODE_BATCH_SIZE": "10",
        "CHEMFM_EXACT_LORA_FASTPATH": "1", "CHEMFM_EXACT_LORA_CUDAGRAPH": "1",
        "CHEMFM_EXACT_LAYER_CUDAGRAPH": "1", "CHEMFM_EXACT_RMSNORM_CUDAGRAPH": "1",
        "CHEMFM_EXACT_ROPE_CUDAGRAPH": "1",
    })
    return env


def checkpoint_for_old(condition: str, seed: int) -> Path:
    if condition == "native":
        return OLD_NATIVE / f"seed_{seed}/native/training/checkpoints/epoch_4"
    return OLD_RELEASED / f"seed_{seed}/stp/training/checkpoints/epoch_4"


def old_evaluation(condition: str, seed: int) -> Path:
    if condition == "native":
        return OLD_NATIVE / f"seed_{seed}/native/evaluation"
    return OLD_RELEASED / f"seed_{seed}/stp/evaluation"


def seed_resume_rows(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for worker in range(4):
        target = destination / f"worker_{worker:02d}.jsonl"
        if not target.exists():
            shutil.copy2(source / target.name, target)
        rows = read_jsonl(target)
        if len(rows) != 64 or any(row["panel_index"] % 4 != worker for row in rows):
            raise ValueError(f"invalid immutable Stage-A resume rows in {target}")


def evaluate(checkpoint: Path, output: Path, *, resume_from: Path | None = None) -> None:
    if (output / "predictions.jsonl").exists():
        rows = read_jsonl(output / "predictions.jsonl")
        if len(rows) == 512:
            return
    if resume_from is not None:
        seed_resume_rows(resume_from, output)
    run([
        sys.executable, "-u", "src/eval_uspto_mit_five_view_a6000.py", "run",
        "--checkpoint", str(checkpoint), "--manifest", str(PANEL),
        "--workers", "4", "--threads-per-worker", "1",
        "--prompt-batch-size", "1", "--batch-mode", "left-pad",
        "--output-dir", str(output),
    ], env=evaluation_env())


def teacher(checkpoint: Path, output: Path) -> None:
    if output.exists():
        return
    run([
        sys.executable, "-u", "scripts/eval_teacher_forced_five_view.py",
        "--checkpoint", str(checkpoint), "--manifest", str(PANEL),
        "--batch-size", "16", "--output", str(output),
    ])


def verify_rank128_evaluation_equivalence(root: Path, checkpoint: Path) -> None:
    output = root / "stage_b/rank128_reference_equivalence"
    reference_path = output / "predictions.jsonl"
    if not reference_path.exists():
        env = {
            key: value for key, value in os.environ.items()
            if not key.startswith("CHEMFM_")
        }
        env.update({
            "PYTHONPATH": "src", "TOKENIZERS_PARALLELISM": "false",
            "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
            "RAYON_NUM_THREADS": "1",
        })
        run([
            sys.executable, "-u", "src/eval_uspto_mit_five_view_a6000.py", "run",
            "--checkpoint", str(checkpoint), "--manifest", str(EQUIVALENCE_PANEL),
            "--workers", "4", "--threads-per-worker", "1",
            "--prompt-batch-size", "1", "--batch-mode", "left-pad",
            "--output-dir", str(output),
        ], env=env)
    reference = read_jsonl(reference_path)
    optimized_path = root / "stage_b/released_r128_l0.02/seed_533/evaluation/predictions.jsonl"
    optimized = read_jsonl(optimized_path)[:24]
    fields = (
        "reaction_identity", "raw_candidates_by_view",
        "canonical_candidates_by_view", "ranked_candidates", "exact",
    )
    equality = {
        field: [row[field] for row in reference] == [row[field] for row in optimized]
        for field in fields
    }
    if len(reference) != 24 or not all(equality.values()):
        raise ValueError(f"rank-128 optimized evaluation changed ordered predictions: {equality}")
    locked_record(root / "stage_b/rank128_evaluation_equivalence.json", {
        "reactions": 24,
        "checkpoint": str(checkpoint.resolve()),
        "reference_fast_paths": False,
        "optimized_fast_paths": True,
        "reference_predictions": str(reference_path.resolve()),
        "reference_sha256": sha256(reference_path),
        "optimized_full_panel_predictions": str(optimized_path.resolve()),
        "optimized_full_panel_sha256": sha256(optimized_path),
        "ordered_equality": equality,
    })


def train_one(
    output_root: Path, *, condition: str, label: str, seed: int,
    rank: int, alpha: int, coefficient: float,
) -> Path:
    training = output_root / label / f"seed_{seed}/training"
    result = training / "result.json"
    if not result.exists():
        run([
            sys.executable, "-u", "src/train.py", "--gate", "5",
            "--dataset", "uspto_mit_synthesis", "--condition", condition,
            "--seed", str(seed), "--learning-rate", "1e-4",
            "--stp-lambda", str(coefficient), "--lora-rank", str(rank),
            "--lora-alpha", str(alpha), "--k", "0", "--lambda-eff", "1.0",
            "--dropout", "0.5", "--epochs", "4", "--stop-after-epoch", "4",
            "--evaluation-epochs", "4", "--final-checkpoint-only",
            "--batch-size", "4", "--gradient-accumulation-steps", "4",
            "--no-gradient-checkpointing", "--fused-adamw",
            "--attention-implementation", "sdpa", "--pin-memory",
            "--eval-generation-batch-size", "1", "--train-manifest", str(TRAIN),
            "--validation-manifest", str(VALIDATION), "--max-validation-rows", "2",
            "--checkpoint-dir", str(training / "checkpoints"), "--no-wandb",
            "--output", str(result),
        ])
    payload = read_json(result)
    if payload["seed"] != seed or payload["config"]["lora_rank"] != rank:
        raise ValueError(f"training identity mismatch: {result}")
    if payload["compute"]["optimizer_steps"] != 320 or payload["selected_epoch"] != 4:
        raise ValueError(f"incomplete trajectory: {result}")
    return training / "checkpoints/epoch_4"


def train_pair(output_root: Path, **kwargs) -> dict[int, Path]:
    label = kwargs["label"]
    resource_path = output_root / label / "training_resources.json"
    already_complete = all(
        (output_root / label / f"seed_{seed}/training/result.json").exists()
        for seed in SEEDS
    )
    samples = []
    stop = threading.Event()

    def monitor():
        psutil.cpu_percent(interval=None)
        while not stop.wait(10.0):
            sample = {"timestamp": time.time(), "cpu_utilization_percent": psutil.cpu_percent(interval=None)}
            try:
                result = subprocess.run([
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used,power.draw",
                    "--format=csv,noheader,nounits",
                ], check=True, capture_output=True, text=True)
                gpu, memory, power = [float(value.strip()) for value in result.stdout.splitlines()[0].split(",")]
                sample.update({
                    "gpu_utilization_percent": gpu,
                    "gpu_memory_mib": memory,
                    "gpu_power_watts": power,
                })
            except Exception:
                pass
            samples.append(sample)

    started = time.perf_counter()
    monitor_thread = None
    if not already_complete:
        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()
    checkpoints = {}
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = {
                pool.submit(train_one, output_root, seed=seed, **kwargs): seed
                for seed in SEEDS
            }
            for future in as_completed(futures):
                checkpoints[futures[future]] = future.result()
    finally:
        stop.set()
        if monitor_thread is not None:
            monitor_thread.join(timeout=2)
    if not already_complete:
        numeric = [sample for sample in samples if "gpu_utilization_percent" in sample]
        write_json(resource_path, {
            "condition_label": label,
            "concurrent_seeds": list(SEEDS),
            "wall_seconds": time.perf_counter() - started,
            "sample_interval_seconds": 10.0,
            "samples": samples,
            "mean_cpu_utilization_percent": (
                sum(sample["cpu_utilization_percent"] for sample in samples) / len(samples)
                if samples else None
            ),
            "mean_gpu_utilization_percent": (
                sum(sample["gpu_utilization_percent"] for sample in numeric) / len(numeric)
                if numeric else None
            ),
            "peak_gpu_memory_mib": max(
                (sample["gpu_memory_mib"] for sample in numeric), default=None
            ),
        })
    return checkpoints


def exact_top1(path: Path) -> list[int]:
    return [int(row["exact"][0]) for row in read_jsonl(path)]


def treatment_effect(native: Path, treatment: Path) -> float:
    left, right = exact_top1(native), exact_top1(treatment)
    if len(left) != 512 or len(right) != 512:
        raise ValueError("treatment decision requires the complete 512 panel")
    return sum(b - a for a, b in zip(left, right)) / len(left)


def stage_a(root: Path) -> None:
    payload = {
        "stage": "A", "panel": str(PANEL), "panel_sha256": sha256(PANEL),
        "seeds": STAGE_A_SEEDS, "rank": 8, "alpha": 8, "lambda": 0.02,
        "conditions": ["native", "released"], "training": "reuse only",
        "primary": "paired released-minus-native exact top-1",
    }
    locked_record(root / "stage_a/preregistration.json", payload)
    for seed in STAGE_A_SEEDS:
        for condition in ("native", "released"):
            destination = root / f"stage_a/r8_l0.02/{condition}/seed_{seed}/evaluation"
            checkpoint = checkpoint_for_old(condition, seed)
            evaluate(checkpoint, destination, resume_from=old_evaluation(condition, seed))
            teacher(checkpoint, destination / "teacher_forced.json")


def paths_for(root: Path, stage: str, label: str, seed: int) -> tuple[Path, Path]:
    base = root / stage / label / f"seed_{seed}"
    return base / "training/checkpoints/epoch_4", base / "evaluation"


def stage_b(root: Path) -> None:
    payload = {
        "stage": "B", "panel_sha256": sha256(PANEL), "seeds": SEEDS,
        "rank": 128, "alpha": 128, "lambda": 0.02,
        "conditions": ["native", "stp_released"], "optimizer_steps": 320,
        "primary": "released-minus-native within rank",
        "evaluation_equivalence": "24 reactions, generic vs optimized ordered beams",
        "rank_selection_rule": "r128 iff mean effect exceeds r8 by >=0.005 and both r128 effects >=0",
    }
    locked_record(root / "stage_b/preregistration.json", payload)
    for condition, label in (("native", "native_r128"), ("stp_released", "released_r128_l0.02")):
        checkpoints = train_pair(
            root / "stage_b", condition=condition, label=label,
            rank=128, alpha=128, coefficient=0.02,
        )
        for seed in SEEDS:
            evaluation = root / f"stage_b/{label}/seed_{seed}/evaluation"
            evaluate(checkpoints[seed], evaluation)
            teacher(checkpoints[seed], evaluation / "teacher_forced.json")

    verify_rank128_evaluation_equivalence(
        root, root / "stage_b/released_r128_l0.02/seed_533/training/checkpoints/epoch_4"
    )

    r8 = [treatment_effect(
        root / f"stage_a/r8_l0.02/native/seed_{seed}/evaluation/predictions.jsonl",
        root / f"stage_a/r8_l0.02/released/seed_{seed}/evaluation/predictions.jsonl",
    ) for seed in SEEDS]
    r128 = [treatment_effect(
        root / f"stage_b/native_r128/seed_{seed}/evaluation/predictions.jsonl",
        root / f"stage_b/released_r128_l0.02/seed_{seed}/evaluation/predictions.jsonl",
    ) for seed in SEEDS]
    selected = 128 if sum(r128) / 2 >= sum(r8) / 2 + 0.005 and min(r128) >= 0 else 8
    locked_record(root / "stage_b/decision.json", {
        "r8_seed_effects": r8, "r128_seed_effects": r128, "selected_rank": selected,
        "rule": payload["rank_selection_rule"],
    })


def selected_rank_paths(root: Path, rank: int, condition: str, seed: int) -> tuple[Path, Path]:
    if rank == 8:
        checkpoint = checkpoint_for_old("native" if condition == "native" else "released", seed)
        evaluation = root / f"stage_a/r8_l0.02/{condition}/seed_{seed}/evaluation"
    else:
        label = "native_r128" if condition == "native" else "released_r128_l0.02"
        checkpoint, evaluation = paths_for(root, "stage_b", label, seed)
    return checkpoint, evaluation


def stage_c(root: Path) -> None:
    rank = int(read_json(root / "stage_b/decision.json")["selected_rank"])
    payload = {
        "stage": "C", "panel_sha256": sha256(PANEL), "seeds": SEEDS,
        "selected_rank": rank, "alpha": rank, "lambda": 0.02,
        "new_condition": "stp_paper", "primary": "objective-definition comparison",
        "formulation_selection_rule": "paper iff mean effect exceeds released by >=0.005 and both paper effects >=0",
    }
    locked_record(root / "stage_c/preregistration.json", payload)
    run([
        sys.executable, "-u", "scripts/diagnose_stp_objectives.py",
        "--rank", str(rank), "--output", str(root / "stage_c/frozen_objective_diagnostics.json"),
    ])
    checkpoints = train_pair(
        root / "stage_c", condition="stp_paper",
        label=f"paper_r{rank}_l0.02", rank=rank, alpha=rank, coefficient=0.02,
    )
    for seed in SEEDS:
        evaluation = root / f"stage_c/paper_r{rank}_l0.02/seed_{seed}/evaluation"
        evaluate(checkpoints[seed], evaluation)
        teacher(checkpoints[seed], evaluation / "teacher_forced.json")

    released, paper = [], []
    for seed in SEEDS:
        _, native_eval = selected_rank_paths(root, rank, "native", seed)
        _, released_eval = selected_rank_paths(root, rank, "released", seed)
        paper_eval = root / f"stage_c/paper_r{rank}_l0.02/seed_{seed}/evaluation"
        released.append(treatment_effect(native_eval / "predictions.jsonl", released_eval / "predictions.jsonl"))
        paper.append(treatment_effect(native_eval / "predictions.jsonl", paper_eval / "predictions.jsonl"))
    formulation = "paper" if sum(paper) / 2 >= sum(released) / 2 + 0.005 and min(paper) >= 0 else "released"
    locked_record(root / "stage_c/decision.json", {
        "rank": rank, "released_seed_effects": released, "paper_seed_effects": paper,
        "selected_formulation": formulation, "rule": payload["formulation_selection_rule"],
    })


def treatment_paths(root: Path, rank: int, formulation: str, coefficient: float, seed: int):
    native_checkpoint, native_eval = selected_rank_paths(root, rank, "native", seed)
    if coefficient == 0.02:
        if formulation == "released":
            treatment_checkpoint, treatment_eval = selected_rank_paths(root, rank, "released", seed)
        else:
            treatment_checkpoint, treatment_eval = paths_for(root, "stage_c", f"paper_r{rank}_l0.02", seed)
    else:
        label = f"{formulation}_r{rank}_l{coefficient:g}"
        treatment_checkpoint, treatment_eval = paths_for(root, "stage_d", label, seed)
    return native_checkpoint, native_eval, treatment_checkpoint, treatment_eval


def stage_d(root: Path) -> None:
    decision = read_json(root / "stage_c/decision.json")
    rank, formulation = int(decision["rank"]), decision["selected_formulation"]
    payload = {
        "stage": "D", "panel_sha256": sha256(PANEL), "seeds": SEEDS,
        "rank": rank, "alpha": rank, "formulation": formulation,
        "lambdas": [0.005, 0.02, 0.08], "lambda_0.02": "exact reuse",
        "primary": "mean paired exact top-1 treatment effect",
        "final_rank_rule": "run iff non-0.02 improves mean effect by >=0.01 and both effects >=0",
    }
    locked_record(root / "stage_d/preregistration.json", payload)
    condition = "stp_paper" if formulation == "paper" else "stp_released"
    for coefficient in (0.005, 0.08):
        label = f"{formulation}_r{rank}_l{coefficient:g}"
        checkpoints = train_pair(
            root / "stage_d", condition=condition, label=label,
            rank=rank, alpha=rank, coefficient=coefficient,
        )
        for seed in SEEDS:
            evaluation = root / f"stage_d/{label}/seed_{seed}/evaluation"
            evaluate(checkpoints[seed], evaluation)
            teacher(checkpoints[seed], evaluation / "teacher_forced.json")

    effects = {}
    for coefficient in (0.005, 0.02, 0.08):
        values = []
        for seed in SEEDS:
            _, native_eval, _, treatment_eval = treatment_paths(
                root, rank, formulation, coefficient, seed
            )
            values.append(treatment_effect(
                native_eval / "predictions.jsonl", treatment_eval / "predictions.jsonl"
            ))
        effects[str(coefficient)] = values
    selected = max((0.005, 0.02, 0.08), key=lambda value: sum(effects[str(value)]) / 2)
    baseline = sum(effects["0.02"]) / 2
    selected_values = effects[str(selected)]
    final_rank = selected != 0.02 and sum(selected_values) / 2 >= baseline + 0.01 and min(selected_values) >= 0
    locked_record(root / "stage_d/decision.json", {
        "rank": rank, "formulation": formulation, "seed_effects": effects,
        "selected_lambda": selected, "run_final_rank_check": final_rank,
        "rule": payload["final_rank_rule"],
    })


def final_rank(root: Path) -> None:
    decision = read_json(root / "stage_d/decision.json")
    if not decision["run_final_rank_check"]:
        locked_record(root / "final_rank/decision.json", {
            "run": False, "reason": "preregistered material-improvement threshold not met",
        })
        return
    selected_rank = int(decision["rank"])
    other_rank = 128 if selected_rank == 8 else 8
    formulation = decision["formulation"]
    coefficient = float(decision["selected_lambda"])
    payload = {
        "stage": "final_rank", "seeds": SEEDS, "formulation": formulation,
        "lambda": coefficient, "new_rank": other_rank, "alpha": other_rank,
        "primary": "same-rank STP-minus-native treatment effects",
    }
    locked_record(root / "final_rank/preregistration.json", payload)
    condition = "stp_paper" if formulation == "paper" else "stp_released"
    # Native at the other rank already exists from Stage A or B.
    label = f"{formulation}_r{other_rank}_l{coefficient:g}"
    checkpoints = train_pair(
        root / "final_rank", condition=condition, label=label,
        rank=other_rank, alpha=other_rank, coefficient=coefficient,
    )
    for seed in SEEDS:
        evaluation = root / f"final_rank/{label}/seed_{seed}/evaluation"
        evaluate(checkpoints[seed], evaluation)
        teacher(checkpoints[seed], evaluation / "teacher_forced.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--through", choices=("a", "b", "c", "d", "final"), default="final"
    )
    args = parser.parse_args()
    root = args.output_root.resolve()
    freeze_panel()
    locked_record(root / "panel.json", {
        "source": str(SOURCE_PANEL), "source_sha256": sha256(SOURCE_PANEL),
        "panel": str(PANEL), "panel_sha256": sha256(PANEL), "reactions": 512,
        "selection": "first 512 rows of the frozen 1280 endpoint",
    })
    stages = (("a", stage_a), ("b", stage_b), ("c", stage_c), ("d", stage_d), ("final", final_rank))
    for name, function in stages:
        function(root)
        if name == args.through:
            break


if __name__ == "__main__":
    main()
