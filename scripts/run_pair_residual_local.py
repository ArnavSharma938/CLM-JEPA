"""Run the preregistered native-versus-pair-residual experiment on one GPU."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (533, 917, 1301)
MANIFEST = ROOT / "data/clm_jepa_uspto_mit_official_endpoint/prespecified_stage1_256.jsonl"
TRAIN_MANIFEST = ROOT / "data/clm_jepa_uspto_mit_pilot_1280/uspto_mit_train.csv"
VALIDATION_MANIFEST = ROOT / "data/clm_jepa_uspto_mit_validation_256/uspto_mit_validation_length_stratified_256.csv"


def run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    print(json.dumps({"launch": command}), flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def validate_training(path: Path, condition: str, seed: int) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload["condition"] != condition or payload["seed"] != seed:
        raise ValueError(f"training identity mismatch: {path}")
    if payload["selected_epoch"] != 4 or payload["compute"]["optimizer_steps"] != 320:
        raise ValueError(f"incomplete fixed trajectory: {path}")
    if condition == "clm_jepa_pair_residual":
        active = [row for row in payload["curves"] if row["jepa_active"]]
        if not active or any(row["gradient_interaction"] is None for row in active):
            raise ValueError(f"missing residual diagnostics: {path}")
    return payload


def prune_unselected_checkpoints(training_root: Path, selected_checkpoint: Path) -> None:
    checkpoints = (training_root / "checkpoints").resolve()
    selected = selected_checkpoint.resolve()
    experiment_root = (ROOT / "runs/pair_residual").resolve()
    if experiment_root not in checkpoints.parents or checkpoints not in selected.parents:
        raise ValueError("refusing to prune checkpoints outside the experiment root")
    for epoch in (1, 2, 3):
        candidate = (checkpoints / f"epoch_{epoch}").resolve()
        if candidate.exists() and candidate != selected:
            shutil.rmtree(candidate)


def train_condition(
    python: str, output_root: Path, seed: int, label: str,
    *, batch_size: int, accumulation_steps: int,
    gradient_checkpointing: bool, pin_memory: bool,
) -> dict:
    condition = "native" if label == "native" else "clm_jepa_pair_residual"
    training_root = output_root / f"seed_{seed}" / label / "training"
    result_path = training_root / "result.json"
    if result_path.exists():
        return validate_training(result_path, condition, seed)
    command = [
        python, "-u", "src/train.py",
        "--gate", "5",
        "--dataset", "uspto_mit_synthesis",
        "--condition", condition,
        "--seed", str(seed),
        "--learning-rate", "1e-4",
        "--k", "0",
        "--lambda-eff", "1.0",
        "--dropout", "0.5",
        "--epochs", "4",
        "--stop-after-epoch", "4",
        "--evaluation-epochs", "4",
        "--batch-size", str(batch_size),
        "--gradient-accumulation-steps", str(accumulation_steps),
        "--fused-adamw",
        "--attention-implementation", "sdpa",
        "--eval-generation-batch-size", "1",
        "--train-manifest", str(TRAIN_MANIFEST),
        "--validation-manifest", str(VALIDATION_MANIFEST),
        "--checkpoint-dir", str(training_root / "checkpoints"),
        "--max-validation-rows", "2",
        "--no-wandb",
        "--output", str(result_path),
    ]
    command.append(
        "--gradient-checkpointing" if gradient_checkpointing
        else "--no-gradient-checkpointing"
    )
    command.append("--pin-memory" if pin_memory else "--no-pin-memory")
    if label == "residual":
        command.extend(("--sigreg-batch-size", "16"))
    run(command)
    payload = validate_training(result_path, condition, seed)
    prune_unselected_checkpoints(training_root, Path(payload["selected_checkpoint"]))
    return payload


def evaluate_condition(
    python: str, output_root: Path, seed: int, label: str, checkpoint: Path,
    *, workers: int, threads_per_worker: int, teacher_batch_size: int,
) -> None:
    evaluation_root = output_root / f"seed_{seed}" / label / "evaluation"
    prediction_path = evaluation_root / "predictions.jsonl"
    if not prediction_path.exists():
        env = os.environ.copy()
        env.update({
            "PYTHONPATH": "src",
            "TOKENIZERS_PARALLELISM": "false",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "RAYON_NUM_THREADS": "1",
            "CHEMFM_EXACT_PREALLOCATED_CACHE": "1",
            "CHEMFM_EXACT_BEAM_SCORER": "1",
            "CHEMFM_DECODE_BATCH_SIZE": "10",
            "CHEMFM_EXACT_LORA_FASTPATH": "1",
            "CHEMFM_EXACT_LORA_CUDAGRAPH": "1",
            "CHEMFM_EXACT_LAYER_CUDAGRAPH": "1",
            "CHEMFM_EXACT_RMSNORM_CUDAGRAPH": "1",
            "CHEMFM_EXACT_ROPE_CUDAGRAPH": "1",
        })
        run([
            python, "-u", "src/eval_uspto_mit_five_view_a6000.py", "run",
            "--checkpoint", str(checkpoint),
            "--manifest", str(MANIFEST),
            "--workers", str(workers),
            "--threads-per-worker", str(threads_per_worker),
            "--prompt-batch-size", "1",
            "--batch-mode", "left-pad",
            "--output-dir", str(evaluation_root),
        ], env=env)
    teacher_path = evaluation_root / "teacher_forced.json"
    if not teacher_path.exists():
        run([
            python, "-u", "scripts/eval_teacher_forced_five_view.py",
            "--checkpoint", str(checkpoint),
            "--manifest", str(MANIFEST),
            "--batch-size", str(teacher_batch_size),
            "--output", str(teacher_path),
        ])


def evaluate_representations(
    python: str, output_root: Path, payloads: dict[int, dict[str, dict]],
    *, batch_size: int,
) -> None:
    output = output_root / "representation_256.json"
    if output.exists():
        return
    command = [
        python, "-u", "src/representation_eval.py",
        "--dataset", "uspto_mit_synthesis",
        "--validation-manifest", str(MANIFEST),
        "--seed", "533", "--k", "0",
        "--diagnostic-limit", "256",
        "--diagnostic-batch-size", str(batch_size),
        "--output", str(output),
    ]
    for seed in SEEDS:
        for label in ("native", "residual"):
            command.extend((
                "--legacy-checkpoint", f"seed_{seed}_{label}",
                str(payloads[seed][label]["selected_checkpoint"]),
            ))
    run(command)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=ROOT / "runs/pair_residual")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--threads-per-worker", type=int, default=1)
    parser.add_argument("--teacher-batch-size", type=int, default=4)
    parser.add_argument("--representation-batch-size", type=int, default=4)
    parser.add_argument("--training-concurrency", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument(
        "--gradient-checkpointing", action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--pin-memory", action=argparse.BooleanOptionalAction, default=False,
    )
    parser.add_argument(
        "--phase", choices=("all", "train", "evaluate", "analyze"), default="all"
    )
    args = parser.parse_args()
    if args.batch_size * args.gradient_accumulation_steps != 16:
        parser.error("physical batch * accumulation must equal the locked logical batch 16")
    if args.training_concurrency < 1 or args.training_concurrency > 3:
        parser.error("training concurrency must be between 1 and 3")
    output_root = args.output_root.resolve()
    python = sys.executable

    payloads: dict[int, dict[str, dict]] = {}
    if args.phase in {"all", "train"}:
        payloads = {seed: {} for seed in SEEDS}
        jobs = [(seed, label) for seed in SEEDS for label in ("native", "residual")]
        with ThreadPoolExecutor(max_workers=args.training_concurrency) as executor:
            futures = {
                executor.submit(
                    train_condition,
                    python, output_root, seed, label,
                    batch_size=args.batch_size,
                    accumulation_steps=args.gradient_accumulation_steps,
                    gradient_checkpointing=args.gradient_checkpointing,
                    pin_memory=args.pin_memory,
                ): (seed, label)
                for seed, label in jobs
            }
            for future in as_completed(futures):
                seed, label = futures[future]
                payloads[seed][label] = future.result()
    if args.phase in {"all", "evaluate"}:
        for seed in SEEDS:
            payloads.setdefault(seed, {})
            for label in ("native", "residual"):
                condition = "native" if label == "native" else "clm_jepa_pair_residual"
                result_path = output_root / f"seed_{seed}" / label / "training/result.json"
                payload = validate_training(result_path, condition, seed)
                payloads[seed][label] = payload
                evaluate_condition(
                    python, output_root, seed, label,
                    Path(payload["selected_checkpoint"]),
                    workers=args.workers,
                    threads_per_worker=args.threads_per_worker,
                    teacher_batch_size=args.teacher_batch_size,
                )
            paired_path = output_root / f"seed_{seed}" / "paired_summary.json"
            if not paired_path.exists():
                run([
                    python, "-u", "src/eval_uspto_mit_five_view_a6000.py", "summarize",
                    "--manifest", str(MANIFEST),
                    "--native-predictions", str(
                        output_root / f"seed_{seed}/native/evaluation/predictions.jsonl"
                    ),
                    "--clm-predictions", str(
                        output_root / f"seed_{seed}/residual/evaluation/predictions.jsonl"
                    ),
                    "--seed", str(seed),
                    "--output", str(paired_path),
                ])
        evaluate_representations(
            python, output_root, payloads,
            batch_size=args.representation_batch_size,
        )
    if args.phase in {"all", "analyze"}:
        run([
            python, "-u", "scripts/analyze_pair_residual_results.py",
            "--root", str(output_root),
            "--output", str(output_root / "summary.json"),
        ])


if __name__ == "__main__":
    main()
