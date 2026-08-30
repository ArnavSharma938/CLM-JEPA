"""Run the preregistered official-STP ChemFM condition on one GPU."""

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
NATIVE_ROOT = ROOT / "runs/pair_residual/a6000/results"


def run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    print(json.dumps({"launch": command}), flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_training(path: Path, seed: int) -> dict:
    payload = read_json(path)
    if payload["condition"] != "stp" or payload["seed"] != seed:
        raise ValueError(f"training identity mismatch: {path}")
    if payload["selected_epoch"] != 4 or payload["compute"]["optimizer_steps"] != 320:
        raise ValueError(f"incomplete fixed STP trajectory: {path}")
    config = payload["config"]
    if (
        config["stp_upstream_commit"]
        != "ea0017c654ad917066ff32afc88276bea8ca5f7e"
        or config["stp_lambda"] != 0.02
        or config["stp_hidden_layer"] != "final"
        or config["stp_spans_per_example"] != 1
    ):
        raise ValueError(f"STP fidelity metadata mismatch: {path}")
    return payload


def native_payload(seed: int) -> dict:
    payload = read_json(NATIVE_ROOT / f"seed_{seed}/native/training/result.json")
    if payload["condition"] != "native" or payload["seed"] != seed:
        raise ValueError(f"native control identity mismatch for seed {seed}")
    return payload


def prune_unselected_checkpoints(training_root: Path, selected_checkpoint: Path) -> None:
    checkpoints = (training_root / "checkpoints").resolve()
    selected = selected_checkpoint.resolve()
    experiment_root = (ROOT / "runs/stp").resolve()
    if experiment_root not in checkpoints.parents or checkpoints not in selected.parents:
        raise ValueError("refusing to prune checkpoints outside the STP experiment root")
    for epoch in (1, 2, 3):
        candidate = (checkpoints / f"epoch_{epoch}").resolve()
        if candidate.exists() and candidate != selected:
            shutil.rmtree(candidate)


def train_seed(
    python: str, output_root: Path, seed: int,
    *, batch_size: int, accumulation_steps: int,
) -> dict:
    training_root = output_root / f"seed_{seed}/stp/training"
    result_path = training_root / "result.json"
    if not result_path.exists():
        run([
            python, "-u", "src/train.py",
            "--gate", "5",
            "--dataset", "uspto_mit_synthesis",
            "--condition", "stp",
            "--seed", str(seed),
            "--learning-rate", "1e-4",
            "--stp-lambda", "0.02",
            "--k", "0",
            "--lambda-eff", "1.0",
            "--dropout", "0.5",
            "--epochs", "4",
            "--stop-after-epoch", "4",
            "--evaluation-epochs", "4",
            "--batch-size", str(batch_size),
            "--gradient-accumulation-steps", str(accumulation_steps),
            "--no-gradient-checkpointing",
            "--fused-adamw",
            "--attention-implementation", "sdpa",
            "--pin-memory",
            "--eval-generation-batch-size", "1",
            "--train-manifest", str(TRAIN_MANIFEST),
            "--validation-manifest", str(VALIDATION_MANIFEST),
            "--max-validation-rows", "2",
            "--checkpoint-dir", str(training_root / "checkpoints"),
            "--no-wandb",
            "--output", str(result_path),
        ])
    payload = validate_training(result_path, seed)
    native = native_payload(seed)
    if (
        payload["config"]["initial_trainable_sha256"]
        != native["config"]["initial_trainable_sha256"]
    ):
        raise ValueError(f"initial trainable state mismatch for seed {seed}")
    for key in (
        "train_manifest_sha256", "optimizer", "fused_adamw",
        "physical_batch_size", "gradient_accumulation_steps",
        "effective_batch_size", "learning_rate", "epochs",
        "attention_implementation", "gradient_checkpointing",
    ):
        if payload["config"][key] != native["config"][key]:
            raise ValueError(f"native/STP configuration mismatch for {key}, seed {seed}")
    prune_unselected_checkpoints(training_root, Path(payload["selected_checkpoint"]))
    return payload


def evaluate_seed(
    python: str, output_root: Path, seed: int, checkpoint: Path,
    *, workers: int, teacher_batch_size: int,
) -> None:
    evaluation_root = output_root / f"seed_{seed}/stp/evaluation"
    prediction_path = evaluation_root / "predictions.jsonl"
    if not prediction_path.exists():
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
        run([
            python, "-u", "src/eval_uspto_mit_five_view_a6000.py", "run",
            "--checkpoint", str(checkpoint),
            "--manifest", str(MANIFEST),
            "--workers", str(workers), "--threads-per-worker", "1",
            "--prompt-batch-size", "1", "--batch-mode", "left-pad",
            "--output-dir", str(evaluation_root),
        ], env=env)
    teacher_path = evaluation_root / "teacher_forced.json"
    if not teacher_path.exists():
        run([
            python, "-u", "scripts/eval_teacher_forced_five_view.py",
            "--checkpoint", str(checkpoint), "--manifest", str(MANIFEST),
            "--batch-size", str(teacher_batch_size), "--output", str(teacher_path),
        ])
    paired = output_root / f"seed_{seed}/paired_summary.json"
    if not paired.exists():
        run([
            python, "-u", "src/eval_uspto_mit_five_view_a6000.py", "summarize",
            "--manifest", str(MANIFEST),
            "--native-predictions", str(
                NATIVE_ROOT / f"seed_{seed}/native/evaluation/predictions.jsonl"
            ),
            "--clm-predictions", str(prediction_path),
            "--seed", str(seed), "--output", str(paired),
        ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=ROOT / "runs/stp/a6000/results")
    parser.add_argument("--phase", choices=("all", "train", "evaluate"), default="all")
    parser.add_argument("--training-concurrency", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--teacher-batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.batch_size * args.gradient_accumulation_steps != 16:
        parser.error("physical batch times accumulation must equal 16")
    if not 1 <= args.training_concurrency <= 3:
        parser.error("training concurrency must be in [1,3]")
    output_root = args.output_root.resolve()
    python = sys.executable
    payloads: dict[int, dict] = {}
    if args.phase in {"all", "train"}:
        with ThreadPoolExecutor(max_workers=args.training_concurrency) as executor:
            futures = {
                executor.submit(
                    train_seed, python, output_root, seed,
                    batch_size=args.batch_size,
                    accumulation_steps=args.gradient_accumulation_steps,
                ): seed for seed in SEEDS
            }
            for future in as_completed(futures):
                seed = futures[future]
                payloads[seed] = future.result()
                print(json.dumps({"training_complete": seed}), flush=True)
    if args.phase in {"all", "evaluate"}:
        for seed in SEEDS:
            payload = payloads.get(seed) or validate_training(
                output_root / f"seed_{seed}/stp/training/result.json", seed
            )
            evaluate_seed(
                python, output_root, seed, Path(payload["selected_checkpoint"]),
                workers=args.workers, teacher_batch_size=args.teacher_batch_size,
            )


if __name__ == "__main__":
    main()
