"""Run the two prespecified faithful-NextLat development cells sequentially."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from chemfm import (MODEL_DIR, TOKENIZER_DIR, ReactionCollator,
                    add_predictor_tokens, load_lora_model, load_reaction_tokenizer)
from decoder_projected import eligible_transitions
from faithful_nextlat import FaithfulNextLatObjective
from run_stp_matrix import PANEL, TRAIN, VALIDATION, evaluation_env, read_jsonl, sha256
from run_stp_completion import native_evaluation
from train import read_rows, validate_serialization_endings

OUT = ROOT / "runs/faithful_nextlat"
PANEL_HASH = "a2e6202a4abaf9a70f4700e04299a09964d38c10fce004022dc43e759aa6057d"


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def launch(command, log, env=None):
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        subprocess.run(command, cwd=ROOT, env=env, stdout=handle,
                       stderr=subprocess.STDOUT, check=True)


def preflight():
    import torch
    rows = read_rows("uspto_mit_synthesis", path=TRAIN)
    groups = Counter(row["group_id"] for row in rows)
    assert len(rows) == 1280 and len(groups) == 256 and set(groups.values()) == {5}
    assert sha256(PANEL) == PANEL_HASH and len(read_jsonl(PANEL)) == 512
    tokenizer = load_reaction_tokenizer(TOKENIZER_DIR)
    native_vocab = len(tokenizer)
    add_predictor_tokens(tokenizer)
    collator = ReactionCollator(tokenizer)
    validate_serialization_endings(collator, rows, tokenizer.eos_token_id)
    transitions = sum(int(eligible_transitions(
        collator([row]), tokenizer.convert_tokens_to_ids("<prostart>"),
        tokenizer.eos_token_id).sum()) for row in rows)
    controls = {}
    panel_ids = [row["reaction_identity"] for row in read_jsonl(PANEL)]
    for seed in (533, 917):
        prediction_path = native_evaluation(8, seed) / "predictions.jsonl"
        predictions = read_jsonl(prediction_path)
        assert [row["reaction_identity"] for row in predictions] == panel_ids
        controls[str(seed)] = {"path": str(prediction_path), "sha256": sha256(prediction_path)}
    value = {
        "train_manifest": str(TRAIN.resolve()), "train_sha256": sha256(TRAIN),
        "validation_manifest": str(VALIDATION.resolve()), "validation_sha256": sha256(VALIDATION),
        "panel": str(PANEL.resolve()), "panel_sha256": PANEL_HASH,
        "rows": 1280, "reactions": 256, "views": 5,
        "eligible_transitions": transitions, "controls": controls,
        "model": str(MODEL_DIR.resolve()), "cuda": torch.cuda.get_device_name(0),
        "vram_bytes": torch.cuda.get_device_properties(0).total_memory,
    }
    write_json(OUT / "preflight.json", value)
    return value


def smoke():
    import time
    import torch
    tokenizer = load_reaction_tokenizer(TOKENIZER_DIR)
    native_vocab = len(tokenizer)
    add_predictor_tokens(tokenizer)
    collator = ReactionCollator(tokenizer)
    rows = read_rows("uspto_mit_synthesis", path=TRAIN)
    row = min(rows, key=lambda value: collator([value])["input_ids"].numel())
    model = load_lora_model(MODEL_DIR, tokenizer, chemfm_vocab_size=native_vocab,
                            attn_implementation="sdpa", lora_rank=8, lora_alpha=8).cuda().train()
    method = FaithfulNextLatObjective(
        model.get_output_embeddings().weight[:native_vocab],
        product_start_token_id=tokenizer.convert_tokens_to_ids("<prostart>"),
        eos_token_id=tokenizer.eos_token_id, native_vocab_size=native_vocab).cuda()
    batch = {k: v.cuda() for k, v in collator([row]).items() if torch.is_tensor(v)}
    torch.cuda.reset_peak_memory_stats(); started = time.perf_counter()
    output = method(model, batch); output.loss.backward()
    assert torch.isfinite(output.loss)
    assert method.predictor.mlp[0].weight.grad.norm() > 0
    assert method.predictor.mlp[-1].weight.grad.norm() > 0
    assert model.get_input_embeddings().modules_to_save["USPTO-MIT-Synthesis"].weight.grad.norm() > 0
    assert model.get_output_embeddings().modules_to_save["USPTO-MIT-Synthesis"].weight.grad is not None
    assert not any("predictor" in key for key in model.state_dict())
    value = {"status": "passed", "sequence_length": int(batch["input_ids"].shape[1]),
             "native_loss": float(output.native_loss), "latent": float(output.lz),
             "kl": float(output.kl), "peak_vram_bytes": torch.cuda.max_memory_allocated(),
             "seconds": time.perf_counter() - started, "optimizer_steps": 0,
             "metadata": method.metadata}
    write_json(OUT / "smoke.json", value)
    return value


def run_seed(seed):
    base = OUT / f"seed_{seed}"
    training = base / "training"
    result = training / "result.json"
    checkpoint = training / "checkpoints/epoch_4"
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "TOKENIZERS_PARALLELISM": "false"}
    if not result.exists():
        command = [sys.executable, "-u", "src/train.py", "--gate", "5",
            "--dataset", "uspto_mit_synthesis", "--condition", "faithful_nextlat",
            "--seed", str(seed), "--learning-rate", "1e-4", "--lora-rank", "8",
            "--lora-alpha", "8", "--epochs", "4", "--stop-after-epoch", "4",
            "--evaluation-epochs", "4", "--batch-size", "4",
            "--gradient-accumulation-steps", "4", "--no-gradient-checkpointing",
            "--fused-adamw", "--attention-implementation", "sdpa", "--pin-memory",
            "--eval-generation-batch-size", "1", "--train-manifest", str(TRAIN),
            "--validation-manifest", str(VALIDATION), "--max-validation-rows", "2",
            "--checkpoint-dir", str(training / "checkpoints"), "--no-wandb",
            "--output", str(result)]
        complete = [training / f"checkpoints/epoch_{epoch}" for epoch in (1, 2, 3, 4)
                    if (training / f"checkpoints/epoch_{epoch}/training_state.pt").exists()]
        if complete:
            command += ["--resume-from", str(complete[-1])]
        launch(command, training / "train.log", env)
    payload = json.loads(result.read_text())
    assert payload["condition"] == "faithful_nextlat" and payload["seed"] == seed
    assert payload["compute"]["optimizer_steps"] == 320 and payload["selected_epoch"] == 4
    assert payload["config"]["train_manifest_sha256"] == sha256(TRAIN)
    evaluation = base / "evaluation"
    if len(read_jsonl(evaluation / "predictions.jsonl")) != 512 if (evaluation / "predictions.jsonl").exists() else True:
        launch([sys.executable, "-u", "src/eval_uspto_mit_five_view_a6000.py", "run",
                "--checkpoint", str(checkpoint), "--manifest", str(PANEL), "--workers", "4",
                "--threads-per-worker", "1", "--prompt-batch-size", "1", "--batch-mode", "left-pad",
                "--output-dir", str(evaluation)], evaluation / "evaluate.log", evaluation_env())
    comparison = base / "comparison.json"
    launch([sys.executable, "src/eval_uspto_mit_five_view_a6000.py", "summarize",
            "--manifest", str(PANEL), "--native-predictions",
            str(native_evaluation(8, seed) / "predictions.jsonl"), "--clm-predictions",
            str(evaluation / "predictions.jsonl"), "--seed", str(seed), "--output",
            str(comparison)], base / "comparison.log")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("preflight", "smoke", "run"))
    args = parser.parse_args()
    preflight()
    if args.stage == "preflight": return
    smoke()
    if args.stage == "smoke": return
    for seed in (533, 917): run_seed(seed)
    write_json(OUT / "run_complete.json", {"status": "complete", "seeds": [533, 917]})


if __name__ == "__main__":
    main()
