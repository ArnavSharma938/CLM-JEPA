from __future__ import annotations

import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from esm2_gate1.modeling import configure_adaptation, dense_targets

from .batching import GenerationCollator, GenerationDataset, length_bucket_batches
from .config import MODEL_ID, MODEL_REVISION
from .attention import patch_esm_sdpa
from .masking import per_example_loss_sums, selected_cross_entropy_sum
from .prepare_data import sha256_file


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_base_model(attention_backend: str = "sdpa"):
    from transformers import AutoTokenizer, EsmForMaskedLM

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    if attention_backend not in {"sdpa", "eager"}:
        raise ValueError(f"unsupported ESM attention backend: {attention_backend}")
    model = EsmForMaskedLM.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    if attention_backend == "sdpa":
        patch_esm_sdpa(model)
    dense_targets(model)
    return model, tokenizer


def make_loader(
    dataset: GenerationDataset,
    token_budget: int,
    *,
    seed: int,
    epoch: int = 0,
    shuffle: bool = False,
    workers: int = 2,
) -> tuple[DataLoader, list[list[int]]]:
    batches = length_bucket_batches(dataset.lengths, token_budget, seed, epoch, shuffle)
    loader = DataLoader(
        dataset,
        batch_sampler=batches,
        collate_fn=GenerationCollator(dataset.cache["special_tokens"]),
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )
    return loader, batches


def expected_optimizer_steps(
    dataset: GenerationDataset,
    token_budget: int,
    supervised_budget: int,
    *,
    seed: int,
    epochs: int,
) -> int:
    total = 0
    for epoch in range(epochs):
        batches = length_bucket_batches(dataset.lengths, token_budget, seed, epoch, True)
        accumulated = 0
        for batch in batches:
            accumulated += sum(len(dataset.items[index]["peptide_tokens"]) for index in batch)
            if accumulated >= supervised_budget:
                total += 1
                accumulated = 0
        if accumulated:
            total += 1
    return total


def learning_rate_at(step: int, total_steps: int, peak_lr: float) -> float:
    warmup_steps = max(1, math.ceil(total_steps * 0.05))
    if step <= warmup_steps:
        return peak_lr * step / warmup_steps
    decay_steps = max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, (step - warmup_steps) / decay_steps))
    return peak_lr * (0.1 + 0.9 * (1.0 + math.cos(math.pi * progress)) / 2.0)


def _atomic_json(path: Path, value: dict | list) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _trainable_state(model) -> dict[str, torch.Tensor]:
    model = getattr(model, "_orig_mod", model)
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters() if parameter.requires_grad
    }


def _save_best(path: Path, model, config: dict, event: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({"config": config, "event": event, "trainable_state": _trainable_state(model)}, temporary)
    os.replace(temporary, path)


def _save_latest(path: Path, model, optimizer, progress: dict, config: dict) -> None:
    raw_model = getattr(model, "_orig_mod", model)
    payload = {
        "config": config,
        "progress": progress,
        "trainable_state": _trainable_state(model),
        "optimizer": optimizer.state_dict(),
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all(),
        },
        "gradient_state": {
            name: parameter.grad.detach().cpu()
            for name, parameter in raw_model.named_parameters()
            if parameter.requires_grad and parameter.grad is not None
        },
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _restore_latest(path: Path, model, optimizer) -> dict:
    raw_model = getattr(model, "_orig_mod", model)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    parameters = dict(raw_model.named_parameters())
    expected = {name for name, value in parameters.items() if value.requires_grad}
    if set(payload["trainable_state"]) != expected:
        raise RuntimeError("resume checkpoint trainable set does not match the configured model")
    with torch.no_grad():
        for name, value in payload["trainable_state"].items():
            parameters[name].copy_(value.to(parameters[name].device))
        for name, value in payload.get("gradient_state", {}).items():
            parameters[name].grad = value.to(parameters[name].device)
    optimizer.load_state_dict(payload["optimizer"])
    random.setstate(payload["rng"]["python"])
    np.random.set_state(payload["rng"]["numpy"])
    torch.set_rng_state(payload["rng"]["torch"])
    torch.cuda.set_rng_state_all(payload["rng"]["cuda"])
    return payload


@torch.no_grad()
def evaluate_examples(
    model,
    examples: list[dict],
    cache: dict,
    *,
    token_budget: int,
    workers: int = 2,
    seed: int = 0,
) -> dict:
    """Evaluate full-span peptide NLL with one vector transfer per microbatch."""
    class EvaluationDataset(torch.utils.data.Dataset):
        def __init__(self):
            self.cache = cache
            self.items = examples
            self.lengths = [len(x["receptor_tokens"]) + len(x["peptide_tokens"]) + 2 for x in examples]

        def __len__(self):
            return len(self.items)

        def __getitem__(self, index):
            return self.items[index]

    dataset = EvaluationDataset()
    sampler = length_bucket_batches(dataset.lengths, token_budget, seed, shuffle=False)
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=GenerationCollator(cache["special_tokens"]),
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )
    device = torch.device("cuda")
    results: list[dict] = []
    model.eval()
    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        sums, counts = per_example_loss_sums(logits, labels)
        combined_cpu = torch.stack((sums, counts.to(sums.dtype)), dim=1).cpu()
        for row, values in zip(batch["rows"], combined_cpu.tolist()):
            loss_sum, token_count = values
            results.append({
                "pair_id": row["pair_id"],
                "receptor_id": row.get("evaluated_receptor_id", row["receptor_id"]),
                "true_receptor_id": row["receptor_id"],
                "receptor_cluster_id": row.get("receptor_cluster_id"),
                "peptide_sequence": row["peptide_sequence"],
                "receptor_sequence": row["receptor_sequence"],
                "loss_sum": loss_sum,
                "supervised_tokens": int(token_count),
                "peptide_nll": loss_sum / token_count,
            })
    total_loss = sum(row["loss_sum"] for row in results)
    total_tokens = sum(row["supervised_tokens"] for row in results)
    return {
        "peptide_nll": total_loss / total_tokens,
        "loss_sum": total_loss,
        "supervised_tokens": total_tokens,
        "pairs": len(results),
        "per_example": results,
    }


def _dataset_examples(dataset: GenerationDataset) -> list[dict]:
    return [
        {
            "row": item["row"],
            "receptor_tokens": item["receptor_tokens"],
            "peptide_tokens": item["peptide_tokens"],
        }
        for item in dataset.items
    ]


def _validate(
    model, dataset: GenerationDataset, *, token_budget: int, workers: int, epoch: float
) -> dict:
    validation = evaluate_examples(
        model, _dataset_examples(dataset), dataset.cache,
        token_budget=token_budget, workers=workers,
    )
    return {key: value for key, value in validation.items() if key != "per_example"} | {"epoch": epoch}


def train(
    *,
    train_manifest: Path,
    validation_manifest: Path,
    cache_path: Path,
    output_dir: Path,
    peak_lr: float,
    seed: int,
    token_budget: int = 12_288,
    supervised_budget: int = 8_192,
    maximum_epochs: int = 4,
    schedule_epochs: int | None = None,
    workers: int = 2,
    attention_backend: str = "sdpa",
    compile_mode: str | None = None,
    adaptation_mode: str = "dtft",
    maximum_optimizer_steps: int | None = None,
    initial_validation: dict | None = None,
    minimum_epochs_for_stopping: int = 0,
    stopping_patience: int | None = None,
    resume: Path | None = None,
    validation_every_half_epoch: bool = True,
) -> dict:
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Propedia DTFT requires a CUDA GPU with BF16 support")
    output_dir.mkdir(parents=True, exist_ok=True)
    wall_start = time.perf_counter()
    seed_everything(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda")
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    training_data = GenerationDataset(cache, train_manifest)
    validation_data = GenerationDataset(cache, validation_manifest)
    config = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "mode": adaptation_mode,
        "seed": seed,
        "peak_learning_rate": peak_lr,
        "optimizer": {"name": "AdamW", "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.01, "fused": True},
        "gradient_clip_norm": 1.0,
        "input_token_budget": token_budget,
        "supervised_peptide_token_budget": supervised_budget,
        "maximum_epochs": maximum_epochs,
        "schedule_epochs": schedule_epochs or maximum_epochs,
        "schedule_warmup_fraction": 0.05,
        "schedule_floor_fraction": 0.1,
        "attention_backend": attention_backend,
        "gradient_checkpointing": False,
        "compile_mode": compile_mode,
        "training_manifest_sha256": sha256_file(train_manifest),
        "validation_manifest_sha256": sha256_file(validation_manifest),
        "token_cache_sha256": sha256_file(cache_path),
    }
    _atomic_json(output_dir / "config.json", config)
    model, tokenizer = load_base_model(attention_backend)
    counts = configure_adaptation(model, adaptation_mode)
    expected_trainables = {
        "dtft": 648_806_400,
        "lora8": 6_082_560,
        "lora64": 48_660_480,
    }
    if adaptation_mode not in expected_trainables:
        raise ValueError(f"unsupported Propedia adaptation mode: {adaptation_mode}")
    if (
        counts["trainable_parameters"] != expected_trainables[adaptation_mode]
        or counts["target_matrix_count"] != 198
    ):
        raise RuntimeError(
            f"{adaptation_mode} trainable set is not exactly the expected 198-matrix target set"
        )
    model.to(device)
    if compile_mode:
        model = torch.compile(model, mode=compile_mode, dynamic=True)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters, lr=peak_lr, betas=(0.9, 0.999), eps=1e-8,
        weight_decay=0.01, fused=True,
    )
    planned_steps = expected_optimizer_steps(
        training_data, token_budget, supervised_budget, seed=seed,
        epochs=int(config["schedule_epochs"]),
    )
    progress = {
        "epoch": 0, "optimizer_steps": 0, "stale_checks": 0,
        "best_validation_nll": float("inf"),
        "accumulated_supervised_tokens": 0,
        "accumulated_input_tokens": 0,
    }
    history_path = output_dir / "history.json"
    history: list[dict] = []
    if resume is not None:
        payload = _restore_latest(resume, model, optimizer)
        progress = payload["progress"]
        history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.exists() else []
        if payload["config"]["training_manifest_sha256"] != config["training_manifest_sha256"]:
            raise RuntimeError("resume checkpoint train split changed")
        if payload["config"]["validation_manifest_sha256"] != config["validation_manifest_sha256"]:
            raise RuntimeError("resume checkpoint validation split changed")
        if payload["config"]["peak_learning_rate"] != peak_lr:
            raise RuntimeError("resume checkpoint peak learning rate changed")
        if payload["config"].get("mode", "dtft") != adaptation_mode:
            raise RuntimeError("resume checkpoint adaptation mode changed")
        if payload["config"].get("schedule_epochs") != config["schedule_epochs"]:
            raise RuntimeError("resume checkpoint schedule horizon changed")
        planned_steps = int(payload["progress"]["planned_steps"])
    else:
        optimizer.zero_grad(set_to_none=True)
        base = (
            _validate(model, validation_data, token_budget=token_budget, workers=workers, epoch=0.0)
            if initial_validation is None
            else {
                "peptide_nll": float(initial_validation["peptide_nll"]),
                "loss_sum": float(initial_validation.get("loss_sum", 0.0)),
                "supervised_tokens": int(initial_validation.get("supervised_tokens", 0)),
                "pairs": int(initial_validation.get("pairs", 0)),
                "epoch": 0.0,
            }
        )
        base["optimizer_step"] = 0
        base["learning_rate"] = 0.0
        history.append(base)
        _atomic_json(history_path, history)

    steps_path = output_dir / "optimizer_steps.jsonl"
    completed_epochs = int(progress["epoch"])
    stale_checks = int(progress.get("stale_checks", 0))
    best_nll = float(progress.get("best_validation_nll", float("inf")))
    torch.cuda.reset_peak_memory_stats()
    loop_start = time.perf_counter()
    total_nonpadding_tokens = 0
    total_padded_tokens = 0
    total_examples = 0
    maximum_gradient_norm = 0.0
    stability = "stable"
    stopped_early = False
    profile_cap_reached = False
    latest_lr = peak_lr
    training_extent_epochs = float(completed_epochs)

    def apply_optimizer_step(epoch_index: int, batch_index: int) -> None:
        nonlocal latest_lr, maximum_gradient_norm
        step = int(progress["optimizer_steps"]) + 1
        latest_lr = learning_rate_at(step, planned_steps, peak_lr)
        for group in optimizer.param_groups:
            group["lr"] = latest_lr
        selected = int(progress["accumulated_supervised_tokens"])
        if selected <= 0:
            raise RuntimeError("optimizer step has no supervised peptide tokens")
        for parameter in parameters:
            if parameter.grad is not None:
                parameter.grad.div_(selected)
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        if not torch.isfinite(norm):
            raise FloatingPointError("non-finite gradient norm after clipping")
        maximum_gradient_norm = max(maximum_gradient_norm, float(norm.detach().cpu()))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        progress["optimizer_steps"] = step
        event = {
            "epoch": epoch_index + 1,
            "batch": batch_index,
            "optimizer_step": step,
            "input_tokens": int(progress["accumulated_input_tokens"]),
            "supervised_peptide_tokens": selected,
            "gradient_norm_before_clip": float(norm.detach().cpu()),
            "learning_rate": latest_lr,
        }
        with steps_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, sort_keys=True) + "\n")
        progress["accumulated_supervised_tokens"] = 0
        progress["accumulated_input_tokens"] = 0

    for epoch_index in range(completed_epochs, maximum_epochs):
        loader, batches = make_loader(
            training_data, token_budget, seed=seed, epoch=epoch_index,
            shuffle=True, workers=workers,
        )
        model.train()
        checks = {max(1, len(batches) // 2), len(batches)} if validation_every_half_epoch else {len(batches)}
        for batch_index, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                loss_sum = selected_cross_entropy_sum(logits, labels)
            if not torch.isfinite(loss_sum):
                stability = "non-finite loss"
                raise FloatingPointError(stability)
            loss_sum.backward()
            progress["accumulated_supervised_tokens"] += int(batch["supervised_tokens"])
            progress["accumulated_input_tokens"] += int(batch["padded_input_tokens"])
            total_nonpadding_tokens += int(batch["real_input_tokens"])
            total_padded_tokens += int(batch["padded_input_tokens"])
            total_examples += len(batch["rows"])
            training_extent_epochs = epoch_index + (batch_index + 1) / len(batches)
            if progress["accumulated_supervised_tokens"] >= supervised_budget:
                apply_optimizer_step(epoch_index, batch_index)
                if maximum_optimizer_steps is not None and int(progress["optimizer_steps"]) >= maximum_optimizer_steps:
                    profile_cap_reached = True
                    break

            batch_number = batch_index + 1
            if batch_number in checks:
                check_epoch = epoch_index + batch_number / len(batches)
                validation = _validate(
                    model, validation_data, token_budget=token_budget,
                    workers=workers, epoch=check_epoch,
                )
                validation["optimizer_step"] = int(progress["optimizer_steps"])
                validation["learning_rate"] = latest_lr
                history.append(validation)
                current_nll = float(validation["peptide_nll"])
                improved_best = current_nll < best_nll
                meaningful = current_nll <= best_nll - 0.001
                if improved_best:
                    best_nll = current_nll
                    _save_best(output_dir / "best.pt", model, config, validation)
                if check_epoch >= minimum_epochs_for_stopping:
                    stale_checks = 0 if meaningful else stale_checks + 1
                progress["best_validation_nll"] = best_nll
                progress["stale_checks"] = stale_checks
                _atomic_json(history_path, history)
                model.train()
                if (
                    stopping_patience is not None
                    and check_epoch >= minimum_epochs_for_stopping
                    and stale_checks >= stopping_patience
                ):
                    stopped_early = True
                    break
            if maximum_optimizer_steps is not None and int(progress["optimizer_steps"]) >= maximum_optimizer_steps:
                profile_cap_reached = True
                break

        # Flush the final partial statistical batch before the epoch checkpoint.
        if profile_cap_reached:
            break
        if progress["accumulated_supervised_tokens"] > 0:
            apply_optimizer_step(epoch_index, len(batches) - 1)
        completed_epochs = epoch_index + 1
        if not stopped_early:
            training_extent_epochs = float(completed_epochs)
        progress["epoch"] = completed_epochs
        progress["stale_checks"] = stale_checks
        progress["planned_steps"] = planned_steps
        progress["best_validation_nll"] = best_nll
        if not stopped_early:
            _save_latest(output_dir / "latest.pt", model, optimizer, progress, config)
        if stopped_early:
            break

    wall_seconds = time.perf_counter() - wall_start
    loop_seconds = time.perf_counter() - loop_start
    history_path.write_text(json.dumps(history, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    measured = [event for event in history if event.get("epoch", 0) > 0]
    best = min(measured or history, key=lambda event: event["peptide_nll"])
    execution = {
        **progress,
        "epochs_completed": training_extent_epochs,
        "completed_full_epochs": int(math.floor(training_extent_epochs)),
        "optimizer_steps": int(progress["optimizer_steps"]),
        "best_epoch": best["epoch"],
        "best_validation_peptide_nll": float(best["peptide_nll"]),
        "terminal_validation_peptide_nll": float(history[-1]["peptide_nll"]),
        "stopped_early": stopped_early,
        "profile_step_cap_reached": profile_cap_reached,
        "maximum_gradient_norm_before_clip": maximum_gradient_norm,
        "numerical_stability": stability,
        "wall_seconds": wall_seconds,
        "training_loop_seconds": loop_seconds,
        "real_input_tokens": total_nonpadding_tokens,
        "padded_input_tokens": total_padded_tokens,
        "examples_seen": total_examples,
        "input_tokens_per_second": total_nonpadding_tokens / max(loop_seconds, 1e-9),
        "peak_vram_bytes": torch.cuda.max_memory_allocated(),
        "last_learning_rate": latest_lr,
        "best_checkpoint": str(output_dir / "best.pt"),
        "latest_checkpoint": str(output_dir / "latest.pt"),
        "training_manifest_sha256": config["training_manifest_sha256"],
        "validation_manifest_sha256": config["validation_manifest_sha256"],
    }
    _atomic_json(output_dir / "execution.json", execution)
    return execution
