from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .batching import TokenBudgetBatchSampler
from .config import CANONICAL_AA, TrainConfig
from .manifests import load_manifest
from .masking import corrupt_tokens
from .modeling import configure_adaptation, load_model


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ProteinDataset(Dataset):
    def __init__(self, manifest: Path, tokenizer) -> None:
        self.rows = load_manifest(manifest)
        sequences = [row["sequence"] for row in self.rows]
        encoded = tokenizer(sequences, add_special_tokens=True, padding=False, truncation=False)
        self.token_ids = [torch.tensor(values, dtype=torch.long) for values in encoded["input_ids"]]
        for row, tokens in zip(self.rows, self.token_ids):
            if tokens.numel() != int(row["length"]) + 2:
                raise RuntimeError(f"tokenization changed residue count for {row['sequence_id']}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        return {"row": self.rows[index], "token_ids": self.token_ids[index]}


class MLMCollator:
    def __init__(self, tokenizer, *, exposure: int, common_seed: int, realization: int = 0) -> None:
        self.pad_id = tokenizer.pad_token_id
        self.mask_id = tokenizer.mask_token_id
        self.canonical_ids = tokenizer.convert_tokens_to_ids(list(CANONICAL_AA))
        if any(value == tokenizer.unk_token_id for value in self.canonical_ids):
            raise RuntimeError("ESM tokenizer does not resolve every canonical amino acid")
        self.exposure = exposure
        self.common_seed = common_seed
        self.realization = realization

    def __call__(self, examples: list[dict]) -> dict:
        maximum = max(item["token_ids"].numel() for item in examples)
        input_ids = torch.full((len(examples), maximum), self.pad_id, dtype=torch.long)
        labels = torch.full_like(input_ids, -100)
        attention_mask = torch.zeros_like(input_ids)
        rows = []
        selected_counts = []
        for index, item in enumerate(examples):
            tokens = item["token_ids"]
            row = item["row"]
            corruption = corrupt_tokens(
                tokens,
                range(1, tokens.numel() - 1),
                self.canonical_ids,
                self.mask_id,
                sequence_id=row["sequence_id"],
                exposure=self.exposure,
                common_seed=self.common_seed,
                realization=self.realization,
            )
            input_ids[index, : tokens.numel()] = corruption.input_ids
            labels[index, : tokens.numel()] = corruption.labels
            attention_mask[index, : tokens.numel()] = 1
            rows.append(row)
            selected_counts.append(len(corruption.selected_positions))
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "rows": rows,
            "selected_counts": selected_counts,
            "nonpadding_residues": sum(int(row["length"]) for row in rows),
        }


def make_loader(dataset: ProteinDataset, tokenizer, config: TrainConfig, epoch: int, training: bool) -> DataLoader:
    sampler = TokenBudgetBatchSampler(
        [int(row["length"]) for row in dataset.rows],
        config.microbatch_tokens,
        config.seed if training else 0,
        shuffle=training,
    )
    sampler.set_epoch(epoch if training else 0)
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=MLMCollator(
            tokenizer, exposure=epoch if training else 0, common_seed=config.common_mask_seed
        ),
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=config.num_workers > 0,
    )


@torch.no_grad()
def evaluate_nll(model, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    loss_sum = 0.0
    token_count = 0
    protein_values = []
    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        losses = F.cross_entropy(logits.float().transpose(1, 2), labels, reduction="none", ignore_index=-100)
        for index, row in enumerate(batch["rows"]):
            selected = labels[index].ne(-100)
            value = float(losses[index][selected].sum().item())
            count = int(selected.sum().item())
            loss_sum += value
            token_count += count
            protein_values.append({"backbone_id": row["backbone_id"], "loss_sum": value, "masked_tokens": count, "nll": value / count})
    return {
        "masked_token_nll": loss_sum / token_count,
        "mean_protein_nll": float(np.mean([item["nll"] for item in protein_values])),
        "masked_tokens": token_count,
        "proteins": protein_values,
    }


def _trainable_state(model) -> dict[str, torch.Tensor]:
    model = getattr(model, "_orig_mod", model)
    return {name: value.detach().cpu() for name, value in model.named_parameters() if value.requires_grad}


def save_checkpoint(path: Path, model, optimizer, config: TrainConfig, progress: dict, best_nll: float) -> None:
    raw_model = getattr(model, "_orig_mod", model)
    payload = {
        "config": config.serializable(),
        "progress": progress,
        "best_validation_nll": best_nll,
        "trainable_state": _trainable_state(model),
        "gradient_state": {
            name: value.grad.detach().cpu()
            for name, value in raw_model.named_parameters()
            if value.requires_grad and value.grad is not None
        },
        "optimizer": optimizer.state_dict(),
        "rng": {
            "python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all(),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def mirror_checkpoint(source: Path, destination: Path) -> None:
    """Make an atomic hard-link alias when best and last are the same state."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    try:
        os.link(source, temporary)
        os.replace(temporary, destination)
    except OSError:
        temporary.unlink(missing_ok=True)
        # Cross-device filesystems are not expected, but correctness beats deduplication.
        import shutil
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)


def classify_trajectory(history: list[dict], tolerance: float = 1e-4) -> str:
    values = [float(item["validation_nll"]) for item in history]
    if len(values) < 2:
        return "mixed"
    recent = values[-min(3, len(values)):]
    if max(recent) - min(recent) <= tolerance:
        return "plateaued"
    if values[-1] <= min(values[:-1]) - tolerance:
        return "improving"
    if values[-1] >= min(values) + tolerance and values[-1] > values[0] + tolerance:
        return "deteriorating"
    return "mixed"


def restore_checkpoint(path: Path, model, optimizer) -> tuple[dict, float]:
    model = getattr(model, "_orig_mod", model)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    parameters = dict(model.named_parameters())
    expected = {name for name, value in parameters.items() if value.requires_grad}
    if set(payload["trainable_state"]) != expected:
        raise RuntimeError("checkpoint trainable set differs from configured model")
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
    return payload["progress"], float(payload["best_validation_nll"])


def train(config: TrainConfig, resume: Path | None = None) -> dict:
    config.validate()
    wall_start = time.perf_counter()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Gate-1 training requires a CUDA GPU with BF16 support")
    seed_everything(config.seed)
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    model, tokenizer = load_model(revision=config.model_revision, attention_backend=config.attention_backend)
    counts = configure_adaptation(model, config.mode)
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if config.mode in {"dtft", "lora8", "lora64"}:
            model.enable_input_require_grads()
    model.to(device)
    if config.compile_model:
        # Match the mode benchmarked by profile_execution; changing compile
        # modes here would invalidate the selected throughput/VRAM setting.
        model = torch.compile(model, mode="reduce-overhead", dynamic=True)
    train_data = ProteinDataset(config.train_manifest, tokenizer)
    validation_data = ProteinDataset(config.validation_manifest, tokenizer)
    parameters = [value for value in model.parameters() if value.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=config.learning_rate,
        weight_decay=0.0,
        fused=True,
    )
    progress = {
        "epoch": 0, "next_batch": 0, "optimizer_steps": 0, "residues": 0,
        "sequences": 0, "accumulated_residues": 0, "accumulated_selected_tokens": 0,
        "stale_checks": 0,
    }
    best_nll = float("inf")
    if resume is not None:
        progress, best_nll = restore_checkpoint(resume, model, optimizer)
    history_path = config.output_dir / "history.json"
    history = (
        json.loads(history_path.read_text(encoding="utf-8"))
        if resume is not None and history_path.exists()
        else []
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "config.json").write_text(json.dumps(config.serializable(), indent=2, sort_keys=True), encoding="utf-8")
    (config.output_dir / "parameter_counts.json").write_text(json.dumps(counts, indent=2, sort_keys=True), encoding="utf-8")
    torch.cuda.reset_peak_memory_stats()
    loop_start = time.perf_counter()
    peak_padding = padded_tokens = nonpadding_tokens = 0
    stale_checks = int(progress.get("stale_checks", 0))
    if resume is None:
        optimizer.zero_grad(set_to_none=True)
    accumulated_residues = int(progress.get("accumulated_residues", 0))
    accumulated_selected = int(progress.get("accumulated_selected_tokens", 0))
    step_residue_budgets = progress.setdefault("step_residue_budgets", [])
    reached_step_cap = False
    for epoch in range(progress["epoch"], config.maximum_epochs):
        loader = make_loader(train_data, tokenizer, config, epoch, True)
        model.train()
        for batch_index, batch in enumerate(loader):
            if epoch == progress["epoch"] and batch_index < progress["next_batch"]:
                continue
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            selected = int(labels.ne(-100).sum().item())
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                loss_sum = F.cross_entropy(
                    logits.transpose(1, 2), labels, reduction="sum", ignore_index=-100
                )
            loss_sum.backward()
            residues = int(batch["nonpadding_residues"])
            accumulated_residues += residues
            accumulated_selected += selected
            progress["accumulated_residues"] = accumulated_residues
            progress["accumulated_selected_tokens"] = accumulated_selected
            progress["residues"] += residues
            progress["sequences"] += len(batch["rows"])
            nonpadding_tokens += sum(int(row["length"]) + 2 for row in batch["rows"])
            padded_tokens += int(input_ids.numel())
            peak_padding = max(peak_padding, input_ids.shape[1])
            if accumulated_residues >= config.effective_residues_per_step:
                completed_budget = accumulated_residues
                for parameter in parameters:
                    if parameter.grad is not None:
                        parameter.grad.div_(accumulated_selected)
                torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                accumulated_residues = 0
                progress["accumulated_residues"] = 0
                progress["accumulated_selected_tokens"] = 0
                step_residue_budgets.append(completed_budget)
                accumulated_selected = 0
                progress["optimizer_steps"] += 1
                progress.update(epoch=epoch, next_batch=batch_index + 1)
                if progress["optimizer_steps"] % config.validation_every_steps == 0:
                    validation = evaluate_nll(model, make_loader(validation_data, tokenizer, config, 0, False), device)
                    event = {
                        "epoch": epoch, "batch": batch_index, "optimizer_step": progress["optimizer_steps"],
                        "validation_nll": validation["masked_token_nll"],
                    }
                    history.append(event)
                    improved = validation["masked_token_nll"] < best_nll
                    stale_checks = 0 if improved else stale_checks + 1
                    progress["stale_checks"] = stale_checks
                    if improved:
                        best_nll = validation["masked_token_nll"]
                        if config.save_checkpoints:
                            save_checkpoint(config.output_dir / "best.pt", model, optimizer, config, progress, best_nll)
                    if config.save_checkpoints:
                        if improved:
                            mirror_checkpoint(config.output_dir / "best.pt", config.output_dir / "last.pt")
                        else:
                            save_checkpoint(config.output_dir / "last.pt", model, optimizer, config, progress, best_nll)
                    (config.output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
                    model.train()
                    if stale_checks >= config.early_stopping_checks:
                        break
                if (
                    config.maximum_optimizer_steps is not None
                    and progress["optimizer_steps"] >= config.maximum_optimizer_steps
                ):
                    reached_step_cap = True
                    break
        progress.update(epoch=epoch + 1, next_batch=0)
        if config.save_checkpoints:
            best_is_current = bool(
                history
                and history[-1]["optimizer_step"] == progress["optimizer_steps"]
                and history[-1]["validation_nll"] == best_nll
            )
            if best_is_current:
                mirror_checkpoint(config.output_dir / "best.pt", config.output_dir / "last.pt")
            else:
                save_checkpoint(config.output_dir / "last.pt", model, optimizer, config, progress, best_nll)
        if stale_checks >= config.early_stopping_checks or reached_step_cap:
            break
    loop_elapsed = time.perf_counter() - loop_start
    wall_elapsed = time.perf_counter() - wall_start
    execution = {
        **progress,
        "wall_seconds": wall_elapsed,
        "training_loop_seconds": loop_elapsed,
        "residues_per_second": progress["residues"] / loop_elapsed,
        "peak_vram_bytes": torch.cuda.max_memory_allocated(),
        "padding_fraction": 1 - nonpadding_tokens / padded_tokens,
        "maximum_padded_length": peak_padding,
        "effective_residues_per_step": {
            "target": config.effective_residues_per_step,
            "minimum": min(step_residue_budgets) if step_residue_budgets else None,
            "mean": float(np.mean(step_residue_budgets)) if step_residue_budgets else None,
            "maximum": max(step_residue_budgets) if step_residue_budgets else None,
        },
        "best_validation_nll": best_nll,
        "stopped_early": stale_checks >= config.early_stopping_checks,
        "trajectory_classification": classify_trajectory(history),
        "reached_optimizer_step_cap": reached_step_cap,
    }
    (config.output_dir / "execution.json").write_text(json.dumps(execution, indent=2, sort_keys=True), encoding="utf-8")
    return execution


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    raw = json.loads(args.config.read_text(encoding="utf-8"))
    for name in ("train_manifest", "validation_manifest", "output_dir"):
        raw[name] = Path(raw[name])
    train(TrainConfig(**raw), args.resume)


if __name__ == "__main__":
    main()
