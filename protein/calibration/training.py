from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import TrainConfig
from .data import (
    BackboneBatchSampler,
    CoordinateDataset,
    EpochRandomSampler,
    IF1Collator,
    load_manifest,
    preload_coordinate_cache,
    sft_training_records,
)
from .lora import adapter_state
from .modeling import autocast_context, configure_adaptation, load_if1
from .optimized_forward import (
    OptimizedIF1Forward,
    enable_if1_training_optimizations,
    model_logits,
    prefetch_batches,
)
from .targets import target_matrices


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _worker_seed(worker_id: int) -> None:
    del worker_id
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def target_token_count(batch: dict[str, Any]) -> int:
    """Count residue and EOS targets without synchronizing an accelerator."""
    return sum(record.sequence_length + 1 for record in batch["records"])


def sequence_nll(
    model: torch.nn.Module,
    batch: dict[str, Any],
    alphabet: Any,
    device: torch.device,
    *,
    reuse_backbones: bool = False,
) -> tuple[torch.Tensor, int]:
    tokens = batch["tokens"].to(device, non_blocking=True)
    target = tokens[:, 1:]
    with autocast_context(device):
        logits = model_logits(model, batch, device, reuse_backbones=reuse_backbones, tokens=tokens)
        loss_sum = F.cross_entropy(
            logits.float(), target, reduction="sum", ignore_index=alphabet.padding_idx
        )
    # Teacher forcing predicts every residue plus EOS. Deriving this from the
    # immutable records avoids a device synchronization on every train step.
    token_count = target_token_count(batch)
    return loss_sum, token_count


@torch.inference_mode()
def validation_nll(model: torch.nn.Module, loader: Iterable[dict[str, Any]], alphabet: Any, device: torch.device) -> float:
    model.eval()
    losses: list[torch.Tensor] = []
    token_count = 0
    for batch in prefetch_batches(loader, device):
        # Keep the exact full-batch encoder arithmetic.  Deduplicating identical
        # backbones changes BF16 GEMM shapes and is not bit-identical on A6000.
        loss, count = sequence_nll(model, batch, alphabet, device, reuse_backbones=False)
        losses.append(loss.detach())
        token_count += count
    if token_count == 0:
        raise RuntimeError("Validation set has zero target tokens")
    # Transfer scalar batch losses together, then preserve the reference
    # Python-float summation order exactly.
    loss_sum = sum(torch.stack(losses).cpu().tolist())
    return loss_sum / token_count


def _cpu_trainables(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().to(device="cpu").clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def _target_snapshot(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        if name.endswith(".base") and hasattr(module, "weight"):
            plain_name = name.removesuffix(".base")
            result[plain_name] = module.weight.detach().to(device="cpu").clone()
    if result:
        return result
    return {
        item.name: item.module.weight.detach().to(device="cpu").clone()
        for item in target_matrices(model)
    }


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    initial_targets: dict[str, torch.Tensor],
    metadata: dict[str, Any],
    *,
    include_optimizer: bool,
) -> None:
    trainable_state = _cpu_trainables(model)
    is_lora = any(name.endswith("lora_A") for name in trainable_state)
    payload: dict[str, Any] = {
        "metadata": metadata,
        "trainable_state": trainable_state,
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        },
    }
    if is_lora:
        payload["adapter"] = adapter_state(model)
    else:
        current_targets = _target_snapshot(model)
        payload["target_weights"] = current_targets
        payload["target_delta"] = {
            name: current_targets[name] - initial_targets[name] for name in initial_targets
        }
    if include_optimizer and optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _make_loader(
    records,
    alphabet,
    *,
    batch_size: int,
    training: bool,
    noise: float,
    seed: int,
    workers: int,
    coordinate_cache=None,
) -> tuple[DataLoader, EpochRandomSampler | BackboneBatchSampler]:
    dataset = CoordinateDataset(records)
    common = {
        "dataset": dataset,
        "collate_fn": IF1Collator(
            alphabet,
            training=training,
            coordinate_noise=noise,
            coordinate_cache=coordinate_cache,
        ),
        "num_workers": workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": workers > 0,
        "worker_init_fn": _worker_seed,
        "generator": torch.Generator().manual_seed(seed),
    }
    if workers > 0:
        common["prefetch_factor"] = 4
    if training:
        sampler: EpochRandomSampler | BackboneBatchSampler = EpochRandomSampler(len(records), seed)
        loader = DataLoader(batch_size=batch_size, sampler=sampler, shuffle=False, drop_last=False, **common)
    else:
        sampler = BackboneBatchSampler(records, batch_size, seed, shuffle=False)
        loader = DataLoader(batch_sampler=sampler, **common)
    return loader, sampler


def train(config: TrainConfig, checkpoint: Path | None = None) -> dict[str, Any]:
    config.validate()
    seed_everything(config.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("Real ESM-IF1 calibration training requires CUDA bf16")
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    model, alphabet = load_if1(checkpoint)
    initial_targets = (
        {
            item.name: item.module.weight.detach().to(device="cpu").clone()
            for item in target_matrices(model)
        }
        if config.save_checkpoints else {}
    )
    counts = configure_adaptation(model, config.mode)
    if config.mode in {"ft", "dtft", "lora64"}:
        enable_if1_training_optimizations(model)
    model.to(device)
    records = load_manifest(config.manifest)
    if config.mode != "base" and config.instrument_checkpoints and config.diagnostic_panel is None:
        raise ValueError("Adaptation runs require the fixed --diagnostic-panel")
    train_records = sft_training_records(records)
    validation_records = [record for record in records if record.split == "validation"]
    if not validation_records:
        raise ValueError("Validation split is empty")
    coordinate_cache = preload_coordinate_cache(records)
    train_loader, sampler = _make_loader(
        train_records,
        alphabet,
        batch_size=config.batch_size,
        training=True,
        noise=config.coordinate_noise_angstrom,
        seed=config.seed,
        workers=config.num_workers,
        coordinate_cache=coordinate_cache,
    )
    validation_loader, _ = _make_loader(
        validation_records,
        alphabet,
        batch_size=config.batch_size,
        training=False,
        noise=0.0,
        seed=config.seed,
        workers=config.num_workers,
        coordinate_cache=coordinate_cache,
    )
    diagnostic_loaders: dict[str, DataLoader] = {}
    if config.instrument_checkpoints and config.diagnostic_panel is not None:
        panel = json.loads(config.diagnostic_panel.read_text(encoding="utf-8"))
        train_ids = {sequence_id for values in panel["train_diagnostic"].values() for sequence_id in values}
        test_ids = {
            sequence_id
            for tier in panel["test_diagnostic"].values()
            for values in tier.values()
            for sequence_id in values
        }
        for label, identifiers in (("train", train_ids), ("test", test_ids)):
            selected = [record for record in records if record.sequence_id in identifiers]
            if len(selected) != len(identifiers):
                raise RuntimeError(f"Fixed {label} diagnostic panel did not resolve exactly")
            diagnostic_loaders[label], _ = _make_loader(
                selected,
                alphabet,
                batch_size=min(8, config.batch_size),
                training=False,
                noise=0.0,
                seed=config.seed,
                workers=config.num_workers,
                coordinate_cache=coordinate_cache,
            )
    run_dir = config.output_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(config.serializable(), indent=2, sort_keys=True), encoding="utf-8")
    (run_dir / "parameter_counts.json").write_text(json.dumps(counts, indent=2, sort_keys=True), encoding="utf-8")
    if config.mode == "base":
        value = validation_nll(model, validation_loader, alphabet, device)
        result = {"validation_nll": value, "parameter_counts": counts}
        (run_dir / "base_validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result

    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=config.lr,
        weight_decay=0.0 if config.mode.startswith("lora") else config.adapter_weight_decay,
        fused=True,
    )
    execution_model: torch.nn.Module = model
    compile_status = {"requested": config.compile_training, "enabled": False, "mode": config.compile_mode}
    if config.compile_training and platform.system() != "Windows":
        # Fixed length-72 batches bound graph compilation shapes without truncating the
        # ProteinDPO MegaScale proteins (reported range 40--72 residues).
        execution_model = torch.compile(
            OptimizedIF1Forward(model), mode=config.compile_mode, dynamic=False, fullgraph=False
        )
        compile_status["enabled"] = True
    (run_dir / "execution.json").write_text(json.dumps(compile_status, indent=2), encoding="utf-8")

    def instrument(checkpoint_path: Path) -> None:
        from .diagnostics import checkpoint_diagnostics

        python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
        cuda_state = torch.cuda.get_rng_state_all()
        try:
            payload = {
                label: checkpoint_diagnostics(model, loader, alphabet, device)
                for label, loader in diagnostic_loaders.items()
            }
            diagnostic_path = checkpoint_path.with_suffix(".diagnostics.pt")
            temporary = diagnostic_path.with_suffix(".pt.tmp")
            torch.save(payload, temporary)
            os.replace(temporary, diagnostic_path)
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.set_rng_state(torch_state)
            torch.cuda.set_rng_state_all(cuda_state)
    steps_per_epoch = len(train_loader)
    milestone_steps = {
        max(1, math.ceil(epoch * steps_per_epoch)): label
        for epoch, label in ((0.25, "epoch_0.25"), (1, "epoch_1"), (3, "epoch_3"), (10, "epoch_10"))
        if epoch <= config.epochs
    }
    history: list[dict[str, Any]] = []
    global_step = 0
    initial_nll = validation_nll(model, validation_loader, alphabet, device)
    history.append({"step": 0, "epoch": 0.0, "validation_nll": initial_nll})
    if config.save_checkpoints:
        save_checkpoint(
            run_dir / "checkpoints" / "epoch_0.pt",
            model,
            optimizer,
            initial_targets,
            history[-1],
            include_optimizer=config.mode.startswith("lora"),
        )
        if config.instrument_checkpoints:
            instrument(run_dir / "checkpoints" / "epoch_0.pt")
    best_nll = initial_nll
    best_step = 0
    checks_without_improvement = 0
    stopped_early = False
    best_path = run_dir / "checkpoints" / "best.pt"
    if config.save_checkpoints:
        save_checkpoint(
            best_path,
            model,
            optimizer,
            initial_targets,
            history[-1],
            include_optimizer=True,
        )
    validation_interval = max(1, math.ceil(config.validation_fraction * steps_per_epoch))
    for epoch_index in range(config.epochs):
        assert sampler is not None
        sampler.set_epoch(epoch_index)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for batch_index, batch in enumerate(prefetch_batches(train_loader, device), start=1):
            # Encoder dropout is active in train mode. Reusing an encoded
            # backbone would change both the number and assignment of random
            # draws, so training deliberately keeps the reference forward.
            loss_sum, _ = sequence_nll(execution_model, batch, alphabet, device, reuse_backbones=False)
            # The registered SFT objective sums tokens within each sequence and
            # averages those sequence losses over the minibatch.
            (loss_sum / len(batch["records"])).backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            fractional_epoch = epoch_index + batch_index / steps_per_epoch
            label = milestone_steps.get(global_step)
            if label and config.save_checkpoints:
                save_checkpoint(
                    run_dir / "checkpoints" / f"{label}.pt",
                    model,
                    optimizer,
                    initial_targets,
                    {"step": global_step, "epoch": fractional_epoch},
                    include_optimizer=config.mode.startswith("lora"),
                )
                if config.instrument_checkpoints:
                    instrument(run_dir / "checkpoints" / f"{label}.pt")
            if batch_index % validation_interval == 0 or batch_index == steps_per_epoch:
                value = validation_nll(model, validation_loader, alphabet, device)
                event = {"step": global_step, "epoch": fractional_epoch, "validation_nll": value}
                history.append(event)
                if value < best_nll:
                    best_nll = value
                    best_step = global_step
                    checks_without_improvement = 0
                    if config.save_checkpoints:
                        save_checkpoint(
                            best_path,
                            model,
                            optimizer,
                            initial_targets,
                            event,
                            include_optimizer=True,
                        )
                else:
                    checks_without_improvement += 1
                model.train()
                if (
                    config.early_stopping_patience_checks is not None
                    and fractional_epoch >= config.minimum_epochs
                    and checks_without_improvement >= config.early_stopping_patience_checks
                ):
                    stopped_early = True
                    break
        (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        if stopped_early:
            break
    completed_epochs = global_step / steps_per_epoch
    terminal = {"step": global_step, "epoch": completed_epochs, "validation_nll": history[-1]["validation_nll"]}
    if config.epochs == 30 and not stopped_early and config.save_checkpoints:
        save_checkpoint(
            run_dir / "checkpoints" / "terminal_epoch_30.pt",
            model,
            optimizer,
            initial_targets,
            terminal,
            include_optimizer=True,
        )
        if config.instrument_checkpoints:
            instrument(run_dir / "checkpoints" / "terminal_epoch_30.pt")
    if config.save_checkpoints:
        from .checkpointing import restore_trainables

        restore_trainables(model, best_path)
        if config.instrument_checkpoints:
            instrument(best_path)
    result = {
        "best_validation_nll": best_nll,
        "best_step": best_step,
        "steps": global_step,
        "epochs_completed": completed_epochs,
        "stopped_early": stopped_early,
        "parameter_counts": counts,
    }
    (run_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def parse_args() -> tuple[TrainConfig, Path | None]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=("base", "ft", "dtft", "lora8", "lora64"))
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--lr", required=True, type=float)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--diagnostic-panel", type=Path)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--no-diagnostics", action="store_true")
    parser.add_argument("--no-checkpoints", action="store_true")
    parser.add_argument("--early-stopping-patience-checks", type=int)
    parser.add_argument("--minimum-epochs", type=float, default=0.0)
    args = parser.parse_args()
    config = TrainConfig(
        mode=args.mode,
        seed=args.seed,
        lr=args.lr,
        manifest=args.manifest,
        output_dir=args.output_dir,
        diagnostic_panel=args.diagnostic_panel,
        epochs=args.epochs,
        compile_training=args.compile,
        instrument_checkpoints=not args.no_diagnostics,
        save_checkpoints=not args.no_checkpoints,
        early_stopping_patience_checks=args.early_stopping_patience_checks,
        minimum_epochs=args.minimum_epochs,
    )
    return config, args.checkpoint


if __name__ == "__main__":
    train(*parse_args())
