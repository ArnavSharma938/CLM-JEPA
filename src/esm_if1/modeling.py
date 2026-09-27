from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .lora import attach_lora
from .targets import freeze_for_dtft, target_matrices, trainable_parameter_report


def load_if1(checkpoint: Path | None = None) -> tuple[nn.Module, Any]:
    import esm

    if checkpoint is None:
        model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
    else:
        model, alphabet = esm.pretrained.load_model_and_alphabet_local(str(checkpoint))
    # This is intentionally measured, not copied from model metadata.
    loaded_count = sum(parameter.numel() for parameter in model.parameters())
    if loaded_count <= 0:
        raise RuntimeError("Loaded ESM-IF1 has no parameters")
    target_matrices(model, enforce_if1=True)
    return model, alphabet


def configure_adaptation(model: nn.Module, mode: str) -> dict[str, int]:
    pretrained_count = sum(parameter.numel() for parameter in model.parameters())
    if mode == "base":
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    elif mode == "ft":
        for parameter in model.parameters():
            parameter.requires_grad_(True)
    elif mode == "dtft":
        freeze_for_dtft(model)
    elif mode in {"lora8", "lora64"}:
        rank = int(mode.removeprefix("lora"))
        attach_lora(model, rank=rank, alpha=rank)
        # fair-esm's torch-MHA shortcut reads q/k/v ``.weight`` directly.
        # That both bypasses adapter forwards and is incompatible with wrapped
        # projections, so LoRA attention must use the explicit projection path.
        from .lora import LoRALinear

        for module in model.modules():
            if hasattr(module, "enable_torch_version") and any(
                isinstance(getattr(module, name, None), LoRALinear)
                for name in ("q_proj", "k_proj", "v_proj")
            ):
                module.enable_torch_version = False
    else:
        raise ValueError(f"Unknown adaptation mode: {mode}")
    report = trainable_parameter_report(model)
    report["loaded"] = pretrained_count
    report["serialized_trainable_scalars"] = report["trainable_scalars"]
    if mode in {"lora8", "lora64"}:
        rank = int(mode.removeprefix("lora"))
        targets = target_matrices_from_wrapped(model)
        report["intrinsic_dimension"] = sum(
            rank * (module.base.out_features + module.base.in_features - rank)
            for module in targets
        )
        report["diagnostic_merged_update_scalars"] = sum(module.base.weight.numel() for module in targets)
    else:
        report["intrinsic_dimension"] = report["trainable_scalars"]
        report["diagnostic_merged_update_scalars"] = 0
    return report


def target_matrices_from_wrapped(model: nn.Module):
    from .lora import LoRALinear

    modules = [module for module in model.modules() if isinstance(module, LoRALinear)]
    if len(modules) != 128:
        raise RuntimeError(f"Expected 128 wrapped LoRA matrices, found {len(modules)}")
    return modules


def autocast_context(device: torch.device):
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")

