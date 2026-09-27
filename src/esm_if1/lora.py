from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .targets import EXPECTED_LORA, target_matrices


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: int) -> None:
        super().__init__()
        if rank <= 0 or alpha <= 0:
            raise ValueError("rank and alpha must be positive")
        if alpha != rank:
            raise ValueError("Calibration LoRA requires alpha/r = 1")
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.base(inputs) + F.linear(F.linear(inputs, self.lora_A), self.lora_B) * self.scaling

    def merged_update(self) -> torch.Tensor:
        return (self.lora_B @ self.lora_A) * self.scaling


def _parent_and_leaf(model: nn.Module, path: str) -> tuple[nn.Module, str]:
    parts = path.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def attach_lora(model: nn.Module, rank: int, alpha: int) -> list[str]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    targets = target_matrices(model)
    for item in targets:
        parent, leaf = _parent_and_leaf(model, item.name)
        setattr(parent, leaf, LoRALinear(item.module, rank, alpha))
    names = [name for name, p in model.named_parameters() if p.requires_grad]
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if count != EXPECTED_LORA[rank]:
        raise RuntimeError(f"rank-{rank} LoRA has {count:,} trainables, expected {EXPECTED_LORA[rank]:,}")
    if not names or any(not (n.endswith("lora_A") or n.endswith("lora_B")) for n in names):
        raise RuntimeError("A non-LoRA parameter remained trainable")
    return names


def lora_modules(model: nn.Module) -> dict[str, LoRALinear]:
    return {name: module for name, module in model.named_modules() if isinstance(module, LoRALinear)}


def adapter_state(model: nn.Module) -> dict[str, torch.Tensor | float | int]:
    state: dict[str, torch.Tensor | float | int] = {}
    for name, module in lora_modules(model).items():
        state[f"{name}.A"] = module.lora_A.detach().cpu()
        state[f"{name}.B"] = module.lora_B.detach().cpu()
        state[f"{name}.BA"] = module.merged_update().detach().cpu()
        state[f"{name}.rank"] = module.rank
        state[f"{name}.scaling"] = module.scaling
    return state
