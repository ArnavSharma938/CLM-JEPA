from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import MODEL_ID, MODEL_REVISION


_TARGET = re.compile(
    r"^esm\.encoder\.layer\.(\d+)\."
    r"(?:attention\.self\.(?:query|key|value)|attention\.output\.dense|"
    r"intermediate\.dense|output\.dense)$"
)
EXPECTED_LAYERS = 33
EXPECTED_TARGETS = EXPECTED_LAYERS * 6


@dataclass(frozen=True)
class DenseTarget:
    name: str
    module: nn.Linear

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self.module.weight.shape)  # type: ignore[return-value]

    def lora_scalars(self, rank: int) -> int:
        rows, columns = self.shape
        return rank * (rows + columns)


def dense_targets(model: nn.Module, *, enforce_650m: bool = True) -> list[DenseTarget]:
    targets = [
        DenseTarget(name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and _TARGET.fullmatch(name)
    ]
    targets.sort(key=lambda item: item.name)
    if enforce_650m:
        layers = {int(_TARGET.fullmatch(item.name).group(1)) for item in targets}  # type: ignore[union-attr]
        per_layer = {layer: 0 for layer in range(EXPECTED_LAYERS)}
        for item in targets:
            per_layer[int(_TARGET.fullmatch(item.name).group(1))] += 1  # type: ignore[union-attr]
        if len(targets) != EXPECTED_TARGETS or layers != set(range(EXPECTED_LAYERS)):
            raise RuntimeError(f"ESM-2 target mismatch: {len(targets)} matrices, layers={sorted(layers)}")
        if any(count != 6 for count in per_layer.values()):
            raise RuntimeError(f"ESM-2 target count per layer is not six: {per_layer}")
    return targets


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: int) -> None:
        super().__init__()
        if rank <= 0 or alpha != rank:
            raise ValueError("vanilla Gate-1 LoRA requires rank > 0 and alpha/r = 1")
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        for parameter in base.parameters():
            parameter.requires_grad_(False)
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.base(values) + F.linear(F.linear(values, self.lora_A), self.lora_B) * self.scaling

    def merged_update(self) -> torch.Tensor:
        return (self.lora_B @ self.lora_A) * self.scaling


def _parent_and_leaf(model: nn.Module, path: str) -> tuple[nn.Module, str]:
    parent = model
    parts = path.split(".")
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def configure_adaptation(model: nn.Module, mode: str) -> dict:
    targets = dense_targets(model)
    target_names = [item.name for item in targets]
    target_weights = {f"{item.name}.weight" for item in targets}
    for parameter in model.parameters():
        parameter.requires_grad_(mode == "ft")
    if mode == "base":
        pass
    elif mode == "ft":
        pass
    elif mode == "dtft":
        parameters = dict(model.named_parameters())
        for name in target_weights:
            parameters[name].requires_grad_(True)
    elif mode in {"lora8", "lora64"}:
        rank = int(mode.removeprefix("lora"))
        for item in targets:
            parent, leaf = _parent_and_leaf(model, item.name)
            setattr(parent, leaf, LoRALinear(item.module, rank, rank))
    else:
        raise ValueError(f"unknown adaptation mode: {mode}")

    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    if mode == "dtft" and trainable != target_weights:
        raise RuntimeError("DTFT trainables are not exactly the LoRA dense target weights")
    if mode.startswith("lora") and (
        not trainable or any(not name.endswith(("lora_A", "lora_B")) for name in trainable)
    ):
        raise RuntimeError("non-adapter parameter remained trainable in LoRA mode")
    return parameter_report(model, target_names)


def parameter_report(model: nn.Module, target_names: Iterable[str] | None = None) -> dict:
    unwrapped = []
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            unwrapped.append((name, module.base))
        elif isinstance(module, nn.Linear) and _TARGET.fullmatch(name):
            unwrapped.append((name, module))
    unwrapped.sort()
    dense_count = sum(module.weight.numel() for _, module in unwrapped)
    total = sum(
        parameter.numel() for name, parameter in model.named_parameters() if "lora_" not in name
    )
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {
        "total_pretrained_parameters": total,
        "ft_trainable_count": total,
        "dtft_trainable_count": dense_count,
        "lora_r8_trainable_count": sum(
            8 * (module.in_features + module.out_features) for _, module in unwrapped
        ),
        "lora_r64_trainable_count": sum(
            64 * (module.in_features + module.out_features) for _, module in unwrapped
        ),
        "trainable_parameters": trainable,
        "target_matrix_count": len(unwrapped),
        "target_scalar_count": dense_count,
        "dtft_ft_coverage_fraction": dense_count / total,
        "target_modules": [
            {"name": name, "shape": list(module.weight.shape)} for name, module in unwrapped
        ],
    }
def load_model(*, revision: str = MODEL_REVISION, attention_backend: str = "eager"):
    from transformers import AutoTokenizer, EsmForMaskedLM

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=revision)
    model = EsmForMaskedLM.from_pretrained(
        MODEL_ID, revision=revision, attn_implementation=attention_backend
    )
    dense_targets(model)
    return model, tokenizer


def sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()
