from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

import torch.nn as nn


EXPECTED_MATRIX_COUNT = 128
EXPECTED_DENSE_WEIGHTS = 58_720_256
EXPECTED_LORA = {8: 1_441_792, 64: 11_534_336}

_TARGET = re.compile(
    r"^(?:encoder|decoder)\.layers\.(\d+)\."
    r"(?:self_attn\.(?:q_proj|k_proj|v_proj|out_proj)|"
    r"encoder_attn\.(?:q_proj|k_proj|v_proj|out_proj)|fc[12])$"
)


@dataclass(frozen=True)
class TargetMatrix:
    name: str
    module: nn.Linear

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self.module.weight.shape)  # type: ignore[return-value]

    def lora_scalars(self, rank: int) -> int:
        m, n = self.shape
        return rank * (m + n)

    def intrinsic_dimension(self, rank: int) -> int:
        m, n = self.shape
        return rank * (m + n - rank)


def target_matrices(model: nn.Module, *, enforce_if1: bool = True) -> list[TargetMatrix]:
    targets = [
        TargetMatrix(name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and _TARGET.fullmatch(name)
    ]
    targets.sort(key=lambda item: item.name)
    if enforce_if1:
        _assert_if1_architecture(targets)
    return targets


def _assert_if1_architecture(targets: Iterable[TargetMatrix]) -> None:
    targets = list(targets)
    names = {item.name for item in targets}
    expected: set[str] = set()
    projections = ("q_proj", "k_proj", "v_proj", "out_proj")
    for stack in ("encoder", "decoder"):
        for layer in range(8):
            expected.update(f"{stack}.layers.{layer}.self_attn.{p}" for p in projections)
            expected.update({f"{stack}.layers.{layer}.fc1", f"{stack}.layers.{layer}.fc2"})
            if stack == "decoder":
                expected.update(f"decoder.layers.{layer}.encoder_attn.{p}" for p in projections)
    if names != expected:
        missing = sorted(expected - names)
        extra = sorted(names - expected)
        raise RuntimeError(
            "Loaded architecture does not match released ESM-IF1 transformer targets; "
            f"missing={missing[:8]}, extra={extra[:8]}"
        )
    dense = sum(item.module.weight.numel() for item in targets)
    if len(targets) != EXPECTED_MATRIX_COUNT or dense != EXPECTED_DENSE_WEIGHTS:
        raise RuntimeError(
            f"ESM-IF1 target budget mismatch: matrices={len(targets)} (expected 128), "
            f"dense={dense:,} (expected {EXPECTED_DENSE_WEIGHTS:,})"
        )
    for rank, expected_count in EXPECTED_LORA.items():
        actual = sum(item.lora_scalars(rank) for item in targets)
        if actual != expected_count:
            raise RuntimeError(
                f"LoRA rank-{rank} budget is {actual:,}; expected {expected_count:,}"
            )


def freeze_for_dtft(model: nn.Module) -> list[str]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    targets = target_matrices(model)
    for item in targets:
        item.module.weight.requires_grad_(True)
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    expected = [f"{item.name}.weight" for item in targets]
    if set(trainable) != set(expected):
        raise RuntimeError("DTFT trainables differ from the exact LoRA target weights")
    return sorted(trainable)


def trainable_parameter_report(model: nn.Module) -> dict[str, int]:
    return {
        "loaded": sum(p.numel() for p in model.parameters()),
        "trainable_scalars": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "in_memory_state_scalars": sum(p.numel() for p in model.state_dict().values()),
    }
