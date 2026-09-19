from __future__ import annotations

from pathlib import Path

import torch


def restore_trainables(model: torch.nn.Module, checkpoint: Path) -> dict:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    parameters = dict(model.named_parameters())
    expected = {name for name, parameter in parameters.items() if parameter.requires_grad}
    supplied = set(payload["trainable_state"])
    if supplied != expected:
        raise RuntimeError(
            f"Checkpoint trainables do not match configured model; missing={sorted(expected-supplied)[:5]}, "
            f"extra={sorted(supplied-expected)[:5]}"
        )
    with torch.no_grad():
        for name, value in payload["trainable_state"].items():
            parameters[name].copy_(value.to(dtype=parameters[name].dtype))
    return payload["metadata"]

