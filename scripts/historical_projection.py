"""Archived projection-head definition for reproducing experiment 08.

The projection experiment is intentionally absent from ``src/`` and the
active trainer.  This module only keeps old checkpoints and diagnostics
loadable.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


class ProjectionHead(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int = 2048, output_dim: int = 64,
    ) -> None:
        super().__init__()
        if min(input_dim, hidden_dim, output_dim) < 1:
            raise ValueError("projection dimensions must be positive")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        if states.ndim != 2 or states.size(-1) != self.input_dim:
            raise ValueError(
                f"projector expects (samples, {self.input_dim}) endpoint states"
            )
        return self.layers(states.float())

    def configuration(self) -> dict[str, int | str | bool]:
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "architecture": (
                f"{self.input_dim}->{self.hidden_dim}->{self.hidden_dim}"
                f"->{self.output_dim}"
            ),
            "hidden_normalization": "BatchNorm1d",
            "hidden_activation": "ReLU",
            "final_activation": False,
            "l2_normalization": False,
        }


def load_projection_head_checkpoint(
    projector: ProjectionHead, checkpoint: Path,
) -> None:
    payload = torch.load(
        checkpoint / "projection_head.pt",
        map_location=next(projector.parameters()).device,
        weights_only=False,
    )
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported projection-head checkpoint schema")
    if payload.get("configuration") != projector.configuration():
        raise ValueError("projection-head checkpoint configuration mismatch")
    projector.load_state_dict(payload["state_dict"])
