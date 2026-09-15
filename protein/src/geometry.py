"""Minimal STP-relevant Euclidean trajectory measurements."""
from __future__ import annotations

import torch
from torch.nn import functional as F


def local_curvature(states: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if states.ndim != 2 or states.shape[0] < 3:
        raise ValueError("states must be LxD with L>=3")
    left = states[1:-1] - states[:-2]
    right = states[2:] - states[1:-1]
    return 1.0 - F.cosine_similarity(left, right, dim=-1, eps=eps)


def adjacent_tangent_correlation(states: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return 1.0 - local_curvature(states, eps=eps)


def chord_decomposition(states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return parallel and orthogonal displacement from the start-to-end chord."""
    chord = states[-1] - states[0]
    unit = chord / chord.norm().clamp_min(1e-8)
    displacement = states - states[0]
    parallel = (displacement @ unit).unsqueeze(-1) * unit
    return parallel, displacement - parallel


def tube_radius(states: torch.Tensor, *, normalized: bool = True) -> torch.Tensor:
    parallel, orthogonal = chord_decomposition(states)
    radius = orthogonal[1:-1].norm(dim=-1)
    if normalized:
        radius = radius / (states[-1] - states[0]).norm().clamp_min(1e-8)
    return radius


def path_efficiency(states: torch.Tensor) -> torch.Tensor:
    path = (states[1:] - states[:-1]).norm(dim=-1).sum()
    chord = (states[-1] - states[0]).norm()
    return chord / path.clamp_min(1e-8)


def attenuate_orthogonal(
    states: torch.Tensor, fraction: float = 0.10, *, restore_norm: bool = True
) -> torch.Tensor:
    if not 0 <= fraction <= 1:
        raise ValueError("fraction must be in [0,1]")
    parallel, orthogonal = chord_decomposition(states)
    changed = states[0] + parallel + (1.0 - fraction) * orthogonal
    if restore_norm:
        norms = states.norm(dim=-1, keepdim=True)
        changed = changed * (norms / changed.norm(dim=-1, keepdim=True).clamp_min(1e-8))
    return changed

