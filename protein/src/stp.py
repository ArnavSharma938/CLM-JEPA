"""Released STP objective adapted to RITA's single residue segment.

RITA has no BOS state, so a protein of L residues supplies L-1 state
displacements, from h[0] through h[L-1]. EOS is never part of the path.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F


def sample_released_span(num_steps: int, generator: torch.Generator) -> tuple[int, int]:
    if num_steps < 2:
        raise ValueError("released STP requires at least two state displacements")
    start = int(torch.randint(0, num_steps, (), generator=generator))
    while True:
        end = int(torch.randint(start + 1, num_steps + 1, (), generator=generator))
        if end - start < num_steps:
            return start, end


def released_stp_terms(states: torch.Tensor, start: int, end: int):
    if states.ndim != 2 or not (0 <= start < end <= states.shape[0] - 1):
        raise ValueError("invalid single-segment STP span")
    before = states[start] - states[0]
    patch = states[end] - states[start]
    after = states[-1] - states[end]
    return before, patch, after


def released_stp_loss(
    state_rows: list[torch.Tensor], *, seed: int = 0
) -> tuple[torch.Tensor, tuple[tuple[int, int], ...]]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    losses, spans = [], []
    for states in state_rows:
        start, end = sample_released_span(states.shape[0] - 1, generator)
        before, patch, after = released_stp_terms(states, start, end)
        losses.append(1.0 - F.cosine_similarity(patch.float(), (before + after).float(), dim=0))
        spans.append((start, end))
    return torch.stack(losses).mean(), tuple(spans)

