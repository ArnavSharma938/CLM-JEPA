"""Exact short-sequence reproduction of the official RITA fitness score."""
from __future__ import annotations

import torch
from torch.nn import functional as F


@torch.no_grad()
def official_rita_fitness(model, tokenizer, sequence: str, device: str | torch.device) -> float:
    """Sum negative mean CE for forward and reversed sequence, as upstream."""
    score = 0.0
    for direction in (sequence, sequence[::-1]):
        ids = torch.tensor([tokenizer.encode(direction)], device=device)
        logits = model(ids[:, :-1]).logits
        score -= float(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1)))
    return score

