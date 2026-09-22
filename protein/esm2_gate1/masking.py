from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Sequence

import torch


def derived_seed(sequence_id: str, exposure: int, common_seed: int, realization: int = 0) -> int:
    payload = f"{common_seed}\0{sequence_id}\0{exposure}\0{realization}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


@dataclass(frozen=True)
class Corruption:
    input_ids: torch.Tensor
    labels: torch.Tensor
    selected_positions: tuple[int, ...]


def corrupt_tokens(
    token_ids: torch.Tensor,
    eligible_positions: Sequence[int],
    canonical_token_ids: Sequence[int],
    mask_token_id: int,
    *,
    sequence_id: str,
    exposure: int,
    common_seed: int,
    realization: int = 0,
) -> Corruption:
    """Apply deterministic ESM-style 15% / 80-10-10 MLM corruption."""
    if not eligible_positions:
        raise ValueError("a protein must expose at least one eligible residue")
    if len(canonical_token_ids) != 20 or len(set(canonical_token_ids)) != 20:
        raise ValueError("canonical_token_ids must contain 20 distinct tokens")
    rng = random.Random(derived_seed(sequence_id, exposure, common_seed, realization))
    count = max(1, int(round(0.15 * len(eligible_positions))))
    selected = tuple(sorted(rng.sample(list(eligible_positions), count)))
    corrupted = token_ids.clone()
    labels = torch.full_like(token_ids, -100)
    for position in selected:
        labels[position] = token_ids[position]
        draw = rng.random()
        if draw < 0.8:
            corrupted[position] = mask_token_id
        elif draw < 0.9:
            corrupted[position] = canonical_token_ids[rng.randrange(20)]
        # final 10% remains unchanged
    return Corruption(corrupted, labels, selected)
