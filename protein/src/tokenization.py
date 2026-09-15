"""Fail-closed RITA tokenization and causal-position contracts.

The published tokenizer has no BOS token and appends EOS through its tokenizer
post-processor.  Although IDs 1 and 2 exist as <PAD>/<EOS>, the published
``special_tokens_map.json`` is empty, so callers must not rely on generic
``tokenizer.pad_token_id`` or ``tokenizer.eos_token_id`` attributes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch

from .constants import CANONICAL_AA_SET, RITA_EOS_ID, RITA_PAD_ID


@dataclass(frozen=True)
class EncodedProtein:
    sequence: str
    input_ids: tuple[int, ...]
    residue_positions: tuple[int, ...]

    @property
    def eos_position(self) -> int:
        return len(self.sequence)


def is_canonical(sequence: str) -> bool:
    return bool(sequence) and set(sequence).issubset(CANONICAL_AA_SET)


def encode_canonical(tokenizer, sequence: str) -> EncodedProtein:
    if not is_canonical(sequence):
        invalid = sorted(set(sequence) - CANONICAL_AA_SET)
        raise ValueError(f"noncanonical or empty protein: {invalid}")
    ids = tuple(int(x) for x in tokenizer.encode(sequence, add_special_tokens=True))
    if len(ids) != len(sequence) + 1:
        raise ValueError("RITA contract violated: expected one token/residue plus EOS")
    if ids[-1] != RITA_EOS_ID:
        raise ValueError("RITA contract violated: trailing token is not EOS id 2")
    if RITA_PAD_ID in ids[:-1] or RITA_EOS_ID in ids[:-1]:
        raise ValueError("PAD/EOS contamination within residue tokens")
    tokens = tokenizer.convert_ids_to_tokens(list(ids[:-1]))
    if list(tokens) != list(sequence):
        raise ValueError(f"RITA residue/token mismatch: {tokens!r} vs {sequence!r}")
    return EncodedProtein(sequence, ids, tuple(range(len(sequence))))


def collate_encoded(items: Sequence[EncodedProtein]) -> dict[str, torch.Tensor]:
    if not items:
        raise ValueError("cannot collate an empty protein batch")
    width = max(len(item.input_ids) for item in items)
    ids = torch.full((len(items), width), RITA_PAD_ID, dtype=torch.long)
    attention = torch.zeros_like(ids)
    residue = torch.zeros_like(ids, dtype=torch.bool)
    for row, item in enumerate(items):
        n = len(item.input_ids)
        ids[row, :n] = torch.tensor(item.input_ids)
        attention[row, :n] = 1
        residue[row, : len(item.sequence)] = True
    return {"input_ids": ids, "attention_mask": attention, "residue_mask": residue}


def next_residue_transition_mask(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Mask ``h_t -> x[t+1]`` where both positions are biological residues."""
    if input_ids.shape != attention_mask.shape or input_ids.ndim != 2:
        raise ValueError("input_ids and attention_mask must be equal BxL tensors")
    current_real = attention_mask[:, :-1].bool()
    future_real = attention_mask[:, 1:].bool()
    current_special = (input_ids[:, :-1] == RITA_PAD_ID) | (input_ids[:, :-1] == RITA_EOS_ID)
    future_special = (input_ids[:, 1:] == RITA_PAD_ID) | (input_ids[:, 1:] == RITA_EOS_ID)
    return current_real & future_real & ~current_special & ~future_special


def eligible_geometry_centers(residue_mask: torch.Tensor) -> torch.Tensor:
    """Centers r for which r-1, r, and r+1 are all residues."""
    if residue_mask.ndim != 2:
        raise ValueError("residue_mask must be BxL")
    return residue_mask[:, :-2] & residue_mask[:, 1:-1] & residue_mask[:, 2:]


def assert_no_special_positions(
    input_ids: torch.Tensor, position_mask: torch.Tensor, *, shifted: bool = False
) -> None:
    ids = input_ids[:, :-1] if shifted else input_ids
    if ids.shape != position_mask.shape:
        raise ValueError("position mask shape mismatch")
    chosen = ids[position_mask]
    if torch.any((chosen == RITA_PAD_ID) | (chosen == RITA_EOS_ID)):
        raise AssertionError("PAD/EOS entered a residue-only diagnostic")

