from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class GenerationExample:
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    target_positions: tuple[int, ...]

    @property
    def supervised_tokens(self) -> int:
        return len(self.target_positions)


def make_full_span_example(
    receptor_tokens: list[int] | tuple[int, ...],
    peptide_tokens: list[int] | tuple[int, ...],
    *,
    bos_token_id: int,
    eos_token_id: int,
    mask_token_id: int,
) -> GenerationExample:
    if not receptor_tokens or not peptide_tokens:
        raise ValueError("receptor and peptide must both contain residues")
    receptor_tokens = tuple(map(int, receptor_tokens))
    peptide_tokens = tuple(map(int, peptide_tokens))
    receptor_length = len(receptor_tokens)
    peptide_length = len(peptide_tokens)
    input_ids = (
        int(bos_token_id), *receptor_tokens, *(int(mask_token_id) for _ in peptide_tokens),
        int(eos_token_id),
    )
    target_positions = tuple(range(receptor_length + 1, receptor_length + peptide_length + 1))
    labels = (-100,) * (receptor_length + 1) + peptide_tokens + (-100,)
    if len(target_positions) != peptide_length or any(input_ids[position] != mask_token_id for position in target_positions):
        raise RuntimeError("the complete peptide span was not masked")
    if sum(label != -100 for label in labels) != peptide_length:
        raise RuntimeError("loss labels do not cover exactly the peptide span")
    return GenerationExample(input_ids, labels, target_positions)


def selected_cross_entropy_sum(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Cross-entropy over supervised peptide positions only."""
    selected = labels.ne(-100)
    return F.cross_entropy(logits[selected].float(), labels[selected], reduction="sum")


def per_example_loss_sums(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return vectorized loss sums and target counts, one item per sequence."""
    selected = labels.ne(-100)
    losses = F.cross_entropy(logits[selected].float(), labels[selected], reduction="none")
    batch_index = torch.arange(labels.shape[0], device=labels.device).unsqueeze(1).expand_as(labels)
    sums = torch.zeros(labels.shape[0], dtype=losses.dtype, device=labels.device)
    sums.index_add_(0, batch_index[selected], losses)
    return sums, selected.sum(dim=1)
