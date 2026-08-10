from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from chemfm import IGNORE_INDEX


PREDICTOR_TOKENS = [f"<|predictor_{index}|>" for index in range(1, 11)]


def add_predictor_tokens(tokenizer, model=None) -> list[int]:
    tokenizer.add_special_tokens({"additional_special_tokens": PREDICTOR_TOKENS})
    if model is not None and model.get_input_embeddings().weight.shape[0] != len(tokenizer):
        # Match pinned LLM-JEPA: keep resize_token_embeddings' independent
        # initialization for every predictor token.
        model.resize_token_embeddings(len(tokenizer))
    return tokenizer.convert_tokens_to_ids(PREDICTOR_TOKENS)


def _unpadded_rows(values: torch.Tensor, mask: torch.Tensor) -> list[torch.Tensor]:
    return [row[row_mask.bool()] for row, row_mask in zip(values, mask)]


def extract_source_and_target(batch: dict[str, torch.Tensor]) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    sources: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    for ids, labels, mask in zip(batch["input_ids"], batch["labels"], batch["attention_mask"]):
        active = mask.bool()
        ids = ids[active]
        labels = labels[active]
        target_positions = labels.ne(IGNORE_INDEX).nonzero(as_tuple=False).flatten()
        if target_positions.numel() == 0:
            raise ValueError("each native row must contain target labels")
        start = int(target_positions[0])
        if not labels[start:].ne(IGNORE_INDEX).all():
            raise ValueError("target labels must be one contiguous suffix")
        sources.append(ids[:start])
        targets.append(ids[start:])
    return sources, targets


def matched_derangement(targets: Sequence[torch.Tensor], seed: int) -> list[int]:
    """Return a reproducible, length-matched target permutation without valid pairs."""
    count = len(targets)
    if count < 2:
        raise ValueError("shuffling requires at least two targets")
    generator = torch.Generator().manual_seed(seed)
    best: tuple[int, tuple[int, ...]] | None = None
    for _ in range(512):
        candidate = tuple(torch.randperm(count, generator=generator).tolist())
        if any(i == j or torch.equal(targets[i], targets[j]) for i, j in enumerate(candidate)):
            continue
        cost = sum(abs(len(targets[i]) - len(targets[j])) for i, j in enumerate(candidate))
        if best is None or (cost, candidate) < best:
            best = (cost, candidate)
    if best is None:
        raise ValueError("no unequal-target derangement exists for this batch")
    return list(best[1])


@dataclass
class CLMJEPAOutput:
    loss: torch.Tensor
    native_loss: torch.Tensor
    jepa_loss: torch.Tensor | None
    logits: torch.Tensor
    source_states: torch.Tensor | None
    target_states: torch.Tensor | None
    source_final_indices: torch.Tensor | None
    target_final_indices: torch.Tensor | None
    shuffle_indices: list[int] | None
    jepa_active: bool


class CLMJEPA:
    def __init__(self, predictor_token_ids: Sequence[int], eos_token_id: int, pad_token_id: int):
        self.predictor_token_ids = list(predictor_token_ids)
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id

    def __call__(
        self,
        model,
        batch: dict[str, torch.Tensor],
        *,
        k: int,
        jepa_weight: float,
        native_weight: float = 1.0,
        monitor_only: bool = False,
        stop_gradient_target: bool = False,
        jepa_ratio: float = -1.0,
        jepa_targets: Sequence[torch.Tensor] | None = None,
        shuffle_seed: int | None = None,
    ) -> CLMJEPAOutput:
        if not -1.0 <= jepa_ratio <= 1.0:
            raise ValueError("jepa_ratio must be -1 or in [0, 1]")
        jepa_active = not (jepa_ratio > 0.0 and torch.rand(1).item() > jepa_ratio)
        if (jepa_weight == 0.0 and not monitor_only) or not jepa_active:
            native = model(**{key: batch[key] for key in ("input_ids", "attention_mask", "labels")})
            return CLMJEPAOutput(
                loss=native_weight * native.loss,
                native_loss=native.loss,
                jepa_loss=None,
                logits=native.logits,
                source_states=None,
                target_states=None,
                source_final_indices=None,
                target_final_indices=None,
                shuffle_indices=None,
                jepa_active=False,
            )

        if k != -1 and not 0 <= k <= len(self.predictor_token_ids):
            raise ValueError(f"k must be -1 or in [0, {len(self.predictor_token_ids)}]")
        sources, native_targets = extract_source_and_target(batch)
        targets = list(jepa_targets) if jepa_targets is not None else native_targets
        if len(targets) != len(sources):
            raise ValueError("JEPA target count must equal native batch size")
        shuffle_indices = None
        if shuffle_seed is not None:
            shuffle_indices = matched_derangement(targets, shuffle_seed)
            targets = [targets[index] for index in shuffle_indices]

        # LLM-JEPA appends <|predictor_k|>, ..., <|predictor_1|> in that order.
        predictor_suffix = [] if k == -1 else list(reversed(self.predictor_token_ids[:k]))
        source_rows = [
            torch.cat((source, source.new_tensor(predictor_suffix))) if predictor_suffix else source
            for source in sources
        ]
        for target in targets:
            if int(target[-1]) != self.eos_token_id:
                raise ValueError("JEPA target view must end at target <eos>")
        native_rows = _unpadded_rows(batch["input_ids"], batch["attention_mask"])
        native_label_rows = _unpadded_rows(batch["labels"], batch["attention_mask"])
        rows = native_rows + source_rows + targets
        padded = pad_sequence(rows, batch_first=True, padding_value=self.pad_token_id)
        attention = padded.ne(self.pad_token_id)
        auxiliary_labels = [
            row.new_full(row.shape, IGNORE_INDEX) for row in source_rows + targets
        ]
        labels = pad_sequence(
            native_label_rows + auxiliary_labels,
            batch_first=True,
            padding_value=IGNORE_INDEX,
        )
        outputs = model(
            input_ids=padded,
            attention_mask=attention,
            labels=labels,
            output_hidden_states=True,
        )
        hidden = outputs.hidden_states[-1]
        batch_size = len(sources)
        source_indices = attention[batch_size:2 * batch_size].sum(dim=1) - (2 if k == -1 else 1)
        if (source_indices < 0).any():
            raise ValueError("k=-1 requires every source to contain at least two active tokens")
        target_indices = attention[2 * batch_size:].sum(dim=1) - 1
        row_indices = torch.arange(batch_size, device=hidden.device)
        source_states = hidden[row_indices + batch_size, source_indices]
        target_states = hidden[row_indices + 2 * batch_size, target_indices]
        jepa_target_states = target_states.detach() if stop_gradient_target else target_states
        jepa_loss = 1.0 - F.cosine_similarity(
            source_states, jepa_target_states, dim=-1
        ).mean()
        applied_jepa = outputs.loss.new_zeros(()) if monitor_only else jepa_weight * jepa_loss
        loss = native_weight * outputs.loss + applied_jepa
        return CLMJEPAOutput(
            loss=loss,
            native_loss=outputs.loss,
            jepa_loss=jepa_loss,
            logits=outputs.logits[:batch_size, :batch["input_ids"].shape[1]],
            source_states=source_states,
            target_states=target_states,
            source_final_indices=source_indices,
            target_final_indices=target_indices,
            shuffle_indices=shuffle_indices,
            jepa_active=True,
        )
