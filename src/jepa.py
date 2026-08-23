from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from chemfm import IGNORE_INDEX


PREDICTOR_TOKENS = [f"<|predictor_{index}|>" for index in range(1, 11)]


class SIGReg:
    """LeJEPA Epps-Pulley SIGReg on random one-dimensional projections."""

    def __init__(
        self, *, knots: int = 17, t_max: float = 3.0,
        num_slices: int = 1024, seed: int = 0,
    ):
        if knots < 3 or knots % 2 == 0:
            raise ValueError("SIGReg knots must be an odd integer of at least three")
        if not 0 < t_max or num_slices < 1:
            raise ValueError("SIGReg t_max and num_slices must be positive")
        self.knots = knots
        self.t_max = t_max
        self.num_slices = num_slices
        self.seed = seed
        self.global_step = 0

    def _draw_parameters(
        self, dimensions: int, device: torch.device, dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        generator = torch.Generator(device=device).manual_seed(
            self.seed + self.global_step
        )
        directions = torch.randn(
            dimensions, self.num_slices,
            device=device, dtype=dtype, generator=generator,
        )
        directions = directions / directions.norm(dim=0, keepdim=True).clamp_min(1e-12)
        self.global_step += 1
        t = torch.linspace(
            0.0, self.t_max, self.knots, device=device, dtype=dtype
        )
        dt = self.t_max / (self.knots - 1)
        quadrature = torch.full_like(t, 2.0 * dt)
        quadrature[[0, -1]] = dt
        normal_cf = torch.exp(-0.5 * t.square())
        weights = quadrature * normal_cf
        return directions, t, normal_cf, weights

    def __call__(self, representations: torch.Tensor) -> torch.Tensor:
        if representations.ndim < 2:
            raise ValueError("SIGReg expects (..., samples, dimensions)")
        sample_count = representations.size(-2)
        if sample_count < 2:
            raise ValueError("SIGReg requires at least two samples")
        values = representations.float()
        directions, t, normal_cf, weights = self._draw_parameters(
            values.size(-1), values.device, values.dtype
        )

        projected = values @ directions
        arguments = projected.unsqueeze(-1) * t
        real_error = arguments.cos().mean(dim=-3) - normal_cf
        imaginary = arguments.sin().mean(dim=-3)
        statistic = ((real_error.square() + imaginary.square()) @ weights) * sample_count
        return statistic.mean()

    def start_streaming(
        self, *, views: int, dimensions: int, expected_samples: int,
        device: torch.device,
    ) -> "StreamingSIGReg":
        """Start an exact sufficient-statistic pass over a logical batch.

        Representations may arrive in memory-sized chunks. ``finalize`` returns
        the exact materialized-batch value and a VJP surrogate for recomputed
        chunks, without a detached queue or stale representations.
        """
        if views < 1 or dimensions < 1 or expected_samples < 2:
            raise ValueError("invalid streaming SIGReg shape")
        directions, t, normal_cf, weights = self._draw_parameters(
            dimensions, device, torch.float32
        )
        return StreamingSIGReg(
            expected_views=views,
            expected_samples=expected_samples,
            directions=directions,
            t=t,
            normal_cf=normal_cf,
            weights=weights,
        )


@dataclass
class PreparedSIGReg:
    """Finalized global ECF statistics for exact chunk-recomputed gradients."""

    loss: torch.Tensor
    directions: torch.Tensor
    t: torch.Tensor
    weights: torch.Tensor
    real_error: torch.Tensor
    imaginary: torch.Tensor

    def representation_gradients(self, representations: torch.Tensor) -> torch.Tensor:
        values = representations.detach().float()
        if values.ndim != 3 or values.size(0) != self.real_error.size(0):
            raise ValueError("streaming SIGReg chunks must have shape (views, samples, dimensions)")
        projected = values @ self.directions
        arguments = projected.unsqueeze(-1) * self.t
        weighted_t = self.weights * self.t
        projected_gradients = (
            2.0
            / (self.real_error.size(0) * self.directions.size(1))
            * (
                -arguments.sin() * self.real_error.unsqueeze(-3)
                + arguments.cos() * self.imaginary.unsqueeze(-3)
            )
            @ weighted_t
        )
        return projected_gradients @ self.directions.transpose(0, 1)

    def surrogate(self, representations: torch.Tensor) -> torch.Tensor:
        """Return a scalar with the exact global SIGReg gradient for this chunk."""
        gradients = self.representation_gradients(representations)
        return (representations * gradients.to(representations.dtype)).sum()


@dataclass
class StreamingSIGReg:
    expected_views: int
    expected_samples: int
    directions: torch.Tensor
    t: torch.Tensor
    normal_cf: torch.Tensor
    weights: torch.Tensor
    samples: int = 0
    cosine_sum: torch.Tensor | None = None
    sine_sum: torch.Tensor | None = None

    def update(self, representations: torch.Tensor) -> None:
        values = representations.detach().float()
        if values.ndim != 3:
            raise ValueError("streaming SIGReg chunks must have shape (views, samples, dimensions)")
        if values.size(0) != self.expected_views or values.size(-1) != self.directions.size(0):
            raise ValueError("streaming SIGReg chunk shape does not match its accumulator")
        projected = values @ self.directions
        arguments = projected.unsqueeze(-1) * self.t
        cosine = arguments.cos().sum(dim=-3)
        sine = arguments.sin().sum(dim=-3)
        self.cosine_sum = cosine if self.cosine_sum is None else self.cosine_sum + cosine
        self.sine_sum = sine if self.sine_sum is None else self.sine_sum + sine
        self.samples += values.size(-2)
        if self.samples > self.expected_samples:
            raise ValueError("streaming SIGReg received more samples than expected")

    def finalize(self) -> PreparedSIGReg:
        if self.samples != self.expected_samples:
            raise ValueError(
                f"streaming SIGReg expected {self.expected_samples} samples, got {self.samples}"
            )
        if self.cosine_sum is None or self.sine_sum is None:
            raise ValueError("streaming SIGReg received no chunks")
        real_error = self.cosine_sum / self.samples - self.normal_cf
        imaginary = self.sine_sum / self.samples
        loss = (
            (real_error.square() + imaginary.square()) @ self.weights
            * self.samples
        ).mean()
        return PreparedSIGReg(
            loss=loss,
            directions=self.directions,
            t=self.t,
            weights=self.weights,
            real_error=real_error,
            imaginary=imaginary,
        )


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
    sigreg_loss: torch.Tensor | None
    jepa_objective_loss: torch.Tensor | None
    logits: torch.Tensor
    source_states: torch.Tensor | None
    target_states: torch.Tensor | None
    source_final_indices: torch.Tensor | None
    target_final_indices: torch.Tensor | None
    shuffle_indices: list[int] | None
    jepa_active: bool


class CLMJEPA:
    def __init__(
        self, predictor_token_ids: Sequence[int], eos_token_id: int,
        pad_token_id: int, *, sigreg_seed: int = 0,
        optimized_native_logits: bool = False,
    ):
        self.predictor_token_ids = list(predictor_token_ids)
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.sigreg = SIGReg(seed=sigreg_seed)
        self.jepa_dropout_generator = torch.Generator().manual_seed(sigreg_seed)
        self.optimized_native_logits = optimized_native_logits

    def sample_jepa_activity(self, ratio: float) -> bool:
        if not 0.0 <= ratio <= 1.0:
            raise ValueError("JEPA activity ratio must be in [0, 1]")
        return bool(
            torch.rand(1, generator=self.jepa_dropout_generator).item() <= ratio
        )

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
        jepa_loss_type: str = "cosine",
        sigreg_tradeoff: float = 0.0,
        sigreg_relative_scale: float = 1.0,
        jepa_ratio: float = -1.0,
        jepa_targets: Sequence[torch.Tensor] | None = None,
        shuffle_seed: int | None = None,
        force_jepa_active: bool | None = None,
        endpoint_only: bool = False,
        representation_only: bool = False,
    ) -> CLMJEPAOutput:
        if not -1.0 <= jepa_ratio <= 1.0:
            raise ValueError("jepa_ratio must be -1 or in [0, 1]")
        if not 0.0 <= sigreg_tradeoff < 1.0:
            raise ValueError("sigreg_tradeoff must be in [0, 1)")
        if jepa_loss_type not in {"cosine", "mse"}:
            raise ValueError("jepa_loss_type must be cosine or mse")
        if sigreg_relative_scale <= 0.0:
            raise ValueError("sigreg_relative_scale must be positive")
        jepa_active = (
            force_jepa_active
            if force_jepa_active is not None
            else not (jepa_ratio > 0.0 and not self.sample_jepa_activity(jepa_ratio))
        )
        if (jepa_weight == 0.0 and not monitor_only) or not jepa_active:
            native = model(**{key: batch[key] for key in ("input_ids", "attention_mask", "labels")})
            return CLMJEPAOutput(
                loss=native_weight * native.loss,
                native_loss=native.loss,
                jepa_loss=None,
                sigreg_loss=None,
                jepa_objective_loss=None,
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
        batch_size = len(sources)
        # Endpoint-only statistics never consume vocabulary logits, so they can
        # always call the causal backbone directly.  This is exact even when the
        # gradient-bearing pass deliberately retains the standard LM forward.
        if self.optimized_native_logits or endpoint_only:
            causal_model = model.get_base_model() if hasattr(model, "get_base_model") else model
            backbone_outputs = causal_model.model(
                input_ids=padded,
                attention_mask=attention,
                use_cache=False,
                return_dict=True,
            )
            hidden = backbone_outputs.last_hidden_state
            if endpoint_only:
                native_loss = hidden.new_zeros(())
                logits = hidden.new_empty((batch_size, 0, 0))
            else:
                native_width = batch["input_ids"].shape[1]
                logits = causal_model.lm_head(hidden[:batch_size, :native_width]).float()
                native_labels = labels[:batch_size, :native_width]
                native_loss = F.cross_entropy(
                    logits[:, :-1].contiguous().view(-1, logits.size(-1)),
                    native_labels[:, 1:].contiguous().view(-1),
                    ignore_index=IGNORE_INDEX,
                )
        else:
            outputs = model(
                input_ids=padded,
                attention_mask=attention,
                labels=labels,
                output_hidden_states=True,
            )
            hidden = outputs.hidden_states[-1]
            native_loss = outputs.loss
            logits = outputs.logits[:batch_size, :batch["input_ids"].shape[1]]
        source_indices = attention[batch_size:2 * batch_size].sum(dim=1) - (2 if k == -1 else 1)
        if (source_indices < 0).any():
            raise ValueError("k=-1 requires every source to contain at least two active tokens")
        target_indices = attention[2 * batch_size:].sum(dim=1) - 1
        row_indices = torch.arange(batch_size, device=hidden.device)
        source_states = hidden[row_indices + batch_size, source_indices]
        target_states = hidden[row_indices + 2 * batch_size, target_indices]
        jepa_target_states = target_states.detach() if stop_gradient_target else target_states
        if representation_only:
            jepa_loss = None
        elif jepa_loss_type == "mse":
            # Exact upstream LLM-JEPA --jepa_mse branch: no endpoint
            # normalization and a mean over examples and representation axes.
            jepa_loss = torch.mean((source_states - jepa_target_states) ** 2)
        else:
            jepa_loss = 1.0 - F.cosine_similarity(
                source_states, jepa_target_states, dim=-1
            ).mean()
        sigreg_loss = None
        jepa_objective_loss = jepa_loss
        if sigreg_tradeoff > 0.0 and not representation_only:
            # LeJEPA regularizes every view distribution independently. Scaling the
            # 0.95/0.05 mixture by 1/0.95 preserves this project's frozen cosine
            # coefficient while retaining LeJEPA's standard relative trade-off.
            sigreg_loss = self.sigreg(torch.stack((source_states, target_states)))
            jepa_objective_loss = (
                jepa_loss
                + (
                    sigreg_relative_scale
                    * sigreg_tradeoff / (1.0 - sigreg_tradeoff)
                ) * sigreg_loss
            )
        applied_jepa = native_loss.new_zeros(())
        if not monitor_only and not representation_only:
            applied_jepa = jepa_weight * jepa_objective_loss
        loss = native_weight * native_loss + applied_jepa
        return CLMJEPAOutput(
            loss=loss,
            native_loss=native_loss,
            jepa_loss=jepa_loss,
            sigreg_loss=sigreg_loss,
            jepa_objective_loss=jepa_objective_loss,
            logits=logits,
            source_states=source_states,
            target_states=target_states,
            source_final_indices=source_indices,
            target_final_indices=target_indices,
            shuffle_indices=shuffle_indices,
            jepa_active=True,
        )
