"""Official Semantic Tube Prediction adapted to ChemFM serialization.

The executable reference is ``galilai-group/llm-jepa/stp.py`` at
``ea0017c654ad917066ff32afc88276bea8ca5f7e``.  The random-span sampler,
content-boundary convention, transition construction, cosine reduction, final
hidden-state selection, and symmetric gradient flow below directly port that
released implementation.  Only discovery of the user/assistant content
boundaries is adapted from chat messages to ChemFM's
``<rstart>reactants<eos><prostart>product<eos>`` serialization.
"""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager

import torch
import torch.nn.functional as F

from chemfm import IGNORE_INDEX


STP_PAPER = "https://arxiv.org/abs/2602.22617"
STP_UPSTREAM_REPOSITORY = "https://github.com/galilai-group/llm-jepa"
STP_UPSTREAM_COMMIT = "ea0017c654ad917066ff32afc88276bea8ca5f7e"


@dataclass
class STPOutput:
    loss: torch.Tensor
    native_loss: torch.Tensor
    jepa_loss: torch.Tensor
    sigreg_loss: None
    jepa_objective_loss: torch.Tensor
    logits: torch.Tensor
    jepa_active: bool
    sampled_spans: tuple[tuple[int, ...], ...]


class SemanticTubePrediction:
    """Released STP random-span loss around a shared causal language model."""

    def __init__(
        self,
        *,
        seed: int,
        reactant_start_token_id: int,
        product_start_token_id: int,
        eos_token_id: int,
    ) -> None:
        self.seed = int(seed)
        self.reactant_start_token_id = int(reactant_start_token_id)
        self.product_start_token_id = int(product_start_token_id)
        self.eos_token_id = int(eos_token_id)
        self._g: torch.Generator | None = None

    def _generator(self, device: torch.device) -> torch.Generator:
        # Upstream constructs its dedicated sampler on TrainingArguments.device
        # and seeds it with the fine-tuning seed on rank zero.
        if self._g is None:
            self._g = torch.Generator(device=device)
            self._g.manual_seed(self.seed)
        elif self._g.device != device:
            raise ValueError(
                f"STP sampler was initialized on {self._g.device}, not {device}"
            )
        return self._g

    def generator_state(self) -> torch.Tensor:
        if self._g is None:
            return torch.Generator().manual_seed(self.seed).get_state()
        return self._g.get_state()

    def set_generator_state(
        self, state: torch.Tensor, device: torch.device
    ) -> None:
        self._g = torch.Generator(device=device)
        self._g.set_state(state.cpu())

    def get_s_t(
        self, full_length: int, *, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Direct port of the released default ``get_s_t`` branch."""
        if full_length < 2:
            raise ValueError("STP requires at least two content tokens")
        generator = self._generator(device)
        patch_start_offset = torch.randint(
            0, full_length, (), generator=generator, device=generator.device
        )
        while True:
            patch_end_offset = torch.randint(
                patch_start_offset + 1,
                full_length + 1,
                (),
                generator=generator,
                device=generator.device,
            )
            if patch_end_offset - patch_start_offset < full_length:
                break
        return patch_start_offset, patch_end_offset

    @staticmethod
    def get_embeddings(
        hidden_states: torch.Tensor,
        user_start_end: torch.Tensor,
        assistant_start_end: torch.Tensor,
        patch_start_offset: int | torch.Tensor,
        patch_end_offset: int | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Direct port of released STP's before/patch/after construction."""
        user_start = user_start_end[0] + 1
        user_end = user_start_end[1] + 1
        assistant_start = assistant_start_end[0] + 1
        assistant_end = assistant_start_end[1] + 1
        if patch_start_offset + user_start < user_end:
            patch_start = user_start + patch_start_offset
        else:
            patch_start = (
                assistant_start + patch_start_offset - (user_end - user_start)
            )
        if patch_end_offset + user_start < user_end:
            patch_end = user_start + patch_end_offset
        else:
            patch_end = assistant_start + patch_end_offset - (user_end - user_start)

        user_start_embedding = hidden_states[user_start - 1]
        user_end_embedding = hidden_states[user_end - 1]
        assistant_start_embedding = hidden_states[assistant_start - 1]
        assistant_end_embedding = hidden_states[assistant_end - 1]
        patch_start_embedding = hidden_states[patch_start - 1]
        patch_end_embedding = hidden_states[patch_end - 1]

        if patch_start >= assistant_start:
            before = (
                user_end_embedding
                - user_start_embedding
                + patch_start_embedding
                - assistant_start_embedding
            )
            patch = patch_end_embedding - patch_start_embedding
            after = assistant_end_embedding - patch_end_embedding
        elif patch_end <= user_end:
            before = patch_start_embedding - user_start_embedding
            patch = patch_end_embedding - patch_start_embedding
            after = (
                user_end_embedding
                - patch_end_embedding
                + assistant_end_embedding
                - assistant_start_embedding
            )
        else:
            before = patch_start_embedding - user_start_embedding
            patch = (
                user_end_embedding
                - patch_start_embedding
                + patch_end_embedding
                - assistant_start_embedding
            )
            after = assistant_end_embedding - patch_end_embedding

        return before, patch, after

    def content_boundaries(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Map official user/assistant content boundaries onto ChemFM tokens."""
        user = []
        assistant = []
        for input_ids, labels, attention_mask in zip(
            batch["input_ids"], batch["labels"], batch["attention_mask"]
        ):
            active_length = int(attention_mask.sum())
            active_ids = input_ids[:active_length]
            active_labels = labels[:active_length]
            target_positions = active_labels.ne(IGNORE_INDEX).nonzero().flatten()
            if target_positions.numel() == 0:
                raise ValueError("STP requires a nonempty product suffix")
            target_start = int(target_positions[0])
            user_start_end = (0, target_start - 2)
            assistant_start_end = (target_start, active_length - 2)
            if user_start_end[1] <= user_start_end[0]:
                raise ValueError("STP requires a nonempty reactant SMILES")
            if assistant_start_end[1] <= assistant_start_end[0]:
                raise ValueError("STP requires a nonempty product SMILES")
            expected = (
                (user_start_end[0], self.reactant_start_token_id),
                (user_start_end[1] + 1, self.eos_token_id),
                (assistant_start_end[0], self.product_start_token_id),
                (assistant_start_end[1] + 1, self.eos_token_id),
            )
            for index, token_id in expected:
                if int(active_ids[index]) != token_id:
                    raise ValueError(
                        "ChemFM serialization does not match STP content boundaries"
                    )
            if not active_labels[target_start:].ne(IGNORE_INDEX).all():
                raise ValueError("ChemFM product labels must be one contiguous suffix")
            user.append(user_start_end)
            assistant.append(assistant_start_end)
        return (
            torch.tensor(user, dtype=torch.long, device=batch["input_ids"].device),
            torch.tensor(
                assistant, dtype=torch.long, device=batch["input_ids"].device
            ),
        )

    def __call__(
        self,
        model,
        batch: dict[str, torch.Tensor],
        *,
        stp_weight: float,
    ) -> STPOutput:
        # The released experiment's default random_span_layer is -1.  Capture
        # the exact post-final-RMSNorm tensor without asking Transformers to
        # retain and return the embedding plus every decoder-layer state.
        with capture_final_hidden_state(model) as captured:
            outputs = model(**{
                key: batch[key]
                for key in ("input_ids", "attention_mask", "labels")
            })
        hidden_states = captured[0]
        user_start_end, assistant_start_end = self.content_boundaries(batch)

        # Upstream uses default-float (FP32) buffers even when model states are
        # BF16, so its cosine is evaluated in FP32. Preserve that exactly.
        user_embedding = torch.zeros(
            (hidden_states.shape[0], hidden_states.shape[-1]),
            device=hidden_states.device,
        )
        assistant_embedding = torch.zeros_like(user_embedding)
        sampled_spans = []
        for index in range(hidden_states.shape[0]):
            user_length = int(user_start_end[index, 1] - user_start_end[index, 0])
            assistant_length = int(
                assistant_start_end[index, 1] - assistant_start_end[index, 0]
            )
            full_length = user_length + assistant_length
            span_start, span_end = self.get_s_t(
                full_length, device=hidden_states.device
            )
            before, patch, after = self.get_embeddings(
                hidden_states[index],
                user_start_end[index],
                assistant_start_end[index],
                span_start,
                span_end,
            )
            user_embedding[index] = before + after
            assistant_embedding[index] = patch
            sampled_spans.append((int(span_start), int(span_end), full_length))

        cosine_similarity = F.cosine_similarity(
            user_embedding, assistant_embedding, dim=-1
        )
        stp_loss = 1.0 - torch.mean(cosine_similarity)
        total_loss = outputs.loss + float(stp_weight) * stp_loss
        return STPOutput(
            loss=total_loss,
            native_loss=outputs.loss,
            jepa_loss=stp_loss,
            sigreg_loss=None,
            jepa_objective_loss=stp_loss,
            logits=outputs.logits,
            jepa_active=True,
            sampled_spans=tuple(sampled_spans),
        )


def _final_norm_module(model):
    """Return ChemFM/Llama's post-decoder norm through optional PEFT wrappers."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    decoder = getattr(base, "model", None)
    norm = getattr(decoder, "norm", None)
    if norm is None:
        raise TypeError(
            "STP final-state capture requires a LlamaForCausalLM-compatible "
            "model exposing model.norm"
        )
    return norm


@contextmanager
def capture_final_hidden_state(model):
    """Capture exactly the tensor formerly returned as ``hidden_states[-1]``."""
    captured: list[torch.Tensor] = []

    def hook(_module, _inputs, output):
        captured.append(output)

    handle = _final_norm_module(model).register_forward_hook(hook)
    try:
        yield captured
    finally:
        handle.remove()
    if len(captured) != 1:
        raise RuntimeError(f"expected one final hidden-state capture, got {len(captured)}")


class PaperSemanticTubePrediction(SemanticTubePrediction):
    """Literal paper equation ``1-cos(h_r-h_s, h_t-h_r)`` for ``s<r<t``."""

    def get_s_r_t(
        self, full_length: int, *, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if full_length < 3:
            raise ValueError("paper STP requires at least three content tokens")
        while True:
            span_start, span_end = self.get_s_t(full_length, device=device)
            if span_end - span_start >= 2:
                break
        generator = self._generator(device)
        interior = torch.randint(
            span_start + 1,
            span_end,
            (),
            generator=generator,
            device=generator.device,
        )
        return span_start, interior, span_end

    @staticmethod
    def boundary_embedding(
        hidden_states: torch.Tensor,
        user_start_end: torch.Tensor,
        assistant_start_end: torch.Tensor,
        offset: int | torch.Tensor,
    ) -> torch.Tensor:
        """Map a concatenated-content boundary offset to its causal state."""
        user_start = user_start_end[0] + 1
        user_end = user_start_end[1] + 1
        assistant_start = assistant_start_end[0] + 1
        if offset + user_start <= user_end:
            boundary = user_start + offset
        else:
            boundary = assistant_start + offset - (user_end - user_start)
        return hidden_states[boundary - 1]

    def __call__(
        self,
        model,
        batch: dict[str, torch.Tensor],
        *,
        stp_weight: float,
    ) -> STPOutput:
        with capture_final_hidden_state(model) as captured:
            outputs = model(**{
                key: batch[key]
                for key in ("input_ids", "attention_mask", "labels")
            })
        hidden_states = captured[0]
        user_start_end, assistant_start_end = self.content_boundaries(batch)

        first_transition = torch.zeros(
            (hidden_states.shape[0], hidden_states.shape[-1]),
            device=hidden_states.device,
        )
        second_transition = torch.zeros_like(first_transition)
        sampled_spans = []
        for index in range(hidden_states.shape[0]):
            user_length = int(user_start_end[index, 1] - user_start_end[index, 0])
            assistant_length = int(
                assistant_start_end[index, 1] - assistant_start_end[index, 0]
            )
            full_length = user_length + assistant_length
            span_start, interior, span_end = self.get_s_r_t(
                full_length, device=hidden_states.device
            )
            h_s = self.boundary_embedding(
                hidden_states[index], user_start_end[index],
                assistant_start_end[index], span_start,
            )
            h_r = self.boundary_embedding(
                hidden_states[index], user_start_end[index],
                assistant_start_end[index], interior,
            )
            h_t = self.boundary_embedding(
                hidden_states[index], user_start_end[index],
                assistant_start_end[index], span_end,
            )
            first_transition[index] = h_r - h_s
            second_transition[index] = h_t - h_r
            sampled_spans.append(
                (int(span_start), int(interior), int(span_end), full_length)
            )

        stp_loss = 1.0 - torch.mean(F.cosine_similarity(
            first_transition, second_transition, dim=-1
        ))
        return STPOutput(
            loss=outputs.loss + float(stp_weight) * stp_loss,
            native_loss=outputs.loss,
            jepa_loss=stp_loss,
            sigreg_loss=None,
            jepa_objective_loss=stp_loss,
            logits=outputs.logits,
            jepa_active=True,
            sampled_spans=tuple(sampled_spans),
        )
