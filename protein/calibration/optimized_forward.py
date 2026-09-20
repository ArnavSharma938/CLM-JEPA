from __future__ import annotations

from contextlib import contextmanager
from types import MethodType
from typing import Any, Iterable, Iterator

import torch


def _fixed_padded_extract_features(self, prev_output_tokens, encoder_out, incremental_state=None):
    """Equivalent IF1 decoder path when fixed-length teacher forcing is padded."""
    bs, _ = prev_output_tokens.size()
    enc = encoder_out["encoder_out"][0]
    if enc.size(1) != bs:
        raise RuntimeError(f"Unexpected encoder batch: {enc.shape}")
    padding_mask = encoder_out["encoder_padding_mask"][0]
    positions = self.embed_positions(prev_output_tokens)
    if incremental_state is not None:
        prev_output_tokens = prev_output_tokens[:, -1:]
        positions = positions[:, -1:]
    x = self.embed_scale * self.embed_tokens(prev_output_tokens)
    if self.project_in_dim is not None:
        x = self.project_in_dim(x)
    x += positions
    x = self.dropout_module(x).transpose(0, 1)
    # The length-72 sentinel guarantees padding in every calibration batch.
    # Avoid the upstream .any() device synchronization and duplicate equality.
    self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)
    inner_states = [x]
    for layer in self.layers:
        self_attn_mask = self.buffered_future_mask(x) if incremental_state is None else None
        x, _, _ = layer(
            x, enc, padding_mask, incremental_state,
            self_attn_mask=self_attn_mask,
            self_attn_padding_mask=self_attn_padding_mask,
            need_attn=False,
            need_head_weights=False,
        )
        inner_states.append(x)
    if self.layer_norm is not None:
        x = self.layer_norm(x)
    return x.transpose(0, 1), {"inner_states": inner_states}


def enable_if1_training_optimizations(model: torch.nn.Module) -> None:
    decoder = model.decoder
    if not getattr(decoder, "_if1_fixed_padded_extract", False):
        decoder.extract_features = MethodType(_fixed_padded_extract_features, decoder)
        decoder._if1_fixed_padded_extract = True


def prefetch_batches(
    loader: Iterable[dict[str, Any]], device: torch.device
) -> Iterator[dict[str, Any]]:
    """Overlap pinned host transfers for the next batch with GPU compute."""
    if device.type != "cuda":
        yield from loader
        return
    stream = torch.cuda.Stream(device=device)
    iterator = iter(loader)

    def preload() -> dict[str, Any] | None:
        try:
            batch = next(iterator)
        except StopIteration:
            return None
        with torch.cuda.stream(stream):
            for key, value in tuple(batch.items()):
                if torch.is_tensor(value):
                    batch[key] = value.to(device, non_blocking=True)
        return batch

    pending = preload()
    while pending is not None:
        torch.cuda.current_stream(device).wait_stream(stream)
        current = pending
        for value in current.values():
            if torch.is_tensor(value):
                value.record_stream(torch.cuda.current_stream(device))
        pending = preload()
        yield current


def enable_if1_evaluation_optimizations(model: torch.nn.Module) -> None:
    """Install parameter-transparent IF1 evaluation fast paths once.

    ESM-IF1's decoder asks every cross-attention block to return attention
    maps in eval mode even though its caller discards them.  It also projects
    identical encoder K/V tensors once per sequence.  The patched projection
    forwards project one representative per backbone and expand only the much
    smaller projected result.  No modules or parameters are replaced, so
    checkpoint names and optimizer state remain unchanged.
    """
    for layer in getattr(model.decoder, "layers", ()):
        layer.need_attn = False
        attention = layer.encoder_attn
        if attention is None:
            continue
        for projection in (attention.k_proj, attention.v_proj):
            if getattr(projection, "_if1_shared_projection", False):
                continue
            original = projection.forward

            def shared_forward(module, inputs, *, _original=original):
                context = getattr(module, "_if1_backbone_context", None)
                if context is None or module.training:
                    return _original(inputs)
                unique_rows, inverse = context
                projected = _original(inputs.index_select(1, unique_rows))
                return projected.index_select(1, inverse)

            projection.forward = MethodType(shared_forward, projection)
            projection._if1_shared_projection = True
            projection._if1_backbone_context = None


@contextmanager
def _shared_projection_context(
    model: torch.nn.Module,
    unique_rows: torch.Tensor,
    inverse: torch.Tensor,
) -> Iterator[None]:
    projections = []
    for layer in getattr(model.decoder, "layers", ()):
        attention = layer.encoder_attn
        if attention is None:
            continue
        for projection in (attention.k_proj, attention.v_proj):
            if getattr(projection, "_if1_shared_projection", False):
                projection._if1_backbone_context = (unique_rows, inverse)
                projections.append(projection)
    try:
        yield
    finally:
        for projection in projections:
            projection._if1_backbone_context = None


class OptimizedIF1Forward(torch.nn.Module):
    """Compilation-friendly IF1 forward with optional pre-encoder deduplication."""

    supports_backbone_inverse = True

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        coords: torch.Tensor,
        padding_mask: torch.Tensor,
        confidence: torch.Tensor,
        previous: torch.Tensor,
        backbone_inverse: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, Any]:
        if backbone_inverse is None:
            return self.model(coords, padding_mask, confidence, previous)
        encoded = self.model.encoder(coords, padding_mask, confidence)
        expanded = expand_encoder_for_decoder(encoded, backbone_inverse)
        return self.model.decoder(previous, encoder_out=expanded)


def backbone_inverse(records: list[Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Return first-occurrence rows and a row-to-unique inverse map."""
    first: dict[tuple[Any, Any], int] = {}
    unique: list[int] = []
    inverse: list[int] = []
    for row, record in enumerate(records):
        key = (record.backbone_path, record.chain_id)
        if key not in first:
            first[key] = len(unique)
            unique.append(row)
        inverse.append(first[key])
    return (
        torch.tensor(unique, dtype=torch.long, device=device),
        torch.tensor(inverse, dtype=torch.long, device=device),
    )


def batch_backbone_indices(batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    """Return cached CPU identity maps, or derive them for legacy/test batches."""
    if "backbone_unique" in batch and "backbone_inverse" in batch:
        return batch["backbone_unique"], batch["backbone_inverse"]
    return backbone_inverse(batch["records"], torch.device("cpu"))


def expand_encoder_for_decoder(encoder_out: dict[str, Any], inverse: torch.Tensor) -> dict[str, Any]:
    """Expand unique encoder rows using the dimensions defined by ESM-IF1."""
    if not encoder_out.get("encoder_out") or not encoder_out.get("encoder_padding_mask"):
        raise RuntimeError("Unexpected ESM-IF1 encoder output schema")
    encoded = encoder_out["encoder_out"][0]
    mask = encoder_out["encoder_padding_mask"][0]
    if encoded.shape[1] == 1:
        # Maximal-reuse batches dominate evaluation. A zero-stride view avoids
        # materializing B copies before the decoder consumes the tensor.
        return {
            "encoder_out": [encoded.expand(-1, inverse.numel(), -1)],
            "encoder_padding_mask": [mask.expand(inverse.numel(), -1)],
        }
    return {
        "encoder_out": [encoded.index_select(1, inverse)],
        "encoder_padding_mask": [mask.index_select(0, inverse)],
    }


def model_logits(
    model: torch.nn.Module,
    batch: dict[str, Any],
    device: torch.device,
    *,
    reuse_backbones: bool,
    tokens: torch.Tensor | None = None,
) -> torch.Tensor:
    if tokens is None:
        tokens = batch["tokens"].to(device, non_blocking=True)
    if not reuse_backbones:
        coords = batch["coords"].to(device, non_blocking=True)
        confidence = batch["confidence"].to(device, non_blocking=True)
        padding_mask = batch["padding_mask"].to(device, non_blocking=True)
        logits, _ = model(coords, padding_mask, confidence, tokens[:, :-1])
        return logits
    unique_cpu, inverse_cpu = batch_backbone_indices(batch)
    if (
        unique_cpu.numel() * 2 > len(batch["records"])
        and type(model).forward is not torch.nn.Module.forward
    ):
        coords = batch["coords"].to(device, non_blocking=True)
        confidence = batch["confidence"].to(device, non_blocking=True)
        padding_mask = batch["padding_mask"].to(device, non_blocking=True)
        logits, _ = model(coords, padding_mask, confidence, tokens[:, :-1])
        return logits
    # Select on the host so duplicate coordinates are never transferred to or
    # simultaneously materialized on the accelerator.
    coords = batch["coords"].index_select(0, unique_cpu).to(device, non_blocking=True)
    confidence = batch["confidence"].index_select(0, unique_cpu).to(device, non_blocking=True)
    padding_mask = batch["padding_mask"].index_select(0, unique_cpu).to(device, non_blocking=True)
    inverse = inverse_cpu.to(device, non_blocking=True)
    if getattr(model, "supports_backbone_inverse", False):
        logits, _ = model(coords, padding_mask, confidence, tokens[:, :-1], inverse)
        return logits
    encoded = model.encoder(coords, padding_mask, confidence)
    expanded = expand_encoder_for_decoder(encoded, inverse)
    unique_rows = unique_cpu.to(device, non_blocking=True)
    with _shared_projection_context(model, unique_rows, inverse):
        logits, _ = model.decoder(tokens[:, :-1], encoder_out=expanded)
    return logits
