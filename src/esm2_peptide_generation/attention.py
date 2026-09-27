from __future__ import annotations

from types import MethodType

import torch
import torch.nn.functional as F


def _esm_sdpa_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    head_mask: torch.Tensor | None = None,
    encoder_hidden_states: torch.Tensor | None = None,
    encoder_attention_mask: torch.Tensor | None = None,
    past_key_value=None,
    output_attentions: bool = False,
):
    """PyTorch SDPA for standard ESM self-attention, preserving ESM's query scaling."""
    if (
        head_mask is not None
        or encoder_hidden_states is not None
        or encoder_attention_mask is not None
        or past_key_value is not None
        or output_attentions
        or self.position_embedding_type not in {"absolute", "rotary"}
    ):
        raise NotImplementedError("the Propedia SDPA path only supports encoder self-attention without attention outputs")

    query = self.transpose_for_scores(self.query(hidden_states))
    key = self.transpose_for_scores(self.key(hidden_states))
    value = self.transpose_for_scores(self.value(hidden_states))
    # Hugging Face ESM scales Q before QK^T; scale=1 keeps that exact convention.
    query = query * (self.attention_head_size ** -0.5)
    if self.position_embedding_type == "rotary":
        query, key = self.rotary_embeddings(query, key)
    context = F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attention_mask,
        dropout_p=self.dropout.p if self.training else 0.0,
        is_causal=False,
        scale=1.0,
    )
    context = context.permute(0, 2, 1, 3).contiguous()
    return (context.view(context.size()[:-2] + (self.all_head_size,)),)


def patch_esm_sdpa(model, *, expected_layers: int | None = 33) -> int:
    """Patch only ESM's self-attention forward; parameter names and matrices stay unchanged."""
    from transformers.models.esm.modeling_esm import EsmSelfAttention

    patched = 0
    for module in model.modules():
        if isinstance(module, EsmSelfAttention):
            module.forward = MethodType(_esm_sdpa_forward, module)
            patched += 1
    if patched == 0 or (expected_layers is not None and patched != expected_layers):
        raise RuntimeError(f"expected {expected_layers} ESM self-attention blocks, patched {patched}")
    return patched
