"""RITA-M loading and hidden-state capture without HF-convention guesses."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

import torch

from .constants import RITA_MODEL_ID, RITA_REVISION, RITA_TOKENIZER_REVISION


@dataclass
class RITALoaded:
    model: object
    tokenizer: object


def load_rita(
    *, device: str = "cuda", dtype: torch.dtype = torch.float16, freeze: bool = True
) -> RITALoaded:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        RITA_MODEL_ID, revision=RITA_TOKENIZER_REVISION
    )
    model = AutoModelForCausalLM.from_pretrained(
        RITA_MODEL_ID,
        revision=RITA_REVISION,
        trust_remote_code=True,
        torch_dtype=dtype,
    ).to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(not freeze)
    return RITALoaded(model, tokenizer)


def rita_layers(model):
    base = getattr(model, "transformer", None)
    layers = getattr(base, "layers", None)
    if layers is None:
        raise TypeError("expected official RITAModelForCausalLM transformer.layers")
    return layers


@contextmanager
def capture_layer(model, layer_index: int) -> Iterator[list[torch.Tensor]]:
    captured: list[torch.Tensor] = []
    handle = rita_layers(model)[layer_index].register_forward_hook(
        lambda _module, _inputs, output: captured.append(output)
    )
    try:
        yield captured
    finally:
        handle.remove()


def tied_weight_audit(model) -> dict[str, object]:
    embedding = model.get_input_embeddings().weight
    head = model.get_output_embeddings().weight
    return {
        "same_shape": tuple(embedding.shape) == tuple(head.shape),
        "same_storage": embedding.data_ptr() == head.data_ptr(),
        "exact_values": bool(torch.equal(embedding.detach(), head.detach())),
    }
