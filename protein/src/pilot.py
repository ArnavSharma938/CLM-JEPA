"""Locked Native/faithful-NextLat LoRA pilot primitives.

The optimized path uses one RITA forward per microbatch.  Its direct autograd
is tested against the upstream two-stage detached-leaf VJP construction.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.nn import functional as F

from .nextlat import FaithfulNextLatPredictor, faithful_losses
from .tokenization import collate_encoded, encode_canonical, next_residue_transition_mask

ATTENTION_TARGETS = ("query", "key", "value", "proj")
EXPECTED_LAYERS = 24


def load_protocol(path: Path) -> dict:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol["status"] != "locked_before_training":
        raise ValueError("causal-pilot protocol is not locked")
    if protocol["prohibited"] != [
        "STP", "coefficient_sweeps", "rank_sweeps",
        "posthoc_rescue_variants", "checkpoint_selection",
    ]:
        raise ValueError("prohibited-analysis contract changed")
    return protocol


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_state_sha256(named: Iterable[tuple[str, torch.Tensor]]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(named):
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(value.detach().float().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def attach_rank32_attention_lora(model, protocol: dict):
    """Attach LoRA only after verifying RITA's pinned module names."""
    expected = [
        f"transformer.layers.{layer}.self_attention.{suffix}"
        for layer in range(EXPECTED_LAYERS)
        for suffix in ATTENTION_TARGETS
    ]
    discovered = [
        name for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear) and name.endswith(ATTENTION_TARGETS)
    ]
    if sorted(discovered) != sorted(expected):
        raise AssertionError({"expected": expected, "discovered": discovered})
    spec = protocol["lora"]
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(spec["rank"]),
        lora_alpha=int(spec["alpha"]),
        lora_dropout=float(spec["dropout"]),
        bias=spec["bias"],
        target_modules=list(spec["target_modules"]),
        init_lora_weights=True,
    )
    wrapped = get_peft_model(model, config)
    trainable = [(name, value) for name, value in wrapped.named_parameters() if value.requires_grad]
    if not trainable or any("lora_" not in name for name, _ in trainable):
        raise AssertionError("only LoRA tensors may be trainable after attachment")
    matched = sorted({name.split(".lora_", 1)[0] for name, _ in trainable})
    if len(matched) != len(expected) or any(".self_attention." not in name for name in matched):
        raise AssertionError(f"unexpected LoRA target set: {matched}")
    return wrapped


def lora_named_parameters(model) -> list[tuple[str, torch.nn.Parameter]]:
    result = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    if not result or any("lora_" not in name for name, _ in result):
        raise AssertionError("trainable model parameter outside LoRA")
    return result


def labels_with_padding_ignored(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    labels = batch["input_ids"].clone()
    labels.masked_fill_(~batch["attention_mask"].bool(), -100)
    return labels


@dataclass
class LossBundle:
    total: torch.Tensor
    ntp: torch.Tensor
    latent: torch.Tensor | None
    kl: torch.Tensor | None
    predicted: torch.Tensor | None
    valid_transitions: int


def causal_pilot_loss(model, predictor, batch: dict[str, torch.Tensor], arm: str) -> LossBundle:
    """One-forward optimized loss with the audited RITA shift and masks."""
    if arm not in {"native", "nextlat"}:
        raise ValueError(f"unknown arm {arm}")
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=labels_with_padding_ignored(batch),
        use_cache=False,
    )
    ntp = outputs.loss.float()
    if arm == "native":
        return LossBundle(ntp, ntp, None, None, None, 0)
    if predictor is None:
        raise ValueError("NextLat arm requires a fresh predictor")
    states = outputs.hidden_states
    if not isinstance(states, torch.Tensor) or states.ndim != 3:
        raise TypeError("pinned RITA must return final hidden states as BxTxD")
    mask = next_residue_transition_mask(batch["input_ids"], batch["attention_mask"])
    next_embedding = model.get_input_embeddings()(batch["input_ids"][:, 1:])
    latent, kl, predicted = faithful_losses(
        predictor,
        states[:, :-1],
        states[:, 1:],
        next_embedding,
        mask,
        model.get_output_embeddings().weight,
    )
    return LossBundle(ntp + latent + kl, ntp, latent, kl, predicted, int(mask.sum()))


def reference_two_stage_backward(model, predictor, batch, loss_divisor: int = 1):
    """Literal upstream detached-leaf/manual-VJP backward for parity tests.

    RITA-specific masking excludes PAD/EOS transitions. The NTP head is frozen
    in the LoRA experiment, so its direct parameter gradient is intentionally
    irrelevant; the hidden-state gradient is propagated exactly as upstream.
    """
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        use_cache=False,
    )
    hidden = outputs.hidden_states
    embedding = model.get_input_embeddings()(batch["input_ids"])
    hidden_leaf = hidden.detach().requires_grad_(True)
    embedding_leaf = embedding.detach().requires_grad_(True)
    labels = labels_with_padding_ignored(batch)
    logits = F.linear(hidden_leaf[:, :-1], model.get_output_embeddings().weight)
    ntp = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        labels[:, 1:].reshape(-1),
        ignore_index=-100,
    )
    mask = next_residue_transition_mask(batch["input_ids"], batch["attention_mask"])
    latent, kl, predicted = faithful_losses(
        predictor,
        hidden_leaf[:, :-1],
        hidden_leaf[:, 1:],
        embedding_leaf[:, 1:],
        mask,
        model.get_output_embeddings().weight,
    )
    total = (ntp + latent + kl) / loss_divisor
    total.backward()
    graph_tensors = [hidden]
    graph_gradients = [hidden_leaf.grad]
    if embedding.requires_grad:
        graph_tensors.append(embedding)
        graph_gradients.append(embedding_leaf.grad)
    torch.autograd.backward(graph_tensors, graph_gradients)
    return LossBundle(total, ntp / loss_divisor, latent / loss_divisor,
                      kl / loss_divisor, predicted, int(mask.sum()))


def directional_examples(primary_rows: Sequence[dict]) -> Iterator[dict]:
    """Yield each primary and exact pre-tokenization reversal in locked order."""
    for row in primary_rows:
        orders = row.get("orientation_order", ["forward", "reverse"])
        if sorted(orders) != ["forward", "reverse"]:
            raise ValueError("each protein must contribute exactly two orientations")
        for orientation in orders:
            sequence = row["sequence"] if orientation == "forward" else row["sequence"][::-1]
            yield {
                "id": row["id"],
                "cluster_id": row["cluster_id"],
                "orientation": orientation,
                "sequence": sequence,
                "primary_length": len(row["sequence"]),
            }


def make_batches(rows: Sequence[dict], tokenizer, batch_size: int):
    examples = list(directional_examples(rows))
    for start in range(0, len(examples), batch_size):
        selected = examples[start:start + batch_size]
        encoded = [encode_canonical(tokenizer, row["sequence"]) for row in selected]
        yield selected, collate_encoded(encoded)


def cosine_multiplier(step: int, total_steps: int, warmup_fraction: float,
                      minimum_ratio: float) -> float:
    warmup = max(1, round(total_steps * warmup_fraction))
    if step < warmup:
        return float(step + 1) / warmup
    progress = (step - warmup) / max(total_steps - warmup - 1, 1)
    return minimum_ratio + (1.0 - minimum_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


def create_optimizer(parameters, protocol: dict, *, cuda: bool):
    spec = protocol["optimization"]
    kwargs = dict(
        lr=float(spec["learning_rate"]),
        betas=tuple(spec["betas"]),
        eps=float(spec["epsilon"]),
        weight_decay=float(spec["weight_decay"]),
    )
    if cuda and spec["fused_adamw_on_cuda"]:
        kwargs["fused"] = True
    return torch.optim.AdamW(parameters, **kwargs)


def deterministic_orientation_order(identifier: str, seed: int) -> list[str]:
    value = hashlib.sha256(f"{seed}\0{identifier}\0orientation".encode()).digest()[0]
    return ["forward", "reverse"] if value % 2 == 0 else ["reverse", "forward"]


def length_bucket_order(rows: Sequence[dict], seed: int, bucket_size: int = 256) -> list[dict]:
    """Deterministic shuffled buckets, then length-sort within each bucket."""
    ordered = [dict(row) for row in rows]
    random.Random(seed).shuffle(ordered)
    result = []
    for start in range(0, len(ordered), bucket_size):
        bucket = sorted(ordered[start:start + bucket_size],
                        key=lambda row: (len(row["sequence"]), row["id"]))
        result.extend(bucket)
    for row in result:
        row["orientation_order"] = deterministic_orientation_order(row["id"], seed)
    return result


def checkpoint_crossings(total_directional_residues: int) -> dict[float, int]:
    return {fraction: math.ceil(total_directional_residues * fraction)
            for fraction in (0.25, 0.50, 1.00)}
