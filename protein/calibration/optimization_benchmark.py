from __future__ import annotations

import argparse
import json
import platform
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .optimized_forward import OptimizedIF1Forward, model_logits


class _Encoder(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.input = nn.Linear(9, width)
        self.layers = nn.ModuleList(nn.Linear(width, width) for _ in range(3))

    def forward(self, coords, padding_mask, confidence):
        del confidence
        value = self.input(coords.flatten(2))
        for layer in self.layers:
            value = value + F.gelu(layer(value))
        return {
            "encoder_out": [value.transpose(0, 1)],
            "encoder_padding_mask": [padding_mask],
        }


class _Decoder(nn.Module):
    def __init__(self, width: int, vocabulary: int) -> None:
        super().__init__()
        self.token = nn.Embedding(vocabulary, width)
        self.hidden = nn.Linear(width, width)
        self.output = nn.Linear(width, vocabulary)

    def forward(self, previous, encoder_out):
        context = encoder_out["encoder_out"][0].transpose(0, 1)
        value = self.token(previous) + context[:, : previous.shape[1]]
        value = F.gelu(self.hidden(value))
        return self.output(value).transpose(1, 2), {}


class _TinyIF1(nn.Module):
    def __init__(self, width: int = 64, vocabulary: int = 33) -> None:
        super().__init__()
        self.encoder = _Encoder(width)
        self.decoder = _Decoder(width, vocabulary)

    def forward(self, coords, padding_mask, confidence, previous):
        encoded = self.encoder(coords, padding_mask, confidence)
        return self.decoder(previous, encoded)


def _batch(batch_size: int, unique_backbones: int, length: int, vocabulary: int = 33) -> dict[str, Any]:
    if not 1 <= unique_backbones <= batch_size:
        raise ValueError("unique_backbones must be in [1, batch_size]")
    generator = torch.Generator().manual_seed(1729 + batch_size + unique_backbones + length)
    base = torch.randn(unique_backbones, length, 3, 3, generator=generator)
    inverse = torch.arange(batch_size, dtype=torch.long).remainder(unique_backbones)
    unique = torch.arange(unique_backbones, dtype=torch.long)
    coords = base.index_select(0, inverse)
    records = [
        type("Record", (), {"backbone_path": Path(f"backbone_{int(i)}.npy"), "chain_id": "A"})()
        for i in inverse
    ]
    return {
        "coords": coords,
        "confidence": torch.ones(batch_size, length),
        "padding_mask": torch.zeros(batch_size, length, dtype=torch.bool),
        "tokens": torch.randint(0, vocabulary, (batch_size, length + 1), generator=generator),
        "records": records,
        "backbone_unique": unique,
        "backbone_inverse": inverse,
    }


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _parity(batch: dict[str, Any], device: torch.device, *, training: bool) -> dict[str, float]:
    reference = _TinyIF1().to(device)
    optimized_base = deepcopy(reference)
    optimized = OptimizedIF1Forward(optimized_base)
    reference.train(training)
    optimized.train(training)
    target = batch["tokens"][:, 1:].to(device)
    with torch.set_grad_enabled(training):
        left = model_logits(reference, batch, device, reuse_backbones=False, tokens=batch["tokens"].to(device))
        right = model_logits(
            optimized, batch, device, reuse_backbones=not training,
            tokens=batch["tokens"].to(device),
        )
        left_loss = F.cross_entropy(left.float(), target, reduction="sum")
        right_loss = F.cross_entropy(right.float(), target, reduction="sum")
        output_error = float((left - right).abs().max())
        loss_error = float((left_loss - right_loss).abs())
        gradient_error = 0.0
        if training:
            left_loss.backward()
            right_loss.backward()
            for left_parameter, right_parameter in zip(reference.parameters(), optimized_base.parameters()):
                gradient_error = max(
                    gradient_error,
                    float((left_parameter.grad - right_parameter.grad).abs().max()),
                )
    torch.testing.assert_close(left, right, rtol=1e-6, atol=2e-6)
    torch.testing.assert_close(left_loss, right_loss, rtol=1e-6, atol=2e-5)
    if training:
        for left_parameter, right_parameter in zip(reference.parameters(), optimized_base.parameters()):
            torch.testing.assert_close(left_parameter.grad, right_parameter.grad, rtol=2e-5, atol=2e-4)
    return {
        "max_abs_output_error": output_error,
        "abs_loss_error": loss_error,
        "relative_loss_error": loss_error / max(
            abs(float(left_loss.detach())), torch.finfo(torch.float32).tiny
        ),
        "max_abs_gradient_error": gradient_error,
    }


def _measure(
    model: nn.Module,
    batch: dict[str, Any],
    device: torch.device,
    *,
    reuse_backbones: bool,
    training: bool,
    iterations: int,
    supply_tokens: bool,
) -> dict[str, float]:
    model.train(training)
    tokens = batch["tokens"].to(device)
    target = tokens[:, 1:]

    def step() -> None:
        model.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model_logits(
                model, batch, device, reuse_backbones=reuse_backbones,
                tokens=tokens if supply_tokens else None,
            )
            if training:
                F.cross_entropy(logits.float(), target, reduction="mean").backward()

    for _ in range(2):
        step()
    _synchronize(device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        baseline = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
    else:
        baseline = 0
    started = time.perf_counter()
    for _ in range(iterations):
        step()
    _synchronize(device)
    elapsed = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated(device) - baseline if device.type == "cuda" else 0
    return {
        "seconds": elapsed,
        "sequences_per_second": iterations * len(batch["records"]) / elapsed,
        "peak_incremental_bytes": int(max(0, peak)),
    }


def run_benchmark(device: torch.device) -> dict[str, Any]:
    torch.manual_seed(1234)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    cases = [
        ("evaluation_maximal_reuse", 1024, 1, 72, False, 50),
        ("evaluation_mixed_odd_batch", 513, 128, 72, False, 50),
        ("evaluation_no_reuse", 257, 257, 72, False, 50),
        ("training_maximal_backbone_multiplicity", 256, 1, 72, True, 20),
        ("training_mixed_odd_batch", 129, 65, 72, True, 20),
    ]
    results: dict[str, Any] = {}
    for name, size, unique_count, length, training, iterations in cases:
        batch = _batch(size, unique_count, length)
        parity = _parity(batch, device, training=training)
        base = _TinyIF1().to(device)
        reference = _measure(
            base, batch, device, reuse_backbones=False, training=training,
            iterations=iterations, supply_tokens=False,
        )
        optimized = _measure(
            base, batch, device,
            reuse_backbones=not training, training=training, iterations=iterations,
            supply_tokens=True,
        )
        results[name] = {
            "batch_size": size,
            "unique_backbones": unique_count,
            "sequence_length": length,
            "training": training,
            "encoder_reuse_enabled": not training,
            "parity": parity,
            "reference": reference,
            "optimized": optimized,
            "throughput_speedup": optimized["sequences_per_second"] / reference["sequences_per_second"],
            "peak_memory_ratio": (
                optimized["peak_incremental_bytes"] / reference["peak_incremental_bytes"]
                if reference["peak_incremental_bytes"] else None
            ),
        }
    return {
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
        "torch_version": torch.__version__,
        "cases": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_benchmark(torch.device(args.device))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
