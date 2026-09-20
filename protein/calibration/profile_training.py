from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import torch

from .data import load_manifest, preload_coordinate_cache, sft_training_records
from .modeling import configure_adaptation, load_if1
from .optimized_forward import enable_if1_training_optimizations, prefetch_batches
from .training import _make_loader, seed_everything, sequence_nll


MODES = ("ft", "dtft", "lora8", "lora64")


def _event() -> torch.cuda.Event:
    return torch.cuda.Event(enable_timing=True)


def profile_mode(mode: str, records, cache, checkpoint: Path, steps: int, optimized: bool) -> dict:
    seed_everything(7)
    model, alphabet = load_if1(checkpoint)
    counts = configure_adaptation(model, mode)
    if optimized:
        enable_if1_training_optimizations(model)
    device = torch.device("cuda")
    model.to(device).train()
    loader, sampler = _make_loader(
        records, alphabet, batch_size=32, training=True, noise=0.1,
        seed=7, workers=6, coordinate_cache=cache,
    )
    sampler.set_epoch(0)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1e-7 if mode in {"ft", "dtft"} else 3e-5,
        weight_decay=0.0,
        fused=True,
    )
    iterator = iter(prefetch_batches(loader, device))

    for _ in range(3):
        batch = next(iterator)
        loss, _ = sequence_nll(model, batch, alphabet, device)
        (loss / len(batch["records"])).backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    stage_ms = {"forward": 0.0, "backward": 0.0, "optimizer": 0.0}
    data_wait_seconds = 0.0
    started = time.perf_counter()
    for _ in range(steps):
        wait_started = time.perf_counter()
        batch = next(iterator)
        data_wait_seconds += time.perf_counter() - wait_started
        marks = [_event() for _ in range(4)]
        marks[0].record()
        loss, _ = sequence_nll(model, batch, alphabet, device)
        marks[1].record()
        (loss / len(batch["records"])).backward()
        marks[2].record()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        marks[3].record()
        torch.cuda.synchronize()
        for label, left, right in zip(stage_ms, marks, marks[1:]):
            stage_ms[label] += left.elapsed_time(right)
    wall_seconds = time.perf_counter() - started

    profiled_iterator = iter(prefetch_batches(loader, device))
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
    ) as profiler:
        for _ in range(min(5, steps)):
            batch = next(profiled_iterator)
            loss, _ = sequence_nll(model, batch, alphabet, device)
            (loss / len(batch["records"])).backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    operators = []
    for event in sorted(
        profiler.key_averages(), key=lambda item: item.self_cuda_time_total, reverse=True
    )[:30]:
        operators.append(
            {
                "name": event.key,
                "calls": event.count,
                "self_cuda_ms": event.self_cuda_time_total / 1000.0,
                "cuda_total_ms": event.cuda_time_total / 1000.0,
                "self_cpu_ms": event.self_cpu_time_total / 1000.0,
                "cuda_memory_bytes": event.cuda_memory_usage,
            }
        )
    result = {
        "mode": mode,
        "steps": steps,
        "wall_seconds": wall_seconds,
        "seconds_per_step": wall_seconds / steps,
        "sequences_per_second": steps * 32 / wall_seconds,
        "data_wait_seconds_per_step": data_wait_seconds / steps,
        "stage_ms_per_step": {key: value / steps for key, value in stage_ms.items()},
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "trainable_scalars": counts["trainable_scalars"],
        "top_operators": operators,
    }
    del profiler, profiled_iterator, iterator, loader, optimizer, model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile real ESM-IF1 training bottlenecks")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--reference", action="store_true")
    args = parser.parse_args()
    records = sft_training_records(load_manifest(args.manifest))
    cache = preload_coordinate_cache(records)
    payload = {
        "device": torch.cuda.get_device_name(),
        "batch_size": 32,
        "results": [profile_mode(mode, records, cache, args.checkpoint, args.steps, not args.reference) for mode in MODES],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
