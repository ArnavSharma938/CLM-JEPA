from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import torch

from .config import HPO_LRS
from .data import load_manifest, preload_coordinate_cache, sft_training_records
from .modeling import configure_adaptation, load_if1
from .optimized_forward import prefetch_batches
from .training import _make_loader, seed_everything, sequence_nll


MODES = ("ft", "dtft", "lora8", "lora64")


def _profile(mode, records, cache, checkpoint, *, force_sync: bool, steps: int) -> dict:
    seed_everything(7)
    model, alphabet = load_if1(checkpoint)
    counts = configure_adaptation(model, mode)
    device = torch.device("cuda")
    model.to(device).train()
    loader, sampler = _make_loader(
        records, alphabet, batch_size=32, training=True, noise=0.1, seed=7,
        workers=6, coordinate_cache=cache,
    )
    sampler.set_epoch(0)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=HPO_LRS[mode][0], weight_decay=0.0, fused=True,
    )
    iterator = iter(prefetch_batches(loader, device))
    for _ in range(3):
        batch = next(iterator)
        loss, _ = sequence_nll(model, batch, alphabet, device)
        if force_sync:
            int(batch["tokens"][:, 1:].ne(alphabet.padding_idx).sum())
        (loss / len(batch["records"])).backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for _ in range(steps):
        batch = next(iterator)
        loss, _ = sequence_nll(model, batch, alphabet, device)
        if force_sync:
            int(batch["tokens"][:, 1:].ne(alphabet.padding_idx).sum())
        (loss / len(batch["records"])).backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    result = {
        "mode": mode,
        "forced_legacy_token_sync": force_sync,
        "steps": steps,
        "seconds": seconds,
        "seconds_per_step": seconds / steps,
        "sequences_per_second": steps * 32 / seconds,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "trainable_scalars": counts["trainable_scalars"],
    }
    del iterator, loader, optimizer, model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=12)
    args = parser.parse_args()
    records = sft_training_records(load_manifest(args.manifest))
    cache = preload_coordinate_cache(records)
    results = []
    for mode in MODES:
        legacy = _profile(mode, records, cache, args.checkpoint, force_sync=True, steps=args.steps)
        optimized = _profile(mode, records, cache, args.checkpoint, force_sync=False, steps=args.steps)
        optimized["speedup_vs_legacy_sync"] = legacy["seconds"] / optimized["seconds"]
        results.append({"legacy": legacy, "optimized": optimized})
    payload = {"device": torch.cuda.get_device_name(), "batch_size": 32, "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
