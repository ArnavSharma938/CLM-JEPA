from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from chemfm import generate_products_batch
from official_five_view_evaluation import (
    BEAM_SIZE,
    load_endpoint,
    prompts_for_group,
    read_jsonl,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--row", type=int, default=0)
    args = parser.parse_args()

    model, tokenizer = load_endpoint(args.checkpoint)
    group = read_jsonl(args.manifest)[args.row]
    prompt = prompts_for_group(group)[0]
    # One unprofiled warmup catches lazy CUDA initialization without changing
    # any generation configuration or candidate semantics.
    generate_products_batch(
        model, tokenizer, [prompt], max_length=1024,
        num_beams=BEAM_SIZE, num_return_sequences=BEAM_SIZE,
    )
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as profiler:
        candidates = generate_products_batch(
            model, tokenizer, [prompt], max_length=1024,
            num_beams=BEAM_SIZE, num_return_sequences=BEAM_SIZE,
        )[0]
    torch.cuda.synchronize()
    wall_seconds = time.perf_counter() - started
    if args.trace is not None:
        args.trace.parent.mkdir(parents=True, exist_ok=True)
        profiler.export_chrome_trace(str(args.trace))
    events = profiler.key_averages()
    rows = []
    for event in events:
        cuda_total = getattr(event, "device_time_total", None)
        if cuda_total is None:
            cuda_total = getattr(event, "cuda_time_total", 0.0)
        self_cuda = getattr(event, "self_device_time_total", None)
        if self_cuda is None:
            self_cuda = getattr(event, "self_cuda_time_total", 0.0)
        rows.append({
            "key": event.key,
            "calls": event.count,
            "cpu_total_us": event.cpu_time_total,
            "cuda_total_us": cuda_total,
            "self_cpu_us": event.self_cpu_time_total,
            "self_cuda_us": self_cuda,
        })
    rows.sort(key=lambda row: row["cuda_total_us"], reverse=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "checkpoint": str(args.checkpoint.resolve()),
        "manifest": str(args.manifest.resolve()),
        "row": args.row,
        "official_group_index": group["official_group_index"],
        "prompt_characters": len(prompt),
        "ordered_candidates": candidates,
        "wall_seconds": wall_seconds,
        "trace": None if args.trace is None else str(args.trace.resolve()),
        "events_by_cuda_total": rows,
    }, indent=2) + "\n", encoding="utf-8")
    print(events.table(sort_by="cuda_time_total", row_limit=30))


if __name__ == "__main__":
    main()
