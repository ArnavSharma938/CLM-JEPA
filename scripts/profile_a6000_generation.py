"""Profile one exact ChemFM beam-10 generation call on an A6000.

The script keeps the official generation function and output semantics intact.
CUDA events around the model's outer forward quantify device work without a
per-token synchronization; cProfile measures host-side beam-search overhead.
"""

from __future__ import annotations

import argparse
import cProfile
import io
import json
import pstats
import time
import types
from pathlib import Path

import torch

from chemfm import generate_products_batch
from eval_uspto_mit_five_view_a6000 import load_endpoint, prompts_for_group, read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--panel-index", type=int, default=0)
    parser.add_argument("--view-index", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    groups = read_jsonl(args.manifest)
    group = next(row for row in groups if row["panel_index"] == args.panel_index)
    prompt = prompts_for_group(group)[args.view_index]
    model, tokenizer = load_endpoint(args.checkpoint)

    # Populate every lazy per-module CUDA graph before measuring.
    generate_products_batch(
        model, tokenizer, [prompt], max_length=1024,
        num_beams=10, num_return_sequences=10,
    )
    torch.cuda.synchronize()

    events: list[tuple[torch.cuda.Event, torch.cuda.Event, tuple[int, ...]]] = []
    # PEFT's generate delegates directly to the wrapped LlamaForCausalLM, so
    # time that forward rather than the unused outer PEFT forward method.
    timed_model = model.get_base_model()
    original_forward = timed_model.forward

    def timed_forward(layer, *forward_args, **forward_kwargs):
        input_ids = forward_kwargs.get("input_ids")
        if input_ids is None and forward_args:
            input_ids = forward_args[0]
        shape = tuple(input_ids.shape) if input_ids is not None else ()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = original_forward(*forward_args, **forward_kwargs)
        end.record()
        events.append((start, end, shape))
        return result

    timed_model.forward = types.MethodType(timed_forward, timed_model)
    profiler = cProfile.Profile()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    profiler.enable()
    candidates = generate_products_batch(
        model, tokenizer, [prompt], max_length=1024,
        num_beams=10, num_return_sequences=10,
    )[0]
    profiler.disable()
    torch.cuda.synchronize()
    wall_seconds = time.perf_counter() - started

    forward_rows = [
        {"shape": list(shape), "cuda_ms": start.elapsed_time(end)}
        for start, end, shape in events
    ]
    stream = io.StringIO()
    pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats("cumulative").print_stats(50)
    payload = {
        "panel_index": args.panel_index,
        "view_index": args.view_index,
        "prompt_characters": len(prompt),
        "wall_seconds": wall_seconds,
        "model_forward_calls": len(forward_rows),
        "model_forward_cuda_seconds": sum(row["cuda_ms"] for row in forward_rows) / 1000.0,
        "non_forward_wall_upper_bound_seconds": max(
            0.0,
            wall_seconds - sum(row["cuda_ms"] for row in forward_rows) / 1000.0,
        ),
        "prefill": next(
            (row for row in forward_rows if row["shape"] and row["shape"][-1] != 1),
            None,
        ),
        "decode_cuda_ms": [
            row["cuda_ms"] for row in forward_rows
            if row["shape"] and row["shape"][-1] == 1
        ],
        "forward_rows": forward_rows,
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()),
        "candidates": candidates,
        "cprofile_top50": stream.getvalue(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "wall_seconds": payload["wall_seconds"],
        "model_forward_calls": payload["model_forward_calls"],
        "model_forward_cuda_seconds": payload["model_forward_cuda_seconds"],
        "non_forward_wall_upper_bound_seconds": payload["non_forward_wall_upper_bound_seconds"],
        "mean_decode_cuda_ms": (
            sum(payload["decode_cuda_ms"]) / len(payload["decode_cuda_ms"])
            if payload["decode_cuda_ms"] else None
        ),
        "output": str(args.output),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
