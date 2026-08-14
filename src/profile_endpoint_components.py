from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
import types
from pathlib import Path

import torch

import official_five_view_evaluation as endpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--start-row", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile-forward", action="store_true")
    args = parser.parse_args()

    groups = endpoint.read_jsonl(args.manifest)
    measured = groups[args.start_row:args.start_row + args.rows]
    model, tokenizer = endpoint.load_endpoint(args.checkpoint)
    endpoint._evaluate_assigned_groups(model, tokenizer, [groups[0]], 1, "left-pad")
    torch.cuda.synchronize()

    generation_calls: list[dict] = []
    original_generate = model.generate

    def measured_generate(self, *positional, **kwargs):
        torch.cuda.synchronize()
        started = time.perf_counter()
        result = original_generate(*positional, **kwargs)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        prompt_width = int(kwargs["input_ids"].shape[1])
        generated = result[:, prompt_width:]
        eos = tokenizer.eos_token_id
        lengths = []
        for row in generated:
            positions = (row == eos).nonzero(as_tuple=False)
            lengths.append(int(positions[0]) + 1 if len(positions) else int(row.numel()))
        generation_calls.append({
            "seconds": elapsed,
            "prompt_tokens": prompt_width,
            "output_width": int(generated.shape[1]),
            "beam_lengths": lengths,
        })
        return result

    model.generate = types.MethodType(measured_generate, model)
    forward_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
    forward_cpu_seconds = 0.0
    if args.profile_forward:
        forward_target = model.get_base_model()
        original_forward = forward_target.forward

        def measured_forward(self, *positional, **kwargs):
            nonlocal forward_cpu_seconds
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            forward_started = time.perf_counter()
            value = original_forward(*positional, **kwargs)
            forward_cpu_seconds += time.perf_counter() - forward_started
            end_event.record()
            forward_events.append((start_event, end_event))
            return value

        forward_target.forward = types.MethodType(measured_forward, forward_target)
    canonical_seconds = 0.0
    rank_seconds = 0.0
    original_canonicalize = endpoint.canonicalize_official
    original_rank = endpoint.official_rank

    def measured_canonicalize(*values, **kwargs):
        nonlocal canonical_seconds
        started = time.perf_counter()
        result = original_canonicalize(*values, **kwargs)
        canonical_seconds += time.perf_counter() - started
        return result

    def measured_rank(*values, **kwargs):
        nonlocal rank_seconds
        started = time.perf_counter()
        result = original_rank(*values, **kwargs)
        rank_seconds += time.perf_counter() - started
        return result

    endpoint.canonicalize_official = measured_canonicalize
    endpoint.official_rank = measured_rank
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    records = endpoint._evaluate_assigned_groups(model, tokenizer, measured, 1, "left-pad")
    torch.cuda.synchronize()
    total_seconds = time.perf_counter() - started

    serialize_started = time.perf_counter()
    for _ in range(20):
        json.dumps(records, sort_keys=True)
    serialization_seconds = (time.perf_counter() - serialize_started) / 20.0
    generation_seconds = sum(call["seconds"] for call in generation_calls)
    forward_gpu_seconds = sum(
        start.elapsed_time(end) for start, end in forward_events
    ) / 1000.0
    output_lengths = [length for call in generation_calls for length in call["beam_lengths"]]
    result = {
        "reactions": len(measured),
        "views": len(generation_calls),
        "total_seconds": total_seconds,
        "seconds_per_reaction": total_seconds / len(measured),
        "generation_seconds": generation_seconds,
        "generation_fraction": generation_seconds / total_seconds,
        "model_forward_calls": len(forward_events),
        "model_forward_cpu_dispatch_seconds": forward_cpu_seconds,
        "model_forward_gpu_seconds": forward_gpu_seconds,
        "model_forward_gpu_fraction_of_generation": (
            forward_gpu_seconds / generation_seconds if generation_seconds else 0.0
        ),
        "canonicalization_seconds": canonical_seconds,
        "ranking_seconds": rank_seconds,
        "other_seconds": total_seconds - generation_seconds - canonical_seconds - rank_seconds,
        "serialization_seconds_per_write": serialization_seconds,
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()),
        "output_length": {
            "mean": statistics.fmean(output_lengths),
            "median": statistics.median(output_lengths),
            "max": max(output_lengths),
        },
        "generation_calls": generation_calls,
        "ordered_candidate_sha256": hashlib.sha256(
            json.dumps(
                [record["raw_candidates_by_view"] for record in records],
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
