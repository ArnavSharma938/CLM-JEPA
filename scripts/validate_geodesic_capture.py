#!/usr/bin/env python
"""Validate selected-state hooks against Transformers hidden-state outputs."""

import json
import time

import torch

from run_geodesic_audit import (
    MODEL_DIR,
    TOKENIZER_DIR,
    SelectedStateCapture,
    add_predictor_tokens,
    load_lora_model,
    load_reaction_tokenizer,
)


def main():
    tokenizer = load_reaction_tokenizer(TOKENIZER_DIR)
    chemfm_vocab_size = len(tokenizer)
    add_predictor_tokens(tokenizer)
    model = load_lora_model(
        MODEL_DIR, tokenizer, chemfm_vocab_size=chemfm_vocab_size,
        attention_dropout=0.0, attn_implementation="sdpa",
        lora_rank=8, lora_alpha=8,
    ).to("cuda").eval()
    capture = SelectedStateCapture(model)
    ids = torch.tensor([tokenizer(
        "<rstart>CCO<eos><prostart>CC=O<eos>", add_special_tokens=False,
    )["input_ids"]], device="cuda")
    capture.clear()
    with torch.inference_mode():
        output = model(
            input_ids=ids, output_hidden_states=True, use_cache=False,
            return_dict=True,
        )
    mapping = {
        "embedding": 0, "layer_6": 6, "layer_16": 16,
        "layer_21": 21, "final_post_norm": 22,
    }
    differences = {
        name: float((capture.values[name] - output.hidden_states[depth]).abs().max())
        for name, depth in mapping.items()
    }
    result = {
        "differences": differences,
        "all_exact": all(value == 0 for value in differences.values()),
        "hook_peak_cuda_bytes": int(torch.cuda.max_memory_allocated()),
    }
    del output
    capture.close()
    # A realistic extraction shape quantifies the retained-state memory and
    # end-to-end forward-time difference.  The input is deterministic and the
    # benchmark does not depend on chemistry labels.
    benchmark_ids = ids.repeat(8, (256 + ids.shape[1] - 1) // ids.shape[1])[:, :256]
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        reference = model(
            input_ids=benchmark_ids, output_hidden_states=True,
            use_cache=False, return_dict=True,
        )
    torch.cuda.synchronize()
    result["reference_seconds"] = time.perf_counter() - started
    result["reference_peak_cuda_bytes"] = int(torch.cuda.max_memory_allocated())
    del reference
    torch.cuda.empty_cache()
    optimized_capture = SelectedStateCapture(model)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        optimized = model(
            input_ids=benchmark_ids, output_hidden_states=False,
            use_cache=False, return_dict=True,
        )
    torch.cuda.synchronize()
    result["optimized_seconds"] = time.perf_counter() - started
    result["optimized_peak_cuda_bytes"] = int(torch.cuda.max_memory_allocated())
    result["wall_speedup"] = result["reference_seconds"] / result["optimized_seconds"]
    result["peak_cuda_reduction_bytes"] = result["reference_peak_cuda_bytes"] - result["optimized_peak_cuda_bytes"]
    optimized_capture.close()
    print(json.dumps(result, indent=2))
    if not result["all_exact"]:
        raise SystemExit("selected state capture is not exact")


if __name__ == "__main__":
    main()
