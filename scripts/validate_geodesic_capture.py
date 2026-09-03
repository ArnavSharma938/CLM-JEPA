#!/usr/bin/env python
"""Validate selected-state hooks against Transformers hidden-state outputs."""

import json

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
    print(json.dumps(result, indent=2))
    if not result["all_exact"]:
        raise SystemExit("selected state capture is not exact")


if __name__ == "__main__":
    main()
