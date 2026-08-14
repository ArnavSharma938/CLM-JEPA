"""A6000 batched wrapper around the pinned upstream LLM-JEPA evaluator.

Formatting, greedy decoding arguments, response cleanup, and GSM8K scoring are
delegated to or copied verbatim from upstream ``evaluate.py``. The only change
is batching left-padded prompts on one GPU. ``--verify-examples`` compares the
batched strings byte-for-byte with upstream's sequential ``generate_response``
before a full evaluation is allowed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import GenerationConfig
from transformers.utils import logging as transformers_logging

UPSTREAM_DIR = Path(__file__).resolve().parents[2] / "references" / "llm-jepa"
sys.path.insert(0, str(UPSTREAM_DIR))

from evaluate import (  # noqa: E402
    eval as upstream_score,
    format_conversation,
    generate_response,
    get_messages,
    load_model_and_tokenizer,
)


def clean_response(tokenizer, token_ids: torch.Tensor) -> str:
    response = tokenizer.decode(token_ids, skip_special_tokens=True).strip()
    if response.endswith("<|end|>"):
        response = response[:-7].strip()
    return response


def batched_generate(model, tokenizer, prompts: list[str], max_new_tokens: int) -> list[str]:
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
        add_special_tokens=True,
    )
    encoded = {key: value.to(model.device) for key, value in encoded.items()}
    prompt_width = encoded["input_ids"].shape[1]
    with torch.inference_mode():
        outputs = model.generate(
            **encoded,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False,
            max_new_tokens=max_new_tokens,
        )
    return [clean_response(tokenizer, row[prompt_width:]) for row in outputs]


def main() -> None:
    transformers_logging.set_verbosity_error()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--original-model-name", required=True)
    parser.add_argument("--input-file", default="gsm8k_test.jsonl")
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--verify-examples", type=int, default=0)
    args = parser.parse_args()

    if not Path(args.input_file).name.startswith("gsm8k"):
        raise ValueError("The upstream scorer dispatches by a gsm8k* filename.")

    model, tokenizer = load_model_and_tokenizer(
        args.model_name, args.original_model_name, device_map="cuda:0"
    )
    tokenizer.padding_side = "left"
    dataset = load_dataset("json", data_files=args.input_file)["train"]
    if args.max_examples is not None:
        dataset = dataset.select(range(min(args.max_examples, len(dataset))))

    prompts = [
        format_conversation(
            get_messages(args.original_model_name, example["messages"]), tokenizer
        )
        for example in dataset
    ]

    if args.verify_examples:
        count = min(args.verify_examples, len(prompts))
        generation_config = GenerationConfig(max_length=512)
        sequential = [
            generate_response(
                model,
                tokenizer,
                prompt,
                generation_config,
                args.max_new_tokens,
            )
            for prompt in prompts[:count]
        ]
        batched = []
        for start in range(0, count, args.batch_size):
            verification_prompts = prompts[start : min(start + args.batch_size, count)]
            original_count = len(verification_prompts)
            # Exercise the intended physical batch shape while limiting the
            # costly sequential reference calls. Repeated fillers are discarded.
            if verification_prompts and original_count < args.batch_size:
                verification_prompts = verification_prompts + [verification_prompts[-1]] * (
                    args.batch_size - original_count
                )
            batched.extend(
                batched_generate(
                    model, tokenizer, verification_prompts, args.max_new_tokens
                )[:original_count]
            )
        mismatches = [i for i, (a, b) in enumerate(zip(sequential, batched)) if a != b]
        print(json.dumps({"verification_examples": count, "mismatches": mismatches}))
        if mismatches:
            raise RuntimeError("Batched generation did not match upstream sequential output")

    started = time.perf_counter()
    correct = 0
    processed = 0
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for start in range(0, len(prompts), args.batch_size):
            stop = min(start + args.batch_size, len(prompts))
            generated = batched_generate(
                model, tokenizer, prompts[start:stop], args.max_new_tokens
            )
            for index, response in enumerate(generated, start=start):
                messages = dataset[index]["messages"]
                is_correct = bool(
                    upstream_score(
                        response,
                        messages,
                        Path(args.input_file).name,
                        spider_path="",
                    )
                )
                correct += int(is_correct)
                processed += 1
                handle.write(
                    json.dumps(
                        {
                            "index": index,
                            "generated": response,
                            "correct": is_correct,
                            "ground_truth": messages[2]["content"],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                handle.flush()
            print(
                json.dumps(
                    {
                        "processed": processed,
                        "correct": correct,
                        "accuracy": correct / processed,
                    }
                ),
                flush=True,
            )

    elapsed = time.perf_counter() - started
    print(
        json.dumps(
            {
                "examples": processed,
                "correct": correct,
                "accuracy": correct / processed,
                "generation_seconds": elapsed,
                "examples_per_second": processed / elapsed,
            }
        )
    )


if __name__ == "__main__":
    main()
