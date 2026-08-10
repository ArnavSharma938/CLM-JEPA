from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from chemfm import (
    MODEL_DIR,
    TOKENIZER_DIR,
    ReactionCollator,
    load_lora_model,
    load_reaction_tokenizer,
)
from jepa import add_predictor_tokens
from train import beam_evaluate, load_adapter_checkpoint, read_rows


def timed_evaluation(
    model, tokenizer, collator, rows, batch_size: int, beam_size: int
) -> dict:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    metrics, predictions = beam_evaluate(
        model,
        tokenizer,
        collator,
        rows,
        "forward",
        windows=beam_size,
        generation_batch_size=batch_size,
    )
    torch.cuda.synchronize()
    return {
        "generation_batch_size": batch_size,
        "seconds": time.perf_counter() - started,
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()),
        "metrics": metrics,
        "predictions": predictions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exact-parity benchmark for optimized ChemFM beam validation"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--groups", type=int, default=4)
    parser.add_argument("--optimized-batch-size", type=int, choices=(2, 4), default=2)
    parser.add_argument(
        "--reference-result", type=Path,
        help="existing sequential train.py result; skips rerunning the slow baseline",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.groups < 1:
        raise ValueError("groups must be positive")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(533)
    tokenizer = load_reaction_tokenizer(TOKENIZER_DIR)
    chemfm_vocab_size = len(tokenizer)
    add_predictor_tokens(tokenizer)
    model = load_lora_model(
        MODEL_DIR, tokenizer, chemfm_vocab_size=chemfm_vocab_size
    ).cuda().eval()
    load_adapter_checkpoint(model, args.checkpoint.resolve())
    collator = ReactionCollator(tokenizer, task="forward")
    rows = read_rows("uspto_mit_synthesis", path=args.manifest.resolve())
    rows = rows[: args.groups * 5]
    if len(rows) != args.groups * 5:
        raise ValueError("benchmark manifest does not contain the requested five-view groups")

    optimized = timed_evaluation(
        model, tokenizer, collator, rows, args.optimized_batch_size, 10
    )
    if args.reference_result is None:
        sequential = timed_evaluation(model, tokenizer, collator, rows, 1, 10)
    else:
        reference = json.loads(args.reference_result.read_text(encoding="utf-8"))
        sequential = {
            "generation_batch_size": 1,
            "seconds": None,
            "peak_cuda_bytes": None,
            "metrics": reference["validation_metrics"],
            "predictions": reference["predictions"][: args.groups],
            "source": str(args.reference_result.resolve()),
        }
    exact_predictions = optimized["predictions"] == sequential["predictions"]
    exact_metrics = optimized["metrics"] == sequential["metrics"]
    cutoff_outcome_parity = all(
        any(optimized_row["exact"][:cutoff]) == any(sequential_row["exact"][:cutoff])
        for optimized_row, sequential_row in zip(
            optimized["predictions"], sequential["predictions"]
        )
        for cutoff in (1, 3, 5, 10)
    )
    top5_candidate_prefix_parity = all(
        optimized_row["candidates"][:5] == sequential_row["candidates"][:5]
        for optimized_row, sequential_row in zip(
            optimized["predictions"], sequential["predictions"]
        )
    )
    output = {
        "checkpoint": str(args.checkpoint.resolve()),
        "manifest": str(args.manifest.resolve()),
        "groups": args.groups,
        "beam_width": 10,
        "optimized": optimized,
        "sequential": sequential,
        "exact_prediction_parity": exact_predictions,
        "exact_metric_parity": exact_metrics,
        "cutoff_outcome_parity": cutoff_outcome_parity,
        "top5_candidate_prefix_parity": top5_candidate_prefix_parity,
        "speedup": (
            None if sequential["seconds"] is None
            else sequential["seconds"] / optimized["seconds"]
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        key: output[key]
        for key in (
            "groups", "exact_prediction_parity", "exact_metric_parity",
            "cutoff_outcome_parity", "top5_candidate_prefix_parity", "speedup"
        )
    }), flush=True)
    if not exact_metrics or not cutoff_outcome_parity:
        raise RuntimeError("optimized beam evaluation changed a reported metric or cutoff outcome")


if __name__ == "__main__":
    main()
