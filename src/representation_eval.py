"""Standard frozen ChemFM endpoint-representation evaluation entrypoint.

It extracts source/target states from a fixed checkpoint and manifest, then
computes the maintained geometry, retrieval, PCA, and relationship metrics.
Training and generation are outside this module.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import set_seed

from chemfm import MODEL_DIR, TOKENIZER_DIR, ReactionCollator, load_lora_model, load_reaction_tokenizer
from jepa import CLMJEPA, add_predictor_tokens
from train import (
    TASKS, load_adapter_checkpoint, read_rows, representation_diagnostics,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute corrected Section 11 diagnostics")
    parser.add_argument("--dataset", choices=sorted(TASKS), required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--run-json", type=Path, action="append", default=[])
    parser.add_argument(
        "--legacy-checkpoint", nargs=2, action="append", default=[],
        metavar=("LABEL", "PATH"),
        help="explicit labeled checkpoint using the historical predictor vocabulary",
    )
    parser.add_argument("--include-pretrained", action="store_true")
    parser.add_argument("--seed", type=int, default=533)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--diagnostic-limit", type=int, default=32)
    parser.add_argument("--diagnostic-batch-size", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest_path = args.validation_manifest.resolve()
    if manifest_path.suffix == ".jsonl":
        records = [
            json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()
            if line
        ]
        rows = [
            {
                "source": row["canonical_source"],
                "target": row["canonical_target"],
                "src": row["canonical_source"],
                "tgt": row["canonical_target"],
                "reaction_identity": row["reaction_identity"],
            }
            for row in records
        ]
    else:
        rows = read_rows(args.dataset, split="validation", path=manifest_path)
    task = TASKS[args.dataset]
    conditions = []
    if args.include_pretrained:
        conditions.append(("pretrained", None, None, False))
    for label, checkpoint in args.legacy_checkpoint:
        conditions.append((label, Path(checkpoint), None, False))
    for path in args.run_json:
        result = json.loads(path.read_text(encoding="utf-8"))
        if result["dataset"] != args.dataset:
            raise ValueError(f"{path} belongs to {result['dataset']}, not {args.dataset}")
        conditions.append((
            result["condition"], Path(result["selected_checkpoint"]), path,
            result["condition"] == "clm_jepa_vjepa2_1",
        ))
    labels = [condition[0] for condition in conditions]
    if len(labels) != len(set(labels)):
        raise ValueError("representation condition labels must be unique")
    output = {
        "dataset": args.dataset,
        "task": task,
        "validation_manifest": str(manifest_path),
        "seed": args.seed,
        "k": args.k,
        "conditions": {},
    }
    for label, checkpoint, source_result, dense_vjepa in conditions:
        set_seed(args.seed)
        tokenizer = load_reaction_tokenizer(TOKENIZER_DIR)
        chemfm_vocab_size = len(tokenizer)
        predictor_ids = [] if dense_vjepa else add_predictor_tokens(tokenizer)
        if dense_vjepa and args.k != 0:
            raise ValueError("dense V-JEPA endpoint diagnostics require --k 0")
        collator = ReactionCollator(tokenizer, task=task)
        model = load_lora_model(
            MODEL_DIR, tokenizer, chemfm_vocab_size=chemfm_vocab_size
        ).cuda().eval()
        if checkpoint is not None:
            load_adapter_checkpoint(model, checkpoint.resolve())
        method = CLMJEPA(predictor_ids, tokenizer.eos_token_id, tokenizer.pad_token_id)
        torch.cuda.reset_peak_memory_stats()
        metrics = representation_diagnostics(
            model, method, collator, rows,
            args.k, args.seed, task, limit=args.diagnostic_limit,
            physical_batch_size=args.diagnostic_batch_size,
        )
        output["conditions"][label] = {
            "source_result": None if source_result is None else str(source_result),
            "checkpoint": None if checkpoint is None else str(checkpoint),
            "metrics": metrics,
            "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
        del method, model
        torch.cuda.empty_cache()

    print(json.dumps({"output": str(args.output), "conditions": list(output["conditions"])}, sort_keys=True))


if __name__ == "__main__":
    main()
