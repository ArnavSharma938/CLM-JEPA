from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from .batching import GenerationDataset
from .config import MODEL_ID, MODEL_REVISION
from .prepare_data import sha256_file
from .token_cache import load_token_cache
from .training import evaluate_examples, load_base_model


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def decoy_examples(test_dataset: GenerationDataset, decoy_manifest: Path, cache: dict) -> list[dict]:
    pair_by_id = {item["row"]["pair_id"]: item for item in test_dataset.items}
    examples = []
    for line in decoy_manifest.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        true_item = pair_by_id[row["pair_id"]]
        eval_row = dict(true_item["row"])
        eval_row["evaluated_receptor_id"] = row["decoy_receptor_id"]
        examples.append({
            "row": eval_row,
            "receptor_tokens": cache["receptor_tokens"][row["decoy_receptor_id"]],
            "peptide_tokens": true_item["peptide_tokens"],
        })
    return examples


def _save_evaluation(result: dict, directory: Path, name: str, metadata: dict) -> dict:
    rows = result.pop("per_example")
    write_jsonl(directory / f"{name}.jsonl", rows)
    summary = {**result, **metadata}
    (directory / f"{name}_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def run_evaluation(
    *,
    manifest_dir: Path,
    cache_path: Path,
    output_dir: Path,
    checkpoint: Path | None = None,
    token_budget: int = 16_384,
    workers: int = 2,
) -> dict:
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("ESM-2 full-span evaluation requires CUDA BF16")
    started = time.perf_counter()
    torch.backends.cuda.matmul.allow_tf32 = True
    cache = load_token_cache(cache_path)
    model, _ = load_base_model("sdpa")
    if checkpoint is not None:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        parameters = dict(model.named_parameters())
        state = payload["trainable_state"]
        if not set(state).issubset(parameters):
            raise RuntimeError("DTFT checkpoint contains weights outside the ESM-2 model")
        with torch.no_grad():
            for name, value in state.items():
                parameters[name].copy_(value)
        model_name = "dtft_seed_11"
    else:
        model_name = "base"
    model.to(torch.device("cuda"))
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for split in (("validation", "test") if checkpoint is None else ("test",)):
        dataset = GenerationDataset(cache, manifest_dir / f"{split}.jsonl")
        evaluation = evaluate_examples(
            model,
            [{"row": item["row"], "receptor_tokens": item["receptor_tokens"], "peptide_tokens": item["peptide_tokens"]} for item in dataset.items],
            cache,
            token_budget=token_budget,
            workers=workers,
        )
        metadata = {
            "model": model_name,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "split": split,
            "manifest_sha256": sha256_file(manifest_dir / f"{split}.jsonl"),
            "checkpoint_sha256": sha256_file(checkpoint) if checkpoint else None,
            "attention_backend": "sdpa",
            "token_budget": token_budget,
        }
        results[f"{split}_true"] = _save_evaluation(evaluation, output_dir, f"{split}_true", metadata)
        if split == "test":
            decoy_path = manifest_dir / "test_decoys.jsonl"
            decoy_eval = evaluate_examples(
                model, decoy_examples(dataset, decoy_path, cache), cache,
                token_budget=token_budget, workers=workers,
            )
            decoy_metadata = {
                **metadata,
                "condition": "fixed_other_heldout_receptor_decoys",
                "decoy_manifest_sha256": sha256_file(decoy_path),
            }
            results["test_decoy"] = _save_evaluation(decoy_eval, output_dir, "test_decoy", decoy_metadata)
    results["total_wall_seconds"] = time.perf_counter() - started
    (output_dir / "evaluation_summary.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    del model
    torch.cuda.empty_cache()
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Base or DTFT on full-span peptide generation")
    parser.add_argument("--manifest-dir", type=Path, default=Path("data/protein/propedia26/manifests"))
    parser.add_argument("--cache", type=Path, default=Path("data/protein/propedia26/token_cache.pt"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--token-budget", type=int, default=16_384)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    print(json.dumps(run_evaluation(
        manifest_dir=args.manifest_dir,
        cache_path=args.cache,
        output_dir=args.output_dir,
        checkpoint=args.checkpoint,
        token_budget=args.token_budget,
        workers=args.workers,
    ), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
