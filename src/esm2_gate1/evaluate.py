from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .batching import TokenBudgetBatchSampler
from .config import COMMON_MASK_SEED, MODEL_REVISION
from .modeling import configure_adaptation, load_model
from .training import MLMCollator, ProteinDataset


def _restore(model, checkpoint: Path) -> None:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    parameters = dict(model.named_parameters())
    expected = {name for name, value in parameters.items() if value.requires_grad}
    if set(payload["trainable_state"]) != expected:
        raise RuntimeError("evaluation checkpoint does not match configured trainables")
    with torch.no_grad():
        for name, value in payload["trainable_state"].items():
            parameters[name].copy_(value)


@torch.no_grad()
def _evaluate_loaded(
    model,
    tokenizer,
    manifest: Path,
    output: Path,
    *,
    realizations: int = 1,
    token_budget: int = 4096,
    num_workers: int = 2,
) -> dict:
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    dataset = ProteinDataset(manifest, tokenizer)
    aggregate: dict[str, dict] = {}
    total_loss = 0.0
    total_tokens = 0
    for realization in range(realizations):
        sampler = TokenBudgetBatchSampler(
            [int(row["length"]) for row in dataset.rows], token_budget, seed=0, shuffle=False
        )
        loader = DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=MLMCollator(
                tokenizer, exposure=0, common_seed=COMMON_MASK_SEED, realization=realization
            ),
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )
        for batch in loader:
            inputs = batch["input_ids"].cuda(non_blocking=True)
            labels = batch["labels"].cuda(non_blocking=True)
            mask = batch["attention_mask"].cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_ids=inputs, attention_mask=mask).logits
            losses = F.cross_entropy(logits.float().transpose(1, 2), labels, reduction="none", ignore_index=-100)
            for index, row in enumerate(batch["rows"]):
                selected = labels[index].ne(-100)
                value = float(losses[index][selected].sum().item())
                count = int(selected.sum().item())
                total_loss += value
                total_tokens += count
                item = aggregate.setdefault(
                    row["backbone_id"],
                    {
                        **row,
                        "loss_sum": 0.0,
                        "masked_tokens": 0,
                        "realizations": realizations,
                    },
                )
                item["loss_sum"] += value
                item["masked_tokens"] += count
    proteins = []
    for item in aggregate.values():
        item["nll"] = item["loss_sum"] / item["masked_tokens"]
        proteins.append(item)
    proteins.sort(key=lambda item: item["backbone_id"])
    elapsed = time.perf_counter() - started
    summary = {
        "manifest": str(manifest),
        "realizations": realizations,
        "token_budget": token_budget,
        "num_workers": num_workers,
        "masked_token_nll": total_loss / total_tokens,
        "mean_protein_nll": float(np.mean([item["nll"] for item in proteins])),
        "proteins": len(proteins),
        "masked_tokens": total_tokens,
        "wall_seconds": elapsed,
        "residues_per_second": (
            sum(int(row["length"]) for row in dataset.rows) * realizations
            / elapsed
        ),
        "peak_vram_bytes": torch.cuda.max_memory_allocated(),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    with (output / "protein_nll.jsonl").open("w", encoding="utf-8") as handle:
        for item in proteins:
            handle.write(json.dumps(item, sort_keys=True) + "\n")
    return summary


def _load_evaluation_model(
    mode: str, checkpoint: Path | None, attention_backend: str, compile_model: bool
):
    if not torch.cuda.is_available():
        raise RuntimeError("real ESM-2 evaluation requires CUDA")
    model, tokenizer = load_model(revision=MODEL_REVISION, attention_backend=attention_backend)
    configure_adaptation(model, mode)
    if checkpoint is not None:
        _restore(model, checkpoint)
    elif mode != "base":
        raise ValueError("a trained method requires a checkpoint")
    model.to("cuda").eval()
    if compile_model:
        model = torch.compile(model, mode="reduce-overhead")
    return model, tokenizer


def evaluate_many(
    manifests: dict[str, Path],
    outputs: dict[str, Path],
    *,
    mode: str,
    checkpoint: Path | None,
    realizations: int = 1,
    token_budget: int = 4096,
    num_workers: int = 2,
    attention_backend: str = "sdpa",
    compile_model: bool = False,
) -> dict[str, dict]:
    """Evaluate several fixed sets with one model/checkpoint load."""
    if set(manifests) != set(outputs):
        raise ValueError("evaluation manifest/output labels differ")
    model, tokenizer = _load_evaluation_model(
        mode, checkpoint, attention_backend, compile_model
    )
    summaries = {}
    for label, manifest in manifests.items():
        summary = _evaluate_loaded(
            model, tokenizer, manifest, outputs[label], realizations=realizations,
            token_budget=token_budget, num_workers=num_workers,
        )
        summary.update({
            "mode": mode,
            "checkpoint": str(checkpoint) if checkpoint else None,
            "torch_compile": compile_model,
        })
        (outputs[label] / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
        )
        summaries[label] = summary
    return summaries


def evaluate(
    manifest: Path,
    output: Path,
    *,
    mode: str,
    checkpoint: Path | None,
    realizations: int = 1,
    token_budget: int = 4096,
    num_workers: int = 2,
    attention_backend: str = "sdpa",
    compile_model: bool = False,
) -> dict:
    return evaluate_many(
        {"single": manifest}, {"single": output}, mode=mode, checkpoint=checkpoint,
        realizations=realizations, token_budget=token_budget, num_workers=num_workers,
        attention_backend=attention_backend, compile_model=compile_model,
    )["single"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mode", required=True, choices=("base", "ft", "dtft", "lora8", "lora64"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--realizations", type=int, default=1)
    parser.add_argument("--attention-backend", default="sdpa")
    parser.add_argument("--token-budget", type=int, default=4096)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--compile-model", action="store_true")
    args = parser.parse_args()
    evaluate(
        args.manifest, args.output, mode=args.mode, checkpoint=args.checkpoint,
        realizations=args.realizations, attention_backend=args.attention_backend,
        token_budget=args.token_budget, num_workers=args.num_workers,
        compile_model=args.compile_model,
    )


if __name__ == "__main__":
    main()
