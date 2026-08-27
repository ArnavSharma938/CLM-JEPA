"""Teacher-forced token diagnostics on the frozen five-view reaction panel."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chemfm import IGNORE_INDEX, ReactionCollator  # noqa: E402
from eval_uspto_mit_five_view_a6000 import (  # noqa: E402
    file_sha256,
    load_endpoint,
    read_jsonl,
)


def token_decision_metrics(
    logits: torch.Tensor, labels: torch.Tensor,
) -> list[dict[str, float | int]]:
    """Return per-row CE, correct-token margin, and greedy-token accuracy."""
    shifted_logits = logits[:, :-1].float()
    shifted_labels = labels[:, 1:]
    rows = []
    for row_logits, row_labels in zip(shifted_logits, shifted_labels):
        active = row_labels.ne(IGNORE_INDEX)
        selected_logits = row_logits[active]
        selected_labels = row_labels[active]
        if selected_labels.numel() == 0:
            raise ValueError("each teacher-forced row must contain target labels")
        log_probabilities = selected_logits.log_softmax(dim=-1)
        correct_log_probabilities = log_probabilities.gather(
            1, selected_labels[:, None]
        ).squeeze(1)
        top_values, top_indices = selected_logits.topk(2, dim=-1)
        correct_logits = selected_logits.gather(
            1, selected_labels[:, None]
        ).squeeze(1)
        best_other = torch.where(
            top_indices[:, 0].eq(selected_labels), top_values[:, 1], top_values[:, 0]
        )
        rows.append({
            "target_tokens": int(selected_labels.numel()),
            "nll_sum": float(-correct_log_probabilities.sum()),
            "ce": float(-correct_log_probabilities.mean()),
            "correct_margin_mean": float((correct_logits - best_other).mean()),
            "teacher_forced_top1": float(
                selected_logits.argmax(dim=-1).eq(selected_labels).float().mean()
            ),
        })
    return rows


def summarize_rows(rows: list[dict]) -> dict:
    tokens = sum(row["target_tokens"] for row in rows)
    return {
        "rows": len(rows),
        "target_tokens": tokens,
        "token_weighted_ce": sum(row["nll_sum"] for row in rows) / tokens,
        "mean_per_row_ce": sum(row["ce"] for row in rows) / len(rows),
        "mean_correct_token_margin": sum(
            row["correct_margin_mean"] for row in rows
        ) / len(rows),
        "mean_teacher_forced_top1": sum(
            row["teacher_forced_top1"] for row in rows
        ) / len(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise EnvironmentError("CUDA is required for ChemFM token diagnostics")
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")

    manifest = read_jsonl(args.manifest)
    flat_rows = []
    identities = []
    for reaction in manifest:
        if len(reaction["sources"]) != 5 or len(reaction["targets"]) != 5:
            raise ValueError("the diagnostic requires exactly five official views")
        for view_index, (source, target) in enumerate(zip(
            reaction["sources"], reaction["targets"]
        )):
            flat_rows.append({"src": source, "tgt": target})
            identities.append((reaction["reaction_identity"], view_index))

    model, tokenizer = load_endpoint(args.checkpoint, predictor_tokens=True)
    collator = ReactionCollator(tokenizer, task="forward")
    results = []
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for start in range(0, len(flat_rows), args.batch_size):
            raw = collator(flat_rows[start:start + args.batch_size])
            batch = {
                key: value.to(model.device)
                for key, value in raw.items() if torch.is_tensor(value)
            }
            output = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            metrics = token_decision_metrics(output.logits, batch["labels"])
            for offset, values in enumerate(metrics):
                reaction_identity, view_index = identities[start + offset]
                results.append({
                    "reaction_identity": reaction_identity,
                    "view_index": view_index,
                    **values,
                })

    checkpoint_weights = (
        args.checkpoint / "USPTO-MIT-Synthesis" / "adapter_model.safetensors"
    )
    output = {
        "protocol": {
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_adapter_sha256": file_sha256(checkpoint_weights),
            "manifest": str(args.manifest.resolve()),
            "manifest_sha256": file_sha256(args.manifest),
            "reactions": len(manifest),
            "views_per_reaction": 5,
            "teacher_forcing": (
                "official aligned source/target serialization for every view; "
                "label-shifted target tokens including target EOS"
            ),
            "batch_size": args.batch_size,
        },
        "overall": summarize_rows(results),
        "by_view": {
            f"view_{view_index + 1}": summarize_rows([
                row for row in results if row["view_index"] == view_index
            ])
            for view_index in range(5)
        },
        "rows": results,
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()),
        "gpu": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "output": str(args.output.resolve()),
        "checkpoint_adapter_sha256": output["protocol"]["checkpoint_adapter_sha256"],
        **output["overall"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
