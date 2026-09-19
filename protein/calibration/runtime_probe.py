"""One-shot real ESM-IF1 before/after probe for an optimization change."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

from .data import IF1Collator, SequenceRecord
from .modeling import autocast_context, configure_adaptation, load_if1
from .optimized_forward import enable_if1_evaluation_optimizations, model_logits


def _records(manifest: Path, size: int) -> list[SequenceRecord]:
    frame = pd.read_parquet(manifest)
    key = frame.groupby(["backbone_path", "chain_id"]).size().idxmax()
    rows = frame[(frame.backbone_path == key[0]) & (frame.chain_id == key[1])].head(size)
    root = manifest.parent
    return [
        SequenceRecord(
            domain_id=str(row.domain_id), sequence_id=str(row.sequence_id), split=str(row.split),
            backbone_path=(root / str(row.backbone_path)).resolve(), chain_id=str(row.chain_id),
            native_sequence=str(row.native_sequence), target_sequence=str(row.target_sequence),
            ddg=float(row.ddg), substitution_count=int(row.substitution_count),
            foldseek_qtm_max_to_train=float(row.foldseek_qtm_max_to_train),
        )
        for row in rows.itertuples(index=False)
    ]


def _run(model, batch, alphabet, device, *, reuse: bool, backward: bool, iterations: int):
    tokens = batch["tokens"].to(device)
    target = tokens[:, 1:]
    model.train(backward)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    final_logits = final_loss = None
    for _ in range(iterations):
        model.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(backward), autocast_context(device):
            final_logits = model_logits(model, batch, device, reuse_backbones=reuse, tokens=tokens)
            final_loss = F.cross_entropy(
                final_logits.float(), target, reduction="sum", ignore_index=alphabet.padding_idx
            )
        if backward:
            final_loss.backward()
    torch.cuda.synchronize(device)
    gradients = {
        name: parameter.grad.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is not None
    }
    return {
        "seconds": time.perf_counter() - started,
        "peak_bytes": torch.cuda.max_memory_allocated(device),
        "logits": final_logits.detach().cpu(),
        "loss": final_loss.detach().cpu(),
        "gradients": gradients,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--eval-batch", type=int, default=64)
    args = parser.parse_args()
    device = torch.device("cuda")
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    model, alphabet = load_if1(args.checkpoint)
    configure_adaptation(model, "lora8")
    records = _records(args.manifest, args.eval_batch)
    batch = IF1Collator(alphabet, training=False)(records)
    train_batch = {key: (value[:4] if torch.is_tensor(value) else value[:4]) for key, value in batch.items()}
    train_batch["backbone_unique"] = torch.tensor([0])
    train_batch["backbone_inverse"] = torch.zeros(4, dtype=torch.long)
    model.to(device)

    with torch.inference_mode():
        reference_eval = _run(model, batch, alphabet, device, reuse=False, backward=False, iterations=2)
    cpu_rng = torch.get_rng_state()
    rng = torch.cuda.get_rng_state_all()
    reference_train = _run(model, train_batch, alphabet, device, reuse=False, backward=True, iterations=1)
    torch.set_rng_state(cpu_rng)
    torch.cuda.set_rng_state_all(rng)
    optimized_train = _run(model, train_batch, alphabet, device, reuse=False, backward=True, iterations=1)

    enable_if1_evaluation_optimizations(model)
    with torch.inference_mode():
        optimized_eval = _run(model, batch, alphabet, device, reuse=True, backward=False, iterations=2)
    def parity(left, right):
        gradient_error = max(
            (left["gradients"][name] - right["gradients"][name]).abs().max().item()
            for name in left["gradients"]
        ) if left["gradients"] else 0.0
        return {
            "max_abs_logits": (left["logits"] - right["logits"]).abs().max().item(),
            "abs_loss": (left["loss"] - right["loss"]).abs().item(),
            "relative_loss": ((left["loss"] - right["loss"]).abs() / left["loss"].abs()).item(),
            "max_abs_gradient": gradient_error,
        }

    result = {"device": torch.cuda.get_device_name(device), "eval_batch": len(records)}
    for label, reference, optimized in (
        ("evaluation", reference_eval, optimized_eval), ("training", reference_train, optimized_train)
    ):
        result[label] = {
            "speedup": reference["seconds"] / optimized["seconds"],
            "reference_seconds": reference["seconds"], "optimized_seconds": optimized["seconds"],
            "reference_peak_bytes": reference["peak_bytes"], "optimized_peak_bytes": optimized["peak_bytes"],
            "memory_reduction_fraction": 1 - optimized["peak_bytes"] / reference["peak_bytes"],
            "parity": parity(reference, optimized),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
