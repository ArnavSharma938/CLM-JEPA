from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import shutil
from pathlib import Path

import torch

from .data import IF1Collator, load_manifest
from .modeling import autocast_context, load_if1
from .optimized_forward import model_logits
from .targets import EXPECTED_DENSE_WEIGHTS, EXPECTED_LORA, target_matrices


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--model-checkpoint", type=Path, required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    records = load_manifest(args.manifest)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA with native bf16 support is required")
    model, alphabet = load_if1(args.model_checkpoint)
    targets = target_matrices(model)
    dense = sum(item.module.weight.numel() for item in targets)
    lora = {rank: sum(item.lora_scalars(rank) for item in targets) for rank in EXPECTED_LORA}
    if dense != EXPECTED_DENSE_WEIGHTS or lora != EXPECTED_LORA:
        raise RuntimeError("Architecture-derived budgets disagree with the preregistration")
    free = shutil.disk_usage(args.output.parent).free
    # This threshold protects the 100 GB instance from known milestone/delta artifacts.
    if free < 50 * 1024**3:
        raise RuntimeError(f"At least 50 GiB free storage is required; found {free / 1024**3:.1f} GiB")

    # Exercise the two numerical optimizations on real ESM-IF1 before any
    # scientific run. This is a bounded two-example inference check, not study
    # evaluation. Shared-encoder execution should be bit-identical; compiled
    # execution is held to BF16 rounding tolerance.
    grouped: dict[tuple[Path, str], list] = {}
    for record in records:
        grouped.setdefault((record.backbone_path, record.chain_id), []).append(record)
    pair = next((values[:2] for values in grouped.values() if len(values) >= 2), None)
    if pair is None:
        raise RuntimeError("Preflight needs two sequences sharing a backbone for parity checking")
    batch = IF1Collator(alphabet, training=False)(pair)
    device = torch.device("cuda")
    model.to(device).eval()
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode(), autocast_context(device):
        eager = model_logits(model, batch, device, reuse_backbones=False)
        shared = model_logits(model, batch, device, reuse_backbones=True)
    if not torch.equal(eager, shared):
        maximum = float((eager.float() - shared.float()).abs().max())
        raise RuntimeError(f"Shared-backbone forward is not bit-identical on real ESM-IF1 (max abs {maximum})")
    compiled = torch.compile(model, mode="max-autotune", dynamic=False, fullgraph=False)
    with torch.inference_mode(), autocast_context(device):
        compiled_logits = model_logits(compiled, batch, device, reuse_backbones=False)
    compile_max_abs = float((eager.float() - compiled_logits.float()).abs().max())
    torch.testing.assert_close(compiled_logits, eager, rtol=0.02, atol=0.02)

    digest = hashlib.sha256()
    with args.model_checkpoint.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    report = {
        "gpu": torch.cuda.get_device_name(0),
        "cuda": torch.version.cuda,
        "bf16": True,
        "loaded_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "target_matrix_count": len(targets),
        "dtft_trainable_scalars": dense,
        "lora_trainable_scalars": lora,
        "manifest_records": len(records),
        "manifest_domains": len({record.domain_id for record in records}),
        "free_storage_gib": free / 1024**3,
        "model_checkpoint": str(args.model_checkpoint.resolve()),
        "model_checkpoint_sha256": digest.hexdigest(),
        "shared_encoder_bit_identical": True,
        "compiled_forward_max_abs_difference": compile_max_abs,
        "preflight_peak_cuda_memory_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("fair-esm", "torch", "numpy", "scipy", "pandas")
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
