from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import torch

from .checkpointing import restore_trainables
from .data import load_manifest, preload_coordinate_cache
from .evaluation import evaluate_sequences, evaluate_variants
from .modeling import configure_adaptation, load_if1
from .training import _make_loader, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=("base", "ft", "dtft", "lora8", "lora64"))
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--model-checkpoint", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Real ESM-IF1 evaluation requires CUDA bf16")
    seed_everything(args.seed)
    model, alphabet = load_if1(args.model_checkpoint)
    configure_adaptation(model, args.mode)
    if args.mode != "base":
        if args.checkpoint is None:
            raise ValueError("Adapted modes require --checkpoint")
        restore_trainables(model, args.checkpoint)
    device = torch.device("cuda")
    model.to(device).eval()
    test = [record for record in load_manifest(args.manifest) if record.split == "test"]
    unique_native = {}
    for record in test:
        key = (record.backbone_path, record.chain_id)
        unique_native.setdefault(key, replace(record, target_sequence=record.native_sequence))
    native = list(unique_native.values())
    coordinate_cache = preload_coordinate_cache(test)
    sequence_loader, _ = _make_loader(
        test, alphabet, batch_size=args.batch_size, training=False, noise=0, seed=args.seed, workers=6,
        coordinate_cache=coordinate_cache,
    )
    native_loader, _ = _make_loader(
        native, alphabet, batch_size=args.batch_size, training=False, noise=0, seed=args.seed, workers=6,
        coordinate_cache=coordinate_cache,
    )
    evaluate_sequences(model, sequence_loader, native_loader, alphabet, device, args.output_dir)


if __name__ == "__main__":
    main()
