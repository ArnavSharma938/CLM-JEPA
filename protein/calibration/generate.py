from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .checkpointing import restore_trainables
from .data import load_manifest, load_record_coords
from .modeling import configure_adaptation, load_if1


def paired_generation_seed(training_seed: int, backbone: str, sample_index: int, checkpoint_label: str) -> int:
    payload = f"{training_seed}|{backbone}|{sample_index}|{checkpoint_label}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=("base", "ft", "dtft", "lora8", "lora64"))
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--panel", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--checkpoint-label", required=True, choices=("best", "epoch_1", "epoch_10"))
    parser.add_argument("--model-checkpoint", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Real ESM-IF1 generation requires CUDA")
    model, _ = load_if1(args.model_checkpoint)
    configure_adaptation(model, args.mode)
    if args.mode != "base":
        if args.checkpoint is None:
            raise ValueError("Adapted modes require --checkpoint")
        restore_trainables(model, args.checkpoint)
    model.cuda().eval()
    panel = json.loads(args.panel.read_text(encoding="utf-8"))
    selected = {domain for domains in panel["generation"].values() for domain in domains}
    by_domain = {}
    for record in load_manifest(args.manifest):
        if record.domain_id in selected:
            by_domain.setdefault(record.domain_id, record)
    if set(by_domain) != selected or len(selected) != 18:
        raise RuntimeError("Generation panel must resolve to exactly 18 test backbones")
    count = 16 if args.checkpoint_label == "best" else 4
    rows = []
    for domain in sorted(selected):
        record = by_domain[domain]
        coords = load_record_coords(record)
        for sample_index in range(count):
            rng_seed = paired_generation_seed(args.seed, domain, sample_index, args.checkpoint_label)
            torch.manual_seed(rng_seed)
            torch.cuda.manual_seed_all(rng_seed)
            with torch.inference_mode():
                sequence = model.sample(coords, temperature=0.5, device="cuda")
            sample_id = hashlib.sha256(
                f"{args.mode}|{args.seed}|{args.checkpoint_label}|{domain}|{sample_index}".encode()
            ).hexdigest()[:20]
            rows.append(
                {
                    "sample_id": sample_id,
                    "generated_sequence": sequence,
                    "native_sequence": record.native_sequence,
                    "conditioning_backbone": str(record.backbone_path),
                    "domain": domain,
                    "model": args.mode,
                    "training_seed": args.seed,
                    "checkpoint": args.checkpoint_label,
                    "generation_rng_seed": rng_seed,
                    "decoding": {"method": "ancestral", "temperature": 0.5, "top_k": None, "top_p": None},
                }
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
