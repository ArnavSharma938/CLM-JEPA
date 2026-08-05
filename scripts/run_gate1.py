from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import set_seed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clm_jepa.chemfm_native import (  # noqa: E402
    IGNORE_INDEX,
    ReactionCollator,
    canonicalize,
    generate_products,
    load_lora_model,
    load_reaction_tokenizer,
)
from clm_jepa.paths import MODEL_DIR, TOKENIZER_DIR  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_manifests(source: Path, output: Path, seed: int) -> list[dict[str, str]]:
    # The official file consists of 20 randomized-SMILES rows per underlying
    # reaction. Sampling complete groups avoids a first-row/order artifact.
    rng = random.Random(seed)
    reservoirs: list[list[dict[str, str]]] = []
    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        group: list[dict[str, str]] = []
        seen_groups = 0
        for row in reader:
            group.append({"src": row["src"], "tgt": row["tgt"]})
            if len(group) == 20:
                seen_groups += 1
                if len(reservoirs) < 7:
                    reservoirs.append(group)
                else:
                    index = rng.randrange(seen_groups)
                    if index < 7:
                        reservoirs[index] = group
                group = []
    rows = [row for group in reservoirs for row in group]
    selected = rows[:128]
    output.mkdir(parents=True, exist_ok=True)
    for size in (32, 128):
        target = output / f"train_{size}.csv"
        with target.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["src", "tgt"])
            writer.writeheader()
            writer.writerows(selected[:size])
    metadata = {
        "source": "ChemFM official Box USPTO-MIT-Synthesis/train.csv",
        "source_url": "https://clemson.app.box.com/s/kct8hy0pc0i7iyjlpmrxng8cyoj12i9v/folder/303440259916",
        "source_sha256": sha256(source),
        "source_bytes": source.stat().st_size,
        "selection": "seeded reservoir sample of seven complete contiguous 20-row augmentation groups; first 128 sampled rows",
        "seed": seed,
        "counts": {"train_32": 32, "train_128": 128},
    }
    (output / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return selected


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def evaluate_loss(model, loader, device) -> float:
    model.eval()
    weighted_loss = 0.0
    target_tokens = 0
    with torch.inference_mode():
        for batch in loader:
            count = int(batch["labels"].ne(IGNORE_INDEX).sum())
            inputs = {
                key: value.to(device)
                for key, value in batch.items()
                if key in {"input_ids", "attention_mask", "labels"}
            }
            loss = model(**inputs).loss
            weighted_loss += float(loss) * count
            target_tokens += count
    return weighted_loss / target_tokens


def cosine_with_floor(optimizer, warmup_steps: int, total_steps: int, floor_ratio: float):
    def scale(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return floor_ratio + (1.0 - floor_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=int, choices=(32, 128), default=32)
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=533)
    parser.add_argument("--eval-generation", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()

    source = ROOT / "data" / "official" / "uspto_mit_synthesis" / "train.csv"
    manifest_dir = ROOT / "data" / "manifests" / "gate1"
    make_manifests(source, manifest_dir, args.seed)
    if args.prepare_only:
        return

    set_seed(args.seed)
    tokenizer = load_reaction_tokenizer(TOKENIZER_DIR)
    collator = ReactionCollator(tokenizer)
    rows = read_rows(manifest_dir / f"train_{args.subset}.csv")
    loader = DataLoader(rows, batch_size=args.batch_size, shuffle=True, collate_fn=collator)
    fixed_loader = DataLoader(rows, batch_size=args.batch_size, shuffle=False, collate_fn=collator)

    first = collator(rows[: min(args.batch_size, len(rows))])
    leakage = bool((first["labels"] != IGNORE_INDEX)[first["labels"] == IGNORE_INDEX].any())
    source_positions = first["labels"].eq(IGNORE_INDEX) & first["attention_mask"]
    source_leakage = bool((first["labels"][source_positions] != IGNORE_INDEX).any())
    if leakage or source_leakage:
        raise AssertionError("source tokens leaked into labels")

    model = load_lora_model(MODEL_DIR, tokenizer).cuda()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.learning_rate,
        weight_decay=0.01,
    )
    scheduler = cosine_with_floor(
        optimizer,
        warmup_steps=max(1, round(args.steps * 0.05)),
        total_steps=args.steps,
        floor_ratio=1e-5 / args.learning_rate,
    )
    initial_loss = evaluate_loss(model, fixed_loader, model.device)
    losses = []
    iterator = iter(loader)
    model.train()
    for step in range(1, args.steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        inputs = {
            key: value.to(model.device)
            for key, value in batch.items()
            if key in {"input_ids", "attention_mask", "labels"}
        }
        loss = model(**inputs).loss
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite training loss at step {step}: {loss.item()}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss.detach()))
        if step == 1 or step % 20 == 0:
            print(json.dumps({"step": step, "loss": losses[-1], "lr": scheduler.get_last_lr()[0]}), flush=True)

    final_loss = evaluate_loss(model, fixed_loader, model.device)
    result = {
        "subset": args.subset,
        "steps": args.steps,
        "initial_target_loss": initial_loss,
        "final_target_loss": final_loss,
        "last_20_train_loss": sum(losses[-20:]) / min(20, len(losses)),
        "source_label_leakage": False,
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
    }
    if args.eval_generation:
        model.eval()
        prompts = collator(rows)["generation_prompts"]
        predictions = []
        # Official ChemFM generation uses batch size one with a right-padding
        # tokenizer. Keep that behavior so shorter prompts never generate from padding.
        for prompt in prompts:
            predictions.extend(generate_products(model, tokenizer, [prompt]))
        canonical_predictions = [canonicalize(x) for x in predictions]
        canonical_targets = [canonicalize(x["tgt"]) for x in rows]
        result.update(
            valid_products=sum(bool(x) for x in canonical_predictions),
            exact_products=sum(p == t for p, t in zip(canonical_predictions, canonical_targets)),
            predictions=predictions,
        )
    output = ROOT / "artifacts" / "gate1"
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / f"subset_{args.subset}_result.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
