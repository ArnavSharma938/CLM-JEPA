#!/usr/bin/env python
"""Benchmark and train the locked paired Native/NextLat RITA-M LoRA pilot."""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.modeling import load_rita
from src.nextlat import FaithfulNextLatPredictor
from src.pilot import (
    attach_rank32_attention_lora, causal_pilot_loss, checkpoint_crossings,
    cosine_multiplier, create_optimizer, length_bucket_order, load_protocol,
    lora_named_parameters, make_batches, sha256_file, tensor_state_sha256,
)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def move_batch(batch: dict[str, torch.Tensor], device: str) -> dict[str, torch.Tensor]:
    return {name: value.to(device, non_blocking=True) for name, value in batch.items()}


def environment() -> dict:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def build(seed: int, arm: str, protocol: dict, device: str):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    loaded = load_rita(device=device, dtype=torch.bfloat16, freeze=True)
    model = attach_rank32_attention_lora(loaded.model, protocol).to(device)
    model.config.use_cache = False
    model.train()
    lora = lora_named_parameters(model)
    initial_hash = tensor_state_sha256(lora)
    predictor = None
    if arm == "nextlat":
        torch.manual_seed(seed + 100_000)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed + 100_000)
        predictor = FaithfulNextLatPredictor(1024, proj_factor=1.6).to(device).train()
    trainable = [parameter for _, parameter in lora]
    if predictor is not None:
        trainable.extend(predictor.parameters())
    if any(parameter.requires_grad for name, parameter in model.named_parameters()
           if "lora_" not in name):
        raise AssertionError("non-LoRA RITA parameter is trainable")
    return loaded.tokenizer, model, predictor, trainable, initial_hash


def save_checkpoint(output: Path, model, predictor, optimizer, scheduler, metadata: dict):
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output / "adapter", safe_serialization=True)
    if predictor is not None:
        torch.save({
            "state_dict": predictor.state_dict(),
            "hidden_size": 1024,
            "architecture": "FaithfulNextLatPredictor(proj_factor=1.6)",
        }, output / "predictor.pt")
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    # One rolling resume state per arm avoids multiplying optimizer-state storage.
    torch.save({
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "metadata": metadata,
    }, output.parent / "latest_optimizer_state.pt")


def train(args):
    protocol = load_protocol(args.protocol)
    seed = protocol["replicates"]["seeds"][args.replicate]
    rows = read_jsonl(args.manifest)
    if any(row.get("replicate") != args.replicate for row in rows):
        raise ValueError("training manifest replicate mismatch")
    rows = length_bucket_order(rows, seed)
    microbatch = protocol["optimization"]["microbatch_directional_sequences"]
    accumulation = protocol["optimization"]["gradient_accumulation_steps"]
    if (2 * len(rows)) % (microbatch * accumulation):
        raise ValueError("manifest must yield complete fixed-size accumulation windows")
    total_directional = 2 * sum(len(row["sequence"]) for row in rows)
    total_microsteps = (2 * len(rows)) // microbatch
    total_steps = total_microsteps // accumulation
    crossings = checkpoint_crossings(total_directional)
    tokenizer, model, predictor, parameters, initial_hash = build(
        seed, args.arm, protocol, args.device
    )
    optimizer = create_optimizer(parameters, protocol, cuda=args.device == "cuda")
    spec = protocol["optimization"]
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: cosine_multiplier(
            step, total_steps, spec["warmup_fraction"], spec["minimum_lr_ratio"]
        ),
    )
    output = args.output / f"replicate_{args.replicate}" / args.arm
    output.mkdir(parents=True, exist_ok=True)
    run = {
        "replicate": args.replicate,
        "seed": seed,
        "arm": args.arm,
        "protocol_sha256": sha256_file(args.protocol),
        "manifest_sha256": sha256_file(args.manifest),
        "initial_lora_sha256": initial_hash,
        "environment": environment(),
        "total_primary_residues": total_directional // 2,
        "total_directional_residues": total_directional,
        "total_optimizer_steps": total_steps,
        "records": [],
        "parameters_updated": True,
    }
    (output / "initialization.json").write_text(
        json.dumps({key: run[key] for key in (
            "replicate", "seed", "arm", "protocol_sha256", "manifest_sha256",
            "initial_lora_sha256", "environment",
        )}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    optimizer.zero_grad(set_to_none=True)
    directional_seen = 0
    optimizer_step = 0
    saved: set[float] = set()
    started = time.perf_counter()
    recent = {"ntp": 0.0, "latent": 0.0, "kl": 0.0, "microsteps": 0}
    for microstep, (examples, cpu_batch) in enumerate(
        make_batches(rows, tokenizer, microbatch), start=1
    ):
        batch = move_batch(cpu_batch, args.device)
        with torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=args.device == "cuda"
        ):
            losses = causal_pilot_loss(model, predictor, batch, args.arm)
            scaled = losses.total / accumulation
        scaled.backward()
        directional_seen += sum(len(row["sequence"]) for row in examples)
        recent["ntp"] += float(losses.ntp)
        recent["latent"] += 0.0 if losses.latent is None else float(losses.latent)
        recent["kl"] += 0.0 if losses.kl is None else float(losses.kl)
        recent["microsteps"] += 1
        if microstep % accumulation:
            continue
        grad_norm = float(torch.nn.utils.clip_grad_norm_(
            parameters, spec["gradient_clip_norm"]
        ))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        optimizer_step += 1
        crossed = [
            fraction for fraction, threshold in crossings.items()
            if fraction not in saved and directional_seen >= threshold
        ]
        if crossed:
            elapsed = time.perf_counter() - started
            record = {
                "step": optimizer_step,
                "directional_residues": directional_seen,
                "fraction_actual": directional_seen / total_directional,
                "elapsed_seconds": elapsed,
                "residues_per_second": directional_seen / elapsed,
                "learning_rate": scheduler.get_last_lr()[0],
                "gradient_norm_preclip": grad_norm,
                "mean_recent_ntp": recent["ntp"] / recent["microsteps"],
                "mean_recent_latent": recent["latent"] / recent["microsteps"],
                "mean_recent_kl": recent["kl"] / recent["microsteps"],
            }
            for fraction in crossed:
                metadata = {**run, "records": run["records"] + [record],
                            "nominal_budget_fraction": fraction, "checkpoint": record}
                save_checkpoint(
                    output / f"checkpoint_{int(fraction * 100):03d}",
                    model, predictor, optimizer, scheduler, metadata,
                )
                saved.add(fraction)
            run["records"].append(record)
            recent = {"ntp": 0.0, "latent": 0.0, "kl": 0.0, "microsteps": 0}
            print(json.dumps({"event": "checkpoint", **record}), flush=True)
    if optimizer_step != total_steps or saved != {0.25, 0.5, 1.0}:
        raise AssertionError((optimizer_step, total_steps, saved))
    run["elapsed_seconds"] = time.perf_counter() - started
    run["final_lora_sha256"] = tensor_state_sha256(lora_named_parameters(model))
    run["completed"] = True
    (output / "training.json").write_text(
        json.dumps(run, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def benchmark(args):
    protocol = load_protocol(args.protocol)
    rows = read_jsonl(args.manifest)
    seed = protocol["replicates"]["seeds"][0]
    rows = length_bucket_order(rows, seed)
    microbatch = protocol["optimization"]["microbatch_directional_sequences"]
    results = {}
    for arm in ("native", "nextlat"):
        tokenizer, model, predictor, parameters, _ = build(seed, arm, protocol, args.device)
        optimizer = create_optimizer(parameters, protocol, cuda=args.device == "cuda")
        # Cover several independent length buckets rather than repeatedly timing
        # one unusually short batch.
        batches = list(make_batches(
            rows[:max(256, microbatch * args.steps)], tokenizer, microbatch
        ))
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        started = time.perf_counter()
        residues = 0
        for step in range(args.steps):
            optimizer.zero_grad(set_to_none=True)
            for accumulation_index in range(
                protocol["optimization"]["gradient_accumulation_steps"]
            ):
                batch_index = (step * protocol["optimization"]["gradient_accumulation_steps"]
                               + accumulation_index) % len(batches)
                examples, cpu_batch = batches[batch_index]
                batch = move_batch(cpu_batch, args.device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                    enabled=args.device == "cuda"):
                    losses = causal_pilot_loss(model, predictor, batch, arm)
                    scaled = losses.total / protocol["optimization"][
                        "gradient_accumulation_steps"
                    ]
                scaled.backward()
                residues += sum(len(row["sequence"]) for row in examples)
            torch.nn.utils.clip_grad_norm_(
                parameters, protocol["optimization"]["gradient_clip_norm"]
            )
            optimizer.step()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        results[arm] = {
            "optimizer_steps": args.steps,
            "microsteps": args.steps * protocol["optimization"]["gradient_accumulation_steps"],
            "directional_residues": residues,
            "elapsed_seconds": elapsed,
            "directional_residues_per_second": residues / elapsed,
            "peak_vram_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
        }
        del model, predictor, optimizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    harmonic_pair_rate = 2.0 / sum(1.0 / results[arm]["directional_residues_per_second"]
                                   for arm in ("native", "nextlat"))
    manifest_paths = sorted(args.manifest.parent.glob("train_replicate_*.jsonl"))
    directional_per_all_replicates = sum(
        2 * sum(len(row["sequence"]) for row in read_jsonl(path))
        for path in manifest_paths
    )
    results["estimate"] = {
        "replicate_manifests_counted": len(manifest_paths),
        "total_directional_residues_all_16_arms": 2 * directional_per_all_replicates,
        "training_seconds": directional_per_all_replicates * sum(
            1.0 / results[arm]["directional_residues_per_second"]
            for arm in ("native", "nextlat")
        ),
        "paired_harmonic_residues_per_second": harmonic_pair_rate,
    }
    results["protocol_sha256"] = sha256_file(args.protocol)
    results["environment"] = environment()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(json.dumps(results, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/causal_pilot.json")
    sub = parser.add_subparsers(dest="command", required=True)
    bench = sub.add_parser("benchmark")
    bench.add_argument("manifest", type=Path)
    bench.add_argument("output", type=Path)
    bench.add_argument("--steps", type=int, default=200)
    bench.add_argument("--device", default="cuda")
    train_parser = sub.add_parser("train")
    train_parser.add_argument("manifest", type=Path)
    train_parser.add_argument("output", type=Path)
    train_parser.add_argument("--replicate", type=int, choices=range(8), required=True)
    train_parser.add_argument("--arm", choices=("native", "nextlat"), required=True)
    train_parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    benchmark(args) if args.command == "benchmark" else train(args)


if __name__ == "__main__":
    main()
