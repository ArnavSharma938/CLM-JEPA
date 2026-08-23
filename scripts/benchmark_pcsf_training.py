"""Archived A6000 benchmark source for the removed PCSF training path.

The measurements remain reproducible from the retained environment/archive,
but this script intentionally does not reconnect PCSF to the active trainer.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import get_scheduler, set_seed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chemfm import MODEL_DIR, TOKENIZER_DIR, ReactionCollator, load_lora_model, load_reaction_tokenizer  # noqa: E402
from jepa import CLMJEPA, add_predictor_tokens  # noqa: E402
from historical_pcsf import PairCenterSpreadFloor  # noqa: E402
from train import (  # noqa: E402
    ADAM_BETAS, ADAM_EPSILON, MIN_LEARNING_RATE, WARMUP_RATIO, WEIGHT_DECAY,
    read_rows,
)


class LimitedLoader:
    def __init__(self, loader, count: int):
        self.loader = loader
        self.count = count
        self.pin_memory = loader.pin_memory

    def __len__(self):
        return self.count

    def __iter__(self):
        return itertools.islice(iter(self.loader), self.count)


def load_model(args, *, optimized: bool):
    set_seed(533)
    tokenizer = load_reaction_tokenizer(TOKENIZER_DIR)
    chemfm_vocab_size = len(tokenizer)
    predictor_ids = add_predictor_tokens(tokenizer)
    model = load_lora_model(
        MODEL_DIR, tokenizer, chemfm_vocab_size=chemfm_vocab_size,
        attn_implementation=args.attention,
    ).cuda()
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.enable_input_require_grads()
    method = CLMJEPA(
        predictor_ids, tokenizer.eos_token_id, tokenizer.pad_token_id,
        sigreg_seed=533, optimized_native_logits=optimized,
    )
    return tokenizer, model, method


def gradient_vector(loss, model) -> torch.Tensor:
    selected = [
        parameter for name, parameter in model.named_parameters()
        if parameter.requires_grad and ".lora_" in name
    ]
    values = torch.autograd.grad(loss, selected, allow_unused=True)
    return torch.cat([
        (torch.zeros_like(parameter) if value is None else value).detach().float().flatten().cpu()
        for parameter, value in zip(selected, values)
    ])


def parity(args) -> dict:
    rows = read_rows("uspto_mit_synthesis", path=args.train_manifest)[: args.physical_batch]
    tokenizer, model, standard = load_model(args, optimized=False)
    optimized = CLMJEPA(
        standard.predictor_token_ids, tokenizer.eos_token_id, tokenizer.pad_token_id,
        sigreg_seed=533, optimized_native_logits=True,
    )
    collator = ReactionCollator(tokenizer, task="forward")
    batch = {
        key: value.cuda() for key, value in collator(rows).items()
        if torch.is_tensor(value)
    }
    model.eval()
    reference = standard(
        model, batch, k=0, jepa_weight=2.0, jepa_loss_type="mse",
        force_jepa_active=True,
    )
    reference_gradient = gradient_vector(reference.loss, model)
    candidate = optimized(
        model, batch, k=0, jepa_weight=2.0, jepa_loss_type="mse",
        force_jepa_active=True,
    )
    candidate_gradient = gradient_vector(candidate.loss, model)
    difference = candidate_gradient - reference_gradient
    result = {
        "native_loss_reference": float(reference.native_loss),
        "native_loss_candidate": float(candidate.native_loss),
        "native_loss_absolute_error": abs(float(reference.native_loss) - float(candidate.native_loss)),
        "mse_reference": float(reference.jepa_loss),
        "mse_candidate": float(candidate.jepa_loss),
        "mse_absolute_error": abs(float(reference.jepa_loss) - float(candidate.jepa_loss)),
        "source_state_max_absolute_error": float(
            (reference.source_states - candidate.source_states).abs().max()
        ),
        "target_state_max_absolute_error": float(
            (reference.target_states - candidate.target_states).abs().max()
        ),
        "logit_max_absolute_error": float((reference.logits - candidate.logits).abs().max()),
        "lora_gradient_cosine": float(F.cosine_similarity(
            reference_gradient.unsqueeze(0), candidate_gradient.unsqueeze(0)
        )),
        "lora_gradient_relative_l2_error": float(
            difference.norm() / reference_gradient.norm().clamp_min(1e-30)
        ),
    }
    result["passed"] = (
        result["source_state_max_absolute_error"] == 0.0
        and result["target_state_max_absolute_error"] == 0.0
        and result["native_loss_absolute_error"] <= 2e-5
        and result["mse_absolute_error"] == 0.0
        and result["lora_gradient_cosine"] >= 0.99999
        and result["lora_gradient_relative_l2_error"] <= 0.005
    )
    del model
    torch.cuda.empty_cache()
    return result


def benchmark(args) -> dict:
    if 16 % args.physical_batch:
        raise ValueError("physical batch must divide logical batch 16")
    tokenizer, model, method = load_model(args, optimized=args.optimized)
    rows = read_rows("uspto_mit_synthesis", path=args.train_manifest)
    reference, metadata = load_pcsf_reference_cache(
        args.reference_cache, rows, args.train_manifest,
    )
    reference = reference.cuda()
    collator = ReactionCollator(tokenizer, task="forward")
    workers = args.workers
    loader_kwargs = {
        "num_workers": workers,
        "persistent_workers": workers > 0,
    }
    if workers:
        loader_kwargs["prefetch_factor"] = 2
    loader = DataLoader(
        rows, batch_size=args.physical_batch, shuffle=True,
        generator=torch.Generator().manual_seed(533), collate_fn=collator,
        pin_memory=args.pin_memory, **loader_kwargs,
    )
    chunks = 16 // args.physical_batch
    limited = LimitedLoader(loader, args.updates * chunks)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=1e-4, betas=ADAM_BETAS, eps=ADAM_EPSILON,
        weight_decay=WEIGHT_DECAY, fused=args.fused_adamw,
    )
    total_steps = 4 * (len(rows) // 16)
    scheduler = get_scheduler(
        "cosine_with_min_lr", optimizer,
        num_warmup_steps=int(total_steps * WARMUP_RATIO),
        num_training_steps=total_steps,
        scheduler_specific_kwargs={"min_lr": MIN_LEARNING_RATE},
    )
    tracker = WandbTracker(
        TrackingContext("forward", "uspto_mit_synthesis", "benchmark", 533, 1.0, {}),
        run_name="pcsf-benchmark", enabled=False,
    )
    torch.cuda.reset_peak_memory_stats()
    optimizer.zero_grad(set_to_none=True)
    started = time.perf_counter()
    curves, _ = train_streaming_pcsf_epoch(
        model=model, method=method, loader=limited,
        optimizer=optimizer, scheduler=scheduler, epoch=1,
        logical_batch_size=16, physical_batch_size=args.physical_batch,
        actual_lambda=2.0, native_weight=1.0, pcsf_beta=args.beta,
        pcsf=PairCenterSpreadFloor(rho=args.rho),
        reference_centers=reference, jepa_ratio=0.5,
        non_embedding_parameters=model.num_parameters(exclude_embeddings=True),
        pin_memory=args.pin_memory, tracker=tracker, global_step=0,
        profile_phases=args.profile_phases,
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    examples = args.updates * 16
    result = {
        "configuration": {
            "physical_batch": args.physical_batch,
            "gradient_accumulation_equivalent": 16 // args.physical_batch,
            "logical_batch": 16,
            "attention": args.attention,
            "optimized_jepa_forward": args.optimized,
            "gradient_checkpointing": args.gradient_checkpointing,
            "fused_adamw": args.fused_adamw,
            "pin_memory": args.pin_memory,
            "workers": workers,
            "profile_phases": args.profile_phases,
            "reference": metadata,
        },
        "updates": args.updates,
        "elapsed_seconds": elapsed,
        "step_seconds": elapsed / args.updates,
        "examples_per_second": examples / elapsed,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "active_updates": sum(row["jepa_active"] for row in curves),
        "mean_native_loss": statistics.fmean(row["native_loss"] for row in curves),
        "mean_mse_loss": statistics.fmean(
            row["jepa_loss"] for row in curves if row["jepa_loss"] is not None
        ),
        "mean_pcsf_loss": statistics.fmean(
            row["pcsf_loss"] for row in curves if row["pcsf_loss"] is not None
        ),
        "phase_means": {
            key: statistics.fmean(row[key] for row in curves)
            for key in (
                "data_seconds", "pcsf_statistics_forward_seconds",
                "gradient_forward_backward_seconds", "optimizer_seconds",
            )
        },
        "projected_four_epoch_minutes": elapsed / args.updates * 320 / 60,
        "curves": curves,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--reference-cache", type=Path, required=True)
    parser.add_argument("--rho", type=float, required=True)
    parser.add_argument("--beta", type=float, required=True)
    parser.add_argument("--physical-batch", type=int, default=4)
    parser.add_argument("--updates", type=int, default=12)
    parser.add_argument("--attention", choices=("eager", "sdpa", "flash_attention_2"), default="sdpa")
    parser.add_argument("--optimized", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--fused-adamw", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--profile-phases", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--parity", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raise RuntimeError(
        "PCSF was removed from src/train.py; this file is retained only as the "
        "historical benchmark source. Use runs/pcsf/benchmark and report 06 "
        "for the frozen measurements."
    )


if __name__ == "__main__":
    main()
