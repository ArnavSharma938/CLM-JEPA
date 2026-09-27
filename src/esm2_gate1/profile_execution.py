from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .config import EVIDENCE_SEED, TrainConfig
from .modeling import configure_adaptation, load_model
from .training import ProteinDataset, make_loader


def _training_candidate(
    manifest: Path,
    backend: str,
    token_budget: int,
    workers: int,
    measured_batches: int,
    gradient_checkpointing: bool,
    compile_model: bool = False,
) -> dict:
    model = tokenizer = dataset = loader = optimizer = None
    try:
        model, tokenizer = load_model(attention_backend=backend)
        configure_adaptation(model, "ft")  # Profile the highest-memory Gate-1 condition.
        if gradient_checkpointing:
            model.gradient_checkpointing_enable()
        model.to("cuda").train()
        if compile_model:
            model = torch.compile(model, mode="reduce-overhead", dynamic=True)
        dataset = ProteinDataset(manifest, tokenizer)
        config = TrainConfig(
            mode="ft",
            seed=EVIDENCE_SEED,
            learning_rate=1e-6,
            train_manifest=manifest,
            validation_manifest=manifest,
            output_dir=Path("profile-only"),
            microbatch_tokens=token_budget,
            effective_residues_per_step=max(32768, token_budget),
            num_workers=workers,
            attention_backend=backend,
            save_checkpoints=False,
        )
        loader = make_loader(dataset, tokenizer, config, 0, True)
        iterator = iter(loader)
        parameters = [value for value in model.parameters() if value.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=1e-6, weight_decay=0.0, fused=True)
        torch.cuda.reset_peak_memory_stats()
        elapsed = residues = padded = nonpadding = sequences = 0
        # The untimed iteration initializes workers, kernels, gradients, and Adam moments.
        total_batches = measured_batches + 1
        for index in range(total_batches):
            fetch_start = time.perf_counter()
            batch = next(iterator)
            input_ids = batch["input_ids"].cuda(non_blocking=True)
            labels = batch["labels"].cuda(non_blocking=True)
            attention_mask = batch["attention_mask"].cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                loss = F.cross_entropy(logits.transpose(1, 2), labels, ignore_index=-100)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            torch.cuda.synchronize()
            if index:
                elapsed += time.perf_counter() - fetch_start
                residues += int(batch["nonpadding_residues"])
                sequences += len(batch["rows"])
                padded += int(input_ids.numel())
                nonpadding += sum(int(row["length"]) + 2 for row in batch["rows"])
        return {
            "passed": True,
            "backend": backend,
            "token_budget": token_budget,
            "workers": workers,
            "gradient_checkpointing": gradient_checkpointing,
            "compile_model": compile_model,
            "measured_batches": measured_batches,
            "residues_per_second": residues / elapsed,
            "sequences_per_second": sequences / elapsed,
            "padding_fraction": 1 - nonpadding / padded,
            "peak_vram_bytes": torch.cuda.max_memory_allocated(),
            "wall_seconds": elapsed,
        }
    except (RuntimeError, StopIteration) as error:
        if isinstance(error, RuntimeError) and "out of memory" not in str(error).lower():
            raise
        return {
            "passed": False,
            "backend": backend,
            "token_budget": token_budget,
            "workers": workers,
            "gradient_checkpointing": gradient_checkpointing,
            "compile_model": compile_model,
            "error": repr(error),
        }
    finally:
        model = tokenizer = dataset = loader = optimizer = None
        gc.collect()
        torch.cuda.empty_cache()


@torch.no_grad()
def _evaluation_candidate(
    manifest: Path,
    backend: str,
    token_budget: int,
    workers: int,
    measured_batches: int,
    compile_model: bool = False,
) -> dict:
    model = tokenizer = dataset = loader = None
    try:
        model, tokenizer = load_model(attention_backend=backend)
        configure_adaptation(model, "base")
        model.to("cuda").eval()
        if compile_model:
            model = torch.compile(model, mode="reduce-overhead", dynamic=True)
        dataset = ProteinDataset(manifest, tokenizer)
        config = TrainConfig(
            mode="base",
            seed=EVIDENCE_SEED,
            learning_rate=0.0,
            train_manifest=manifest,
            validation_manifest=manifest,
            output_dir=Path("profile-only"),
            microbatch_tokens=token_budget,
            effective_residues_per_step=max(32768, token_budget),
            num_workers=workers,
            attention_backend=backend,
            save_checkpoints=False,
        )
        loader = make_loader(dataset, tokenizer, config, 0, False)
        iterator = iter(loader)
        torch.cuda.reset_peak_memory_stats()
        elapsed = residues = padded = nonpadding = sequences = 0
        for index in range(measured_batches + 1):
            fetch_start = time.perf_counter()
            batch = next(iterator)
            input_ids = batch["input_ids"].cuda(non_blocking=True)
            labels = batch["labels"].cuda(non_blocking=True)
            attention_mask = batch["attention_mask"].cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            F.cross_entropy(logits.float().transpose(1, 2), labels, ignore_index=-100)
            torch.cuda.synchronize()
            if index:
                elapsed += time.perf_counter() - fetch_start
                residues += int(batch["nonpadding_residues"])
                sequences += len(batch["rows"])
                padded += int(input_ids.numel())
                nonpadding += sum(int(row["length"]) + 2 for row in batch["rows"])
        return {
            "passed": True,
            "backend": backend,
            "token_budget": token_budget,
            "workers": workers,
            "compile_model": compile_model,
            "measured_batches": measured_batches,
            "residues_per_second": residues / elapsed,
            "sequences_per_second": sequences / elapsed,
            "padding_fraction": 1 - nonpadding / padded,
            "peak_vram_bytes": torch.cuda.max_memory_allocated(),
            "wall_seconds": elapsed,
        }
    except (RuntimeError, StopIteration) as error:
        if isinstance(error, RuntimeError) and "out of memory" not in str(error).lower():
            raise
        return {
            "passed": False,
            "backend": backend,
            "token_budget": token_budget,
            "workers": workers,
            "compile_model": compile_model,
            "error": repr(error),
        }
    finally:
        model = tokenizer = dataset = loader = None
        gc.collect()
        torch.cuda.empty_cache()


def profile(
    manifest: Path,
    output: Path,
    backend: str,
    train_budgets: tuple[int, ...],
    eval_budgets: tuple[int, ...],
    workers: tuple[int, ...],
    measured_batches: int,
    gradient_checkpointing_options: tuple[bool, ...],
    comparison_backends: tuple[str, ...],
) -> dict:
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("execution profiling requires a BF16 CUDA GPU")
    training = [
        _training_candidate(
            manifest, candidate_backend, budget, count, measured_batches,
            gradient_checkpointing,
        )
        for candidate_backend in comparison_backends
        for gradient_checkpointing in gradient_checkpointing_options
        for count in workers
        for budget in train_budgets
    ]
    evaluation = [
        _evaluation_candidate(
            manifest, candidate_backend, budget, count, measured_batches
        )
        for candidate_backend in comparison_backends
        for count in workers
        for budget in eval_budgets
    ]
    train_passing = [item for item in training if item["passed"]]
    eval_passing = [item for item in evaluation if item["passed"]]
    if not train_passing or not eval_passing:
        raise RuntimeError("no complete training/evaluation profile candidate passed")
    vram_limit = int(torch.cuda.get_device_properties(0).total_memory * 0.90)

    def select_with_headroom(candidates: list[dict]) -> dict:
        safe = [item for item in candidates if item["peak_vram_bytes"] <= vram_limit]
        return max(safe or candidates, key=lambda item: item["residues_per_second"])

    selected_training = select_with_headroom(train_passing)
    selected_backend = selected_training["backend"]
    selected_evaluation = select_with_headroom(
        [item for item in eval_passing if item["backend"] == selected_backend]
    )
    backend_path = output.parent / "backend_benchmark.json"
    compile_eligible = False
    if backend_path.exists():
        backend_profile = json.loads(backend_path.read_text(encoding="utf-8"))
        compile_eligible = bool(backend_profile.get("torch_compile_benchmark", {}).get("passed"))
    if compile_eligible and selected_backend == backend:
        compiled_training = _training_candidate(
            manifest, selected_backend, selected_training["token_budget"],
            selected_training["workers"], measured_batches,
            selected_training["gradient_checkpointing"], True,
        )
        training.append(compiled_training)
        if (
            compiled_training["passed"]
            and compiled_training["peak_vram_bytes"] <= vram_limit
            and compiled_training["residues_per_second"]
            > selected_training["residues_per_second"] * 1.05
        ):
            selected_training = compiled_training
        compiled_evaluation = _evaluation_candidate(
            manifest, selected_backend, selected_evaluation["token_budget"],
            selected_evaluation["workers"], measured_batches, True,
        )
        evaluation.append(compiled_evaluation)
        if (
            compiled_evaluation["passed"]
            and compiled_evaluation["peak_vram_bytes"] <= vram_limit
            and compiled_evaluation["residues_per_second"]
            > selected_evaluation["residues_per_second"] * 1.05
        ):
            selected_evaluation = compiled_evaluation
    payload = {
        "backend": selected_backend,
        "profiled_backends": list(comparison_backends),
        "manifest": str(manifest),
        "manifest_sha256": json.loads(
            manifest.with_suffix(manifest.suffix + ".metadata.json").read_text(encoding="utf-8")
        )["manifest_sha256"],
        "training": training,
        "evaluation": evaluation,
        "selected_training": selected_training,
        "selected_evaluation": selected_evaluation,
        "selection_rule": (
            "highest end-to-end residue throughput among candidates retaining 10% VRAM headroom; "
            "torch.compile requires at least 5% improvement"
        ),
        "vram_headroom_fraction": 0.10,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--attention-backend", required=True)
    parser.add_argument(
        "--compare-attention-backends", nargs="+", default=("sdpa",)
    )
    parser.add_argument("--train-budgets", nargs="+", type=int, default=(8192, 12288, 16384))
    parser.add_argument("--eval-budgets", nargs="+", type=int, default=(16384, 24576, 32768))
    parser.add_argument("--workers", nargs="+", type=int, default=(2,))
    parser.add_argument("--measured-batches", type=int, default=6)
    parser.add_argument(
        "--gradient-checkpointing", choices=("both", "on", "off"), default="off"
    )
    args = parser.parse_args()
    checkpointing = {
        "both": (False, True), "on": (True,), "off": (False,)
    }[args.gradient_checkpointing]
    backends = tuple(dict.fromkeys((args.attention_backend, *args.compare_attention_backends)))
    profile(
        args.manifest,
        args.output,
        args.attention_backend,
        tuple(args.train_budgets),
        tuple(args.eval_budgets),
        tuple(args.workers),
        args.measured_batches,
        checkpointing,
        backends,
    )


if __name__ == "__main__":
    main()
