from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from .batching import GenerationCollator, GenerationDataset
from .prepare_data import sha256_file
from .training import load_base_model, seed_everything, train
from .token_cache import load_token_cache
from esm2_gate1.modeling import configure_adaptation


def _first_batch(cache: dict, manifest: Path, token_budget: int) -> dict:
    data = GenerationDataset(cache, manifest)
    # A reproducible HPO-subset batch, using the same length bucketing and collator as training.
    from .training import make_loader
    loader, _ = make_loader(data, token_budget, seed=7, epoch=0, shuffle=True, workers=0)
    return next(iter(loader))


def _gradient_snapshot(mode: str | None, cache: dict, batch: dict) -> tuple[float, dict[str, torch.Tensor], float]:
    seed_everything(314159)
    model, _ = load_base_model("sdpa")
    report = configure_adaptation(model, "dtft")
    if report["trainable_parameters"] != 648_806_400 or report["target_matrix_count"] != 198:
        raise RuntimeError("profile model does not have the exact DTFT target matrices")
    model.to("cuda")
    if mode:
        model = torch.compile(model, mode=mode, dynamic=True)
    torch.cuda.reset_peak_memory_stats()
    model.train()
    seed_everything(271828)
    ids = batch["input_ids"].to("cuda", non_blocking=True)
    labels = batch["labels"].to("cuda", non_blocking=True)
    mask = batch["attention_mask"].to("cuda", non_blocking=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(input_ids=ids, attention_mask=mask).logits
        selected = labels.ne(-100)
        loss = torch.nn.functional.cross_entropy(logits[selected].float(), labels[selected], reduction="sum")
    loss.backward()
    denominator = int(batch["supervised_tokens"])
    gradients = {
        name: parameter.grad.detach().float().div_(denominator).cpu().clone()
        for name, parameter in getattr(model, "_orig_mod", model).named_parameters()
        if parameter.requires_grad and parameter.grad is not None
    }
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
    value = float((loss.detach() / denominator).cpu())
    del model
    torch.cuda.empty_cache()
    return value, gradients, peak


def _compare_gradients(reference: dict[str, torch.Tensor], candidate: dict[str, torch.Tensor]) -> dict:
    if set(reference) != set(candidate) or len(reference) != 198:
        return {"passed": False, "reason": "trainable gradient-name mismatch", "matrices": len(candidate)}
    worst_relative = 0.0
    passed = True
    for name, left in reference.items():
        right = candidate[name]
        if not torch.isfinite(right).all():
            return {"passed": False, "reason": f"non-finite gradient: {name}"}
        difference = (left - right).abs()
        scale = left.abs().maximum(right.abs()).clamp_min(1e-6)
        worst_relative = max(worst_relative, float((difference / scale).max()))
        passed = passed and torch.allclose(left, right, rtol=0.02, atol=5e-3)
    return {"passed": passed, "matrices": len(reference), "worst_elementwise_relative_error": worst_relative}


def _compile_modes() -> list[str | None]:
    modes: list[str | None] = [None, "default"]
    try:
        torch._dynamo.list_backends()
        modes.append("max-autotune-no-cudagraphs")
    except Exception:
        pass
    return modes


def profile(
    *, manifest_dir: Path, cache_path: Path, base_evaluation_path: Path, output_dir: Path,
    token_budget: int = 12_288, supervised_budget: int = 8_192,
    workers: int = 2, optimizer_steps: int = 12,
) -> dict:
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("L40 profiling requires CUDA BF16 support")
    output_dir.mkdir(parents=True, exist_ok=True)
    cache = load_token_cache(cache_path)
    hpo_manifest = manifest_dir / "hpo_4096.jsonl"
    validation_manifest = manifest_dir / "validation.jsonl"
    base_evaluation = json.loads(base_evaluation_path.read_text(encoding="utf-8"))
    baseline = base_evaluation["validation_true"]
    if baseline["manifest_sha256"] != sha256_file(validation_manifest):
        raise RuntimeError("profile baseline does not match the frozen validation manifest")
    batch = _first_batch(cache, hpo_manifest, token_budget)
    reference_loss, reference_gradients, baseline_peak = _gradient_snapshot(None, cache, batch)
    profile_results: dict[str, dict] = {}
    for mode in _compile_modes():
        label = mode or "uncompiled_sdpa"
        parity = {"passed": True, "loss": reference_loss, "peak_vram_gib": baseline_peak}
        if mode:
            try:
                value, gradients, peak = _gradient_snapshot(mode, cache, batch)
                parity = {
                    **_compare_gradients(reference_gradients, gradients),
                    "loss": value,
                    "loss_abs_difference": abs(value - reference_loss),
                    "peak_vram_gib": peak,
                }
                parity["passed"] = bool(parity["passed"] and abs(value - reference_loss) <= 0.01)
                del gradients
            except Exception as error:
                parity = {"passed": False, "failure": repr(error)}
                torch.cuda.empty_cache()

        result = {"gradient_parity": parity}
        if not parity.get("passed"):
            result["status"] = "rejected_parity_or_numerical_failure"
            profile_results[label] = result
            continue
        started = time.perf_counter()
        dynamo_counters = None
        try:
            torch._dynamo.utils.counters.clear()
            execution = train(
                train_manifest=hpo_manifest,
                validation_manifest=validation_manifest,
                cache_path=cache_path,
                output_dir=output_dir / label,
                peak_lr=1e-5,
                seed=7,
                token_budget=token_budget,
                supervised_budget=supervised_budget,
                maximum_epochs=1,
                schedule_epochs=10,
                workers=workers,
                attention_backend="sdpa",
                compile_mode=mode,
                maximum_optimizer_steps=optimizer_steps,
                initial_validation=baseline,
                validation_every_half_epoch=False,
            )
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            dynamo_counters = {
                key: dict(value) for key, value in torch._dynamo.utils.counters.items()
            }
            profile_results[label] = {
                **result,
                "status": "completed",
                "execution": execution,
                "wall_seconds_including_model_compile": elapsed,
                "measured_input_tokens_per_second_including_compile": execution["real_input_tokens"] / max(elapsed, 1e-9),
                "estimated_peak_vram_gib": execution["peak_vram_bytes"] / (1024 ** 3),
                "training_optimizer_steps": execution["optimizer_steps"],
                "dynamo_counters": dynamo_counters,
                "complete_profile_artifacts": str(output_dir / label),
            }
        except Exception as error:
            profile_results[label] = {
                **result,
                "status": "failed",
                "failure": repr(error),
                "wall_seconds_including_model_compile": time.perf_counter() - started,
                "dynamo_counters": {
                    key: dict(value) for key, value in torch._dynamo.utils.counters.items()
                },
            }
            torch.cuda.empty_cache()

    dataset = GenerationDataset(cache, manifest_dir / "train.jsonl")
    hpo_data = GenerationDataset(cache, hpo_manifest)
    lengths = sorted(dataset.lengths)
    peptide_lengths = [len(item["peptide_tokens"]) for item in dataset.items]
    baseline_profile = profile_results.get("uncompiled_sdpa", {})
    if baseline_profile.get("status") != "completed":
        raise RuntimeError("uncompiled SDPA baseline failed; re-profile with a smaller padded-token budget")
    full_epoch_tokens = sum(dataset.lengths)
    baseline_execution = baseline_profile["execution"]
    baseline_rate = float(baseline_execution["input_tokens_per_second"])
    baseline_overhead = max(
        0.0,
        float(baseline_profile["wall_seconds_including_model_compile"])
        - float(baseline_execution["training_loop_seconds"]),
    )
    ten_epoch_baseline_seconds = baseline_overhead + 10 * full_epoch_tokens / max(baseline_rate, 1e-9)
    memory_limit_gib = 0.9 * torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    baseline_peak_gib = max(
        float(baseline_profile.get("estimated_peak_vram_gib", 0.0)),
        float(baseline_profile["gradient_parity"].get("peak_vram_gib", 0.0)),
    )
    baseline_memory_safe = baseline_peak_gib <= memory_limit_gib
    compile_decisions = {}
    compile_candidates = []
    for label, item in profile_results.items():
        if label == "uncompiled_sdpa" or item.get("status") != "completed":
            continue
        execution = item["execution"]
        rate = float(execution["input_tokens_per_second"])
        overhead = max(
            0.0,
            float(item["wall_seconds_including_model_compile"])
            - float(execution["training_loop_seconds"]),
        )
        projected_total = overhead + 10 * full_epoch_tokens / max(rate, 1e-9)
        effective_rate = 10 * full_epoch_tokens / max(projected_total, 1e-9)
        counters = item.get("dynamo_counters", {})
        graph_breaks = sum(counters.get("graph_break", {}).values())
        peak = max(
            float(item.get("estimated_peak_vram_gib", 0.0)),
            float(item["gradient_parity"].get("peak_vram_gib", 0.0)),
        )
        speedup = effective_rate / max(10 * full_epoch_tokens / ten_epoch_baseline_seconds, 1e-9) - 1.0
        eligible = bool(
            item["gradient_parity"].get("passed")
            and graph_breaks == 0
            and peak <= memory_limit_gib
            and speedup >= 0.05
        )
        compile_decisions[label] = {
            "extra_setup_or_compile_seconds": overhead,
            "projected_10_epoch_speedup_after_amortization": speedup,
            "effective_input_tokens_per_second": effective_rate,
            "graph_break_count": graph_breaks,
            "peak_vram_gib_including_parity": peak,
            "eligible": eligible,
        }
        if eligible:
            compile_candidates.append((effective_rate, label))
    report = {
        "hardware": torch.cuda.get_device_name(0),
        "gpu_total_memory_gib": torch.cuda.get_device_properties(0).total_memory / (1024 ** 3),
        "train_pairs": len(dataset),
        "hpo_pairs": len(hpo_data),
        "mean_total_tokenized_length": sum(lengths) / len(lengths),
        "median_total_tokenized_length": lengths[len(lengths) // 2],
        "mean_supervised_peptide_tokens_per_example": sum(peptide_lengths) / len(peptide_lengths),
        "median_supervised_peptide_tokens_per_example": sorted(peptide_lengths)[len(peptide_lengths) // 2],
        "validation_baseline_nll": baseline["peptide_nll"],
        "train_manifest_sha256": sha256_file(manifest_dir / "train.jsonl"),
        "hpo_manifest_sha256": sha256_file(hpo_manifest),
        "validation_manifest_sha256": sha256_file(validation_manifest),
        "base_evaluation_sha256": sha256_file(base_evaluation_path),
        "token_cache_sha256": sha256_file(cache_path),
        "selected_microbatch_padded_token_budget": token_budget,
        "memory_headroom_limit_gib": memory_limit_gib,
        "baseline_peak_vram_gib_including_parity": baseline_peak_gib,
        "baseline_keeps_10_percent_vram_headroom": baseline_memory_safe,
        "selected_supervised_token_optimizer_step_budget": supervised_budget,
        "workers": workers,
        "profile_optimizer_steps_per_mode": optimizer_steps,
        "profile_results": profile_results,
        "uncompiled_projected_10_epoch_seconds": ten_epoch_baseline_seconds,
        "compile_decisions": compile_decisions,
        "selected_execution_mode": (
            max(compile_candidates)[1] if compile_candidates else "uncompiled_sdpa"
        ) if baseline_memory_safe else "reprofile_with_lower_microbatch_budget",
        "selected_microbatch_budget_is_safe": baseline_memory_safe,
        "selection_rule": "select compile only if gradient parity passes, graph behavior is stable, peak VRAM <=90% of total, and projected end-to-end throughput after amortizing compile overhead over 10 epochs improves >=5%; otherwise use uncompiled SDPA",
    }
    report_path = output_dir / "l40_profile_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Real Propedia DTFT L40 execution profile")
    parser.add_argument("--manifest-dir", type=Path, default=Path("manifests"))
    parser.add_argument("--cache", type=Path, default=Path("token_cache.pt"))
    parser.add_argument("--base-evaluation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--token-budget", type=int, default=12_288)
    parser.add_argument("--supervised-budget", type=int, default=8_192)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--optimizer-steps", type=int, default=12)
    args = parser.parse_args()
    print(json.dumps(profile(
        manifest_dir=args.manifest_dir, cache_path=args.cache,
        base_evaluation_path=args.base_evaluation, output_dir=args.output_dir,
        token_budget=args.token_budget, supervised_budget=args.supervised_budget,
        workers=args.workers, optimizer_steps=args.optimizer_steps,
    ), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
