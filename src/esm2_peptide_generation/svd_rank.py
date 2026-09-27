from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from esm2_gate1.modeling import configure_adaptation, dense_targets

from .analyze_conditioning import analyze, read_jsonl
from .config import MODEL_ID, MODEL_REVISION
from .evaluate import decoy_examples, write_jsonl
from .prepare_data import sha256_file
from .token_cache import load_token_cache
from .batching import GenerationDataset
from .training import evaluate_examples, load_base_model


def construct_rank64_factors(checkpoint_path: Path, output_path: Path) -> dict:
    started = time.perf_counter()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    trained = checkpoint["trainable_state"]
    model, _ = load_base_model("sdpa")
    report = configure_adaptation(model, "base")
    if report["target_matrix_count"] != 198:
        raise RuntimeError("SVD base model target set is not exactly 198 matrices")
    targets = dense_targets(model)
    factors: dict[str, dict[str, torch.Tensor]] = {}
    matrix_report = []
    torch.set_num_threads(6)
    for target in targets:
        name = f"{target.name}.weight"
        if name not in trained:
            raise RuntimeError(f"DTFT checkpoint is missing target matrix {name}")
        base_weight = target.module.weight.detach().cpu().float()
        delta = trained[name].detach().cpu().float() - base_weight
        u, singular, vh = torch.linalg.svd(delta, full_matrices=False)
        factors[target.name] = {
            "u64": u[:, :64].contiguous(),
            "s64": singular[:64].contiguous(),
            "vh64": vh[:64, :].contiguous(),
        }
        matrix_report.append({
            "name": target.name,
            "shape": list(delta.shape),
            "delta_frobenius_norm": float(torch.linalg.vector_norm(delta)),
            "rank8_captured_squared_singular_value_fraction": float(singular[:8].square().sum() / singular.square().sum()),
            "rank64_captured_squared_singular_value_fraction": float(singular[:64].square().sum() / singular.square().sum()),
        })
    payload = {
        "format_version": 1,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "algorithm": "torch.linalg.svd(DeltaW, full_matrices=False); retain U[:, :64], S[:64], Vh[:64, :] per target matrix",
        "matrix_count": len(factors),
        "rank_limit": 64,
        "factors": factors,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    del model, checkpoint, trained, factors
    return {
        "factor_file": str(output_path),
        "factor_file_sha256": sha256_file(output_path),
        "checkpoint_sha256": payload["checkpoint_sha256"],
        "matrix_count": len(matrix_report),
        "algorithm": payload["algorithm"],
        "matrices": matrix_report,
        "wall_seconds": time.perf_counter() - started,
    }


def evaluate_rank(
    rank: int, *, factors_path: Path, dtft_checkpoint: Path, manifest_dir: Path,
    cache_path: Path, output_dir: Path, token_budget: int = 16_384, workers: int = 2,
) -> dict:
    if rank not in {8, 64}:
        raise ValueError("diagnostic rank must be 8 or 64")
    factors_payload = torch.load(factors_path, map_location="cpu", weights_only=False)
    # The source DTFT checkpoint is only used for lineage validation; no fitted update is applied.
    if factors_payload["checkpoint_sha256"] != sha256_file(dtft_checkpoint):
        raise RuntimeError("SVD factors were made from a different DTFT checkpoint")
    cache = load_token_cache(cache_path)
    model, _ = load_base_model("sdpa")
    configure_adaptation(model, "base")
    targets = {target.name: target for target in dense_targets(model)}
    if set(targets) != set(factors_payload["factors"]):
        raise RuntimeError("SVD factor matrix names do not match the pinned Base model")
    with torch.no_grad():
        for name, target in targets.items():
            factor = factors_payload["factors"][name]
            update = (factor["u64"][:, :rank] * factor["s64"][:rank]) @ factor["vh64"][:rank, :]
            target.module.weight.add_(update)
            del update
    model.to("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    test_data = GenerationDataset(cache, manifest_dir / "test.jsonl")
    true_examples = [
        {"row": item["row"], "receptor_tokens": item["receptor_tokens"], "peptide_tokens": item["peptide_tokens"]}
        for item in test_data.items
    ]
    true = evaluate_examples(model, true_examples, cache, token_budget=token_budget, workers=workers)
    decoys = evaluate_examples(
        model, decoy_examples(test_data, manifest_dir / "test_decoys.jsonl", cache),
        cache, token_budget=token_budget, workers=workers,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "test_true.jsonl", true.pop("per_example"))
    write_jsonl(output_dir / "test_decoy.jsonl", decoys.pop("per_example"))
    metadata = {
        "model": f"svd_r{rank}",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "rank_per_matrix": rank,
        "checkpoint_sha256": sha256_file(dtft_checkpoint),
        "svd_factor_sha256": sha256_file(factors_path),
        "test_manifest_sha256": sha256_file(manifest_dir / "test.jsonl"),
        "decoy_manifest_sha256": sha256_file(manifest_dir / "test_decoys.jsonl"),
        "attention_backend": "sdpa",
        "trained_after_truncation": False,
    }
    summary = {"true": {**true, **metadata}, "decoy": {**decoys, **metadata}}
    (output_dir / "evaluation_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    del model, targets, factors_payload
    torch.cuda.empty_cache()
    return summary


def summarize_existing_rank_evaluations(
    *, base_eval_dir: Path, stage4: dict, output_dir: Path,
    factor_provenance_sha256: str | None = None, factors_sha256: str | None = None,
) -> dict:
    """Recompute rank retention from frozen score files using cluster-equal estimands."""
    if not stage4["stage4_passed"]:
        raise RuntimeError("rank truncation is prohibited because Stage 4 failed")
    rank_reports = {}
    for rank in (8, 64):
        comparison = analyze(base_eval_dir, output_dir / f"svd_r{rank}")
        c_rank = comparison["C_DTFT"]
        rank_reports[f"r{rank}"] = {
            "true_pair_peptide_nll": comparison["dtft_true_peptide_nll"],
            "adaptation_gain_equal_cluster": comparison["A_DTFT"],
            "specificity_increment_over_Base": c_rank,
            "conditional_signal_equal_cluster": c_rank,
            "R_rank_adaptation_gain_retained": (
                comparison["A_DTFT"] / stage4["A_DTFT"]
                if stage4["A_DTFT"] != 0 else None
            ),
            "Q_rank_conditional_signal_retained": (
                c_rank / stage4["C_DTFT"] if stage4["C_DTFT"] != 0 else None
            ),
        }
    r64 = rank_reports["r64"]["R_rank_adaptation_gain_retained"]
    q64 = rank_reports["r64"]["Q_rank_conditional_signal_retained"]
    low_rank_easy = bool(r64 is not None and q64 is not None and r64 >= 0.90 and q64 >= 0.90)
    result = {
        "primary_estimand": "equal-receptor-cluster-weighted mean peptide-NLL gain",
        "rank8_rank64": rank_reports,
        "rank64_preserves_at_least_90_percent_of_both": low_rank_easy,
        "classification": "LOW-RANK-EASY" if low_rank_easy else "passes_rank64_screen",
        "factor_provenance_sha256": factor_provenance_sha256,
        "factors_sha256": factors_sha256,
        "svd_method": "best truncated SVD independently on each of the 198 DTFT delta matrices; no optimization after truncation",
        "retention_estimand": "rank-specific equal-cluster gain/signal divided by Stage-4 equal-cluster gain/signal",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "rank_screen_report.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def run_svd_screen(
    *, checkpoint_path: Path, manifest_dir: Path, cache_path: Path,
    base_eval_dir: Path, dtft_eval_dir: Path, stage4_path: Path, output_dir: Path,
    token_budget: int = 16_384, workers: int = 2,
) -> dict:
    stage4 = json.loads(stage4_path.read_text(encoding="utf-8"))
    if not stage4["stage4_passed"]:
        raise RuntimeError("rank truncation is prohibited because Stage 4 failed")
    factors_path = output_dir / "rank64_factors.pt"
    factor_report_path = output_dir / "svd_factor_provenance.json"
    factor_report = construct_rank64_factors(checkpoint_path, factors_path)
    factor_report_path.write_text(json.dumps(factor_report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for rank in (8, 64):
        rank_dir = output_dir / f"svd_r{rank}"
        evaluate_rank(
            rank, factors_path=factors_path, dtft_checkpoint=checkpoint_path,
            manifest_dir=manifest_dir, cache_path=cache_path,
            output_dir=rank_dir, token_budget=token_budget, workers=workers,
        )
    return summarize_existing_rank_evaluations(
        base_eval_dir=base_eval_dir, stage4=stage4, output_dir=output_dir,
        factor_provenance_sha256=sha256_file(factor_report_path),
        factors_sha256=sha256_file(factors_path),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-hoc rank-8/rank-64 SVD screen")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, default=Path("manifests"))
    parser.add_argument("--cache", type=Path, default=Path("token_cache.pt"))
    parser.add_argument("--base-eval-dir", type=Path, required=True)
    parser.add_argument("--dtft-eval-dir", type=Path, required=True)
    parser.add_argument("--stage4", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--token-budget", type=int, default=16_384)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    print(json.dumps(run_svd_screen(
        checkpoint_path=args.checkpoint, manifest_dir=args.manifest_dir, cache_path=args.cache,
        base_eval_dir=args.base_eval_dir, dtft_eval_dir=args.dtft_eval_dir,
        stage4_path=args.stage4, output_dir=args.output_dir,
        token_budget=args.token_budget, workers=args.workers,
    ), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
