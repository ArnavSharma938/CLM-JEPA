from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from .prepare_data import sha256_file
from .training import train


INITIAL_CANDIDATES = (1e-6, 3e-6, 1e-5, 3e-5, 1e-4)


def best_two_consecutive(history: list[dict]) -> float:
    values = [float(row["peptide_nll"]) for row in history if row.get("epoch", 0) > 0]
    if len(values) < 2:
        return math.inf
    return min((left + right) / 2 for left, right in zip(values, values[1:]))


def still_improving(history: list[dict]) -> bool:
    values = [float(row["peptide_nll"]) for row in history if row.get("epoch", 0) > 0]
    if len(values) < 4:
        return False
    return (values[-1] == min(values) and values[-2] - values[-1] >= 0.001) or (
        values[-3] - values[-1] >= 0.002
    )


def candidate_summary(directory: Path) -> dict:
    history = json.loads((directory / "history.json").read_text(encoding="utf-8"))
    execution = json.loads((directory / "execution.json").read_text(encoding="utf-8"))
    values = [float(row["peptide_nll"]) for row in history if row.get("epoch", 0) > 0]
    best_index = min(range(len(values)), key=values.__getitem__) if values else None
    return {
        "trajectory": history,
        "minimum_validation_nll": min(values) if values else None,
        "terminal_validation_nll": values[-1] if values else None,
        "best_two_consecutive_check_mean_nll": best_two_consecutive(history),
        "deterioration_from_minimum": values[-1] - min(values) if values else None,
        "best_check_epoch": next((
            row["epoch"] for row in history
            if row.get("epoch", 0) > 0 and row["peptide_nll"] == values[best_index]
        ), None) if values else None,
        "still_improving": still_improving(history),
        "numerical_stability": execution["numerical_stability"],
        "maximum_gradient_norm_before_clip": execution["maximum_gradient_norm_before_clip"],
        "optimizer_steps": execution["optimizer_steps"],
        "epochs_completed": execution["epochs_completed"],
    }


def run_hpo(
    *, manifest_dir: Path, cache_path: Path, base_evaluation_path: Path, output_dir: Path,
    token_budget: int = 12_288, supervised_budget: int = 8_192, workers: int = 2,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    hpo_manifest = manifest_dir / "hpo_4096.jsonl"
    validation_manifest = manifest_dir / "validation.jsonl"
    baseline_summary = json.loads(base_evaluation_path.read_text(encoding="utf-8"))["validation_true"]
    if baseline_summary["manifest_sha256"] != sha256_file(validation_manifest):
        raise RuntimeError("Base HPO validation was not computed on the frozen validation manifest")
    baseline = {
        key: baseline_summary[key]
        for key in ("peptide_nll", "loss_sum", "supervised_tokens", "pairs")
    } | {"epoch": 0.0, "optimizer_step": 0, "learning_rate": 0.0}
    outcomes: dict[float, dict] = {}

    def run_one(lr: float, *, resume: Path | None = None, maximum_epochs: int = 4) -> dict:
        directory = output_dir / f"lr_{lr:.0e}"
        train(
            train_manifest=hpo_manifest,
            validation_manifest=validation_manifest,
            cache_path=cache_path,
            output_dir=directory,
            peak_lr=lr,
            seed=7,
            token_budget=token_budget,
            supervised_budget=supervised_budget,
            maximum_epochs=maximum_epochs,
            schedule_epochs=4,
            workers=workers,
            attention_backend="sdpa",
            resume=resume,
            initial_validation=baseline if resume is None else None,
        )
        outcomes[lr] = candidate_summary(directory)
        return outcomes[lr]

    for lr in INITIAL_CANDIDATES:
        try:
            run_one(lr)
        except (FloatingPointError, torch.cuda.OutOfMemoryError) as error:
            outcomes[lr] = {"numerical_stability": "failed", "failure": repr(error)}
            torch.cuda.empty_cache()

    stable = [lr for lr in INITIAL_CANDIDATES if outcomes[lr].get("numerical_stability") == "stable"]
    finalists = sorted(stable, key=lambda lr: (outcomes[lr]["best_two_consecutive_check_mean_nll"], lr))[:2]
    if any(outcomes[lr]["still_improving"] for lr in finalists):
        for lr in finalists:
            run_one(lr, resume=output_dir / f"lr_{lr:.0e}" / "latest.pt", maximum_epochs=6)

    def choose(candidates: list[float]) -> float:
        scores = {lr: outcomes[lr]["best_two_consecutive_check_mean_nll"] for lr in candidates}
        best_score = min(scores.values())
        practical = [lr for lr in candidates if scores[lr] <= best_score + 0.002]
        selected = min(practical)
        # If a higher-rate winner has materially deteriorated, prefer a stable lower-rate near-tie.
        for lr in sorted(candidates):
            if lr < selected and scores[lr] <= scores[selected] + 0.002:
                selected = lr
        return selected

    if not stable:
        raise RuntimeError("all initial DTFT LR candidates failed numerically")
    selected_lr = choose([lr for lr in finalists if outcomes[lr].get("numerical_stability") == "stable"] or stable)
    expansion = None
    edge = INITIAL_CANDIDATES[0] if selected_lr == INITIAL_CANDIDATES[0] else (
        INITIAL_CANDIDATES[-1] if selected_lr == INITIAL_CANDIDATES[-1] else None
    )
    if edge is not None:
        adjacent = INITIAL_CANDIDATES[1] if edge == INITIAL_CANDIDATES[0] else INITIAL_CANDIDATES[-2]
        if outcomes[adjacent]["best_two_consecutive_check_mean_nll"] - outcomes[edge]["best_two_consecutive_check_mean_nll"] > 0.002:
            expansion = 3e-7 if edge == INITIAL_CANDIDATES[0] else 3e-4
            try:
                expansion_summary = run_one(expansion)
                old_finalists = list(finalists)
                comparison = sorted(old_finalists + [expansion], key=lambda lr: outcomes[lr]["best_two_consecutive_check_mean_nll"])
                if expansion in comparison[:2] and expansion_summary["still_improving"]:
                    run_one(expansion, resume=output_dir / f"lr_{expansion:.0e}" / "latest.pt", maximum_epochs=6)
                finalists = comparison[:2]
                selected_lr = choose(finalists)
            except (FloatingPointError, torch.cuda.OutOfMemoryError) as error:
                outcomes[expansion] = {"numerical_stability": "failed", "failure": repr(error)}

    report = {
        "seed": 7,
        "evidence_seed": 11,
        "manifest_sha256": sha256_file(hpo_manifest),
        "validation_manifest_sha256": sha256_file(validation_manifest),
        "base_evaluation_sha256": sha256_file(base_evaluation_path),
        "baseline_validation": baseline,
        "initial_candidates": list(INITIAL_CANDIDATES),
        "candidate_results": {f"{lr:.0e}": result for lr, result in sorted(outcomes.items())},
        "finalists": [f"{lr:.0e}" for lr in finalists],
        "boundary_expansion": f"{expansion:.0e}" if expansion is not None else None,
        "selected_peak_learning_rate": selected_lr,
        "selection_rule": "best two-consecutive-check mean; <=0.002 practical tie resolved to lower LR; deteriorated higher LR yields to near-tied stable lower LR",
        "input_token_budget": token_budget,
        "supervised_peptide_token_budget": supervised_budget,
    }
    (output_dir / "hpo_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Protocol-defined Propedia DTFT LR selection")
    parser.add_argument("--manifest-dir", type=Path, default=Path("manifests"))
    parser.add_argument("--cache", type=Path, default=Path("token_cache.pt"))
    parser.add_argument("--base-evaluation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--token-budget", type=int, default=12_288)
    parser.add_argument("--supervised-budget", type=int, default=8_192)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    print(json.dumps(run_hpo(
        manifest_dir=args.manifest_dir, cache_path=args.cache, base_evaluation_path=args.base_evaluation,
        output_dir=args.output_dir,
        token_budget=args.token_budget, supervised_budget=args.supervised_budget, workers=args.workers,
    ), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
