from __future__ import annotations

import argparse
import json
from pathlib import Path

from .prepare_data import sha256_file
from .training import train


def still_improving(history: list[dict]) -> bool:
    values = [float(row["peptide_nll"]) for row in history if row.get("epoch", 0) > 0]
    if len(values) < 8:
        return False
    newest_best_index = min(range(len(values)), key=values.__getitem__)
    previous_best = min(values[:newest_best_index], default=float("inf"))
    condition_one = newest_best_index >= len(values) - 2 and values[newest_best_index] <= previous_best - 0.001
    condition_two = sum(values[-4:]) / 4 <= sum(values[-8:-4]) / 4 - 0.001
    return condition_one or condition_two


def run_evidence(
    *, manifest_dir: Path, cache_path: Path, hpo_report_path: Path,
    base_evaluation_path: Path, output_dir: Path,
    token_budget: int = 8_192, supervised_budget: int = 8_192, workers: int = 2,
) -> dict:
    hpo = json.loads(hpo_report_path.read_text(encoding="utf-8"))
    base = json.loads(base_evaluation_path.read_text(encoding="utf-8"))
    initial_validation = base["validation_true"]
    if initial_validation["manifest_sha256"] != sha256_file(manifest_dir / "validation.jsonl"):
        raise RuntimeError("Base validation was not evaluated on the frozen validation manifest")
    selected_lr = float(hpo["selected_peak_learning_rate"])
    train_manifest = manifest_dir / "train.jsonl"
    validation_manifest = manifest_dir / "validation.jsonl"
    executions = []
    result = train(
        train_manifest=train_manifest,
        validation_manifest=validation_manifest,
        cache_path=cache_path,
        output_dir=output_dir,
        peak_lr=selected_lr,
        seed=11,
        token_budget=token_budget,
        supervised_budget=supervised_budget,
        maximum_epochs=12,
        schedule_epochs=10,
        workers=workers,
        attention_backend="sdpa",
        initial_validation={
            key: initial_validation[key]
            for key in ("peptide_nll", "loss_sum", "supervised_tokens", "pairs")
        },
        minimum_epochs_for_stopping=5,
        stopping_patience=4,
        validation_every_half_epoch=True,
    )
    executions.append(result)
    for ceiling in (12, 16):
        history = json.loads((output_dir / "history.json").read_text(encoding="utf-8"))
        latest = json.loads((output_dir / "execution.json").read_text(encoding="utf-8"))
        if latest["stopped_early"]:
            classification = "early_stopping"
            break
        if latest["completed_full_epochs"] < ceiling:
            raise RuntimeError(f"training did not reach ceiling {ceiling} and did not early stop")
        improving = still_improving(history)
        if not improving:
            classification = "plateau_at_ceiling"
            break
        if ceiling == 16:
            continuation = train(
                train_manifest=train_manifest,
                validation_manifest=validation_manifest,
                cache_path=cache_path,
                output_dir=output_dir,
                peak_lr=selected_lr,
                seed=11,
                token_budget=token_budget,
                supervised_budget=supervised_budget,
                maximum_epochs=20,
                schedule_epochs=10,
                workers=workers,
                attention_backend="sdpa",
                minimum_epochs_for_stopping=5,
                stopping_patience=4,
                resume=output_dir / "latest.pt",
                validation_every_half_epoch=True,
            )
            executions.append(continuation)
            history = json.loads((output_dir / "history.json").read_text(encoding="utf-8"))
            latest = json.loads((output_dir / "execution.json").read_text(encoding="utf-8"))
            if latest["stopped_early"]:
                classification = "early_stopping"
            elif latest["completed_full_epochs"] >= 20 and still_improving(history):
                classification = "hard_unresolved_ceiling"
            else:
                classification = "plateau_at_ceiling"
            break
        continuation = train(
            train_manifest=train_manifest,
            validation_manifest=validation_manifest,
            cache_path=cache_path,
            output_dir=output_dir,
            peak_lr=selected_lr,
            seed=11,
            token_budget=token_budget,
            supervised_budget=supervised_budget,
            maximum_epochs=16,
            schedule_epochs=10,
            workers=workers,
            attention_backend="sdpa",
            minimum_epochs_for_stopping=5,
            stopping_patience=4,
            resume=output_dir / "latest.pt",
            validation_every_half_epoch=True,
        )
        executions.append(continuation)
    else:
        classification = "hard_unresolved_ceiling"

    history = json.loads((output_dir / "history.json").read_text(encoding="utf-8"))
    observations = [row for row in history if row.get("epoch", 0) > 0]
    best = min(observations, key=lambda row: row["peptide_nll"])
    last = observations[-1]
    aggregate_wall = sum(item["wall_seconds"] for item in executions)
    report = {
        "status": "UNRESOLVED" if classification == "hard_unresolved_ceiling" else "converged",
        "convergence_classification": classification,
        "selected_learning_rate": selected_lr,
        "hpo_report_sha256": sha256_file(hpo_report_path),
        "evidence_seed": 11,
        "training_manifest_sha256": sha256_file(train_manifest),
        "validation_manifest_sha256": sha256_file(validation_manifest),
        "full_training_pairs": len([line for line in train_manifest.read_text(encoding="utf-8").splitlines() if line]),
        "epochs_completed": latest["epochs_completed"],
        "optimizer_steps": latest["optimizer_steps"],
        "best_epoch": best["epoch"],
        "best_validation_peptide_nll": best["peptide_nll"],
        "terminal_validation_peptide_nll": last["peptide_nll"],
        "best_checkpoint": str(output_dir / "best.pt"),
        "best_checkpoint_sha256": sha256_file(output_dir / "best.pt"),
        "last_eight_validation_observations": observations[-8:],
        "total_wall_seconds": aggregate_wall,
        "peak_vram_bytes": max(item["peak_vram_bytes"] for item in executions),
        "training_runs": executions,
        "still_improving_at_final_ceiling": still_improving(history),
    }
    (output_dir / "convergence_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Single seed-11 Propedia DTFT evidence run")
    parser.add_argument("--manifest-dir", type=Path, default=Path("manifests"))
    parser.add_argument("--cache", type=Path, default=Path("token_cache.pt"))
    parser.add_argument("--hpo-report", type=Path, required=True)
    parser.add_argument("--base-evaluation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--token-budget", type=int, default=8_192)
    parser.add_argument("--supervised-budget", type=int, default=8_192)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    print(json.dumps(run_evidence(
        manifest_dir=args.manifest_dir, cache_path=args.cache,
        hpo_report_path=args.hpo_report, base_evaluation_path=args.base_evaluation,
        output_dir=args.output_dir, token_budget=args.token_budget,
        supervised_budget=args.supervised_budget, workers=args.workers,
    ), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
