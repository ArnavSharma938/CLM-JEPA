"""Run the preregistered paper-STP completion matrix on one A6000."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_stp_matrix import (  # noqa: E402
    PANEL, SEEDS, evaluate, locked_record, read_json, read_jsonl, run, sha256,
    teacher, train_pair, treatment_effect,
)


MATRIX = ROOT / "runs/stp_matrix/a6000"
DEFAULT_OUTPUT = ROOT / "runs/stp_completion/a6000"
PAPER_REQUIRED = (
    ("paper_r128_l0.02", 128, 0.02),
    ("paper_r8_l0.08", 8, 0.08),
    ("paper_r8_l0.12", 8, 0.12),
)


def native_evaluation(rank: int, seed: int) -> Path:
    if rank == 8:
        return MATRIX / f"stage_a/r8_l0.02/native/seed_{seed}/evaluation"
    return MATRIX / f"stage_b/native_r128/seed_{seed}/evaluation"


def released_evaluation(rank: int, seed: int) -> Path:
    if rank == 8:
        return MATRIX / f"stage_a/r8_l0.02/released/seed_{seed}/evaluation"
    return MATRIX / f"stage_b/released_r128_l0.02/seed_{seed}/evaluation"


def paper_existing(rank: int, coefficient: float, seed: int) -> tuple[Path, Path]:
    if rank != 8 or coefficient != 0.02:
        raise ValueError("only paper rank-8/lambda-0.02 exists in Report 08")
    base = MATRIX / f"stage_c/paper_r8_l0.02/seed_{seed}"
    return base / "training/checkpoints/epoch_4", base / "evaluation"


def label(rank: int, coefficient: float) -> str:
    return f"paper_r{rank}_l{coefficient:g}"


def completion_paths(
    root: Path, rank: int, coefficient: float, seed: int,
) -> tuple[Path, Path]:
    base = root / f"trajectories/{label(rank, coefficient)}/seed_{seed}"
    return base / "training/checkpoints/epoch_4", base / "evaluation"


def paper_paths(
    root: Path, rank: int, coefficient: float, seed: int,
) -> tuple[Path, Path]:
    if rank == 8 and coefficient == 0.02:
        return paper_existing(rank, coefficient, seed)
    return completion_paths(root, rank, coefficient, seed)


def verify_reuse() -> None:
    if sha256(PANEL) != "a2e6202a4abaf9a70f4700e04299a09964d38c10fce004022dc43e759aa6057d":
        raise ValueError("development panel changed")
    paths = []
    for seed in SEEDS:
        paths.extend([
            native_evaluation(8, seed) / "predictions.jsonl",
            native_evaluation(128, seed) / "predictions.jsonl",
            released_evaluation(8, seed) / "predictions.jsonl",
            released_evaluation(128, seed) / "predictions.jsonl",
            paper_existing(8, 0.02, seed)[1] / "predictions.jsonl",
        ])
    for path in paths:
        if len(read_jsonl(path)) != 512:
            raise ValueError(f"missing complete reusable predictions: {path}")


def diagnose(checkpoint: Path, output: Path, coefficient: float) -> None:
    if output.exists():
        return
    run([
        sys.executable, "-u", "scripts/diagnose_stp_checkpoint.py",
        "--checkpoint", str(checkpoint), "--formulation", "paper",
        "--coefficient", str(coefficient), "--reactions", "16",
        "--output", str(output),
    ])


def run_condition(root: Path, rank: int, coefficient: float) -> None:
    condition_label = label(rank, coefficient)
    checkpoints = train_pair(
        root / "trajectories", condition="stp_paper", label=condition_label,
        rank=rank, alpha=rank, coefficient=coefficient,
    )
    for seed in SEEDS:
        _, evaluation = completion_paths(root, rank, coefficient, seed)
        evaluate(checkpoints[seed], evaluation)
        teacher(checkpoints[seed], evaluation / "teacher_forced.json")
        diagnose(
            checkpoints[seed], evaluation / "fixed_span_paper_stp_diagnostic.json",
            coefficient,
        )


def effect(root: Path, rank: int, coefficient: float, seed: int) -> float:
    _, paper_evaluation = paper_paths(root, rank, coefficient, seed)
    return treatment_effect(
        native_evaluation(rank, seed) / "predictions.jsonl",
        paper_evaluation / "predictions.jsonl",
    )


def effects(root: Path, rank: int, coefficient: float) -> list[float]:
    return [effect(root, rank, coefficient, seed) for seed in SEEDS]


def should_run_lambda_016(values: dict[float, list[float]]) -> bool:
    mean_012 = statistics.fmean(values[0.12])
    return (
        mean_012 >= max(
            statistics.fmean(values[0.02]), statistics.fmean(values[0.08])
        ) + 0.01
        and min(values[0.12]) >= 0
    )


def select_lambda(values: dict[float, list[float]], ran_016: bool) -> float:
    baseline = statistics.fmean(values[0.02])
    eligible = [
        coefficient for coefficient in (0.08, 0.12)
        if statistics.fmean(values[coefficient]) >= baseline + 0.005
        and min(values[coefficient]) >= 0
    ]
    if eligible:
        best_mean = max(statistics.fmean(values[value]) for value in eligible)
        selected = min(
            value for value in eligible
            if best_mean - statistics.fmean(values[value]) < 0.005
        )
    else:
        selected = 0.02
    if ran_016:
        if (
            statistics.fmean(values[0.16])
            >= statistics.fmean(values[selected]) + 0.005
            and min(values[0.16]) >= 0
        ):
            selected = 0.16
    return selected


def select_rank(r8: list[float], r128: list[float]) -> tuple[int, list[float]]:
    interaction = [right - left for left, right in zip(r8, r128)]
    selected = (
        128 if statistics.fmean(interaction) >= 0.005 and min(interaction) >= 0
        else 8
    )
    return selected, interaction


def required(root: Path) -> None:
    verify_reuse()
    locked_record(root / "required/preregistration.json", {
        "stage": "required", "source_commit": "e58a45a",
        "panel_sha256": sha256(PANEL), "seeds": list(SEEDS),
        "new_trajectories": [value[0] for value in PAPER_REQUIRED],
        "optimizer_steps": 320,
        "primary": "paper-minus-same-rank-native exact five-view top-1",
    })
    for _, rank, coefficient in PAPER_REQUIRED:
        run_condition(root, rank, coefficient)
    # The existing paper lambda-0.02 checkpoints need the same trained-state
    # diagnostic as the new lambda cells, but are never retrained.
    for seed in SEEDS:
        checkpoint, evaluation = paper_existing(8, 0.02, seed)
        diagnose(
            checkpoint, evaluation / "fixed_span_paper_stp_diagnostic.json", 0.02
        )


def lambda_stage(root: Path) -> None:
    values = {
        coefficient: effects(root, 8, coefficient)
        for coefficient in (0.02, 0.08, 0.12)
    }
    run_016 = should_run_lambda_016(values)
    locked_record(root / "lambda/edge_decision.json", {
        "effects": {str(key): value for key, value in values.items()},
        "run_lambda_0.16": run_016,
        "rule": (
            "0.12 mean >= max(0.02,0.08) + 0.01 and both 0.12 effects >= 0"
        ),
    })
    if run_016:
        run_condition(root, 8, 0.16)
        values[0.16] = effects(root, 8, 0.16)
    selected = select_lambda(values, run_016)
    locked_record(root / "lambda/decision.json", {
        "effects": {str(key): value for key, value in values.items()},
        "selected_lambda": selected,
        "rule": (
            "default 0.02; alternative mean >= 0.02 + 0.005 and both effects >= 0; "
            "within 0.005 choose lower; 0.16 must additionally beat selected by 0.005"
        ),
    })


def rank_stage(root: Path) -> None:
    selected_lambda = float(read_json(root / "lambda/decision.json")["selected_lambda"])
    if selected_lambda != 0.02:
        run_condition(root, 128, selected_lambda)
    r8 = effects(root, 8, selected_lambda)
    r128 = effects(root, 128, selected_lambda)
    selected_rank, interaction = select_rank(r8, r128)
    locked_record(root / "rank/decision.json", {
        "lambda": selected_lambda,
        "r8_effects": r8, "r128_effects": r128,
        "interaction_effects": interaction, "selected_rank": selected_rank,
        "rule": "rank128 iff mean interaction >= 0.005 and both interactions >= 0",
    })


def final_stage(root: Path) -> None:
    lambda_decision = read_json(root / "lambda/decision.json")
    rank_decision = read_json(root / "rank/decision.json")
    paper_rank = int(rank_decision["selected_rank"])
    paper_lambda = float(lambda_decision["selected_lambda"])
    paper = effects(root, paper_rank, paper_lambda)
    released = [
        treatment_effect(
            native_evaluation(8, seed) / "predictions.jsonl",
            released_evaluation(8, seed) / "predictions.jsonl",
        )
        for seed in SEEDS
    ]
    contrasts = [right - left for left, right in zip(released, paper)]
    locked_record(root / "formulation/decision.json", {
        "released_reference": {"rank": 8, "lambda": 0.02, "effects": released},
        "paper_selected": {
            "rank": paper_rank, "lambda": paper_lambda, "effects": paper,
        },
        "paper_minus_released_treatment_contrasts": contrasts,
        "superiority_rule": (
            "abs mean contrast >= 0.005, both seed contrasts same sign, and "
            "two-way seed/reaction bootstrap interval excludes zero"
        ),
        "superiority_decision_deferred_to_full_paired_analysis": True,
    })
    locked_record(root / "completion.json", {
        "required_complete": True,
        "lambda_0.16_run": bool(
            read_json(root / "lambda/edge_decision.json")["run_lambda_0.16"]
        ),
        "selected_lambda": paper_lambda,
        "selected_rank": paper_rank,
        "panel_sha256": sha256(PANEL),
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--through", choices=("required", "lambda", "rank", "final"),
        default="final",
    )
    args = parser.parse_args()
    root = args.output_root.resolve()
    stages = (
        ("required", required), ("lambda", lambda_stage),
        ("rank", rank_stage), ("final", final_stage),
    )
    for name, function in stages:
        function(root)
        print(json.dumps({"milestone": f"{name}_complete"}), flush=True)
        if name == args.through:
            break


if __name__ == "__main__":
    main()
