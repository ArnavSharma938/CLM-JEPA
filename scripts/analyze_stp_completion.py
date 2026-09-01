"""Analyze the preregistered paper-STP completion experiment."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from analyze_stp_beams import compare as beam_compare, read_jsonl  # noqa: E402
from analyze_stp_matrix import aggregate, compare  # noqa: E402
from chemfm import TOKENIZER_DIR, load_reaction_tokenizer  # noqa: E402
from run_stp_completion import (  # noqa: E402
    MATRIX, SEEDS, completion_paths, native_evaluation, paper_paths,
    released_evaluation,
)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def two_way_contrast(
    left: dict[str, dict], right: dict[str, dict], seed: int,
) -> dict:
    ordered = [str(value) for value in SEEDS]
    matrix = np.asarray([
        np.asarray(right[value]["per_reaction_top1_difference"], dtype=np.int8)
        - np.asarray(left[value]["per_reaction_top1_difference"], dtype=np.int8)
        for value in ordered
    ])
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(20000)
    for index in range(len(bootstrap)):
        selected_seeds = rng.integers(0, matrix.shape[0], matrix.shape[0])
        selected_reactions = rng.integers(0, matrix.shape[1], matrix.shape[1])
        bootstrap[index] = matrix[selected_seeds][:, selected_reactions].mean()
    seed_effects = matrix.mean(axis=1).tolist()
    return {
        "seed_effects": seed_effects,
        "mean_effect": statistics.fmean(seed_effects),
        "two_way_seed_reaction_bootstrap_95_ci": [
            float(value) for value in np.quantile(bootstrap, (0.025, 0.975))
        ],
        "all_positive": all(value > 0 for value in seed_effects),
        "all_negative": all(value < 0 for value in seed_effects),
    }


def comparisons_for(
    root: Path, rank: int, coefficient: float, tokenizer,
) -> dict[str, dict]:
    output = {}
    for seed in SEEDS:
        _, treatment = paper_paths(root, rank, coefficient, seed)
        output[str(seed)] = compare(
            native_evaluation(rank, seed), treatment, seed, tokenizer
        )
    return output


def direct_comparisons(
    left_paths: dict[int, Path], right_paths: dict[int, Path], tokenizer,
) -> dict[str, dict]:
    return {
        str(seed): compare(left_paths[seed], right_paths[seed], seed, tokenizer)
        for seed in SEEDS
    }


def condition_payload(
    root: Path, rank: int, coefficient: float, tokenizer,
) -> dict:
    comparisons = comparisons_for(root, rank, coefficient, tokenizer)
    beam = {}
    diagnostics = {}
    training = {}
    for seed in SEEDS:
        _, treatment = paper_paths(root, rank, coefficient, seed)
        beam[str(seed)] = beam_compare(
            read_jsonl(native_evaluation(rank, seed) / "predictions.jsonl"),
            read_jsonl(treatment / "predictions.jsonl"),
            seed,
        )
        diagnostic = treatment / "fixed_span_paper_stp_diagnostic.json"
        if diagnostic.exists():
            diagnostics[str(seed)] = read_json(diagnostic)
        checkpoint, _ = paper_paths(root, rank, coefficient, seed)
        result = checkpoint.parents[1] / "result.json"
        if result.exists():
            training[str(seed)] = read_json(result)
    return {
        "rank": rank, "lambda": coefficient,
        "comparisons": comparisons,
        "aggregate": aggregate(comparisons, 9000 + rank + int(coefficient * 1000)),
        "beam_diagnostics": beam,
        "fixed_span_diagnostics": diagnostics,
        "training_results": training,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    tokenizer = load_reaction_tokenizer(TOKENIZER_DIR)
    completion = read_json(root / "completion.json")
    lambda_decision = read_json(root / "lambda/decision.json")
    rank_decision = read_json(root / "rank/decision.json")
    paper_rank = int(rank_decision["selected_rank"])
    paper_lambda = float(lambda_decision["selected_lambda"])

    condition_keys = {(8, 0.02), (8, 0.08), (8, 0.12), (128, 0.02)}
    if completion["lambda_0.16_run"]:
        condition_keys.add((8, 0.16))
    if paper_lambda != 0.02:
        condition_keys.add((128, paper_lambda))
    conditions = {
        f"paper_r{rank}_l{coefficient:g}": condition_payload(
            root, rank, coefficient, tokenizer
        )
        for rank, coefficient in sorted(condition_keys)
    }

    r8_002 = conditions["paper_r8_l0.02"]["comparisons"]
    r128_002 = conditions["paper_r128_l0.02"]["comparisons"]
    selected_r8 = conditions[f"paper_r8_l{paper_lambda:g}"]["comparisons"]
    selected_r128 = conditions[f"paper_r128_l{paper_lambda:g}"]["comparisons"]

    released_r8 = {
        str(seed): compare(
            native_evaluation(8, seed), released_evaluation(8, seed),
            seed, tokenizer,
        ) for seed in SEEDS
    }
    released_r128 = {
        str(seed): compare(
            native_evaluation(128, seed), released_evaluation(128, seed),
            seed, tokenizer,
        ) for seed in SEEDS
    }
    selected_paper = conditions[f"paper_r{paper_rank}_l{paper_lambda:g}"]["comparisons"]

    direct_best = direct_comparisons(
        {seed: released_evaluation(8, seed) for seed in SEEDS},
        {seed: paper_paths(root, paper_rank, paper_lambda, seed)[1] for seed in SEEDS},
        tokenizer,
    )
    direct_r8_selected = direct_comparisons(
        {seed: released_evaluation(8, seed) for seed in SEEDS},
        {seed: paper_paths(root, 8, paper_lambda, seed)[1] for seed in SEEDS},
        tokenizer,
    )
    direct_same_lambda = {}
    for rank in (8, 128):
        direct = direct_comparisons(
            {seed: released_evaluation(rank, seed) for seed in SEEDS},
            {seed: paper_paths(root, rank, 0.02, seed)[1] for seed in SEEDS},
            tokenizer,
        )
        direct_same_lambda[f"rank_{rank}"] = {
            "comparisons": direct, "aggregate": aggregate(direct, 9400 + rank)
        }

    treatment_contrast = two_way_contrast(released_r8, selected_paper, 9501)
    interval = treatment_contrast["two_way_seed_reaction_bootstrap_95_ci"]
    superiority = (
        abs(treatment_contrast["mean_effect"]) >= 0.005
        and (treatment_contrast["all_positive"] or treatment_contrast["all_negative"])
        and (interval[0] > 0 or interval[1] < 0)
    )

    seed1301 = root / "existing_diagnostics/released_r8_l0.02_seed1301_beams.json"
    token1301 = root / "existing_diagnostics/released_r8_l0.02_seed1301_teacher_tokens.json"
    payload = {
        "type": "paper_stp_completion_analysis",
        "root": str(root),
        "report08_analysis": read_json(MATRIX / "analysis.json"),
        "decisions": {
            "completion": completion,
            "lambda": lambda_decision,
            "rank": rank_decision,
            "formulation": read_json(root / "formulation/decision.json"),
        },
        "conditions": conditions,
        "interactions": {
            "paper_rank_at_lambda_0.02": two_way_contrast(r8_002, r128_002, 9301),
            "paper_rank_at_selected_lambda": two_way_contrast(
                selected_r8, selected_r128, 9302
            ),
        },
        "released_references": {
            "rank8_lambda_0.02": {
                "comparisons": released_r8,
                "aggregate": aggregate(released_r8, 9303),
            },
            "rank128_lambda_0.02": {
                "comparisons": released_r128,
                "aggregate": aggregate(released_r128, 9304),
            },
        },
        "formulation_comparison": {
            "selected_paper": {"rank": paper_rank, "lambda": paper_lambda},
            "treatment_effect_contrast_paper_minus_released_r8_l0.02": treatment_contrast,
            "superiority_rule_satisfied": superiority,
            "direct_best_model_comparisons": direct_best,
            "direct_best_model_aggregate": aggregate(direct_best, 9502),
            "direct_r8_selected_lambda_comparisons": direct_r8_selected,
            "direct_r8_selected_lambda_aggregate": aggregate(
                direct_r8_selected, 9503
            ),
            "direct_same_lambda_0.02": direct_same_lambda,
        },
        "existing_seed1301_beam_diagnostic": (
            read_json(seed1301) if seed1301.exists() else None
        ),
        "existing_seed1301_teacher_token_diagnostic": (
            read_json(token1301) if token1301.exists() else None
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "selected_paper": payload["formulation_comparison"]["selected_paper"],
        "paper_conditions": {
            key: value["aggregate"] for key, value in conditions.items()
        },
        "paper_rank_selected": payload["interactions"]["paper_rank_at_selected_lambda"],
        "paper_minus_released": treatment_contrast,
        "superiority_rule_satisfied": superiority,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
