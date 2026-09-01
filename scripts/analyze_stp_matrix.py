"""Analyze paired generation and teacher-forced endpoints for the STP matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binomtest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chemfm import TOKENIZER_DIR, load_reaction_tokenizer  # noqa: E402


SEEDS = (533, 917)
STAGE_A_SEEDS = (533, 917, 1301)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bootstrap_interval(values: np.ndarray, seed: int, repetitions: int = 20000):
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(repetitions, len(values)))
    samples = values[indices].mean(axis=1)
    return [float(value) for value in np.quantile(samples, (0.025, 0.975))]


def paired_stats(native: np.ndarray, treatment: np.ndarray, seed: int) -> dict:
    differences = treatment.astype(np.int8) - native.astype(np.int8)
    native_only = int(np.sum((native == 1) & (treatment == 0)))
    treatment_only = int(np.sum((native == 0) & (treatment == 1)))
    discordant = native_only + treatment_only
    p_value = (
        float(binomtest(min(native_only, treatment_only), discordant, 0.5).pvalue)
        if discordant else 1.0
    )
    return {
        "n": len(native),
        "native_correct": int(native.sum()),
        "treatment_correct": int(treatment.sum()),
        "native_accuracy": float(native.mean()),
        "treatment_accuracy": float(treatment.mean()),
        "absolute_difference": float(differences.mean()),
        "paired_bootstrap_95_ci": bootstrap_interval(differences, seed),
        "native_only_correct": native_only,
        "treatment_only_correct": treatment_only,
        "discordant": discordant,
        "exact_mcnemar_two_sided_p": p_value,
    }


def topk(rows: list[dict], cutoff: int) -> np.ndarray:
    return np.asarray([any(row["exact"][:cutoff]) for row in rows], dtype=np.int8)


def view_top1(rows: list[dict], view: int) -> np.ndarray:
    return np.asarray([
        bool(row["canonical_candidates_by_view"][view][0])
        and row["canonical_candidates_by_view"][view][0] == row["target"]
        for row in rows
    ], dtype=np.int8)


def view_gold_survival(rows: list[dict], view: int) -> np.ndarray:
    return np.asarray([
        row["target"] in row["canonical_candidates_by_view"][view]
        for row in rows
    ], dtype=np.int8)


def first_divergence(native_rows, treatment_rows, tokenizer) -> dict:
    positions = []
    normalized = []
    changed = 0
    for native, treatment in zip(native_rows, treatment_rows):
        for view in range(5):
            left = native["raw_candidates_by_view"][view][0]
            right = treatment["raw_candidates_by_view"][view][0]
            if left == right:
                continue
            changed += 1
            left_ids = tokenizer(left, add_special_tokens=False)["input_ids"]
            right_ids = tokenizer(right, add_special_tokens=False)["input_ids"]
            limit = min(len(left_ids), len(right_ids))
            position = next(
                (index for index in range(limit) if left_ids[index] != right_ids[index]),
                limit,
            )
            positions.append(position)
            normalized.append(position / max(1, max(len(left_ids), len(right_ids))))
    return {
        "definition": "first token mismatch between the two final raw top-1 beams; not an online beam-pruning trace",
        "view_pairs": len(native_rows) * 5,
        "changed_top1_view_pairs": changed,
        "changed_fraction": changed / (len(native_rows) * 5),
        "median_first_divergence_token": statistics.median(positions) if positions else None,
        "mean_normalized_first_divergence": statistics.fmean(normalized) if normalized else None,
    }


def teacher_comparison(native_path: Path, treatment_path: Path, seed: int) -> dict:
    native = read_json(native_path)
    treatment = read_json(treatment_path)
    left, right = native["rows"], treatment["rows"]
    identities = [(row["reaction_identity"], row["view_index"]) for row in left]
    if identities != [(row["reaction_identity"], row["view_index"]) for row in right]:
        raise ValueError("teacher-forced rows are not paired")
    by_reaction = {}
    for nrow, trow in zip(left, right):
        item = by_reaction.setdefault(nrow["reaction_identity"], {"nll_n": 0, "nll_t": 0, "tokens": 0, "margin": []})
        item["nll_n"] += nrow["nll_sum"]
        item["nll_t"] += trow["nll_sum"]
        item["tokens"] += nrow["target_tokens"]
        item["margin"].append(trow["correct_margin_mean"] - nrow["correct_margin_mean"])
    ce_differences = np.asarray([
        (value["nll_t"] - value["nll_n"]) / value["tokens"]
        for value in by_reaction.values()
    ])
    margin_differences = np.asarray([
        statistics.fmean(value["margin"]) for value in by_reaction.values()
    ])
    return {
        "native": native["overall"], "treatment": treatment["overall"],
        "token_weighted_ce_difference": (
            treatment["overall"]["token_weighted_ce"]
            - native["overall"]["token_weighted_ce"]
        ),
        "correct_token_rate_difference": (
            treatment["overall"]["correct_token_rate"]
            - native["overall"]["correct_token_rate"]
        ),
        "mean_margin_difference": (
            treatment["overall"]["mean_correct_token_margin"]
            - native["overall"]["mean_correct_token_margin"]
        ),
        "reaction_paired_ce_mean_difference": float(ce_differences.mean()),
        "reaction_paired_ce_bootstrap_95_ci": bootstrap_interval(ce_differences, seed + 101),
        "reaction_paired_margin_mean_difference": float(margin_differences.mean()),
        "reaction_paired_margin_bootstrap_95_ci": bootstrap_interval(margin_differences, seed + 103),
    }


def compare(native_dir: Path, treatment_dir: Path, seed: int, tokenizer) -> dict:
    native_path = native_dir / "predictions.jsonl"
    treatment_path = treatment_dir / "predictions.jsonl"
    native, treatment = read_jsonl(native_path), read_jsonl(treatment_path)
    identities = [row["reaction_identity"] for row in native]
    if identities != [row["reaction_identity"] for row in treatment] or len(native) != 512:
        raise ValueError("generation rows are not the same complete 512 panel")
    output = {
        "seed": seed,
        "raw": {
            "native_predictions": str(native_path.resolve()),
            "native_sha256": sha256(native_path),
            "treatment_predictions": str(treatment_path.resolve()),
            "treatment_sha256": sha256(treatment_path),
        },
        "top1": paired_stats(topk(native, 1), topk(treatment, 1), seed),
        "top3": paired_stats(topk(native, 3), topk(treatment, 3), seed + 3),
        "top5": paired_stats(topk(native, 5), topk(treatment, 5), seed + 5),
        "top10": paired_stats(topk(native, 10), topk(treatment, 10), seed + 10),
        "views": {},
        "final_gold_beam_survival": {},
        "first_generation_divergence": first_divergence(native, treatment, tokenizer),
        "per_reaction_top1_difference": (
            topk(treatment, 1) - topk(native, 1)
        ).tolist(),
    }
    for view in range(5):
        output["views"][str(view)] = paired_stats(
            view_top1(native, view), view_top1(treatment, view), seed + 20 + view
        )
        output["final_gold_beam_survival"][str(view)] = paired_stats(
            view_gold_survival(native, view), view_gold_survival(treatment, view),
            seed + 30 + view,
        )
    output["teacher_forced"] = teacher_comparison(
        native_dir / "teacher_forced.json",
        treatment_dir / "teacher_forced.json",
        seed,
    )
    return output


def aggregate(comparisons: dict[str, dict], seed: int) -> dict:
    values = list(comparisons.values())
    matrix = np.asarray([value["per_reaction_top1_difference"] for value in values])
    rng = np.random.default_rng(seed)
    bootstrap = []
    for _ in range(20000):
        selected_seeds = rng.integers(0, matrix.shape[0], matrix.shape[0])
        selected_reactions = rng.integers(0, matrix.shape[1], matrix.shape[1])
        bootstrap.append(float(matrix[selected_seeds][:, selected_reactions].mean()))
    seed_effects = [value["top1"]["absolute_difference"] for value in values]
    return {
        "seeds": [value["seed"] for value in values],
        "seed_effects": seed_effects,
        "mean_seed_effect": statistics.fmean(seed_effects),
        "seed_effect_range": [min(seed_effects), max(seed_effects)],
        "two_way_seed_reaction_bootstrap_95_ci": [
            float(value) for value in np.quantile(bootstrap, (0.025, 0.975))
        ],
        "all_seed_effects_positive": all(value > 0 for value in seed_effects),
        "all_seed_effects_nonnegative": all(value >= 0 for value in seed_effects),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    tokenizer = load_reaction_tokenizer(TOKENIZER_DIR)
    report = {"root": str(root), "stages": {}}

    stage_a = {}
    for seed in STAGE_A_SEEDS:
        stage_a[str(seed)] = compare(
            root / f"stage_a/r8_l0.02/native/seed_{seed}/evaluation",
            root / f"stage_a/r8_l0.02/released/seed_{seed}/evaluation",
            seed, tokenizer,
        )
    report["stages"]["A_rank8_released"] = {
        "comparisons": stage_a, "aggregate": aggregate(stage_a, 8001)
    }

    if (root / "stage_b/decision.json").exists():
        stage_b = {}
        for seed in SEEDS:
            stage_b[str(seed)] = compare(
                root / f"stage_b/native_r128/seed_{seed}/evaluation",
                root / f"stage_b/released_r128_l0.02/seed_{seed}/evaluation",
                seed, tokenizer,
            )
        report["stages"]["B_rank128_released"] = {
            "decision": read_json(root / "stage_b/decision.json"),
            "comparisons": stage_b, "aggregate": aggregate(stage_b, 8002),
        }

    if (root / "stage_c/decision.json").exists():
        decision = read_json(root / "stage_c/decision.json")
        rank = decision["rank"]
        native_stage = "stage_a/r8_l0.02/native" if rank == 8 else "stage_b/native_r128"
        paper = {}
        for seed in SEEDS:
            paper[str(seed)] = compare(
                root / f"{native_stage}/seed_{seed}/evaluation",
                root / f"stage_c/paper_r{rank}_l0.02/seed_{seed}/evaluation",
                seed, tokenizer,
            )
        report["stages"]["C_paper"] = {
            "decision": decision, "comparisons": paper,
            "aggregate": aggregate(paper, 8003),
            "frozen_objective_diagnostics": read_json(
                root / "stage_c/frozen_objective_diagnostics.json"
            ),
        }

    if (root / "stage_d/decision.json").exists():
        decision = read_json(root / "stage_d/decision.json")
        rank, formulation = decision["rank"], decision["formulation"]
        native_stage = "stage_a/r8_l0.02/native" if rank == 8 else "stage_b/native_r128"
        lambdas = {}
        for coefficient in (0.005, 0.02, 0.08):
            comparisons = {}
            for seed in SEEDS:
                if coefficient == 0.02:
                    if formulation == "released":
                        treatment = (
                            root / f"stage_a/r8_l0.02/released/seed_{seed}/evaluation"
                            if rank == 8 else
                            root / f"stage_b/released_r128_l0.02/seed_{seed}/evaluation"
                        )
                    else:
                        treatment = root / f"stage_c/paper_r{rank}_l0.02/seed_{seed}/evaluation"
                else:
                    treatment = root / f"stage_d/{formulation}_r{rank}_l{coefficient:g}/seed_{seed}/evaluation"
                comparisons[str(seed)] = compare(
                    root / f"{native_stage}/seed_{seed}/evaluation",
                    treatment, seed, tokenizer,
                )
            lambdas[str(coefficient)] = {
                "comparisons": comparisons,
                "aggregate": aggregate(comparisons, 8100 + int(coefficient * 1000)),
            }
        report["stages"]["D_lambda"] = {"decision": decision, "lambdas": lambdas}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        name: value.get("aggregate", value.get("decision"))
        for name, value in report["stages"].items()
    }))


if __name__ == "__main__":
    main()
