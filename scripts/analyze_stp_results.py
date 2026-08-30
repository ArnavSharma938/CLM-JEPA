"""Consolidate the preregistered native-versus-official-STP experiment."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

import numpy as np
from scipy.stats import t


SEEDS = (533, 917, 1301)
VIEWS = 5


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def exact_flags(rows: list[dict], view_index: int | None = None) -> np.ndarray:
    if view_index is None:
        return np.asarray([bool(row["exact"][0]) for row in rows], dtype=np.int8)
    return np.asarray([
        int(
            bool(row["canonical_candidates_by_view"][view_index][0])
            and row["canonical_candidates_by_view"][view_index][0] == row["target"]
        ) for row in rows
    ], dtype=np.int8)


def crossed_bootstrap_interval(
    differences: np.ndarray, *, seed: int, repetitions: int,
) -> list[float]:
    differences = np.asarray(differences, dtype=np.float64)
    seed_count, identity_count = differences.shape
    rng = np.random.default_rng(seed)
    values = np.empty(repetitions, dtype=np.float64)
    for start in range(0, repetitions, 500):
        size = min(500, repetitions - start)
        seed_indices = rng.integers(0, seed_count, size=(size, seed_count))
        identity_indices = rng.integers(0, identity_count, size=(size, identity_count))
        sampled = differences[seed_indices[:, :, None], identity_indices[:, None, :]]
        values[start:start + size] = sampled.mean(axis=(1, 2))
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def seed_effect_summary(
    effects: list[float], matrix: np.ndarray, *, seed: int, repetitions: int,
) -> dict:
    mean = statistics.fmean(effects)
    sample_sd = statistics.stdev(effects)
    half_width = float(t.ppf(0.975, len(effects) - 1)) * sample_sd / math.sqrt(len(effects))
    return {
        "seed_effects": effects, "mean": mean, "sample_sd": sample_sd,
        "seed_t_95_ci": [mean - half_width, mean + half_width],
        "crossed_seed_identity_bootstrap_95_ci": crossed_bootstrap_interval(
            matrix, seed=seed, repetitions=repetitions,
        ),
        "positive_seeds": sum(value > 0 for value in effects),
        "zero_seeds": sum(value == 0 for value in effects),
        "negative_seeds": sum(value < 0 for value in effects),
    }


def teacher_reaction_differences(
    native_rows: list[dict], stp_rows: list[dict], field: str,
) -> list[float]:
    identities = list(dict.fromkeys(row["reaction_identity"] for row in native_rows))
    output = []
    for identity in identities:
        native = [row for row in native_rows if row["reaction_identity"] == identity]
        stp = [row for row in stp_rows if row["reaction_identity"] == identity]
        if len(native) != VIEWS or len(stp) != VIEWS:
            raise ValueError(f"teacher diagnostic requires five views for {identity}")
        if field == "ce":
            native_value = sum(row["nll_sum"] for row in native) / sum(row["target_tokens"] for row in native)
            stp_value = sum(row["nll_sum"] for row in stp) / sum(row["target_tokens"] for row in stp)
        elif field == "correct_token_rate":
            native_value = sum(row["correct_tokens"] for row in native) / sum(row["target_tokens"] for row in native)
            stp_value = sum(row["correct_tokens"] for row in stp) / sum(row["target_tokens"] for row in stp)
        else:
            native_value = statistics.fmean(row[field] for row in native)
            stp_value = statistics.fmean(row[field] for row in stp)
        output.append(stp_value - native_value)
    return output


def trajectory_summary(curves: list[dict]) -> dict:
    output = {"updates": len(curves), "by_epoch": {}}
    for epoch in range(1, 5):
        rows = [row for row in curves if row["epoch"] == epoch]
        output["by_epoch"][str(epoch)] = {
            "updates": len(rows),
            "native_loss_mean": statistics.fmean(row["native_loss"] for row in rows),
            "stp_loss_mean": statistics.fmean(row["jepa_loss"] for row in rows),
            "total_loss_mean": statistics.fmean(row["total_loss"] for row in rows),
            "sampled_span_fraction_mean": statistics.fmean(
                row["stp"]["mean_sampled_span_fraction"] for row in rows
            ),
            "gradient_norm_mean": statistics.fmean(row["gradient_norm"] for row in rows),
        }
    output["stp_loss_change_epoch4_minus_epoch1"] = (
        output["by_epoch"]["4"]["stp_loss_mean"]
        - output["by_epoch"]["1"]["stp_loss_mean"]
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--native-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=20000)
    args = parser.parse_args()

    predictions = {}
    training = {}
    teacher = {}
    paired = {}
    for seed in SEEDS:
        stp_root = args.root / f"seed_{seed}/stp"
        native_root = args.native_root / f"seed_{seed}/native"
        predictions[seed] = {
            "native": read_jsonl(native_root / "evaluation/predictions.jsonl"),
            "stp": read_jsonl(stp_root / "evaluation/predictions.jsonl"),
        }
        training[seed] = {
            "native": read_json(native_root / "training/result.json"),
            "stp": read_json(stp_root / "training/result.json"),
        }
        teacher[seed] = {
            "native": read_json(native_root / "evaluation/teacher_forced.json"),
            "stp": read_json(stp_root / "evaluation/teacher_forced.json"),
        }
        paired[seed] = read_json(args.root / f"seed_{seed}/paired_summary.json")
        native_ids = [row["reaction_identity"] for row in predictions[seed]["native"]]
        if [row["reaction_identity"] for row in predictions[seed]["stp"]] != native_ids:
            raise ValueError(f"prediction identity mismatch for seed {seed}")
        if (
            training[seed]["native"]["config"]["initial_trainable_sha256"]
            != training[seed]["stp"]["config"]["initial_trainable_sha256"]
        ):
            raise ValueError(f"initial trainable state mismatch for seed {seed}")

    aggregate_matrix = np.stack([
        exact_flags(predictions[seed]["stp"]) - exact_flags(predictions[seed]["native"])
        for seed in SEEDS
    ])
    primary = seed_effect_summary(
        aggregate_matrix.mean(axis=1).tolist(), aggregate_matrix,
        seed=8101, repetitions=args.bootstrap_repetitions,
    )
    primary["per_seed"] = {str(seed): paired[seed]["primary_top1"] for seed in SEEDS}

    per_view = {}
    for view_index in range(VIEWS):
        matrix = np.stack([
            exact_flags(predictions[seed]["stp"], view_index)
            - exact_flags(predictions[seed]["native"], view_index)
            for seed in SEEDS
        ])
        summary = seed_effect_summary(
            matrix.mean(axis=1).tolist(), matrix,
            seed=8201 + view_index, repetitions=args.bootstrap_repetitions,
        )
        summary["per_seed"] = {
            str(seed): paired[seed]["individual_view_exact_top1"][f"view_{view_index + 1}"]
            for seed in SEEDS
        }
        per_view[f"view_{view_index + 1}"] = summary

    teacher_matrices = {"ce": [], "correct_margin_mean": [], "correct_token_rate": []}
    teacher_per_seed = {}
    for seed in SEEDS:
        native_rows = teacher[seed]["native"]["rows"]
        stp_rows = teacher[seed]["stp"]["rows"]
        if (
            [(row["reaction_identity"], row["view_index"]) for row in native_rows]
            != [(row["reaction_identity"], row["view_index"]) for row in stp_rows]
        ):
            raise ValueError(f"teacher-forced row mismatch for seed {seed}")
        for field in teacher_matrices:
            teacher_matrices[field].append(
                teacher_reaction_differences(native_rows, stp_rows, field)
            )
        native_overall = teacher[seed]["native"]["overall"]
        stp_overall = teacher[seed]["stp"]["overall"]
        teacher_per_seed[str(seed)] = {
            "native": native_overall, "stp": stp_overall,
            "token_weighted_ce_delta": stp_overall["token_weighted_ce"] - native_overall["token_weighted_ce"],
            "mean_margin_delta": stp_overall["mean_correct_token_margin"] - native_overall["mean_correct_token_margin"],
            "correct_token_rate_delta": stp_overall["correct_token_rate"] - native_overall["correct_token_rate"],
        }
    teacher_matrices = {key: np.asarray(value) for key, value in teacher_matrices.items()}

    crossed = primary["crossed_seed_identity_bootstrap_95_ci"]
    if primary["positive_seeds"] == 3 and primary["mean"] > 0 and crossed[0] > 0:
        verdict = "PASS"
    elif primary["mean"] <= 0 and primary["positive_seeds"] <= 1:
        verdict = "FAIL"
    else:
        verdict = "INCONCLUSIVE"

    output = {
        "schema_version": 1,
        "protocol": {
            "seeds": list(SEEDS), "reactions": aggregate_matrix.shape[1],
            "views": VIEWS, "bootstrap_repetitions": args.bootstrap_repetitions,
            "primary": "official five-view aggregated exact generated top-1",
            "upstream_repository": "https://github.com/galilai-group/llm-jepa",
            "upstream_commit": "ea0017c654ad917066ff32afc88276bea8ca5f7e",
            "stp_lambda": 0.02,
        },
        "primary_exact_top1": primary,
        "individual_view_exact_top1": per_view,
        "teacher_forced": {
            "per_seed": teacher_per_seed,
            "ce_stp_minus_native": seed_effect_summary(
                teacher_matrices["ce"].mean(axis=1).tolist(), teacher_matrices["ce"],
                seed=8301, repetitions=args.bootstrap_repetitions,
            ),
            "margin_stp_minus_native": seed_effect_summary(
                teacher_matrices["correct_margin_mean"].mean(axis=1).tolist(),
                teacher_matrices["correct_margin_mean"], seed=8303,
                repetitions=args.bootstrap_repetitions,
            ),
            "correct_token_rate_stp_minus_native": seed_effect_summary(
                teacher_matrices["correct_token_rate"].mean(axis=1).tolist(),
                teacher_matrices["correct_token_rate"], seed=8305,
                repetitions=args.bootstrap_repetitions,
            ),
        },
        "trajectory": {
            str(seed): trajectory_summary(training[seed]["stp"]["curves"])
            for seed in SEEDS
        },
        "training": {
            str(seed): {
                condition: {
                    "initial_trainable_sha256": training[seed][condition]["config"]["initial_trainable_sha256"],
                    "optimizer_steps": training[seed][condition]["compute"]["optimizer_steps"],
                    "wall_time_seconds": training[seed][condition]["compute"]["wall_time_seconds"],
                    "peak_vram_bytes": training[seed][condition]["compute"]["peak_vram_bytes"],
                } for condition in ("native", "stp")
            } for seed in SEEDS
        },
        "verdict": verdict,
        "verdict_rule": (
            "PASS requires 3/3 positive seeds, positive mean, crossed CI >0; "
            "FAIL requires nonpositive mean and <=1 positive seed; otherwise INCONCLUSIVE"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output.resolve()), "verdict": verdict,
        "mean_top1_difference": primary["mean"], "crossed_95_ci": crossed,
        "positive_seeds": primary["positive_seeds"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
