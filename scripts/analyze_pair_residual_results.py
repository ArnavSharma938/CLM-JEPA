"""Consolidate the preregistered multi-seed pair-residual experiment."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

import numpy as np
from scipy.stats import t


SEEDS = (533, 917, 1301, 2027, 4099)
CONDITIONS = ("native", "residual")
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
        )
        for row in rows
    ], dtype=np.int8)


def crossed_bootstrap_interval(
    differences: np.ndarray, *, seed: int, repetitions: int,
) -> list[float]:
    """Resample training seeds and reaction identities as crossed clusters."""
    differences = np.asarray(differences, dtype=np.float64)
    if differences.ndim != 2:
        raise ValueError("crossed bootstrap requires a seed-by-identity matrix")
    seed_count, identity_count = differences.shape
    rng = np.random.default_rng(seed)
    values = np.empty(repetitions, dtype=np.float64)
    for start in range(0, repetitions, 500):
        size = min(500, repetitions - start)
        seed_indices = rng.integers(0, seed_count, size=(size, seed_count))
        identity_indices = rng.integers(
            0, identity_count, size=(size, identity_count)
        )
        sampled = differences[
            seed_indices[:, :, None], identity_indices[:, None, :]
        ]
        values[start:start + size] = sampled.mean(axis=(1, 2))
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def seed_effect_summary(
    effects: list[float], matrix: np.ndarray, *, seed: int, repetitions: int,
) -> dict:
    mean = statistics.fmean(effects)
    sample_sd = statistics.stdev(effects)
    half_width = float(t.ppf(0.975, len(effects) - 1)) * sample_sd / math.sqrt(len(effects))
    return {
        "seed_effects": effects,
        "mean": mean,
        "sample_sd": sample_sd,
        "seed_t_95_ci": [mean - half_width, mean + half_width],
        "crossed_seed_identity_bootstrap_95_ci": crossed_bootstrap_interval(
            matrix, seed=seed, repetitions=repetitions,
        ),
        "positive_seeds": sum(value > 0 for value in effects),
        "zero_seeds": sum(value == 0 for value in effects),
        "negative_seeds": sum(value < 0 for value in effects),
    }


def trajectory_summary(curves: list[dict]) -> dict:
    active = [row for row in curves if row["jepa_active"]]
    fields = {
        "residual_ntp_cosine": "cosine",
        "raw_residual_to_ntp_norm_ratio": "raw_auxiliary_to_main_norm_ratio",
        "applied_residual_to_ntp_norm_ratio": "auxiliary_to_main_norm_ratio",
        "residual_adamw_update_effect_to_native_ratio": (
            "residual_to_native_adaptive_update_norm_ratio"
        ),
        "residual_adamw_effect_native_update_cosine": (
            "residual_effect_native_update_cosine"
        ),
        "adamw_preconditioning_amplification": (
            "adamw_preconditioning_amplification"
        ),
        "endpoint_true_shuffle_gradient_cosine": "endpoint_true_shuffle_gradient_cosine",
        "endpoint_residual_over_true_norm": "endpoint_residual_over_true_norm",
    }

    def summarize(values: list[float]) -> dict:
        return {
            "mean": statistics.fmean(values),
            "sample_sd": statistics.stdev(values) if len(values) > 1 else 0.0,
            "minimum": min(values),
            "maximum": max(values),
        }

    output = {
        "active_updates": len(active),
        "active_fraction": len(active) / len(curves),
        "true_mse": summarize([row["pair_residual_true_mse"] for row in active]),
        "shuffled_mse": summarize([
            row["pair_residual_shuffled_mse"] for row in active
        ]),
        "residual_scalar": summarize([
            row["pair_residual_scalar"] for row in active
        ]),
        "target_length_assignment_cost": summarize([
            row["pair_residual_target_length_cost"] for row in active
        ]),
    }
    for output_name, source_name in fields.items():
        output[output_name] = summarize([
            row["gradient_interaction"][source_name] for row in active
        ])
    output["by_epoch"] = {}
    for epoch in range(1, 5):
        selected = [row for row in active if row["epoch"] == epoch]
        output["by_epoch"][str(epoch)] = {
            "active_updates": len(selected),
            **{
                name: statistics.fmean([
                    row["gradient_interaction"][source] for row in selected
                ])
                for name, source in fields.items()
            },
            "residual_scalar": statistics.fmean([
                row["pair_residual_scalar"] for row in selected
            ]),
        }
    return output


def teacher_reaction_differences(
    native_rows: list[dict], residual_rows: list[dict], field: str,
) -> list[float]:
    """Return residual-minus-native diagnostics clustered by reaction."""
    identities = list(dict.fromkeys(row["reaction_identity"] for row in native_rows))
    output = []
    for identity in identities:
        native = [row for row in native_rows if row["reaction_identity"] == identity]
        residual = [row for row in residual_rows if row["reaction_identity"] == identity]
        if len(native) != VIEWS or len(residual) != VIEWS:
            raise ValueError(f"teacher diagnostic requires five views for {identity}")
        if field == "ce":
            native_value = sum(row["nll_sum"] for row in native) / sum(
                row["target_tokens"] for row in native
            )
            residual_value = sum(row["nll_sum"] for row in residual) / sum(
                row["target_tokens"] for row in residual
            )
        elif field == "correct_token_rate":
            native_value = sum(row["correct_tokens"] for row in native) / sum(
                row["target_tokens"] for row in native
            )
            residual_value = sum(row["correct_tokens"] for row in residual) / sum(
                row["target_tokens"] for row in residual
            )
        else:
            native_value = statistics.fmean(row[field] for row in native)
            residual_value = statistics.fmean(row[field] for row in residual)
        output.append(residual_value - native_value)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=20000)
    args = parser.parse_args()

    predictions: dict[int, dict[str, list[dict]]] = {}
    training = {}
    teacher = {}
    paired = {}
    for seed in SEEDS:
        seed_root = args.root / f"seed_{seed}"
        predictions[seed] = {}
        training[seed] = {}
        teacher[seed] = {}
        for condition in CONDITIONS:
            condition_root = seed_root / condition
            predictions[seed][condition] = read_jsonl(
                condition_root / "evaluation" / "predictions.jsonl"
            )
            training[seed][condition] = read_json(
                condition_root / "training" / "result.json"
            )
            teacher[seed][condition] = read_json(
                condition_root / "evaluation" / "teacher_forced.json"
            )
        paired[seed] = read_json(seed_root / "paired_summary.json")
        expected = [row["reaction_identity"] for row in predictions[seed]["native"]]
        if [row["reaction_identity"] for row in predictions[seed]["residual"]] != expected:
            raise ValueError(f"prediction identity mismatch for seed {seed}")
        if (
            training[seed]["native"]["config"]["initial_trainable_sha256"]
            != training[seed]["residual"]["config"]["initial_trainable_sha256"]
        ):
            raise ValueError(f"initial trainable state mismatch for seed {seed}")

    aggregate_matrix = np.stack([
        exact_flags(predictions[seed]["residual"])
        - exact_flags(predictions[seed]["native"])
        for seed in SEEDS
    ])
    seed_effects = aggregate_matrix.mean(axis=1).tolist()
    primary = seed_effect_summary(
        seed_effects, aggregate_matrix, seed=7103,
        repetitions=args.bootstrap_repetitions,
    )
    primary["per_seed"] = {
        str(seed): paired[seed]["primary_top1"] for seed in SEEDS
    }

    per_view = {}
    for view_index in range(VIEWS):
        matrix = np.stack([
            exact_flags(predictions[seed]["residual"], view_index)
            - exact_flags(predictions[seed]["native"], view_index)
            for seed in SEEDS
        ])
        per_view[f"view_{view_index + 1}"] = seed_effect_summary(
            matrix.mean(axis=1).tolist(), matrix,
            seed=7201 + view_index, repetitions=args.bootstrap_repetitions,
        )
        per_view[f"view_{view_index + 1}"]["per_seed"] = {
            str(seed): paired[seed]["individual_view_exact_top1"][
                f"view_{view_index + 1}"
            ]
            for seed in SEEDS
        }

    teacher_ce_matrix = []
    teacher_margin_matrix = []
    teacher_correct_matrix = []
    teacher_seed = {}
    for seed in SEEDS:
        native_rows = teacher[seed]["native"]["rows"]
        residual_rows = teacher[seed]["residual"]["rows"]
        native_keys = [(row["reaction_identity"], row["view_index"]) for row in native_rows]
        residual_keys = [(row["reaction_identity"], row["view_index"]) for row in residual_rows]
        if native_keys != residual_keys:
            raise ValueError(f"teacher-forced row mismatch for seed {seed}")
        teacher_ce_matrix.append(teacher_reaction_differences(
            native_rows, residual_rows, "ce",
        ))
        teacher_margin_matrix.append(teacher_reaction_differences(
            native_rows, residual_rows, "correct_margin_mean",
        ))
        teacher_correct_matrix.append(teacher_reaction_differences(
            native_rows, residual_rows, "correct_token_rate",
        ))
        teacher_seed[str(seed)] = {
            "native": teacher[seed]["native"]["overall"],
            "residual": teacher[seed]["residual"]["overall"],
            "token_weighted_ce_delta": (
                teacher[seed]["residual"]["overall"]["token_weighted_ce"]
                - teacher[seed]["native"]["overall"]["token_weighted_ce"]
            ),
            "mean_margin_delta": (
                teacher[seed]["residual"]["overall"]["mean_correct_token_margin"]
                - teacher[seed]["native"]["overall"]["mean_correct_token_margin"]
            ),
            "correct_token_rate_delta": (
                teacher[seed]["residual"]["overall"]["correct_token_rate"]
                - teacher[seed]["native"]["overall"]["correct_token_rate"]
            ),
        }
    teacher_ce_matrix = np.asarray(teacher_ce_matrix)
    teacher_margin_matrix = np.asarray(teacher_margin_matrix)
    teacher_correct_matrix = np.asarray(teacher_correct_matrix)

    crossed = primary["crossed_seed_identity_bootstrap_95_ci"]
    if primary["positive_seeds"] >= 4 and primary["mean"] > 0 and crossed[0] > 0:
        verdict = "PASS"
    elif primary["mean"] <= 0 and primary["positive_seeds"] <= 2:
        verdict = "FAIL"
    else:
        verdict = "INCONCLUSIVE"

    output = {
        "schema_version": 1,
        "protocol": {
            "seeds": list(SEEDS),
            "conditions": list(CONDITIONS),
            "reactions": aggregate_matrix.shape[1],
            "views": VIEWS,
            "bootstrap_repetitions": args.bootstrap_repetitions,
            "primary": "official five-view aggregated exact generated top-1",
        },
        "primary_exact_top1": primary,
        "individual_view_exact_top1": per_view,
        "teacher_forced": {
            "per_seed": teacher_seed,
            "ce_residual_minus_native": seed_effect_summary(
                teacher_ce_matrix.mean(axis=1).tolist(), teacher_ce_matrix,
                seed=7307, repetitions=args.bootstrap_repetitions,
            ),
            "margin_residual_minus_native": seed_effect_summary(
                teacher_margin_matrix.mean(axis=1).tolist(), teacher_margin_matrix,
                seed=7309, repetitions=args.bootstrap_repetitions,
            ),
            "correct_token_rate_residual_minus_native": seed_effect_summary(
                teacher_correct_matrix.mean(axis=1).tolist(),
                teacher_correct_matrix, seed=7311,
                repetitions=args.bootstrap_repetitions,
            ),
        },
        "trajectory": {
            str(seed): trajectory_summary(training[seed]["residual"]["curves"])
            for seed in SEEDS
        },
        "training": {
            str(seed): {
                condition: {
                    "initial_trainable_sha256": training[seed][condition]["config"][
                        "initial_trainable_sha256"
                    ],
                    "adapter_checkpoint": training[seed][condition]["selected_checkpoint"],
                    "optimizer_steps": training[seed][condition]["compute"]["optimizer_steps"],
                    "wall_time_seconds": training[seed][condition]["compute"][
                        "wall_time_seconds"
                    ],
                    "peak_vram_bytes": training[seed][condition]["compute"][
                        "peak_vram_bytes"
                    ],
                }
                for condition in CONDITIONS
            }
            for seed in SEEDS
        },
        "verdict": verdict,
        "verdict_rule": (
            "PASS requires >=4 positive seeds, positive mean, crossed CI >0; "
            "FAIL requires nonpositive mean and <=2 positive seeds; otherwise INCONCLUSIVE"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "output": str(args.output.resolve()),
        "verdict": verdict,
        "mean_top1_difference": primary["mean"],
        "crossed_95_ci": crossed,
        "positive_seeds": primary["positive_seeds"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
