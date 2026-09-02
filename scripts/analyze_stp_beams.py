"""Diagnose paired five-view ChemFM beams without changing official scoring."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import binomtest


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def stable_full_rank(canonical_views: list[list[str]]) -> dict:
    """Recompute official reciprocal-rank aggregation before its top-10 cut."""
    scores: dict[str, float] = {}
    positions: dict[str, list[tuple[int, int]]] = {}
    encounter_order: list[str] = []
    unique_views = []
    invalid = 0
    for view_index, view in enumerate(canonical_views):
        invalid += sum(not value for value in view)
        unique = []
        for value in view:
            if value and value not in unique:
                unique.append(value)
        unique_views.append(unique)
        for rank, value in enumerate(unique, start=1):
            if value not in scores:
                scores[value] = 0.0
                positions[value] = []
                encounter_order.append(value)
            scores[value] += 1.0 / rank
            positions[value].append((view_index, rank))
    ranked = sorted(encounter_order, key=scores.__getitem__, reverse=True)
    return {
        "ranked": ranked,
        "scores": scores,
        "positions": positions,
        "unique_views": unique_views,
        "invalid": invalid,
    }


def row_detail(row: dict) -> dict:
    aggregate = stable_full_rank(row["canonical_candidates_by_view"])
    expected_stored_rank = aggregate["ranked"][:10]
    expected_stored_rank += [""] * (10 - len(expected_stored_rank))
    if expected_stored_rank != row["ranked_candidates"]:
        raise ValueError(f"stored official rank differs at panel {row['panel_index']}")
    for candidate, score in row["rank_scores"].items():
        if abs(aggregate["scores"][candidate] - score) > 1e-12:
            raise ValueError(f"stored score differs at panel {row['panel_index']}")
    target = row["target"]
    gold_rank = (
        aggregate["ranked"].index(target) + 1
        if target in aggregate["ranked"] else None
    )
    # Per-view beam ranks retain the original ten ordered beam slots. Official
    # cross-view aggregation separately removes invalids and duplicates before
    # assigning reciprocal ranks; conflating those two notions can promote a
    # beam-2 gold product to an apparent per-view top-1 after an invalid beam.
    view_ranks = [
        view.index(target) + 1 if target in view else None
        for view in row["canonical_candidates_by_view"]
    ]
    view_sets = [set(view) for view in aggregate["unique_views"]]
    jaccards = [
        len(view_sets[left] & view_sets[right])
        / len(view_sets[left] | view_sets[right])
        if view_sets[left] | view_sets[right] else 1.0
        for left in range(len(view_sets))
        for right in range(left + 1, len(view_sets))
    ]
    view_top1 = [
        view[0] if view else "" for view in row["canonical_candidates_by_view"]
    ]
    winner = aggregate["ranked"][0] if aggregate["ranked"] else ""
    return {
        **aggregate,
        "gold_rank": gold_rank,
        "gold_score": aggregate["scores"].get(target, 0.0),
        "view_ranks": view_ranks,
        "view_top1": view_top1,
        "distinct_valid_candidates": len(set().union(*view_sets)),
        "mean_pairwise_view_jaccard": statistics.fmean(jaccards),
        "distinct_view_top1": len({value for value in view_top1 if value}),
        "view_top1_consensus": max(
            Counter(value for value in view_top1 if value).values(), default=0
        ),
        "winner": winner,
        "winner_view_count": len(aggregate["positions"].get(winner, [])),
    }


def topk(detail: dict, target: str, cutoff: int) -> bool:
    return target in detail["ranked"][:cutoff]


def paired_binary(left: np.ndarray, right: np.ndarray, seed: int) -> dict:
    left = np.asarray(left, dtype=bool)
    right = np.asarray(right, dtype=bool)
    differences = right.astype(np.int8) - left.astype(np.int8)
    left_only = int(np.sum(left & ~right))
    right_only = int(np.sum(~left & right))
    discordant = left_only + right_only
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(20000)
    for start in range(0, len(bootstrap), 1000):
        size = min(1000, len(bootstrap) - start)
        indices = rng.integers(0, len(differences), size=(size, len(differences)))
        bootstrap[start:start + size] = differences[indices].mean(axis=1)
    return {
        "native_correct": int(left.sum()),
        "treatment_correct": int(right.sum()),
        "absolute_difference": float(differences.mean()),
        "paired_bootstrap_95_ci": [
            float(value) for value in np.quantile(bootstrap, (0.025, 0.975))
        ],
        "native_only": left_only,
        "treatment_only": right_only,
        "exact_mcnemar_two_sided_p": (
            1.0 if not discordant else float(
                binomtest(min(left_only, right_only), discordant, 0.5).pvalue
            )
        ),
    }


def summarize_model(rows: list[dict], details: list[dict]) -> dict:
    return {
        "topk": {
            str(cutoff): sum(
                topk(detail, row["target"], cutoff)
                for row, detail in zip(rows, details)
            )
            for cutoff in (1, 3, 5, 10)
        },
        "individual_view_top1": [
            sum(detail["view_ranks"][view] == 1 for detail in details)
            for view in range(5)
        ],
        "gold": {
            "present_in_any_view": sum(
                any(rank is not None for rank in detail["view_ranks"])
                for detail in details
            ),
            "top1_in_any_view": sum(
                any(rank == 1 for rank in detail["view_ranks"])
                for detail in details
            ),
            "mean_views_containing_gold": statistics.fmean(
                sum(rank is not None for rank in detail["view_ranks"])
                for detail in details
            ),
            "aggregate_rank_histogram": dict(sorted(Counter(
                str(detail["gold_rank"]) if detail["gold_rank"] is not None else "absent"
                for detail in details
            ).items())),
        },
        "validity": {
            "invalid_candidates": sum(detail["invalid"] for detail in details),
            "invalid_rate": sum(detail["invalid"] for detail in details)
            / (len(details) * 50),
        },
        "cross_view": {
            "mean_distinct_valid_candidates": statistics.fmean(
                detail["distinct_valid_candidates"] for detail in details
            ),
            "mean_pairwise_view_jaccard": statistics.fmean(
                detail["mean_pairwise_view_jaccard"] for detail in details
            ),
            "mean_distinct_view_top1": statistics.fmean(
                detail["distinct_view_top1"] for detail in details
            ),
            "mean_view_top1_consensus": statistics.fmean(
                detail["view_top1_consensus"] for detail in details
            ),
            "mean_aggregate_winner_view_count": statistics.fmean(
                detail["winner_view_count"] for detail in details
            ),
        },
    }


def compare(native_rows: list[dict], treatment_rows: list[dict], seed: int) -> dict:
    if len(native_rows) != len(treatment_rows):
        raise ValueError("paired prediction files differ in length")
    identities = [row["reaction_identity"] for row in native_rows]
    if identities != [row["reaction_identity"] for row in treatment_rows]:
        raise ValueError("paired prediction identities differ")
    native_details = [row_detail(row) for row in native_rows]
    treatment_details = [row_detail(row) for row in treatment_rows]
    paired_topk = {}
    for cutoff in (1, 3, 5, 10):
        paired_topk[str(cutoff)] = paired_binary(
            [topk(detail, row["target"], cutoff) for row, detail in zip(native_rows, native_details)],
            [topk(detail, row["target"], cutoff) for row, detail in zip(treatment_rows, treatment_details)],
            seed + cutoff,
        )
    paired_views = {}
    for view in range(5):
        paired_views[str(view)] = paired_binary(
            [detail["view_ranks"][view] == 1 for detail in native_details],
            [detail["view_ranks"][view] == 1 for detail in treatment_details],
            seed + 20 + view,
        )

    native_only = []
    treatment_only = []
    view_aggregation_table = Counter()
    for native, treatment, left, right in zip(
        native_rows, treatment_rows, native_details, treatment_details
    ):
        native_correct = topk(left, native["target"], 1)
        treatment_correct = topk(right, treatment["target"], 1)
        native_view_flags = [rank == 1 for rank in left["view_ranks"]]
        treatment_view_flags = [rank == 1 for rank in right["view_ranks"]]
        aggregate_delta = int(treatment_correct) - int(native_correct)
        view_delta = sum(treatment_view_flags) - sum(native_view_flags)
        view_aggregation_table[(aggregate_delta, view_delta)] += 1
        if native_correct == treatment_correct:
            continue
        destination = native_only if native_correct else treatment_only
        if native_correct:
            if not any(rank is not None for rank in right["view_ranks"]):
                failure_class = "beam_entry_absent"
            elif any(rank == 1 for rank in right["view_ranks"]):
                failure_class = "cross_view_aggregation"
            else:
                failure_class = "within_beam_ranking"
        else:
            failure_class = None
        native_gold_rank = left["gold_rank"] or len(left["ranked"]) + 1
        treatment_gold_rank = right["gold_rank"] or len(right["ranked"]) + 1
        native_ahead = set(left["ranked"][:native_gold_rank - 1])
        newly_promoted = [
            {
                "candidate": candidate,
                "score": right["scores"][candidate],
                "view_positions": right["positions"][candidate],
            }
            for candidate in right["ranked"][:treatment_gold_rank - 1]
            if candidate not in native_ahead
        ]
        destination.append({
            "panel_index": native["panel_index"],
            "reaction_identity": native["reaction_identity"],
            "target": native["target"],
            "failure_class": failure_class,
            "native_view_gold_ranks": left["view_ranks"],
            "treatment_view_gold_ranks": right["view_ranks"],
            "individual_views_gained": [
                index for index, (before, after) in enumerate(
                    zip(native_view_flags, treatment_view_flags)
                ) if not before and after
            ],
            "individual_views_lost": [
                index for index, (before, after) in enumerate(
                    zip(native_view_flags, treatment_view_flags)
                ) if before and not after
            ],
            "native_aggregate_gold_rank": left["gold_rank"],
            "treatment_aggregate_gold_rank": right["gold_rank"],
            "native_gold_score": left["gold_score"],
            "treatment_gold_score": right["gold_score"],
            "treatment_winner": right["winner"],
            "treatment_winner_score": right["scores"].get(right["winner"]),
            "treatment_winner_view_positions": right["positions"].get(
                right["winner"], []
            ),
            "newly_promoted_above_gold": newly_promoted,
        })

    return {
        "reactions": len(native_rows),
        "native": summarize_model(native_rows, native_details),
        "treatment": summarize_model(treatment_rows, treatment_details),
        "paired_topk": paired_topk,
        "paired_individual_views": paired_views,
        "view_aggregation_delta_table": {
            f"aggregate_{aggregate:+d}_view_count_{view:+d}": count
            for (aggregate, view), count in sorted(view_aggregation_table.items())
        },
        "native_only_top1": {
            "count": len(native_only),
            "failure_classes": dict(Counter(
                row["failure_class"] for row in native_only
            )),
            "any_individual_view_gain": sum(
                bool(row["individual_views_gained"]) for row in native_only
            ),
            "net_individual_view_gain": sum(
                len(row["individual_views_gained"])
                > len(row["individual_views_lost"])
                for row in native_only
            ),
            "rows": native_only,
        },
        "treatment_only_top1": {
            "count": len(treatment_only), "rows": treatment_only,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--treatment", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = {
        "type": "paired_five_view_beam_diagnostic",
        "native_predictions": str(args.native.resolve()),
        "treatment_predictions": str(args.treatment.resolve()),
        "comparison": compare(
            read_jsonl(args.native), read_jsonl(args.treatment), args.seed
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output.resolve()),
        "paired_topk": payload["comparison"]["paired_topk"],
        "native_only_top1": payload["comparison"]["native_only_top1"]["failure_classes"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
