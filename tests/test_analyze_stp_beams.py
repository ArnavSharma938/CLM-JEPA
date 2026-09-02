import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_stp_beams import compare, row_detail, stable_full_rank


def row(panel, target, views):
    ranked = stable_full_rank(views)
    stored = ranked["ranked"][:10]
    stored += [""] * (10 - len(stored))
    return {
        "panel_index": panel,
        "reaction_identity": f"reaction-{panel}",
        "target": target,
        "canonical_candidates_by_view": views,
        "ranked_candidates": stored,
        "rank_scores": {
            value: ranked["scores"][value] for value in ranked["ranked"][:10]
        },
    }


def test_full_rank_matches_stable_reciprocal_rank_and_deduplicates_views():
    result = stable_full_rank([
        ["gold", "gold", "a"], ["a", "gold"], ["", "b"]
    ])
    assert result["scores"]["gold"] == 1.5
    assert result["scores"]["a"] == 1.5
    assert result["ranked"][:2] == ["gold", "a"]
    assert result["invalid"] == 1


def test_row_detail_accepts_official_empty_padding():
    views = [["gold", "gold", ""] + [""] * 7 for _ in range(5)]
    detail = row_detail({
        "panel_index": 0,
        "target": "gold",
        "canonical_candidates_by_view": views,
        "ranked_candidates": ["gold"] + [""] * 9,
        "rank_scores": {"gold": 5.0},
    })
    assert detail["ranked"] == ["gold"]
    assert detail["gold_rank"] == 1


def test_view_rank_preserves_original_invalid_beam_slot():
    views = [["", "gold"] + [""] * 8 for _ in range(5)]
    ranked = stable_full_rank(views)
    detail = row_detail({
        "panel_index": 0,
        "target": "gold",
        "canonical_candidates_by_view": views,
        "ranked_candidates": ["gold"] + [""] * 9,
        "rank_scores": {"gold": ranked["scores"]["gold"]},
    })
    assert detail["view_ranks"] == [2] * 5
    assert detail["view_top1"] == [""] * 5
    assert detail["gold_rank"] == 1


def test_native_only_failure_classes_are_distinct():
    native = [
        row(0, "g0", [["g0"], ["g0"], ["g0"], ["x"], ["x"]]),
        row(1, "g1", [["g1"], ["g1"], ["g1"], ["x"], ["x"]]),
        row(2, "g2", [["g2"], ["g2"], ["g2"], ["x"], ["x"]]),
    ]
    treatment = [
        row(0, "g0", [["x"], ["x"], ["x"], ["x"], ["x"]]),
        row(1, "g1", [["x", "g1"], ["x", "g1"], ["x"], ["x"], ["x"]]),
        row(2, "g2", [["g2"], ["x"], ["x"], ["x"], ["x"]]),
    ]
    result = compare(native, treatment, seed=9)
    assert result["native_only_top1"]["failure_classes"] == {
        "beam_entry_absent": 1,
        "within_beam_ranking": 1,
        "cross_view_aggregation": 1,
    }
