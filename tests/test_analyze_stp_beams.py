import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_stp_beams import compare, stable_full_rank


def row(panel, target, views):
    ranked = stable_full_rank(views)
    return {
        "panel_index": panel,
        "reaction_identity": f"reaction-{panel}",
        "target": target,
        "canonical_candidates_by_view": views,
        "ranked_candidates": ranked["ranked"][:10],
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
