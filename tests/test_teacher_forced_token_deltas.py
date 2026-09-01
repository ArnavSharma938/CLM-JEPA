import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from eval_teacher_forced_token_deltas import paired_row


def test_paired_row_uses_frozen_material_event_definitions():
    native = {
        "reaction_identity": "r", "view_index": 0,
        "labels": [1, 2, 3, 4],
        "correct_ranks": [1, 1, 2, 9],
        "correct_margins": [2.0, 1.0, -0.1, -1.0],
    }
    treatment = {
        "reaction_identity": "r", "view_index": 0,
        "labels": [1, 2, 3, 4],
        "correct_ranks": [1, 2, 7, 1],
        "correct_margins": [2.1, -0.2, -0.7, 0.4],
    }
    result = paired_row(native, treatment)
    assert result["first_top1_change"] == 1
    assert result["first_rank_threshold_crossing"] == {"3": 2, "5": 2, "10": None}
    assert result["first_rank_delta_ge_5"] == 2
    assert result["first_abs_margin_delta_ge_0.5"] == 1
    assert result["first_material_change"] == 1
