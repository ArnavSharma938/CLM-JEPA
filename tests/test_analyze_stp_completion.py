import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_stp_completion import two_way_contrast


def comparison(values):
    return {
        str(seed): {"per_reaction_top1_difference": row}
        for seed, row in zip((533, 917), values)
    }


def test_two_way_contrast_is_right_minus_left():
    left = comparison([[0, 0, 0], [0, 0, 0]])
    right = comparison([[1, 1, 1], [1, 1, 1]])
    result = two_way_contrast(left, right, 4)
    assert result["seed_effects"] == [1.0, 1.0]
    assert result["mean_effect"] == 1.0
    assert result["two_way_seed_reaction_bootstrap_95_ci"] == [1.0, 1.0]
