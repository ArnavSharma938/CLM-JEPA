import pytest

from protein.src.pilot_stats import (
    bh_adjust, exact_sign_flip_pvalue, hierarchical_paired_ci,
    paired_bootstrap_ci, paired_summary,
)


def test_exact_sign_flip_all_same_direction_for_eight_pairs():
    assert exact_sign_flip_pvalue([1] * 8) == pytest.approx(2 / 256)


def test_bh_adjust_is_returned_in_input_order():
    assert bh_adjust([.01, .04]) == pytest.approx([.02, .04])


def test_paired_summary_uses_replicates():
    result = paired_summary([1, 2, 3], [2, 3, 4])
    assert result["nextlat_minus_native"] == 1
    assert result["ci95"] == pytest.approx((1, 1))


def test_hierarchical_bootstrap_preserves_pairing():
    ci = hierarchical_paired_ci([[1, 2], [3, 4]], [[2, 3], [4, 5]], draws=100)
    assert ci == pytest.approx((1, 1))
