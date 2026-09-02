import numpy as np

from analyze_stp_representations import bootstrap, rank_biserial


def test_bootstrap_preserves_paired_mean_and_interval_order():
    result = bootstrap(np.asarray([-1.0, 0.0, 2.0]), seed=7, samples=1000)
    assert result["n"] == 3
    assert result["mean"] == 1 / 3
    assert result["ci95"][0] <= result["mean"] <= result["ci95"][1]


def test_rank_biserial_auc_has_direction():
    result = rank_biserial(np.asarray([3.0, 2.0, 0.0, 1.0]), np.asarray([1, 1, -1, -1]))
    assert result["auc"] == 1.0
    assert result["win_minus_loss"] == 2.0
