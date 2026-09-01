import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_stp_completion import (
    select_lambda, select_rank, should_run_lambda_016,
)


def test_lambda_016_requires_material_edge_and_nonnegative_seeds():
    assert should_run_lambda_016({
        0.02: [0.01, 0.01], 0.08: [0.015, 0.015], 0.12: [0.03, 0.025]
    })
    assert not should_run_lambda_016({
        0.02: [0.01, 0.01], 0.08: [0.015, 0.015], 0.12: [0.03, -0.001]
    })


def test_lambda_selection_retains_baseline_for_submaterial_change():
    assert select_lambda({
        0.02: [0.01, 0.01], 0.08: [0.012, 0.013], 0.12: [0.01, 0.012]
    }, False) == 0.02


def test_lambda_selection_uses_lower_lambda_within_noise_band():
    assert select_lambda({
        0.02: [0.01, 0.01], 0.08: [0.02, 0.02], 0.12: [0.023, 0.023]
    }, False) == 0.08


def test_rank_selection_requires_consistent_treatment_interaction():
    assert select_rank([0.01, 0.01], [0.02, 0.02])[0] == 128
    assert select_rank([0.01, 0.01], [0.03, 0.005])[0] == 8
