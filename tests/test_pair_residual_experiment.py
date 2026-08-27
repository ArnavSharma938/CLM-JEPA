import sys
from pathlib import Path

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_pair_residual_results import crossed_bootstrap_interval
from eval_teacher_forced_five_view import token_decision_metrics


def test_teacher_forced_metrics_use_shifted_supervised_labels_and_true_margin():
    logits = torch.tensor([[[0.0, 0.0, 0.0], [0.0, 3.0, 1.0], [2.0, 1.0, 4.0], [0.0, 0.0, 0.0]]])
    labels = torch.tensor([[-100, -100, 1, 0]])
    result = token_decision_metrics(logits, labels)[0]
    assert result["target_tokens"] == 2
    assert result["teacher_forced_top1"] == 0.5
    assert result["correct_margin_mean"] == pytest.approx(0.0)
    expected_ce = -torch.stack((
        logits[0, 1].log_softmax(0)[1],
        logits[0, 2].log_softmax(0)[0],
    )).mean()
    assert result["ce"] == pytest.approx(float(expected_ce))


def test_crossed_bootstrap_preserves_seed_and_identity_axes():
    matrix = np.ones((5, 8), dtype=np.float64) * 0.25
    assert crossed_bootstrap_interval(
        matrix, seed=17, repetitions=1000,
    ) == pytest.approx([0.25, 0.25])
