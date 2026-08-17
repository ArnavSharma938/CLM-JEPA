from __future__ import annotations

import torch

from audit_contraction_ntp_direction import effect


def test_spread_effect_sign_matches_gradient_descent_direction():
    spread = {"x": torch.tensor([1.0, 0.0])}
    expansive = {"x": torch.tensor([-2.0, 0.0])}
    contractive = {"x": torch.tensor([3.0, 0.0])}
    assert effect(spread, expansive)["descent_change_per_unit_learning_rate"] > 0
    assert effect(spread, contractive)["descent_change_per_unit_learning_rate"] < 0


def test_eval_ntp_effect_negative_means_helpful_descent_step():
    eval_ntp = {"x": torch.tensor([2.0, 0.0])}
    aligned_objective = {"x": torch.tensor([1.0, 0.0])}
    opposed_objective = {"x": torch.tensor([-1.0, 0.0])}
    assert effect(eval_ntp, aligned_objective)["descent_change_per_unit_learning_rate"] < 0
    assert effect(eval_ntp, opposed_objective)["descent_change_per_unit_learning_rate"] > 0
