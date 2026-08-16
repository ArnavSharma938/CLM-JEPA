from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from audit_sigreg_pair_specificity import (  # noqa: E402
    add_vectors,
    gradient_descent_effect,
    vector_difference_ratio,
)
from jepa import SIGReg  # noqa: E402


def test_active_sigreg_cancels_from_true_shuffle_residual():
    true = {"x": torch.tensor([1.0, 2.0])}
    shuffled = {"x": torch.tensor([0.5, -1.0])}
    sigreg = {"x": torch.tensor([3.0, 4.0])}
    full_true = add_vectors((true, 2.0), (sigreg, 0.0808080808))
    full_shuffle = add_vectors((shuffled, 2.0), (sigreg, 0.0808080808))
    observed = add_vectors((full_true, 1.0), (full_shuffle, -1.0))
    expected = add_vectors((true, 2.0), (shuffled, -2.0))
    assert torch.equal(observed["x"], expected["x"])


def test_gradient_descent_effect_sign():
    metric = {"x": torch.tensor([1.0, 0.0])}
    improving_objective = {"x": torch.tensor([-2.0, 0.0])}
    damaging_objective = {"x": torch.tensor([3.0, 0.0])}
    assert gradient_descent_effect(metric, improving_objective)["descent_effect_per_unit_learning_rate"] > 0
    assert gradient_descent_effect(metric, damaging_objective)["descent_effect_per_unit_learning_rate"] < 0


def test_difference_ratio_uses_absolute_pair_residual():
    true = {"x": torch.tensor([2.0, 0.0])}
    shuffled = {"x": torch.tensor([1.0, 0.0])}
    assert vector_difference_ratio(true, shuffled) == 0.5


def test_sigreg_resamples_slices_on_every_call():
    values = torch.Generator().manual_seed(9)
    representations = torch.randn(2, 16, 12, generator=values)
    sigreg = SIGReg(num_slices=31, seed=533)
    first = sigreg(representations)
    second = sigreg(representations)
    assert sigreg.global_step == 2
    assert not torch.equal(first, second)
