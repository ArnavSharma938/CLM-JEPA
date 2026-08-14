import math

import torch

from jepa import SIGReg
from diagnose_sigreg_gradients_rtx4050 import (
    EXPECTED_REGULARIZER_COEFFICIENT,
    LEJEPA_TRADEOFF,
    fixed_sigreg_loss,
    fixed_sigreg_parameters,
    visreg_scale_loss,
)


def test_fixed_sigreg_matches_existing_verified_implementation():
    values = torch.randn(2, 16, 11, generator=torch.Generator().manual_seed(41))
    direct_values = values.clone().requires_grad_(True)
    fixed_values = values.clone().requires_grad_(True)
    direct = SIGReg(num_slices=37, seed=533)(direct_values)
    parameters = fixed_sigreg_parameters(
        11, seed=533, device=torch.device("cpu"), num_slices=37
    )
    fixed = fixed_sigreg_loss(fixed_values, parameters)
    direct_gradient = torch.autograd.grad(direct, direct_values)[0]
    fixed_gradient = torch.autograd.grad(fixed, fixed_values)[0]
    torch.testing.assert_close(fixed, direct, rtol=0, atol=0)
    torch.testing.assert_close(fixed_gradient, direct_gradient, rtol=0, atol=0)


def test_visreg_scale_matches_published_per_view_standard_deviation_formula():
    values = torch.randn(2, 16, 13, generator=torch.Generator().manual_seed(71))
    centered = values - values.mean(dim=1, keepdim=True)
    std = centered.norm(dim=1) / math.sqrt(16) + 1e-6
    reference = (1.0 - std).square().mean()
    torch.testing.assert_close(visreg_scale_loss(values), reference, rtol=0, atol=0)


def test_expected_sigreg_coefficient_preserves_cosine_weight():
    assert LEJEPA_TRADEOFF == 0.01
    assert EXPECTED_REGULARIZER_COEFFICIENT == 0.01 / 0.99
