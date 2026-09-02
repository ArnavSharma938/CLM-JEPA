import math

import torch

from src.geodesic_audit import (
    acceleration_decomposition,
    categorical_fisher_squared,
    chord_coordinates,
    curvature_removal_intervention,
    fisher_rao_distance,
    fisher_rao_path_efficiency,
    fisher_rao_triangle_excess,
    gold_logprob_gradient,
    local_tangent_acceleration,
    matched_geodesic_displacement,
    released_objective_anatomy,
    tube_scale_space,
)


def test_tube_radius_is_zero_for_nonuniformly_parameterized_line():
    path = torch.tensor([[0., 0.], [1., 0.], [3., 0.], [8., 0.]])
    rows = tube_scale_space(path)
    assert [row["span_length"] for row in rows] == [2, 3]
    assert max(row["maximum"] for row in rows) < 1e-7
    # Monotone but nonuniform movement remains within the endpoint segment.
    assert max(row["monotonicity_violation"] for row in rows) == 0


def test_tube_radius_and_alpha_match_definition():
    s = torch.tensor([0., 0.])
    r = torch.tensor([1., 1.])
    t = torch.tensor([2., 0.])
    alpha, q, rho = chord_coordinates(s, r, t)
    assert torch.allclose(alpha, torch.tensor(.5))
    assert torch.allclose(q, torch.tensor([0., 1.]))
    assert torch.allclose(rho, torch.tensor(.5))


def test_acceleration_separates_speed_from_turning():
    straight = torch.tensor([[0., 0.], [1., 0.], [3., 0.]])
    bent = torch.tensor([[0., 0.], [1., 0.], [1., 1.]])
    a = acceleration_decomposition(straight)
    b = acceleration_decomposition(bent)
    assert a["acceleration_parallel"].item() == 1
    assert a["acceleration_normal"].item() == 0
    assert b["acceleration_normal"].item() > 0


def test_categorical_fisher_matches_explicit_matrix():
    torch.manual_seed(4)
    p = torch.softmax(torch.randn(7), dim=0)
    w = torch.randn(7, 5)
    u = torch.randn(5)
    dz = w @ u
    explicit = u @ (w.T @ (torch.diag(p) - p[:, None] * p[None, :]) @ w) @ u
    assert torch.allclose(categorical_fisher_squared(dz, p), explicit, atol=2e-6)


def test_fisher_rao_spherical_distance_and_triangle_excess():
    p = torch.tensor([1., 0., 0.])
    q = torch.tensor([0., 1., 0.])
    midpoint = torch.tensor([.5, .5, 0.])
    assert torch.allclose(fisher_rao_distance(p, q), torch.tensor(math.pi))
    assert abs(fisher_rao_triangle_excess(p, midpoint, q).item()) < 1e-6
    probs = torch.stack([p, midpoint, q])
    assert abs(fisher_rao_path_efficiency(probs).item() - 1) < 1e-6


def test_analytic_gold_gradient_matches_autograd():
    torch.manual_seed(7)
    h = torch.randn(6, requires_grad=True)
    w = torch.randn(9, 6)
    expected = torch.autograd.grad(torch.log_softmax(h @ w.T, -1)[3], h)[0]
    actual, _, _ = gold_logprob_gradient(h.detach(), w, 3)
    assert torch.allclose(actual, expected, atol=2e-6)


def test_curvature_removal_uses_exact_lm_head_and_norm_restore():
    h = torch.tensor([2., 0.])
    q = torch.tensor([1., 1.])
    w = torch.tensor([[1., 0.], [0., 1.]])
    raw = curvature_removal_intervention(h, q, w, 0, .5, False)
    restored = curvature_removal_intervention(h, q, w, 0, .5, True)
    assert raw["gold_margin"] == 2.0
    assert restored["gold_margin"] > raw["gold_margin"]


def test_released_anatomy_exposes_complement_cancellation():
    b = torch.tensor([[1., 4.]])
    a = torch.tensor([[1., -4.]])
    p = torch.tensor([[2., 0.]])
    result = released_objective_anatomy(b, p, a)
    assert result["loss"].item() < 1e-7
    assert result["cos_patch_before"].item() < .3
    assert result["cos_patch_after"].item() < .3
    assert result["cancellation_ratio"].item() < .3


def test_matched_displacement_identifies_midpoint_correction():
    native = (torch.tensor([0., 0.]), torch.tensor([1., 1.]), torch.tensor([2., 0.]))
    treatment = (torch.tensor([0., 0.]), torch.tensor([1., 0.]), torch.tensor([2., 0.]))
    result = matched_geodesic_displacement(native, treatment)
    assert result["delta_rho"].item() == -.5
    assert result["correction_cosine"].item() > .999
    assert result["endpoint_middle_ratio"].item() == 0


def test_local_tangent_acceleration_separates_tangent_and_normal():
    x = torch.linspace(-2, 2, 64)
    neighbors = torch.stack([x, torch.zeros_like(x), torch.zeros_like(x)], dim=1)
    query = torch.zeros(3)
    velocity = torch.tensor([1., 0., 0.])
    tangent = local_tangent_acceleration(query, velocity, torch.tensor([2., 0., 0.]), neighbors, 1)
    normal = local_tangent_acceleration(query, velocity, torch.tensor([0., 3., 0.]), neighbors, 1)
    assert tangent["normal_acceleration"].item() < 1e-6
    assert tangent["geodesic_violation"].item() < 1e-6
    assert normal["normal_acceleration"].item() > 2.99
