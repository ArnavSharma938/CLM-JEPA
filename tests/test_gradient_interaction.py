import torch

from gradient_interaction import apply_combination, combine_gradients


def _combined(method, main, auxiliary):
    parameter = torch.nn.Parameter(torch.zeros_like(main))
    result = combine_gradients(method, [main], [auxiliary])
    apply_combination([parameter], [main], [auxiliary], result)
    return parameter.grad, result


def test_weighted_sum_is_exact_raw_gradient_addition():
    main = torch.tensor([1.0, 2.0])
    auxiliary = torch.tensor([3.0, -1.0])
    gradient, result = _combined("weighted_sum", main, auxiliary)
    torch.testing.assert_close(gradient, main + auxiliary)
    assert result.modification_norm == 0.0


def test_asymmetric_pcgrad_projects_only_conflicting_auxiliary_component():
    main = torch.tensor([1.0, 0.0])
    auxiliary = torch.tensor([-2.0, 3.0])
    gradient, result = _combined("pcgrad", main, auxiliary)
    torch.testing.assert_close(gradient, torch.tensor([1.0, 3.0]))
    assert result.conflict
    assert result.main_coefficient == 3.0
    assert result.auxiliary_coefficient == 1.0


def test_pcgrad_leaves_nonconflicting_pair_unchanged():
    main = torch.tensor([1.0, 0.0])
    auxiliary = torch.tensor([2.0, 3.0])
    gradient, result = _combined("pcgrad", main, auxiliary)
    torch.testing.assert_close(gradient, main + auxiliary)
    assert not result.conflict


def test_du_weighted_rule_uses_positive_part_of_cosine():
    main = torch.tensor([1.0, 0.0])
    conflicting = torch.tensor([-1.0, 2.0])
    gradient, result = _combined("aux_similarity", main, conflicting)
    torch.testing.assert_close(gradient, main)
    assert result.auxiliary_gate == 0.0

    aligned = torch.tensor([1.0, 1.0])
    gradient, result = _combined("aux_similarity", main, aligned)
    expected_cosine = 2.0 ** -0.5
    torch.testing.assert_close(
        gradient, main + expected_cosine * aligned, rtol=1e-6, atol=1e-7,
    )


def test_cagrad_matches_official_two_task_formula_and_is_finite():
    main = torch.tensor([1.0, 2.0, -1.0])
    auxiliary = torch.tensor([-2.0, 0.5, 3.0])
    gradient, result = _combined("cagrad", main, auxiliary)
    expected = (
        result.main_coefficient * main
        + result.auxiliary_coefficient * auxiliary
    )
    torch.testing.assert_close(gradient, expected)
    assert torch.isfinite(gradient).all()
    assert 0.0 <= result.cagrad_weight_main <= 1.0
    assert 0.0 <= result.cagrad_weight_auxiliary <= 1.0
    assert result.cagrad_lambda >= 0.0


def test_gradient_statistics_span_multiple_parameter_tensors_exactly():
    main = [torch.tensor([1.0, 0.0]), torch.tensor([0.0, 2.0])]
    auxiliary = [torch.tensor([0.0, 3.0]), torch.tensor([-4.0, 0.0])]
    result = combine_gradients("weighted_sum", main, auxiliary)
    flattened_main = torch.cat(main)
    flattened_auxiliary = torch.cat(auxiliary)
    expected_cosine = torch.nn.functional.cosine_similarity(
        flattened_main, flattened_auxiliary, dim=0,
    )
    assert abs(result.cosine - float(expected_cosine)) < 1e-7
