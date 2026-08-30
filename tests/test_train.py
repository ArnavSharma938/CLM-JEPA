import copy
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from train import (
    ADAM_BETAS, ADAM_EPSILON, MIN_LEARNING_RATE, WARMUP_RATIO,
    WEIGHT_DECAY, adamw_residual_update_diagnostics, attach_matched_targets,
    restore_training_checkpoint, raw_auxiliary_vjp, raw_pair_residual_vjp,
    condition_family,
    _largest_target_overlap_component, _representation_sample, validation_selector,
)
from jepa import SIGReg


class LengthTokenizer:
    def __call__(self, values, add_special_tokens=False):
        return {"input_ids": [list(range(len(value))) for value in values]}


def test_training_conditions_have_explicit_implementation_families():
    assert condition_family("native") == "native"
    assert condition_family("stp") == "semantic_tube_prediction"
    assert condition_family("clm_jepa_mse_sigreg") == "endpoint_clm_jepa"
    assert condition_family("clm_jepa_pair_residual") == "endpoint_clm_jepa"
    assert condition_family("clm_jepa_vjepa2_1") == "dense_vjepa2_1"


def shuffled_rows():
    return [
        {"example_id": "a0", "tgt": "CC"},
        {"example_id": "a1", "tgt": "CC"},
        {"example_id": "b0", "tgt": "CCC"},
        {"example_id": "b1", "tgt": "CCC"},
        {"example_id": "c0", "tgt": "CO"},
        {"example_id": "c1", "tgt": "CO"},
    ]


def test_matched_shuffle_is_reproducible_and_chemically_deranged():
    first = shuffled_rows()
    second = copy.deepcopy(first)
    first_hash = attach_matched_targets(first, LengthTokenizer(), "forward", 533)
    second_hash = attach_matched_targets(second, LengthTokenizer(), "forward", 533)
    assert first_hash == second_hash
    assert [row["jepa_target_example_id"] for row in first] == [
        row["jepa_target_example_id"] for row in second
    ]
    assert all(row["tgt"] != row["jepa_tgt"] for row in first)


def test_chemfm_optimizer_and_scheduler_settings_are_fully_resolved():
    assert ADAM_BETAS == (0.9, 0.999)
    assert ADAM_EPSILON == 1e-8
    assert WEIGHT_DECAY == 0.01
    assert WARMUP_RATIO == 0.05
    assert MIN_LEARNING_RATE == 1e-5


def test_raw_auxiliary_vjp_matches_materialized_objective_and_reaches_both_views():
    torch.manual_seed(31)
    sources = torch.randn(8, 12)
    targets = torch.randn(8, 12)
    sigreg = SIGReg(num_slices=13, seed=533)
    relative = 4.0 * 0.01 / 0.99
    result = raw_auxiliary_vjp(
        sigreg, sources, targets, sigreg_coefficient=relative,
    )
    torch.testing.assert_close(
        result["objective"], result["mse"] + relative * result["sigreg"],
    )
    assert result["source_gradients"].abs().sum() > 0
    assert result["target_gradients"].abs().sum() > 0


def test_pair_residual_vjp_is_exact_true_minus_shuffled_gradient():
    torch.manual_seed(37)
    sources = torch.randn(4, 7)
    targets = torch.randn(4, 7)
    permutation = [1, 0, 3, 2]
    result = raw_pair_residual_vjp(sources, targets, permutation)

    source_leaf = sources.clone().requires_grad_(True)
    target_leaf = targets.clone().requires_grad_(True)
    materialized = (
        torch.nn.functional.mse_loss(source_leaf, target_leaf)
        - torch.nn.functional.mse_loss(source_leaf, target_leaf[permutation])
    )
    source_gradient, target_gradient = torch.autograd.grad(
        materialized, (source_leaf, target_leaf),
    )
    torch.testing.assert_close(result["residual_objective"], materialized)
    torch.testing.assert_close(result["source_gradients"], source_gradient)
    torch.testing.assert_close(result["target_gradients"], target_gradient)
    torch.testing.assert_close(
        result["source_gradients"],
        result["true_source_gradients"] - result["shuffled_source_gradients"],
    )
    torch.testing.assert_close(
        result["target_gradients"],
        result["true_target_gradients"] - result["shuffled_target_gradients"],
    )


def test_pair_residual_vjp_rejects_non_deranged_controls():
    values = torch.randn(3, 5)
    with np.testing.assert_raises(ValueError):
        raw_pair_residual_vjp(values, values + 1, [0, 2, 1])


def test_adamw_residual_diagnostic_matches_counterfactual_optimizer_steps():
    main_parameter = torch.nn.Parameter(torch.tensor([0.7, -0.4]))
    combined_parameter = torch.nn.Parameter(main_parameter.detach().clone())
    main_optimizer = torch.optim.AdamW(
        [main_parameter], lr=3e-3, betas=(0.8, 0.95), eps=1e-7,
        weight_decay=0.0,
    )
    combined_optimizer = torch.optim.AdamW(
        [combined_parameter], lr=3e-3, betas=(0.8, 0.95), eps=1e-7,
        weight_decay=0.0,
    )
    warmup_gradient = torch.tensor([0.2, -0.3])
    for parameter, optimizer in (
        (main_parameter, main_optimizer),
        (combined_parameter, combined_optimizer),
    ):
        parameter.grad = warmup_gradient.clone()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    main_gradient = torch.tensor([0.11, -0.08])
    residual = torch.tensor([-0.04, 0.07])
    diagnostic = adamw_residual_update_diagnostics(
        main_optimizer, [main_parameter], [main_gradient], [residual],
    )
    main_before = main_parameter.detach().clone()
    combined_before = combined_parameter.detach().clone()
    main_parameter.grad = main_gradient.clone()
    combined_parameter.grad = main_gradient + residual
    main_optimizer.step()
    combined_optimizer.step()
    native_update = main_before - main_parameter.detach()
    combined_update = combined_before - combined_parameter.detach()
    effect = combined_update - native_update
    assert diagnostic["native_adaptive_update_norm"] == pytest.approx(
        float(native_update.norm()), rel=1e-5,
    )
    assert diagnostic["residual_adaptive_update_effect_norm"] == pytest.approx(
        float(effect.norm()), rel=1e-4,
    )


def test_adamw_residual_diagnostic_applies_distinct_global_clip_scales():
    native_parameter = torch.nn.Parameter(torch.tensor([0.2, -0.5]))
    combined_parameter = torch.nn.Parameter(native_parameter.detach().clone())
    native_optimizer = torch.optim.AdamW(
        [native_parameter], lr=1e-3, weight_decay=0.0,
    )
    combined_optimizer = torch.optim.AdamW(
        [combined_parameter], lr=1e-3, weight_decay=0.0,
    )
    main_gradient = torch.tensor([4.0, -2.0])
    residual = torch.tensor([-1.0, 3.0])
    native_scale = 0.2
    combined_scale = 0.1
    diagnostic = adamw_residual_update_diagnostics(
        native_optimizer, [native_parameter], [main_gradient], [residual],
        native_gradient_scale=native_scale,
        combined_gradient_scale=combined_scale,
    )
    native_before = native_parameter.detach().clone()
    combined_before = combined_parameter.detach().clone()
    native_parameter.grad = main_gradient * native_scale
    combined_parameter.grad = (main_gradient + residual) * combined_scale
    native_optimizer.step()
    combined_optimizer.step()
    native_update = native_before - native_parameter.detach()
    effect = (combined_before - combined_parameter.detach()) - native_update
    assert diagnostic["native_adaptive_update_norm"] == pytest.approx(
        float(native_update.norm()), rel=1e-4,
    )
    assert diagnostic["residual_adaptive_update_effect_norm"] == pytest.approx(
        float(effect.norm()), rel=1e-4,
    )


def test_checkpoint_selector_uses_only_frozen_task_metric():
    assert validation_selector(
        {"exact_top1": 0.4, "valid_rate": 0.1}, "forward"
    ) == (0.4,)
    assert validation_selector(
        {"recall_at5": 0.3, "lower_bound_precision_at5": 0.2}, "metabolism"
    ) == (0.3, 0.2)


def test_checkpoint_restore_moves_loader_rng_state_back_to_cpu(monkeypatch, tmp_path):
    expected_state = torch.Generator().get_state()

    class RelocatedState:
        def cpu(self):
            return expected_state

    class Stateful:
        def load_state_dict(self, state):
            self.state = state

    checkpoint_state = {
        "planned_epochs": 4,
        "optimizer": {"optimizer": True},
        "scheduler": {"scheduler": True},
        "loader_generator_state": RelocatedState(),
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_states": [],
    }
    monkeypatch.setattr("train.load_adapter_checkpoint", lambda *args: None)
    monkeypatch.setattr("train.torch.load", lambda *args, **kwargs: checkpoint_state)
    monkeypatch.setattr("train.torch.cuda.set_rng_state_all", lambda states: None)
    generator = torch.Generator()
    restored = restore_training_checkpoint(
        tmp_path,
        type("Model", (), {"device": torch.device("cpu")})(),
        Stateful(), Stateful(), generator, 4,
    )
    assert restored is checkpoint_state
    assert torch.equal(generator.get_state(), expected_state)


def test_representation_sample_deduplicates_canonical_r_smiles_identities():
    rows = [
        {"src": "CC.O", "tgt": "CCO"},
        {"src": "O.CC", "tgt": "OCC"},
        {"src": "CN", "tgt": "CCN"},
        {"src": "CCC", "tgt": "CCC"},
    ]
    sample = _representation_sample(rows, "forward")
    assert [row["tgt"] for row in sample] == ["CCO", "CCN", "CCC"]


def test_target_overlap_component_is_not_serialization_last_component():
    index, component = _largest_target_overlap_component(
        {"src": "CCOc1ccccc1.O.[Na+]", "tgt": "Oc1ccccc1"}
    )
    assert index == 0
    assert component == "CCOc1ccccc1"
