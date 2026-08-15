import torch

from audit_chemfm_mechanism import (
    add_vectors,
    default_swap_groups,
    endpoint_objective_gradients,
    hybrid_adapter_state,
    parameter_groups,
    partition_target_regions,
    validate_hybrid,
    vector_difference_ratio,
    vector_relation,
)
from jepa import SIGReg


def test_target_regions_partition_each_true_target_by_rank():
    labels = torch.tensor([
        [-100, -100, 1, 2, 3, 4, 5, 6],
        [-100, 7, 8, 9, 10, -100, -100, -100],
    ])
    regions = partition_target_regions(labels)
    reconstructed = torch.full_like(labels[:, 1:], -100)
    counts = []
    for values in regions.values():
        active = values.ne(-100)
        assert not (reconstructed.ne(-100) & active).any()
        reconstructed[active] = values[active]
        counts.append(int(active.sum()))
    torch.testing.assert_close(reconstructed, labels[:, 1:])
    assert counts == [4, 3, 3]


def test_vector_relation_reports_signed_energy_decomposition():
    native = {"x": torch.tensor([1.0, 0.0])}
    auxiliary = {"x": torch.tensor([-1.0, 1.0])}
    result = vector_relation(auxiliary, native)
    assert result["dot_sign"] == "negative"
    assert abs(result["cosine_similarity"] + 2**-0.5) < 1e-7
    assert result["parallel_energy_fraction"] == 0.0
    assert abs(result["opposed_energy_fraction"] - 0.5) < 1e-7
    assert abs(result["orthogonal_energy_fraction"] - 0.5) < 1e-7


def test_endpoint_mse_branch_gradients_sum_and_shuffle_is_pair_specific():
    source = torch.randn(4, 5, generator=torch.Generator().manual_seed(3))
    target = torch.randn(4, 5, generator=torch.Generator().manual_seed(5))
    accumulator = SIGReg(num_slices=13, seed=7).start_streaming(
        views=2, dimensions=5, expected_samples=4, device=torch.device("cpu")
    )
    accumulator.update(torch.stack((source, target)))
    prepared = accumulator.finalize()
    gradients, _ = endpoint_objective_gradients(source, target, prepared, [1, 0, 3, 2])
    true_source = gradients["mse_source"][0]
    true_target = gradients["mse_target"][1]
    direct_source = source.clone().requires_grad_(True)
    direct_target = target.clone().requires_grad_(True)
    direct = torch.nn.functional.mse_loss(direct_source, direct_target)
    expected_source, expected_target = torch.autograd.grad(direct, (direct_source, direct_target))
    torch.testing.assert_close(true_source, expected_source)
    torch.testing.assert_close(true_target, expected_target)
    true = {"s": true_source, "t": true_target}
    shuffled = {"s": gradients["mse_shuffled"][0], "t": gradients["mse_shuffled"][1]}
    assert vector_difference_ratio(true, shuffled) > 0


def test_parameter_grouping_separates_lora_depth_and_token_io():
    names = [
        "base.model.layers.3.self_attn.q_proj.lora_A.adapter.weight",
        "base.model.layers.3.mlp.up_proj.lora_B.adapter.weight",
        "base.model.embed_tokens.modules_to_save.adapter.weight",
    ]
    groups = parameter_groups(names, 4)
    assert len(groups["global"]["lora_only"]) == 2
    assert len(groups["layers"]["layer_03"]) == 2
    assert len(groups["module_families"]["attention_all"]) == 1
    assert len(groups["module_families"]["mlp_all"]) == 1
    assert len(groups["module_families"]["modules_to_save"]) == 1


def test_hybrid_adapter_changes_only_selected_depth_group():
    groups = default_swap_groups(8)
    keys = {
        "base.model.layers.0.mlp.up_proj.lora_A.weight": torch.tensor([0.0]),
        "base.model.layers.3.mlp.up_proj.lora_A.weight": torch.tensor([0.0]),
        "base.model.layers.7.mlp.up_proj.lora_A.weight": torch.tensor([0.0]),
        "base.model.embed_tokens.weight": torch.tensor([0.0]),
    }
    donor = {name: value + 1 for name, value in keys.items()}
    hybrid, changed = hybrid_adapter_state(keys, donor, "depth_q2", groups)
    validation = validate_hybrid(hybrid, keys, donor, changed)
    assert changed == ["base.model.layers.3.mlp.up_proj.lora_A.weight"]
    assert validation["exact_state_validation"]
    assert torch.equal(hybrid["base.model.layers.0.mlp.up_proj.lora_A.weight"], torch.tensor([0.0]))
    assert torch.equal(hybrid["base.model.layers.3.mlp.up_proj.lora_A.weight"], torch.tensor([1.0]))


def test_add_vectors_is_exact_linear_combination():
    first = {"x": torch.tensor([1.0, 2.0])}
    second = {"x": torch.tensor([3.0, 5.0])}
    result = add_vectors((first, 2.0), (second, -1.0))
    torch.testing.assert_close(result["x"], torch.tensor([-1.0, -1.0]))
