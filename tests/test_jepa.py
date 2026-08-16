import copy
import sys
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import LlamaConfig, LlamaForCausalLM

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chemfm import TOKENIZER_DIR, ReactionCollator, load_reaction_tokenizer
from jepa import (
    CLMJEPA, PairCenterSpreadFloor, SIGReg, add_predictor_tokens, extract_source_and_target,
    matched_derangement,
)


def setup_case():
    torch.manual_seed(17)
    tokenizer = load_reaction_tokenizer(TOKENIZER_DIR)
    predictor_ids = add_predictor_tokens(tokenizer)
    config = LlamaConfig(
        vocab_size=len(tokenizer), hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=256, pad_token_id=tokenizer.pad_token_id,
    )
    model = LlamaForCausalLM(config).eval()
    rows = [
        {"src": "CCO.O=O", "tgt": "CC(=O)O"},
        {"src": "CCBr.N", "tgt": "CCN"},
        {"src": "CO.Cl", "tgt": "CCl"},
    ]
    batch = ReactionCollator(tokenizer)(rows)
    tensor_batch = {key: value for key, value in batch.items() if torch.is_tensor(value)}
    method = CLMJEPA(predictor_ids, tokenizer.eos_token_id, tokenizer.pad_token_id)
    return model, tokenizer, method, tensor_batch


def test_lambda_zero_is_exact_native_equivalence():
    model, _, method, batch = setup_case()
    native = model(**batch)
    result = method(model, batch, k=3, jepa_weight=0.0)
    assert torch.equal(result.logits, native.logits)
    assert torch.equal(result.loss, native.loss)
    assert result.jepa_loss is None


def test_predictor_resize_keeps_distinct_upstream_initialization():
    torch.manual_seed(17)
    tokenizer = load_reaction_tokenizer(TOKENIZER_DIR)
    model = LlamaForCausalLM(LlamaConfig(
        vocab_size=len(tokenizer), hidden_size=32, intermediate_size=64,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=64, pad_token_id=tokenizer.pad_token_id,
    ))
    predictor_ids = add_predictor_tokens(tokenizer, model)
    embeddings = model.get_input_embeddings().weight.detach()[predictor_ids]
    assert torch.unique(embeddings, dim=0).shape[0] == len(predictor_ids)
    assert torch.all(embeddings.norm(dim=1) > 0)


def test_active_jepa_uses_one_concatenated_model_call():
    model, _, method, batch = setup_case()
    calls = 0
    original = model.forward

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    model.forward = counted
    method(model, batch, k=1, jepa_weight=1.0)
    assert calls == 1


def test_optimized_active_forward_matches_standard_loss_states_and_gradients():
    standard_model, tokenizer, standard_method, batch = setup_case()
    optimized_model = copy.deepcopy(standard_model)
    optimized_method = CLMJEPA(
        standard_method.predictor_token_ids,
        tokenizer.eos_token_id,
        tokenizer.pad_token_id,
        optimized_native_logits=True,
    )
    standard_model.eval()
    optimized_model.eval()
    standard = standard_method(
        standard_model, batch, k=0, jepa_weight=2.0,
        jepa_loss_type="mse", force_jepa_active=True,
    )
    optimized = optimized_method(
        optimized_model, batch, k=0, jepa_weight=2.0,
        jepa_loss_type="mse", force_jepa_active=True,
    )
    torch.testing.assert_close(optimized.logits, standard.logits, rtol=2e-5, atol=1e-7)
    torch.testing.assert_close(optimized.source_states, standard.source_states, rtol=0.0, atol=0.0)
    torch.testing.assert_close(optimized.target_states, standard.target_states, rtol=0.0, atol=0.0)
    torch.testing.assert_close(optimized.native_loss, standard.native_loss, rtol=1e-7, atol=1e-8)
    torch.testing.assert_close(optimized.loss, standard.loss, rtol=1e-7, atol=1e-8)
    standard.loss.backward()
    optimized.loss.backward()
    for (standard_name, standard_parameter), (optimized_name, optimized_parameter) in zip(
        standard_model.named_parameters(), optimized_model.named_parameters(),
    ):
        assert standard_name == optimized_name
        if standard_parameter.grad is None:
            assert optimized_parameter.grad is None
        else:
            torch.testing.assert_close(
                optimized_parameter.grad, standard_parameter.grad,
                rtol=2e-5, atol=2e-6, msg=standard_name,
            )


def test_optimized_active_forward_supports_peft_modules_to_save():
    base_model, tokenizer, standard_method, batch = setup_case()
    model = get_peft_model(
        base_model,
        LoraConfig(
            task_type="CAUSAL_LM", r=4, lora_alpha=4,
            target_modules=["q_proj", "v_proj"],
            modules_to_save=["embed_tokens", "lm_head"],
        ),
        adapter_name="USPTO-MIT-Synthesis",
    ).eval()
    optimized_method = CLMJEPA(
        standard_method.predictor_token_ids,
        tokenizer.eos_token_id,
        tokenizer.pad_token_id,
        optimized_native_logits=True,
    )
    standard = standard_method(
        model, batch, k=0, jepa_weight=2.0,
        jepa_loss_type="mse", force_jepa_active=True,
    )
    optimized = optimized_method(
        model, batch, k=0, jepa_weight=2.0,
        jepa_loss_type="mse", force_jepa_active=True,
    )
    torch.testing.assert_close(optimized.loss, standard.loss, rtol=2e-6, atol=2e-7)
    torch.testing.assert_close(
        optimized.source_states, standard.source_states, rtol=0.0, atol=0.0,
    )


def test_endpoint_only_optimized_forward_preserves_endpoint_states():
    model, tokenizer, standard_method, batch = setup_case()
    optimized = CLMJEPA(
        standard_method.predictor_token_ids,
        tokenizer.eos_token_id,
        tokenizer.pad_token_id,
        optimized_native_logits=True,
    )
    model.eval()
    full = optimized(
        model, batch, k=0, jepa_weight=0.0, monitor_only=True,
        jepa_loss_type="mse", force_jepa_active=True,
    )
    endpoints = optimized(
        model, batch, k=0, jepa_weight=0.0, monitor_only=True,
        jepa_loss_type="mse", force_jepa_active=True, endpoint_only=True,
    )
    assert endpoints.logits.numel() == 0
    torch.testing.assert_close(endpoints.source_states, full.source_states, rtol=0.0, atol=0.0)
    torch.testing.assert_close(endpoints.target_states, full.target_states, rtol=0.0, atol=0.0)


def test_endpoint_only_is_independent_of_compact_native_logits():
    model, _, method, batch = setup_case()
    assert not method.optimized_native_logits
    model.eval()
    full = method(
        model, batch, k=0, jepa_weight=0.0, monitor_only=True,
        jepa_loss_type="mse", force_jepa_active=True,
    )
    endpoints = method(
        model, batch, k=0, jepa_weight=0.0, monitor_only=True,
        jepa_loss_type="mse", force_jepa_active=True, endpoint_only=True,
    )
    assert endpoints.logits.numel() == 0
    torch.testing.assert_close(endpoints.source_states, full.source_states, rtol=0.0, atol=0.0)
    torch.testing.assert_close(endpoints.target_states, full.target_states, rtol=0.0, atol=0.0)


def test_predictors_do_not_change_native_logits_or_preceding_source_states():
    model, _, method, batch = setup_case()
    zero = method(model, batch, k=0, jepa_weight=1.0)
    three = method(model, batch, k=3, jepa_weight=1.0)
    assert torch.equal(zero.logits, three.logits)
    sources, _ = extract_source_and_target(batch)
    padded = torch.nn.utils.rnn.pad_sequence(sources, batch_first=True, padding_value=model.config.pad_token_id)
    standalone = model(input_ids=padded, attention_mask=padded.ne(model.config.pad_token_id), output_hidden_states=True)
    # k=3's predictor state changes, while all preceding states remain exactly causal.
    official_suffix = list(reversed(method.predictor_token_ids[:3]))
    combined_rows = [torch.cat((row, row.new_tensor(official_suffix))) for row in sources]
    combined = torch.nn.utils.rnn.pad_sequence(combined_rows, batch_first=True, padding_value=model.config.pad_token_id)
    extended = model(input_ids=combined, attention_mask=combined.ne(model.config.pad_token_id), output_hidden_states=True)
    for index, source in enumerate(sources):
        torch.testing.assert_close(
            standalone.hidden_states[-1][index, : len(source)],
            extended.hidden_states[-1][index, : len(source)],
            rtol=0.0,
            atol=1e-6,
        )


def test_auxiliary_target_cannot_change_source_or_native_logits():
    model, _, method, batch = setup_case()
    _, targets = extract_source_and_target(batch)
    baseline = method(model, batch, k=2, jepa_weight=1.0)
    changed = method(model, batch, k=2, jepa_weight=1.0, jepa_targets=list(reversed(targets)))
    assert torch.equal(baseline.logits, changed.logits)
    assert torch.equal(baseline.source_states, changed.source_states)


def test_source_changes_predictor_state_and_target_eos_is_extracted():
    model, tokenizer, method, batch = setup_case()
    baseline = method(model, batch, k=2, jepa_weight=1.0)
    changed_batch = copy.deepcopy(batch)
    changed_batch["input_ids"][0, 1] = method.predictor_token_ids[-1]
    changed = method(model, changed_batch, k=2, jepa_weight=1.0)
    assert not torch.equal(baseline.source_states[0], changed.source_states[0])
    _, targets = extract_source_and_target(batch)
    for row, index in zip(targets, baseline.target_final_indices):
        assert int(index) == len(row) - 1
        assert int(row[index]) == tokenizer.eos_token_id


def test_k_minus_one_selects_second_to_last_source_token():
    model, _, method, batch = setup_case()
    result = method(model, batch, k=-1, jepa_weight=1.0)
    sources, targets = extract_source_and_target(batch)
    for source, index in zip(sources, result.source_final_indices):
        assert int(index) == len(source) - 2
    for target, index in zip(targets, result.target_final_indices):
        assert int(index) == len(target) - 1


def test_k_zero_uses_existing_source_eos_without_predictor_token():
    model, tokenizer, method, batch = setup_case()
    result = method(model, batch, k=0, jepa_weight=1.0)
    sources, _ = extract_source_and_target(batch)
    for source, index in zip(sources, result.source_final_indices):
        assert int(index) == len(source) - 1
        assert int(source[index]) == tokenizer.eos_token_id


def test_sigreg_matches_lejepa_epps_pulley_formulation():
    torch.manual_seed(29)
    values = torch.randn(2, 7, 11, requires_grad=True)
    sigreg = SIGReg(knots=17, t_max=3.0, num_slices=23, seed=533)
    actual = sigreg(values)

    generator = torch.Generator().manual_seed(533)
    directions = torch.randn(11, 23, generator=generator)
    directions = directions / directions.norm(dim=0, keepdim=True)
    t = torch.linspace(0.0, 3.0, 17)
    dt = 3.0 / 16
    quadrature = torch.full((17,), 2.0 * dt)
    quadrature[[0, -1]] = dt
    phi = torch.exp(-0.5 * t.square())
    arguments = (values @ directions).unsqueeze(-1) * t
    error = (
        (arguments.cos().mean(dim=-3) - phi).square()
        + arguments.sin().mean(dim=-3).square()
    )
    expected = ((error @ (quadrature * phi)) * values.size(-2)).mean()
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)


def test_sigreg_symmetric_objective_updates_both_jepa_views():
    model, _, method, batch = setup_case()
    model.eval()
    result = method(
        model, batch, k=1, jepa_weight=1.0, native_weight=0.0,
        sigreg_tradeoff=0.05,
    )
    assert result.sigreg_loss is not None
    torch.testing.assert_close(
        result.jepa_objective_loss,
        result.jepa_loss + (0.05 / 0.95) * result.sigreg_loss,
    )
    result.source_states.retain_grad()
    result.target_states.retain_grad()
    result.loss.backward()
    assert result.source_states.grad is not None
    assert result.target_states.grad is not None
    assert result.source_states.grad.abs().sum() > 0
    assert result.target_states.grad.abs().sum() > 0


def test_mse_matches_upstream_llm_jepa_definition_exactly():
    model, _, method, batch = setup_case()
    result = method(
        model, batch, k=0, jepa_weight=1.0, native_weight=0.0,
        jepa_loss_type="mse", force_jepa_active=True,
    )
    expected = torch.mean((result.source_states - result.target_states) ** 2)
    torch.testing.assert_close(result.jepa_loss, expected, rtol=0.0, atol=0.0)
    torch.testing.assert_close(result.loss, expected, rtol=0.0, atol=0.0)


def test_two_view_lejepa_mapping_preserves_raw_pairwise_mse_coefficient():
    model, _, method, batch = setup_case()
    tradeoff = 0.01
    result = method(
        model, batch, k=0, jepa_weight=2.0, native_weight=0.0,
        jepa_loss_type="mse", sigreg_tradeoff=tradeoff,
        sigreg_relative_scale=4.0, force_jepa_active=True,
    )
    center = (result.source_states + result.target_states) / 2
    center_loss = torch.stack((
        (result.source_states - center).square(),
        (result.target_states - center).square(),
    )).mean()
    torch.testing.assert_close(center_loss, result.jepa_loss / 4, rtol=1e-6, atol=1e-7)
    expected = result.jepa_loss + (4 * tradeoff / (1 - tradeoff)) * result.sigreg_loss
    torch.testing.assert_close(result.jepa_objective_loss, expected)


def test_streaming_sigreg_matches_materialized_value_and_gradients():
    torch.manual_seed(41)
    direct_values = torch.randn(2, 13, 19, requires_grad=True)
    streaming_values = direct_values.detach().clone().requires_grad_(True)
    direct = SIGReg(knots=17, t_max=3.0, num_slices=31, seed=533)
    streamed = SIGReg(knots=17, t_max=3.0, num_slices=31, seed=533)

    direct_loss = direct(direct_values)
    direct_loss.backward()
    accumulator = streamed.start_streaming(
        views=2, dimensions=19, expected_samples=13,
        device=streaming_values.device,
    )
    for chunk in streaming_values.detach().split((3, 4, 6), dim=1):
        accumulator.update(chunk)
    prepared = accumulator.finalize()
    surrogate = sum(
        prepared.surrogate(chunk)
        for chunk in streaming_values.split((3, 4, 6), dim=1)
    )
    surrogate.backward()

    torch.testing.assert_close(prepared.loss, direct_loss.detach(), rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(
        streaming_values.grad, direct_values.grad, rtol=2e-5, atol=2e-6,
    )


def test_forced_jepa_activity_does_not_consume_dropout_rng():
    model, tokenizer, forced_method, batch = setup_case()
    reference_method = CLMJEPA(
        forced_method.predictor_token_ids,
        tokenizer.eos_token_id,
        tokenizer.pad_token_id,
    )
    before = forced_method.jepa_dropout_generator.get_state().clone()
    forced = forced_method(
        model, batch, k=0, jepa_weight=1.0, jepa_ratio=0.5,
        force_jepa_active=True,
    )
    after = forced_method.jepa_dropout_generator.get_state()
    assert forced.jepa_active
    assert torch.equal(before, after)
    sampled_forced = forced_method(
        model, batch, k=0, jepa_weight=1.0, jepa_ratio=0.5,
    )
    sampled_reference = reference_method(
        model, batch, k=0, jepa_weight=1.0, jepa_ratio=0.5,
    )
    assert sampled_forced.jepa_active == sampled_reference.jepa_active


def test_streaming_recomputation_matches_materialized_tiny_clm_jepa_gradients():
    direct_model, _, direct_method, direct_batch = setup_case()
    streamed_model, _, streamed_method, streamed_batch = setup_case()
    direct_model.eval()
    streamed_model.eval()

    direct = direct_method(
        direct_model, direct_batch, k=0, jepa_weight=2.0,
        sigreg_tradeoff=0.05, force_jepa_active=True,
    )
    direct.loss.backward()

    with torch.no_grad():
        first = streamed_method(
            streamed_model, streamed_batch, k=0, jepa_weight=0.0,
            monitor_only=True, force_jepa_active=True,
        )
        states = torch.stack((first.source_states, first.target_states))
        accumulator = streamed_method.sigreg.start_streaming(
            views=2, dimensions=states.size(-1),
            expected_samples=states.size(1), device=states.device,
        )
        accumulator.update(states)
        prepared = accumulator.finalize()
    streamed = streamed_method(
        streamed_model, streamed_batch, k=0, jepa_weight=2.0,
        sigreg_tradeoff=0.0, force_jepa_active=True,
    )
    streamed_loss = (
        streamed.loss
        + 2.0 * (0.05 / 0.95) * prepared.surrogate(
            torch.stack((streamed.source_states, streamed.target_states))
        )
    )
    streamed_loss.backward()

    expected_value = (
        streamed.loss.detach() + 2.0 * (0.05 / 0.95) * prepared.loss
    )
    torch.testing.assert_close(expected_value, direct.loss.detach(), rtol=2e-5, atol=2e-6)
    direct_gradients = {
        name: parameter.grad
        for name, parameter in direct_model.named_parameters()
        if parameter.grad is not None
    }
    streamed_gradients = {
        name: parameter.grad
        for name, parameter in streamed_model.named_parameters()
        if parameter.grad is not None
    }
    assert direct_gradients.keys() == streamed_gradients.keys()
    for name in direct_gradients:
        torch.testing.assert_close(
            streamed_gradients[name], direct_gradients[name],
            rtol=3e-4, atol=3e-6,
            msg=lambda message, name=name: f"{name}: {message}",
        )


def test_streaming_mse_sigreg_matches_materialized_value_and_gradients():
    direct_model, _, direct_method, direct_batch = setup_case()
    streamed_model, _, streamed_method, streamed_batch = setup_case()
    direct_model.eval()
    streamed_model.eval()
    tradeoff = 0.01
    relative = 4.0 * tradeoff / (1.0 - tradeoff)

    direct = direct_method(
        direct_model, direct_batch, k=0, jepa_weight=2.0,
        jepa_loss_type="mse", sigreg_tradeoff=tradeoff,
        sigreg_relative_scale=4.0, force_jepa_active=True,
    )
    direct.loss.backward()

    with torch.no_grad():
        first = streamed_method(
            streamed_model, streamed_batch, k=0, jepa_weight=0.0,
            monitor_only=True, jepa_loss_type="mse", force_jepa_active=True,
        )
        states = torch.stack((first.source_states, first.target_states))
        accumulator = streamed_method.sigreg.start_streaming(
            views=2, dimensions=states.size(-1),
            expected_samples=states.size(1), device=states.device,
        )
        accumulator.update(states)
        prepared = accumulator.finalize()
    streamed = streamed_method(
        streamed_model, streamed_batch, k=0, jepa_weight=2.0,
        jepa_loss_type="mse", force_jepa_active=True,
    )
    streamed_loss = streamed.loss + 2.0 * relative * prepared.surrogate(
        torch.stack((streamed.source_states, streamed.target_states))
    )
    streamed_loss.backward()

    expected_value = streamed.loss.detach() + 2.0 * relative * prepared.loss
    torch.testing.assert_close(expected_value, direct.loss.detach(), rtol=2e-5, atol=2e-6)
    direct_gradients = {
        name: parameter.grad for name, parameter in direct_model.named_parameters()
        if parameter.grad is not None
    }
    streamed_gradients = {
        name: parameter.grad for name, parameter in streamed_model.named_parameters()
        if parameter.grad is not None
    }
    assert direct_gradients.keys() == streamed_gradients.keys()
    for name in direct_gradients:
        torch.testing.assert_close(
            streamed_gradients[name], direct_gradients[name],
            rtol=3e-4, atol=3e-6,
            msg=lambda message, name=name: f"{name}: {message}",
        )


def test_streaming_mse_pcsf_matches_materialized_value_and_parameter_gradients():
    direct_model, _, direct_method, direct_batch = setup_case()
    streamed_model, _, streamed_method, streamed_batch = setup_case()
    direct_model.eval()
    streamed_model.eval()
    beta = 3.25
    regularizer = PairCenterSpreadFloor(rho=0.8)

    direct = direct_method(
        direct_model, direct_batch, k=0, jepa_weight=2.0,
        jepa_loss_type="mse", force_jepa_active=True,
    )
    reference = (
        (direct.source_states.detach() + direct.target_states.detach()) * 0.5 / 0.2
    )
    direct_pcsf, _, _ = regularizer(
        direct.source_states, direct.target_states, reference,
    )
    direct_loss = direct.loss + 2.0 * beta * direct_pcsf
    direct_loss.backward()

    with torch.no_grad():
        first = streamed_method(
            streamed_model, streamed_batch, k=0, jepa_weight=0.0,
            monitor_only=True, jepa_loss_type="mse", force_jepa_active=True,
        )
        accumulator = regularizer.start_streaming(
            expected_samples=3, dimensions=first.source_states.size(-1),
        )
        for indices in (slice(0, 1), slice(1, 3)):
            accumulator.update(
                first.source_states[indices], first.target_states[indices],
                reference[indices],
            )
        prepared = accumulator.finalize()
    streamed = streamed_method(
        streamed_model, streamed_batch, k=0, jepa_weight=2.0,
        jepa_loss_type="mse", force_jepa_active=True,
    )
    streamed_loss = streamed.loss + 2.0 * beta * sum(
        prepared.surrogate(
            streamed.source_states[indices], streamed.target_states[indices],
        )
        for indices in (slice(0, 1), slice(1, 3))
    )
    streamed_loss.backward()

    torch.testing.assert_close(prepared.loss, direct_pcsf.detach(), rtol=2e-5, atol=2e-6)
    expected_value = streamed.loss.detach() + 2.0 * beta * prepared.loss
    torch.testing.assert_close(expected_value, direct_loss.detach(), rtol=2e-5, atol=2e-6)
    direct_gradients = {
        name: parameter.grad for name, parameter in direct_model.named_parameters()
        if parameter.grad is not None
    }
    streamed_gradients = {
        name: parameter.grad for name, parameter in streamed_model.named_parameters()
        if parameter.grad is not None
    }
    assert direct_gradients.keys() == streamed_gradients.keys()
    for name in direct_gradients:
        torch.testing.assert_close(
            streamed_gradients[name], direct_gradients[name],
            rtol=4e-4, atol=4e-6, msg=name,
        )


def test_jepa_gradients_reach_shared_backbone_but_monitor_only_adds_none():
    model, _, method, batch = setup_case()
    model.train()
    result = method(model, batch, k=2, jepa_weight=1.0, native_weight=0.0)
    result.loss.backward()
    parameter = model.model.layers[0].self_attn.q_proj.weight
    assert parameter.grad is not None and parameter.grad.abs().sum() > 0

    model.zero_grad(set_to_none=True)
    monitored = method(model, batch, k=2, jepa_weight=1.0, native_weight=0.0, monitor_only=True)
    monitored.loss.backward()
    assert parameter.grad is not None and torch.count_nonzero(parameter.grad) == 0


def test_target_stop_gradient_changes_only_jepa_gradient_path():
    symmetric_model, _, symmetric_method, batch = setup_case()
    asymmetric_model = copy.deepcopy(symmetric_model)
    asymmetric_method = CLMJEPA(
        symmetric_method.predictor_token_ids,
        symmetric_method.eos_token_id,
        symmetric_method.pad_token_id,
    )
    symmetric_model.eval()
    asymmetric_model.eval()

    symmetric = symmetric_method(
        symmetric_model, batch, k=1, jepa_weight=1.0, native_weight=0.0
    )
    asymmetric = asymmetric_method(
        asymmetric_model,
        batch,
        k=1,
        jepa_weight=1.0,
        native_weight=0.0,
        stop_gradient_target=True,
    )
    torch.testing.assert_close(asymmetric.loss, symmetric.loss, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        asymmetric.target_states, symmetric.target_states, rtol=0.0, atol=0.0
    )

    symmetric.source_states.retain_grad()
    symmetric.target_states.retain_grad()
    asymmetric.source_states.retain_grad()
    asymmetric.target_states.retain_grad()
    symmetric.loss.backward()
    asymmetric.loss.backward()

    assert symmetric.source_states.grad is not None
    assert symmetric.target_states.grad is not None
    assert asymmetric.source_states.grad is not None
    assert asymmetric.target_states.grad is None
    assert asymmetric.source_states.grad.abs().sum() > 0


def test_target_stop_gradient_preserves_native_parameter_updates():
    combined_model, _, combined_method, batch = setup_case()
    native_model = copy.deepcopy(combined_model)
    jepa_model = copy.deepcopy(combined_model)
    combined_model.eval()
    native_model.eval()
    jepa_model.eval()

    combined = combined_method(
        combined_model,
        batch,
        k=1,
        jepa_weight=1.3,
        native_weight=1.0,
        stop_gradient_target=True,
    )
    native = native_model(**batch)
    jepa_method = CLMJEPA(
        combined_method.predictor_token_ids,
        combined_method.eos_token_id,
        combined_method.pad_token_id,
    )
    jepa = jepa_method(
        jepa_model,
        batch,
        k=1,
        jepa_weight=1.3,
        native_weight=0.0,
        stop_gradient_target=True,
    )
    combined.loss.backward()
    native.loss.backward()
    jepa.loss.backward()

    combined_parameters = dict(combined_model.named_parameters())
    native_parameters = dict(native_model.named_parameters())
    jepa_parameters = dict(jepa_model.named_parameters())
    checked = 0
    for name, parameter in combined_parameters.items():
        if parameter.grad is None:
            continue
        expected = native_parameters[name].grad + jepa_parameters[name].grad
        torch.testing.assert_close(parameter.grad, expected, rtol=2e-5, atol=2e-6, msg=name)
        checked += 1
    assert checked > 0


def test_shuffled_targets_are_reproducible_and_never_correct():
    _, _, _, batch = setup_case()
    _, targets = extract_source_and_target(batch)
    first = matched_derangement(targets, seed=533)
    second = matched_derangement(targets, seed=533)
    assert first == second
    assert all(index != shuffled for index, shuffled in enumerate(first))
    assert all(not torch.equal(targets[index], targets[shuffled]) for index, shuffled in enumerate(first))
