import copy
import sys
from pathlib import Path

import torch
from transformers import LlamaConfig, LlamaForCausalLM

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chemfm import TOKENIZER_DIR, ReactionCollator, load_reaction_tokenizer
from jepa import CLMJEPA, add_predictor_tokens, extract_source_and_target, matched_derangement


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
