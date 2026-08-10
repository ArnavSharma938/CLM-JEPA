"""Executable parity checks against galilai-group/llm-jepa finetune.py.

Reference formulas mirror commit ea0017c, lines 538-564, 603-665, 710-715,
735, and 739. ReactionCollator handles chemical marker placement.
"""

import copy

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from chemfm import IGNORE_INDEX
from jepa import extract_source_and_target
from test_jepa import setup_case


def reference_pack(batch, sources, targets, predictor_ids, k, pad_id):
    suffix = list(reversed(predictor_ids[:k]))
    source_rows = [
        torch.cat((source, source.new_tensor(suffix))) if suffix else source
        for source in sources
    ]
    native_rows = [
        row[mask.bool()] for row, mask in zip(batch["input_ids"], batch["attention_mask"])
    ]
    native_labels = [
        row[mask.bool()] for row, mask in zip(batch["labels"], batch["attention_mask"])
    ]
    rows = native_rows + source_rows + targets
    labels = native_labels + [
        row.new_full(row.shape, IGNORE_INDEX) for row in source_rows + targets
    ]
    return (
        pad_sequence(rows, batch_first=True, padding_value=pad_id),
        pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX),
    )


def reference_forward(model, batch, predictor_ids, k, pad_id, native_weight, jepa_weight):
    """Independent executable transcription of upstream finetune.py's core path."""
    sources, targets = extract_source_and_target(batch)
    packed_ids, packed_labels = reference_pack(
        batch, sources, targets, predictor_ids, k, pad_id
    )
    attention = packed_ids.ne(pad_id)
    outputs = model(
        input_ids=packed_ids,
        attention_mask=attention,
        labels=packed_labels,
        output_hidden_states=True,
    )
    batch_size = len(sources)
    hidden = outputs.hidden_states[-1]
    rows = torch.arange(batch_size, device=hidden.device)
    source_indices = attention[batch_size:2 * batch_size].sum(dim=1) - 1
    target_indices = attention[2 * batch_size:].sum(dim=1) - 1
    source_states = hidden[rows + batch_size, source_indices]
    target_states = hidden[rows + 2 * batch_size, target_indices]
    jepa_loss = 1.0 - F.cosine_similarity(source_states, target_states, dim=-1).mean()
    loss = native_weight * outputs.loss + jepa_weight * jepa_loss
    return loss, outputs.loss, jepa_loss, source_states, target_states


def test_packed_rows_indices_and_cosine_loss_match_official_formulas():
    model, _, method, batch = setup_case()
    captured = {}
    original = model.forward

    def forward(*args, **kwargs):
        captured.update({
            key: value.detach().clone()
            for key, value in kwargs.items() if torch.is_tensor(value)
        })
        return original(*args, **kwargs)

    model.forward = forward
    result = method(model, batch, k=3, jepa_weight=2.0)
    sources, targets = extract_source_and_target(batch)
    expected_ids, expected_labels = reference_pack(
        batch, sources, targets, method.predictor_token_ids, 3, method.pad_token_id
    )
    assert torch.equal(captured["input_ids"], expected_ids)
    assert torch.equal(captured["labels"], expected_labels)
    assert result.source_final_indices.tolist() == [len(row) + 2 for row in sources]
    assert result.target_final_indices.tolist() == [len(row) - 1 for row in targets]
    reference_jepa = 1.0 - F.cosine_similarity(
        result.source_states, result.target_states, dim=-1
    ).mean()
    torch.testing.assert_close(result.jepa_loss, reference_jepa)
    torch.testing.assert_close(result.loss, result.native_loss + 2.0 * reference_jepa)


def test_jepa_ratio_uses_official_strict_greater_than_skip_rule(monkeypatch):
    model, _, method, batch = setup_case()
    monkeypatch.setattr(torch, "rand", lambda *args, **kwargs: torch.tensor([0.8]))
    skipped = method(model, batch, k=1, jepa_weight=1.0, jepa_ratio=0.5)
    assert skipped.jepa_active is False
    assert skipped.jepa_loss is None
    monkeypatch.setattr(torch, "rand", lambda *args, **kwargs: torch.tensor([0.5]))
    active = method(model, batch, k=1, jepa_weight=1.0, jepa_ratio=0.5)
    assert active.jepa_active is True
    assert active.jepa_loss is not None


def test_exact_loss_states_and_parameter_gradients_match_executable_reference():
    local_model, _, method, batch = setup_case()
    reference_model = copy.deepcopy(local_model)
    local_model.eval()
    reference_model.eval()

    local = method(
        local_model, batch, k=2, native_weight=0.6, jepa_weight=1.7
    )
    reference = reference_forward(
        reference_model,
        batch,
        method.predictor_token_ids,
        2,
        method.pad_token_id,
        0.6,
        1.7,
    )
    for actual, expected in zip(
        (local.loss, local.native_loss, local.jepa_loss, local.source_states, local.target_states),
        reference,
    ):
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    local.loss.backward()
    reference[0].backward()
    local_parameters = dict(local_model.named_parameters())
    reference_parameters = dict(reference_model.named_parameters())
    assert local_parameters.keys() == reference_parameters.keys()
    for name in local_parameters:
        actual_grad = local_parameters[name].grad
        expected_grad = reference_parameters[name].grad
        assert (actual_grad is None) == (expected_grad is None), name
        if actual_grad is not None:
            torch.testing.assert_close(actual_grad, expected_grad, rtol=0.0, atol=0.0, msg=name)
