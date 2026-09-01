"""Parity tests for the pinned official STP random-span implementation."""

import copy
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import LlamaConfig, LlamaForCausalLM


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chemfm import TOKENIZER_DIR, ReactionCollator, load_reaction_tokenizer
from stp import (
    STP_UPSTREAM_COMMIT, PaperSemanticTubePrediction,
    SemanticTubePrediction,
)


def setup_case():
    torch.manual_seed(29)
    tokenizer = load_reaction_tokenizer(TOKENIZER_DIR)
    model = LlamaForCausalLM(LlamaConfig(
        vocab_size=len(tokenizer), hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=256, pad_token_id=tokenizer.pad_token_id,
    )).eval()
    batch = ReactionCollator(tokenizer)([
        {"src": "CCO.O=O", "tgt": "CC(=O)O"},
        {"src": "CCBr.N", "tgt": "CCN"},
        {"src": "CO.Cl", "tgt": "CCl"},
    ])
    batch = {key: value for key, value in batch.items() if torch.is_tensor(value)}
    method = SemanticTubePrediction(
        seed=82,
        reactant_start_token_id=tokenizer.convert_tokens_to_ids("<rstart>"),
        product_start_token_id=tokenizer.convert_tokens_to_ids("<prostart>"),
        eos_token_id=tokenizer.eos_token_id,
    )
    return model, tokenizer, method, batch


def upstream_get_s_t(full_length, generator):
    """Pinned stp.py default branch, kept independent for executable parity."""
    patch_start_offset = torch.randint(
        0, full_length, (), generator=generator, device=generator.device
    )
    while True:
        patch_end_offset = torch.randint(
            patch_start_offset + 1,
            full_length + 1,
            (),
            generator=generator,
            device=generator.device,
        )
        if patch_end_offset - patch_start_offset < full_length:
            break
    return patch_start_offset, patch_end_offset


def upstream_get_embeddings(
    hidden_states, user_start_end, assistant_start_end,
    patch_start_offset, patch_end_offset,
):
    """Pinned stp.py function, copied without the local method call."""
    user_start = user_start_end[0] + 1
    user_end = user_start_end[1] + 1
    assistant_start = assistant_start_end[0] + 1
    assistant_end = assistant_start_end[1] + 1
    if patch_start_offset + user_start < user_end:
        patch_start = user_start + patch_start_offset
    else:
        patch_start = assistant_start + patch_start_offset - (user_end - user_start)
    if patch_end_offset + user_start < user_end:
        patch_end = user_start + patch_end_offset
    else:
        patch_end = assistant_start + patch_end_offset - (user_end - user_start)

    user_start_embedding = hidden_states[user_start - 1]
    user_end_embedding = hidden_states[user_end - 1]
    assistant_start_embedding = hidden_states[assistant_start - 1]
    assistant_end_embedding = hidden_states[assistant_end - 1]
    patch_start_embedding = hidden_states[patch_start - 1]
    patch_end_embedding = hidden_states[patch_end - 1]

    if patch_start >= assistant_start:
        before = user_end_embedding - user_start_embedding + patch_start_embedding - assistant_start_embedding
        patch = patch_end_embedding - patch_start_embedding
        after = assistant_end_embedding - patch_end_embedding
    elif patch_end <= user_end:
        before = patch_start_embedding - user_start_embedding
        patch = patch_end_embedding - patch_start_embedding
        after = user_end_embedding - patch_end_embedding + assistant_end_embedding - assistant_start_embedding
    else:
        before = patch_start_embedding - user_start_embedding
        patch = user_end_embedding - patch_start_embedding + patch_end_embedding - assistant_start_embedding
        after = assistant_end_embedding - patch_end_embedding
    return before, patch, after


def upstream_reference(model, batch, method, seed=82, weight=0.02):
    outputs = model(**batch, output_hidden_states=True)
    hidden_states = outputs.hidden_states[-1]
    user_start_end, assistant_start_end = method.content_boundaries(batch)
    generator = torch.Generator(device=hidden_states.device).manual_seed(seed)
    user_embedding = torch.zeros(
        hidden_states.shape[0], hidden_states.shape[-1],
        device=hidden_states.device,
    )
    assistant_embedding = torch.zeros_like(user_embedding)
    spans = []
    for index in range(hidden_states.shape[0]):
        full_length = int(
            user_start_end[index, 1] - user_start_end[index, 0]
            + assistant_start_end[index, 1] - assistant_start_end[index, 0]
        )
        start, end = upstream_get_s_t(full_length, generator)
        before, patch, after = upstream_get_embeddings(
            hidden_states[index], user_start_end[index], assistant_start_end[index],
            start, end,
        )
        user_embedding[index] = before + after
        assistant_embedding[index] = patch
        spans.append((int(start), int(end), full_length))
    stp_loss = 1.0 - F.cosine_similarity(
        user_embedding, assistant_embedding, dim=-1
    ).mean()
    return outputs.loss + weight * stp_loss, outputs.loss, stp_loss, tuple(spans)


def test_upstream_commit_is_pinned():
    assert STP_UPSTREAM_COMMIT == "ea0017c654ad917066ff32afc88276bea8ca5f7e"


def test_chemfm_boundaries_exclude_only_serialization_framing():
    _, tokenizer, method, batch = setup_case()
    user, assistant = method.content_boundaries(batch)
    expected_sources = ("CCO.O=O", "CCBr.N", "CO.Cl")
    expected_targets = ("CC(=O)O", "CCN", "CCl")
    for index, ids in enumerate(batch["input_ids"]):
        active = ids[: int(batch["attention_mask"][index].sum())]
        assert active[user[index, 0] + 1:user[index, 1] + 1].tolist() == tokenizer(
            expected_sources[index], add_special_tokens=False
        )["input_ids"]
        assert active[
            assistant[index, 0] + 1:assistant[index, 1] + 1
        ].tolist() == tokenizer(
            expected_targets[index], add_special_tokens=False
        )["input_ids"]


def test_sampler_matches_upstream_draws_and_never_returns_full_span():
    _, _, method, _ = setup_case()
    reference = torch.Generator().manual_seed(82)
    for full_length in (2, 3, 17, 63):
        for _ in range(50):
            expected = upstream_get_s_t(full_length, reference)
            observed = method.get_s_t(full_length, device=torch.device("cpu"))
            assert tuple(map(int, observed)) == tuple(map(int, expected))
            assert 0 <= int(observed[0]) < int(observed[1]) <= full_length
            assert int(observed[1] - observed[0]) < full_length


def test_loss_states_spans_and_parameter_gradients_match_upstream_executable():
    local_model, _, local_method, batch = setup_case()
    reference_model = copy.deepcopy(local_model)
    local = local_method(local_model, batch, stp_weight=0.02)
    expected_loss, expected_native, expected_stp, expected_spans = upstream_reference(
        reference_model, batch, local_method,
    )
    torch.testing.assert_close(local.loss, expected_loss, rtol=0.0, atol=0.0)
    torch.testing.assert_close(local.native_loss, expected_native, rtol=0.0, atol=0.0)
    torch.testing.assert_close(local.jepa_loss, expected_stp, rtol=0.0, atol=0.0)
    assert local.sampled_spans == expected_spans

    local.loss.backward()
    expected_loss.backward()
    for (local_name, local_parameter), (reference_name, reference_parameter) in zip(
        local_model.named_parameters(), reference_model.named_parameters()
    ):
        assert local_name == reference_name
        if local_parameter.grad is None:
            assert reference_parameter.grad is None
        else:
            torch.testing.assert_close(
                local_parameter.grad, reference_parameter.grad,
                rtol=0.0, atol=0.0, msg=local_name,
            )


def test_released_objective_is_patch_versus_complement_not_paper_three_point():
    hidden = torch.arange(12 * 4, dtype=torch.float32).reshape(12, 4).square()
    user = torch.tensor([0, 4])
    assistant = torch.tensor([6, 10])
    before, patch, after = upstream_get_embeddings(hidden, user, assistant, 2, 5)
    released = 1.0 - F.cosine_similarity(
        (before + after).unsqueeze(0), patch.unsqueeze(0)
    )[0]
    paper_three_point = 1.0 - F.cosine_similarity(
        (hidden[5] - hidden[3]).unsqueeze(0),
        (hidden[3] - hidden[1]).unsqueeze(0),
    )[0]
    assert not torch.isclose(released, paper_three_point)


def test_paper_objective_matches_literal_three_point_equation():
    model, tokenizer, _, batch = setup_case()
    method = PaperSemanticTubePrediction(
        seed=82,
        reactant_start_token_id=tokenizer.convert_tokens_to_ids("<rstart>"),
        product_start_token_id=tokenizer.convert_tokens_to_ids("<prostart>"),
        eos_token_id=tokenizer.eos_token_id,
    )
    reference_model = copy.deepcopy(model)
    observed = method(model, batch, stp_weight=0.02)

    outputs = reference_model(**batch, output_hidden_states=True)
    hidden = outputs.hidden_states[-1]
    user, assistant = method.content_boundaries(batch)
    reference_method = PaperSemanticTubePrediction(
        seed=82,
        reactant_start_token_id=method.reactant_start_token_id,
        product_start_token_id=method.product_start_token_id,
        eos_token_id=method.eos_token_id,
    )
    transitions = []
    expected_spans = []
    for index in range(hidden.shape[0]):
        full = int(
            user[index, 1] - user[index, 0]
            + assistant[index, 1] - assistant[index, 0]
        )
        s, r, t = reference_method.get_s_r_t(full, device=hidden.device)
        def semantic_state(offset):
            user_start = user[index, 0] + 1
            user_end = user[index, 1] + 1
            assistant_start = assistant[index, 0] + 1
            user_length = user_end - user_start
            if offset <= user_length:
                return hidden[index, user_start + offset - 1]
            assistant_boundary = assistant_start + offset - user_length
            return (
                hidden[index, user_end - 1]
                + hidden[index, assistant_boundary - 1]
                - hidden[index, assistant_start - 1]
            )

        h_s = semantic_state(s)
        h_r = semantic_state(r)
        h_t = semantic_state(t)
        transitions.append(((h_r - h_s).float(), (h_t - h_r).float()))
        expected_spans.append((int(s), int(r), int(t), full))
    expected = 1.0 - F.cosine_similarity(
        torch.stack([value[0] for value in transitions]),
        torch.stack([value[1] for value in transitions]),
        dim=-1,
    ).mean()
    torch.testing.assert_close(observed.jepa_loss, expected, rtol=0.0, atol=0.0)
    assert observed.sampled_spans == tuple(expected_spans)
    assert all(s < r < t for s, r, t, _ in observed.sampled_spans)


def test_paper_semantic_path_exactly_recovers_released_patch_transitions():
    hidden = torch.randn(14, 7)
    user = torch.tensor([0, 5])
    assistant = torch.tensor([7, 12])
    method = PaperSemanticTubePrediction
    full = int(user[1] - user[0] + assistant[1] - assistant[0])
    for start in range(full):
        for end in range(start + 1, full + 1):
            _, released_patch, _ = upstream_get_embeddings(
                hidden, user, assistant, start, end
            )
            paper_transition = (
                method.semantic_path_embedding(hidden, user, assistant, end)
                - method.semantic_path_embedding(hidden, user, assistant, start)
            )
            torch.testing.assert_close(
                paper_transition, released_patch, rtol=0.0, atol=1e-6
            )


def test_released_and_paper_objectives_are_genuinely_distinct():
    released_model, tokenizer, released, batch = setup_case()
    paper_model = copy.deepcopy(released_model)
    paper = PaperSemanticTubePrediction(
        seed=82,
        reactant_start_token_id=tokenizer.convert_tokens_to_ids("<rstart>"),
        product_start_token_id=tokenizer.convert_tokens_to_ids("<prostart>"),
        eos_token_id=tokenizer.eos_token_id,
    )
    released_value = released(released_model, batch, stp_weight=0.02).jepa_loss
    paper_value = paper(paper_model, batch, stp_weight=0.02).jepa_loss
    assert not torch.isclose(released_value, paper_value)
