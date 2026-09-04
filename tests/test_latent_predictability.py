import json

import pytest
import torch
from transformers import LlamaConfig, LlamaModel

from latent_predictability import (
    SPLIT_COUNTS,
    assert_disjoint_confirmation,
    chemical_pair_id,
    canonical_atom_correspondence,
    decoder_distribution_metrics,
    history_positions,
    invariance_metrics,
    latent_metrics,
    locked_reaction_split,
    shuffled_reaction_targets,
    suffix_replay_one_position,
    build_suffix_cache,
    replay_suffix_from_cache,
)


def test_locked_split_is_exact_deterministic_and_disjoint():
    identities = [f"reaction-{index:04d}" for index in range(1024)]
    first = locked_reaction_split(identities)
    second = locked_reaction_split(list(reversed(identities)))
    assert first == second
    assert {name: sum(value == name for value in first.values()) for name in SPLIT_COUNTS} == SPLIT_COUNTS


def test_confirmation_overlap_is_fatal(tmp_path):
    manifest = tmp_path / "confirmation.jsonl"
    held_out_pair = chemical_pair_id("A.B", "C")
    manifest.write_text(json.dumps({
        "reaction_identity": "official-group-hash-unrelated-to-chemistry",
        "chemical_pair_id": held_out_pair,
    }) + "\n", encoding="utf-8")
    assert assert_disjoint_confirmation({chemical_pair_id("D", "E")}, manifest)["chemical_pair_overlap"] == 0
    with pytest.raises(RuntimeError, match="leakage"):
        # Different identity namespace, identical canonical chemistry: reject.
        assert_disjoint_confirmation({held_out_pair}, manifest, {"validation-source-target-string"})


def test_history_positions_never_use_future_input():
    rows = history_positions([10, 11, 12, 13, 14, 15, 16], horizon=2, history=3)
    assert rows == [([10, 11, 12, 13], 15), ([11, 12, 13, 14], 16)]
    assert all(max(past) < future for past, future in rows)


def test_latent_metrics_have_unit_constant_nmse_and_zero_r2():
    target = torch.tensor([[0.0, 2.0], [2.0, 0.0]])
    mean = torch.tensor([1.0, 1.0])
    result = latent_metrics(target, mean.expand_as(target), mean)
    assert result["normalized_mse"] == pytest.approx(1.0)
    assert result["r2"] == pytest.approx(0.0)


def test_decoder_metrics_direction_and_overlap():
    logits = torch.tensor([[3.0, 1.0, -1.0]])
    same = decoder_distribution_metrics(logits, logits.clone(), torch.tensor([0]), topk=(2,))
    assert same["kl_true_predicted"].item() == pytest.approx(0.0, abs=1e-7)
    assert same["js"].item() == pytest.approx(0.0, abs=1e-7)
    assert same["gold_rank"].item() == 1
    assert same["top2_overlap"].item() == 1


def test_invariance_ratio_is_zero_for_identical_views():
    values = torch.tensor([[[1.0, 0.0], [1.0, 0.0]], [[0.0, 1.0], [0.0, 1.0]]])
    result = invariance_metrics(values)
    assert result["within_between_ratio"] == pytest.approx(0.0)
    assert result["matched_view_cosine"] == pytest.approx(1.0)


def test_graph_correspondence_is_complete_under_serialization_change():
    mapping = canonical_atom_correspondence("CCO", "OCC")
    assert sorted(mapping) == [0, 1, 2]
    assert sorted(mapping.values()) == [0, 1, 2]


def test_shuffled_targets_never_use_same_reaction():
    metadata = [
        {"reaction_identity": "a", "sequence_length": 10, "future_index": 3},
        {"reaction_identity": "b", "sequence_length": 11, "future_index": 4},
        {"reaction_identity": "c", "sequence_length": 12, "future_index": 5},
    ]
    target = torch.arange(6).reshape(3, 2)
    shuffled, donors = shuffled_reaction_targets(target, metadata, 7)
    assert torch.equal(shuffled, target[donors])
    assert all(metadata[i]["reaction_identity"] != metadata[j]["reaction_identity"] for i, j in enumerate(donors))


def test_true_state_suffix_replay_matches_full_causal_forward():
    config = LlamaConfig(
        vocab_size=20, hidden_size=16, intermediate_size=32,
        num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=32,
    )
    model = LlamaModel(config).eval()
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    output = model(ids, output_hidden_states=True, use_cache=False)
    replay = suffix_replay_one_position(
        model, 1, output.hidden_states[1], output.hidden_states[1][0, 3], 3,
    )
    assert torch.allclose(replay[0], output.last_hidden_state[0, 3], atol=1e-6, rtol=1e-6)
    cache, final = build_suffix_cache(model, 1, output.hidden_states[1])
    alternatives = torch.stack((output.hidden_states[1][0, 3], output.hidden_states[1][0, 3] + 0.01))
    batched = replay_suffix_from_cache(model, 1, cache, alternatives, 3)
    scalar = replay_suffix_from_cache(model, 1, cache, alternatives[:1], 3)
    assert torch.allclose(final[0, 3], output.last_hidden_state[0, 3], atol=1e-6, rtol=1e-6)
    assert torch.allclose(batched[0], scalar[0], atol=1e-6, rtol=1e-6)
