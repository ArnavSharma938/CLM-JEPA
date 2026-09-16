from types import SimpleNamespace

import torch
import pytest
from torch import nn
from torch.nn import functional as F

from protein.src.nextlat import FaithfulNextLatPredictor
from protein.src.pilot import (
    causal_pilot_loss, checkpoint_crossings, directional_examples,
    labels_with_padding_ignored, reference_two_stage_backward,
)
from protein.src.pilot_eval import bidirectional_summary, sample_top_p_uniform
from protein.scripts.prepare_causal_pilot import iter_json_array, normalize_fraction
from protein.scripts.run_esmfold_pilot import structure_metrics
from protein.scripts.summarize_causal_pilot import per_sequence_generation_value
from protein.scripts.evaluate_causal_mechanism import stable_decoder_transition_relative_js


class TinyCausalModel(nn.Module):
    def __init__(self, vocab=26, hidden=8):
        super().__init__()
        self.embedding = nn.Embedding(vocab, hidden)
        self.trunk = nn.Linear(hidden, hidden, bias=False)
        self.head = nn.Linear(hidden, vocab, bias=False)

    def get_input_embeddings(self):
        return self.embedding

    def get_output_embeddings(self):
        return self.head

    def forward(self, input_ids, attention_mask=None, labels=None, use_cache=False):
        hidden = self.trunk(self.embedding(input_ids))
        logits = self.head(hidden)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]),
                labels[:, 1:].reshape(-1), ignore_index=-100,
            )
        return SimpleNamespace(hidden_states=hidden, logits=logits, loss=loss)


def batch():
    return {
        "input_ids": torch.tensor([[4, 5, 6, 7, 2], [8, 9, 10, 2, 1]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 1, 0]]),
    }


def test_optimized_matches_literal_upstream_two_stage_vjp():
    torch.manual_seed(17)
    direct_model = TinyCausalModel()
    direct_predictor = FaithfulNextLatPredictor(8, proj_factor=8.0)
    reference_model = TinyCausalModel()
    reference_predictor = FaithfulNextLatPredictor(8, proj_factor=8.0)
    reference_model.load_state_dict(direct_model.state_dict())
    reference_predictor.load_state_dict(direct_predictor.state_dict())

    direct = causal_pilot_loss(direct_model, direct_predictor, batch(), "nextlat")
    direct.total.backward()
    reference = reference_two_stage_backward(reference_model, reference_predictor, batch())
    torch.testing.assert_close(direct.total, reference.total, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(direct.ntp, reference.ntp, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(direct.latent, reference.latent, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(direct.kl, reference.kl, rtol=1e-6, atol=1e-7)
    for first, second in zip(direct_model.parameters(), reference_model.parameters()):
        if first.grad is not None or second.grad is not None:
            torch.testing.assert_close(first.grad, second.grad, rtol=2e-5, atol=2e-6)
    for first, second in zip(direct_predictor.parameters(), reference_predictor.parameters()):
        torch.testing.assert_close(first.grad, second.grad, rtol=2e-5, atol=2e-6)


def test_native_is_exact_ntp_and_padding_is_ignored():
    model = TinyCausalModel()
    result = causal_pilot_loss(model, None, batch(), "native")
    labels = labels_with_padding_ignored(batch())
    expected = model(**batch(), labels=labels).loss
    torch.testing.assert_close(result.total, expected)
    assert labels[1, -1] == -100


def test_bidirectional_reversal_precedes_tokenization_and_is_exact():
    rows = [{"id": "p", "cluster_id": "c", "sequence": "ACDE",
             "orientation_order": ["reverse", "forward"]}]
    output = list(directional_examples(rows))
    assert [(row["orientation"], row["sequence"]) for row in output] == [
        ("reverse", "EDCA"), ("forward", "ACDE")
    ]


def test_checkpoint_targets_are_fixed_residue_fractions():
    assert checkpoint_crossings(6_000_000) == {
        .25: 1_500_000, .5: 3_000_000, 1.0: 6_000_000
    }


def test_bidirectional_aggregation_is_per_protein_not_per_token():
    rows = [
        {"id": "a", "cluster_id": "x", "orientation": "forward", "length": 10,
         "ntp_loss": 1.0, "perplexity": 2.0, "correct_probability": .4, "top1": .2},
        {"id": "a", "cluster_id": "x", "orientation": "reverse", "length": 10,
         "ntp_loss": 3.0, "perplexity": 4.0, "correct_probability": .2, "top1": .0},
    ]
    result = bidirectional_summary(rows)
    assert result[0]["bidirectional_ntp_loss"] == 2.0
    assert result[0]["correct_probability"] == pytest.approx(.3)


def test_streaming_json_array_and_fraction_normalization(tmp_path):
    path = tmp_path / "large.json"
    path.write_text('[ {"a": 1},\n {"a": [2, 3]} ]', encoding="utf-8")
    assert list(iter_json_array(path, chunk_size=5)) == [{"a": 1}, {"a": [2, 3]}]
    assert normalize_fraction("80") == pytest.approx(.8)
    assert normalize_fraction("0.8") == pytest.approx(.8)


def test_inverse_cdf_sampler_is_rowwise_and_deterministic():
    logits = torch.tensor([[3.0, 2.0, 1.0], [1.0, 2.0, 3.0]])
    first = sample_top_p_uniform(logits, torch.tensor([0.01, 0.01]), 1.0, .95)
    last = sample_top_p_uniform(logits, torch.tensor([0.99, 0.99]), 1.0, .95)
    assert first.squeeze(1).tolist() == [0, 2]
    assert last.squeeze(1).tolist() == [2, 0]


def test_esmfold_plddt_threshold_uses_unit_interval_scale():
    plddt = torch.zeros(1, 3, 37)
    plddt[0, :, 1] = torch.tensor([.69, .70, .80])
    positions = torch.zeros(1, 1, 3, 37, 3)
    output = SimpleNamespace(plddt=plddt, positions=positions)
    metrics = structure_metrics(output, length=3)
    assert metrics["mean_plddt"] == pytest.approx(.73)
    assert metrics["fraction_plddt_ge_70"] == pytest.approx(2 / 3)


def test_generation_termination_rate_uses_boolean_record_field():
    record = {"terminated": True, "length": 93}
    assert per_sequence_generation_value(record, "termination_rate") == 1.0
    assert per_sequence_generation_value(record, "length") == 93


def test_decoder_transition_relative_js_aggregates_before_division():
    diagnostics = {
        "decoder_js": torch.tensor([-1e-9, 1.0]),
        "decoder_transition_js": torch.tensor([0.0, 1.0]),
    }
    value = stable_decoder_transition_relative_js(diagnostics)
    assert value.item() == pytest.approx(1.0)
