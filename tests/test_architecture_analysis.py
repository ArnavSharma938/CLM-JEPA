import sys
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from architecture_analysis import (
    decoder_coordinates, functional_distribution_metrics,
    orthonormal_decoder_basis, product_transition_rows,
)


def record(states):
    return {"reaction_identity": "r", "split": "test",
            "input_ids": torch.tensor([8, 1, 4, 5, 6, 3]),
            "product_indices": [2, 3, 4], "final_states": states}


def test_product_states_are_absolute_position_sliced_and_tokens_are_exact():
    states = torch.arange(12).reshape(3, 4)
    rows = product_transition_rows([record(states)])
    assert len(rows) == 1
    assert rows[0]["absolute_t"] == 2 and rows[0]["absolute_future"] == 3
    assert rows[0]["oracle_token"] == 5 and rows[0]["gold_token"] == 6
    torch.testing.assert_close(rows[0]["h_t"], states[0].float())
    torch.testing.assert_close(rows[0]["h_next"], states[1].float())
    with pytest.raises(ValueError, match="H\\[product_indices\\]"):
        product_transition_rows([record(torch.arange(24).reshape(6, 4))])


def test_coordinates_use_unshifted_state_and_separate_q_z_null():
    vt = torch.tensor([[1., 0., 0.], [0., 1., 0.]])
    sigma = torch.tensor([3., 2.])
    projection = sigma[:, None] * vt
    recovered = orthonormal_decoder_basis(projection, sigma)
    hidden = torch.tensor([[4., 5., 6.]])
    value = decoder_coordinates(hidden, projection, recovered)
    torch.testing.assert_close(value["orthonormal"], torch.tensor([[4., 5.]]))
    torch.testing.assert_close(value["functional"], torch.tensor([[12., 10.]]))
    torch.testing.assert_close(value["parallel"], torch.tensor([[4., 5., 0.]]))
    torch.testing.assert_close(value["null"], torch.tensor([[0., 0., 6.]]))


def test_js_has_single_half_factor_and_gold_metrics_use_predicted_logits():
    true = torch.tensor([[8., 0., -1.]])
    predicted = torch.tensor([[0., 1., 3.]])
    gold = torch.tensor([0])
    metrics = functional_distribution_metrics(true, predicted, gold)
    p, q = F.softmax(true, -1), F.softmax(predicted, -1)
    m = .5 * (p + q)
    expected = .5 * ((p * (p.log() - m.log())).sum(-1) +
                     (q * (q.log() - m.log())).sum(-1))
    torch.testing.assert_close(metrics["js"], expected)
    assert metrics["gold_rank"].item() == 3
    assert metrics["gold_margin"].item() == -3
    assert not metrics["top1_agreement"].item()
