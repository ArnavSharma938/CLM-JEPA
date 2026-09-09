"""Parity and gradient invariants for the official-faithful NextLat adapter."""
import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from faithful_nextlat import FaithfulNextLatObjective, NextLatDynamicsModel


def small_objective(hidden=80, vocab=7):
    torch.manual_seed(3)
    return FaithfulNextLatObjective(
        torch.randn(vocab, hidden), product_start_token_id=1,
        eos_token_id=3, native_vocab_size=vocab,
    )


def test_official_1b_predictor_shape_initialization_and_order():
    torch.manual_seed(7)
    model = NextLatDynamicsModel(2048)
    linears = [module for module in model.mlp if isinstance(module, nn.Linear)]
    assert model.input_dim == 4096 and model.hidden_dim == 6528
    assert [(m.in_features, m.out_features) for m in linears] == [
        (4096, 6528), (6528, 6528), (6528, 2048)]
    assert all(m.bias is None for m in linears)
    assert model.norm_x.weight.shape == (4096,)
    assert linears[-1].weight.count_nonzero() > 0
    for layer in linears:
        assert abs(float(layer.weight.std()) - .02) < .001


def test_forward_matches_independent_upstream_formula():
    torch.manual_seed(11)
    model = NextLatDynamicsModel(80)
    current, embed = torch.randn(2, 3, 80), torch.randn(2, 3, 80)
    joined = torch.cat([embed, current], -1)
    normalized = F.layer_norm(joined, (160,), model.norm_x.weight, None, 1e-5)
    expected = current + model.mlp(normalized)
    torch.testing.assert_close(model(current, embed), expected)


def test_global_valid_reductions_and_teacher_to_student_kl():
    method = small_objective()
    current, future, embed = [torch.randn(2, 3, 80) for _ in range(3)]
    mask = torch.tensor([[True, False, False], [True, True, True]])
    head = torch.randn(7, 80)
    latent, kl, predicted = method.losses(current, future, embed, mask, head)
    chosen = mask.unsqueeze(-1).expand_as(predicted)
    expected_latent = F.smooth_l1_loss(predicted, future, reduction="none")[chosen].mean()
    qlog = F.log_softmax(F.linear(future, head), -1)
    plog = F.log_softmax(F.linear(predicted, head), -1)
    expected_kl = F.kl_div(plog, qlog, log_target=True, reduction="none").sum(-1)[mask].mean()
    torch.testing.assert_close(latent, expected_latent)
    torch.testing.assert_close(kl, expected_kl)


def test_auxiliary_gradient_semantics_and_live_detached_head():
    method = small_objective()
    current = torch.randn(2, 3, 80, requires_grad=True)
    future = torch.randn(2, 3, 80, requires_grad=True)
    embed = torch.randn(2, 3, 80, requires_grad=True)
    head = torch.randn(7, 80, requires_grad=True)
    mask = torch.tensor([[True, False, True], [True, True, True]])
    latent, kl, _ = method.losses(current, future, embed, mask, head)
    (latent + kl).backward()
    assert current.grad is not None and current.grad.norm() > 0
    assert embed.grad is not None and embed.grad.norm() > 0
    assert future.grad is None and head.grad is None
    with torch.no_grad():
        baseline = method.losses(current, future, embed, mask, head)[1]
        changed = method.losses(current, future, embed, mask, head * 2.3)[1]
    assert not torch.isclose(baseline, changed)


def test_training_state_round_trip(tmp_path):
    torch.manual_seed(19)
    head = torch.randn(7, 80)
    method = FaithfulNextLatObjective(
        head, product_start_token_id=1, eos_token_id=3, native_vocab_size=7)
    path = tmp_path / "auxiliary_training_state.pt"
    method.save_training_state(path)
    restored = FaithfulNextLatObjective(
        head, product_start_token_id=1, eos_token_id=3,
        native_vocab_size=7, resume_state=path,
    )
    assert restored.metadata == method.metadata
    for key, value in method.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[key])
