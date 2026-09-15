import torch
from torch import nn
from torch.nn import functional as F

from src.nextlat import FaithfulNextLatPredictor, faithful_losses


def test_faithful_nextlat_shape_order_and_residual():
    torch.manual_seed(2)
    predictor = FaithfulNextLatPredictor(80)
    current, embedding = torch.randn(2, 3, 80), torch.randn(2, 3, 80)
    joined = F.layer_norm(
        torch.cat([embedding, current], -1),
        (160,), predictor.norm_x.weight, None, 1e-5,
    )
    torch.testing.assert_close(predictor(current, embedding), current + predictor.mlp(joined))
    assert predictor.input_dim == 160
    assert predictor.hidden_dim == 256
    assert all(layer.bias is None for layer in predictor.mlp if isinstance(layer, nn.Linear))


def test_faithful_shift_detach_kl_direction_and_gradient_paths():
    torch.manual_seed(3)
    predictor = FaithfulNextLatPredictor(80)
    current = torch.randn(2, 4, 80, requires_grad=True)
    future = torch.randn(2, 4, 80, requires_grad=True)
    embedding = torch.randn(2, 4, 80, requires_grad=True)
    head = torch.randn(7, 80, requires_grad=True)
    mask = torch.tensor([[True, True, False, False], [True, False, True, False]])
    latent, kl, predicted = faithful_losses(
        predictor, current, future, embedding, mask, head
    )
    qlog = F.log_softmax(F.linear(future.detach(), head.detach()), -1)
    plog = F.log_softmax(F.linear(predicted, head.detach()), -1)
    expected_kl = F.kl_div(plog, qlog, log_target=True, reduction="none").sum(-1)[mask].mean()
    torch.testing.assert_close(kl, expected_kl)
    (latent + kl).backward()
    assert current.grad is not None and current.grad.norm() > 0
    assert embedding.grad is not None and embedding.grad.norm() > 0
    assert future.grad is None
    assert head.grad is None
    assert any(p.grad is not None and p.grad.norm() > 0 for p in predictor.parameters())


def test_causal_tensor_shift_is_current_t_to_future_t_plus_one():
    hidden = torch.arange(6 * 3).view(1, 6, 3)
    current, future = hidden[:, :-1], hidden[:, 1:]
    assert torch.equal(current[0, 2], hidden[0, 2])
    assert torch.equal(future[0, 2], hidden[0, 3])

