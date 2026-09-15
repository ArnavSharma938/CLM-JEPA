import torch
from torch.nn import functional as F

from src.stp import released_stp_loss, released_stp_terms


def test_released_stp_single_segment_parity_on_synthetic_tensor():
    torch.manual_seed(4)
    states = torch.randn(9, 7, requires_grad=True)
    before, patch, after = released_stp_terms(states, 2, 6)
    expected = 1 - F.cosine_similarity(
        (states[6] - states[2]).float(),
        ((states[2] - states[0]) + (states[-1] - states[6])).float(),
        dim=0,
    )
    actual = 1 - F.cosine_similarity(patch.float(), (before + after).float(), dim=0)
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert states.grad is not None and states.grad.norm() > 0


def test_released_stp_sampling_is_deterministic_and_excludes_full_span():
    rows = [torch.randn(10, 5), torch.randn(7, 5)]
    loss1, spans1 = released_stp_loss(rows, seed=17)
    loss2, spans2 = released_stp_loss(rows, seed=17)
    torch.testing.assert_close(loss1, loss2)
    assert spans1 == spans2
    for states, (start, end) in zip(rows, spans1):
        assert end - start < states.shape[0] - 1

