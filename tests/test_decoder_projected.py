"""Scientific invariants for the new auxiliary objective (CPU only)."""
import sys
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from decoder_projected import DecoderProjectedObjective, decoder_coordinates, eligible_transitions


def small_objective():
    method = DecoderProjectedObjective.__new__(DecoderProjectedObjective)
    nn.Module.__init__(method)
    torch.manual_seed(71)
    u, projection, method.metadata = decoder_coordinates(torch.randn(5, 8))
    method.register_buffer('u', u)
    method.register_buffer('projection', projection)
    method.predictor = nn.Sequential(nn.LayerNorm(16), nn.Linear(16, 6), nn.GELU(), nn.Linear(6, 4))
    method.alpha = 1.0
    method.calibration = None
    return method


def test_svd_preserves_decoder_distribution():
    torch.manual_seed(11)
    weight = torch.randn(7, 12)
    u, projection, metadata = decoder_coordinates(weight)
    h = torch.randn(9, 12)
    assert metadata['rank'] == 6
    torch.testing.assert_close(F.softmax(F.linear(h, weight), -1),
        F.softmax(F.linear(F.linear(h, projection), u), -1), atol=2e-6, rtol=2e-5)


def test_product_first_included_end_and_padding_excluded():
    ids = torch.tensor([[8, 9, 3, 1, 4, 5, 3], [8, 3, 1, 6, 3, 0, 0]])
    labels = torch.tensor([[-100, -100, -100, 1, 4, 5, 3], [-100, -100, 1, 6, 3, -100, -100]])
    batch = dict(input_ids=ids, labels=labels, attention_mask=ids != 0)
    assert eligible_transitions(batch, 1, 3).tolist() == [
        [False, False, False, True, True, False], [False, False, True, False, False, False]]
    batch['input_ids'][0, -1] = 7
    with pytest.raises(ValueError, match='product end'):
        eligible_transitions(batch, 1, 3)


def test_auxiliary_gradients_only_scale_current_state():
    method = small_objective()
    current, future, embedding = [torch.randn(2, 3, 8, requires_grad=True) for _ in range(3)]
    mask = torch.tensor([[True, False, False], [True, True, True]])
    def gradients(alpha):
        method.alpha = alpha
        losses = method.losses(current, future, embedding, mask)
        return torch.autograd.grad(sum(losses), [current, future, embedding, *method.parameters()], allow_unused=True)
    full, scaled, zero = gradients(1), gradients(.03), gradients(0)
    assert full[1] is None and full[2] is None
    torch.testing.assert_close(scaled[0], .03 * full[0])
    assert zero[0].count_nonzero() == 0
    for a, b, c in zip(full[3:], scaled[3:], zero[3:]):
        torch.testing.assert_close(a, b)
        torch.testing.assert_close(a, c)
    assert full[-1].norm() > 0


def test_row_mean_and_kl_direction_against_direct_probability_formula():
    method = small_objective()
    current, future, embedding = [torch.randn(2, 3, 8) for _ in range(3)]
    mask = torch.tensor([[True, False, False], [True, True, True]])
    actual = method.losses(current, future, embedding, mask)
    rows = []
    for i in range(2):
        h, target, e = current[i, mask[i]], future[i, mask[i]], embedding[i, mask[i]]
        z = F.linear(target, method.projection)
        pred = F.linear(h, method.projection) + method.predictor(torch.cat((h, e), -1))
        q = F.softmax(F.linear(z, method.u), -1)
        p = F.softmax(F.linear(pred, method.u), -1)
        rows.append((F.smooth_l1_loss(pred, z), (q * (q.log() - p.log())).sum(-1).mean()))
    for j in range(2):
        torch.testing.assert_close(actual[j], (rows[0][j] + rows[1][j]) / 2)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        assert method.losses(current, future, embedding, mask)[0].dtype == torch.float32


def test_training_state_round_trip(tmp_path):
    method = small_objective()
    method.alpha = .02
    method.calibration = {'alpha': .02}
    path = tmp_path / 'auxiliary_training_state.pt'
    method.save_training_state(path)
    restored = small_objective()
    restored.restore_training_state(path)
    assert restored.alpha == .02 and restored.calibration == method.calibration
    for key, value in method.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[key])


def test_calibration_and_unchanged_llama_inference(tmp_path):
    from transformers import LlamaConfig, LlamaForCausalLM
    config = LlamaConfig(vocab_size=12, hidden_size=8, intermediate_size=16,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
        attention_dropout=0., bos_token_id=1, eos_token_id=3, pad_token_id=0)
    model = LlamaForCausalLM(config)
    method = small_objective()
    method.product_start, method.eos = 1, 3
    ids = torch.tensor([[8, 3, 1, 4, 5, 3]])
    batch = dict(input_ids=ids, attention_mask=torch.ones_like(ids),
                 labels=torch.tensor([[-100, -100, 1, 4, 5, 3]]))
    measured = method.calibrate(model, batch)
    assert 0 < measured['alpha'] <= 1
    assert measured['expected_auxiliary_ntp_ratio'] <= .0500001
    assert all(p.grad is None for p in model.parameters())
    assert all(p.grad is None for p in method.parameters())
    with pytest.raises(RuntimeError, match='one-time'):
        method.calibrate(model, batch)
    model.eval()
    with torch.no_grad():
        direct = model(**batch)
        output = method(model, batch)
    torch.testing.assert_close(output.native_loss, direct.loss, rtol=0, atol=0)
    torch.testing.assert_close(output.logits, direct.logits, rtol=0, atol=0)
    assert not model.model.norm._forward_hooks
    assert not any('predictor' in k or 'projection' in k for k in model.state_dict())
    model.save_pretrained(tmp_path)
    method.save_training_state(tmp_path / 'auxiliary_training_state.pt')
    restored = LlamaForCausalLM.from_pretrained(tmp_path).eval()
    prompt = ids[:, :3]
    with torch.no_grad():
        torch.testing.assert_close(model.generate(prompt, max_new_tokens=2, do_sample=False),
                                   restored.generate(prompt, max_new_tokens=2, do_sample=False))


def test_zero_final_layer_keeps_residual_gradient():
    method = small_objective()
    nn.init.zeros_(method.predictor[-1].weight)
    nn.init.zeros_(method.predictor[-1].bias)
    h = torch.randn(1, 2, 8, requires_grad=True)
    target, e = torch.randn_like(h), torch.randn_like(h)
    loss = sum(method.losses(h, target, e, torch.ones(1, 2, dtype=torch.bool)))
    loss.backward()
    assert h.grad.norm() > 0
    assert method.predictor[-1].weight.grad.norm() > 0
    assert method.predictor[1].weight.grad.count_nonzero() == 0


def test_frozen_snapshot_architecture_and_resume_without_svd(monkeypatch, tmp_path):
    import decoder_projected
    weight = torch.randn(392, 2048)
    monkeypatch.setattr(decoder_projected, 'decoder_coordinates', lambda w:
        (torch.randn(392, 4), torch.randn(4, 2048), {'rank': 4}))
    method = DecoderProjectedObjective(weight, product_start_token_id=1, eos_token_id=3)
    snapshot = method.frozen_decoder.clone()
    weight.zero_()
    torch.testing.assert_close(method.frozen_decoder, snapshot)
    assert method.predictor[0].normalized_shape == (4096,)
    assert [(m.in_features, m.out_features) for m in method.predictor if isinstance(m, nn.Linear)] == [(4096,256),(256,256),(256,4)]
    assert method.predictor[-1].weight.count_nonzero() == 0
    assert method.predictor[-1].bias.count_nonzero() == 0
    path = tmp_path / 'auxiliary_training_state.pt'
    method.save_training_state(path)
    def forbidden(w):
        raise AssertionError('resume must reuse existing SVD')
    monkeypatch.setattr(decoder_projected, 'decoder_coordinates', forbidden)
    resumed = DecoderProjectedObjective(weight, product_start_token_id=1, eos_token_id=3, resume_state=path)
    torch.testing.assert_close(resumed.frozen_decoder, snapshot)
