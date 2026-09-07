"""Training-only token-conditioned decoder-visible transition objective.

Source audit and intentional NextLat differences: docs/reports/04_DECODER_PROJECTED.md.
"""
from types import SimpleNamespace
import hashlib

import torch
from torch import nn
from torch.nn import functional as F

from stp import capture_final_hidden_state

RANK_RTOL = 1e-6


def decoder_coordinates(weight):
    """One compact CPU FP32 SVD; fixed relative tolerance, no variance cutoff."""
    weight = weight.detach().float().cpu().clone()
    centered = weight - weight.mean(0, keepdim=True)
    u, s, vh = torch.linalg.svd(centered, full_matrices=False)
    rank = int((s > RANK_RTOL * s[0]).sum())
    if not 0 < rank <= weight.shape[0] - 1:
        raise ValueError(f'invalid centered numerical rank: {rank}')
    projection = s[:rank, None] * vh[:rank]
    error = float((centered - u[:, :rank] @ projection).norm() / centered.norm())
    return u[:, :rank].contiguous(), projection, {
        'rank': rank, 'rank_rtol': RANK_RTOL, 'rank_atol': 0.0,
        'frozen_decoder_fp32_sha256': hashlib.sha256(weight.contiguous().numpy().tobytes()).hexdigest(),
        'rank_threshold': float(s[0] * RANK_RTOL),
        'singular_values': s.tolist(), 'relative_reconstruction_error': error,
    }


def eligible_transitions(batch, product_start, eos):
    """Return target positions: actual product tokens, including the first."""
    ids, labels, attention = (batch[k] for k in ('input_ids', 'labels', 'attention_mask'))
    starts = (ids == product_start) & (labels != -100) & attention.bool()
    if not torch.all(starts.sum(1) == 1):
        raise ValueError('expected exactly one supervised product-start per row')
    positions = torch.arange(ids.shape[1], device=ids.device)[None, :]
    start = starts.long().argmax(1)[:, None]
    ends = (ids == eos) & (positions > start) & attention.bool()
    if not torch.all(ends.sum(1) == 1):
        raise ValueError('missing or ambiguous product end (possibly truncated)')
    end = ends.long().argmax(1)[:, None]
    mask = (positions > start) & (positions < end) & (labels != -100) & attention.bool()
    if not torch.all(mask.sum(1) > 0):
        raise ValueError('each row requires actual product tokens')
    return mask[:, 1:]


def gradient_norm(loss, parameters, *, retain_graph=True):
    grads = torch.autograd.grad(loss, parameters, retain_graph=retain_graph, allow_unused=True)
    return float(torch.stack([g.detach().float().square().sum() for g in grads if g is not None]).sum().sqrt())


class DecoderProjectedObjective(nn.Module):
    def __init__(self, weight, *, product_start_token_id, eos_token_id, resume_state=None):
        super().__init__()
        if tuple(weight.shape) != (392, 2048):
            raise ValueError(f'expected native decoder 392 x 2048, got {tuple(weight.shape)}')
        if resume_state is None:
            u, projection, self.metadata = decoder_coordinates(weight)
            frozen_decoder = weight.detach().float().cpu().clone()
        else:
            saved = torch.load(resume_state, map_location='cpu', weights_only=False)
            u, projection = saved['state_dict']['u'], saved['state_dict']['projection']
            self.metadata = saved['metadata']
            frozen_decoder = saved['state_dict']['frozen_decoder']
        self.register_buffer('u', u)
        self.register_buffer('projection', projection)
        self.register_buffer('frozen_decoder', frozen_decoder)
        self.predictor = nn.Sequential(
            nn.LayerNorm(4096), nn.Linear(4096, 256), nn.GELU(),
            nn.Linear(256, 256), nn.GELU(), nn.Linear(256, self.metadata['rank']),
        )
        nn.init.zeros_(self.predictor[-1].weight)
        nn.init.zeros_(self.predictor[-1].bias)
        self.product_start = product_start_token_id
        self.eos = eos_token_id
        self.alpha = 1.0
        self.calibration = None
        if resume_state is not None:
            self.restore_training_state(resume_state)

    def losses(self, current, future, embedding, mask):
        # Explicitly disable any outer BF16 context for every auxiliary operation.
        with torch.autocast(device_type=current.device.type, enabled=False):
            h = current.float()
            h = h.detach() + self.alpha * (h - h.detach())
            target = F.linear(future.detach().float(), self.projection)
            predicted = F.linear(h, self.projection) + self.predictor(
                torch.cat((h, embedding.detach().float()), dim=-1))
            lz = F.smooth_l1_loss(predicted, target, reduction='none').mean(-1)
            log_q = F.log_softmax(F.linear(target, self.u), dim=-1).detach()
            log_p = F.log_softmax(F.linear(predicted, self.u), dim=-1)
            kl = F.kl_div(log_p, log_q, log_target=True, reduction='none').sum(-1)
            counts = mask.sum(1)
            if not torch.all(counts > 0):
                raise ValueError('empty transition row')
            return ((lz * mask).sum(1) / counts).mean(), ((kl * mask).sum(1) / counts).mean()

    def forward(self, model, batch):
        mask = eligible_transitions(batch, self.product_start, self.eos)
        with capture_final_hidden_state(model) as captured:
            output = model(**{k: batch[k] for k in ('input_ids', 'attention_mask', 'labels')})
        with torch.no_grad():
            embedding = model.get_input_embeddings()(batch['input_ids'][:, 1:])
        lz, kl = self.losses(captured[0][:, :-1], captured[0][:, 1:], embedding, mask)
        auxiliary = lz + kl
        return SimpleNamespace(
            loss=output.loss + auxiliary, native_loss=output.loss, logits=output.logits,
            jepa_loss=auxiliary, sigreg_loss=None, jepa_objective_loss=auxiliary,
            jepa_active=True, lz=lz, kl=kl,
        )

    def calibrate(self, model, batch):
        if self.calibration is not None:
            raise RuntimeError('auxiliary gradient calibration is one-time only')
        parameters = tuple(p for p in model.parameters() if p.requires_grad)
        output = self(model, batch)
        gn = gradient_norm(output.native_loss, parameters)
        ga = gradient_norm(output.jepa_loss, parameters, retain_graph=False)
        if not (gn > 0 and ga > 0 and torch.isfinite(torch.tensor([gn, ga])).all()):
            raise FloatingPointError('invalid calibration gradient norms')
        self.alpha = min(1.0, 0.05 * gn / ga)
        self.calibration = {'g_N': gn, 'g_A_unscaled': ga, 'alpha': self.alpha,
                            'expected_auxiliary_ntp_ratio': self.alpha * ga / gn}
        return self.calibration

    def save_training_state(self, path):
        torch.save({'state_dict': self.state_dict(), 'metadata': self.metadata,
                    'calibration': self.calibration, 'alpha': self.alpha}, path)

    def restore_training_state(self, path):
        state = torch.load(path, map_location=self.u.device, weights_only=False)
        self.load_state_dict(state['state_dict'])
        self.metadata, self.calibration, self.alpha = state['metadata'], state['calibration'], state['alpha']


class FullStateObjective(DecoderProjectedObjective):
    """Matched NextLat-style full-hidden baseline; training-only."""
    def __init__(self, weight, *, product_start_token_id, eos_token_id, resume_state=None):
        nn.Module.__init__(self)
        self.register_buffer('frozen_decoder', weight.detach().float().cpu().clone())
        self.register_buffer('u', weight.detach().float().cpu().clone().t())
        self.register_buffer('projection', torch.empty(0))
        self.metadata = {'method': 'full_state_nextlat_style', 'hidden_size': int(weight.shape[1]),
                         'frozen_decoder_fp32_sha256': hashlib.sha256(self.frozen_decoder.numpy().tobytes()).hexdigest()}
        self.predictor = nn.Sequential(nn.LayerNorm(4096), nn.Linear(4096, 256), nn.GELU(),
                                       nn.Linear(256, 256), nn.GELU(), nn.Linear(256, weight.shape[1]))
        nn.init.zeros_(self.predictor[-1].weight); nn.init.zeros_(self.predictor[-1].bias)
        self.product_start, self.eos, self.alpha, self.calibration = product_start_token_id, eos_token_id, 1.0, None
        if resume_state is not None: self.restore_training_state(resume_state)

    def losses(self, current, future, embedding, mask):
        with torch.autocast(device_type=current.device.type, enabled=False):
            h = current.float(); h = h.detach() + self.alpha * (h - h.detach())
            target = future.detach().float(); predicted = h + self.predictor(torch.cat((h, embedding.detach().float()), -1))
            lz = F.smooth_l1_loss(predicted, target, reduction='none').mean(-1)
            log_q = F.log_softmax(F.linear(target, self.frozen_decoder), -1).detach()
            log_p = F.log_softmax(F.linear(predicted, self.frozen_decoder), -1)
            kl = F.kl_div(log_p, log_q, log_target=True, reduction='none').sum(-1)
            counts = mask.sum(1); return ((lz*mask).sum(1)/counts).mean(), ((kl*mask).sum(1)/counts).mean()

    def save_training_state(self, path):
        torch.save({'state_dict': self.state_dict(), 'metadata': self.metadata, 'calibration': self.calibration, 'alpha': self.alpha}, path)
