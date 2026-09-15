"""Faithful horizon-1 NextLat predictor and frozen-probe metrics for RITA."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class BiaslessLayerNorm(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(value, (value.shape[-1],), self.weight, None, 1e-5)


class FaithfulNextLatPredictor(nn.Module):
    """Official ordering, width rule, residual, biases, and initialization."""

    def __init__(self, hidden_size: int, proj_factor: float = 1.6):
        super().__init__()
        input_dim = 2 * hidden_size
        hidden_dim = 128 * round(proj_factor * input_dim / 128)
        self.norm_x = BiaslessLayerNorm(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_size, bias=False),
        )
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

    def forward(self, current: torch.Tensor, next_embedding: torch.Tensor) -> torch.Tensor:
        joined = self.norm_x(torch.cat([next_embedding, current], dim=-1))
        return current + self.mlp(joined)


def faithful_losses(
    predictor: nn.Module,
    current: torch.Tensor,
    future: torch.Tensor,
    next_embedding: torch.Tensor,
    mask: torch.Tensor,
    live_head_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not torch.any(mask):
        raise ValueError("empty NextLat transition batch")
    predicted = predictor(current, next_embedding)
    target = future.detach()
    element = F.smooth_l1_loss(predicted, target, reduction="none")
    expanded = mask.unsqueeze(-1).expand_as(element)
    latent = element[expanded].mean()
    head = live_head_weight.detach()
    teacher_log = F.log_softmax(F.linear(target, head), dim=-1).detach()
    predicted_log = F.log_softmax(F.linear(predicted, head), dim=-1)
    per_position = F.kl_div(
        predicted_log, teacher_log, log_target=True, reduction="none"
    ).sum(-1)
    kl = per_position[mask].mean()
    return latent, kl, predicted


def decoder_metrics(true: torch.Tensor, predicted: torch.Tensor, head: torch.Tensor) -> dict[str, torch.Tensor]:
    true_logits = F.linear(true.float(), head.float())
    pred_logits = F.linear(predicted.float(), head.float())
    true_logp = F.log_softmax(true_logits, -1)
    pred_logp = F.log_softmax(pred_logits, -1)
    true_p, pred_p = true_logp.exp(), pred_logp.exp()
    kl = (true_p * (true_logp - pred_logp)).sum(-1)
    return {
        "kl_true_predicted": kl,
        "top1_agreement": true_logits.argmax(-1).eq(pred_logits.argmax(-1)).float(),
    }


def transition_diagnostics(
    current: torch.Tensor,
    future: torch.Tensor,
    predicted: torch.Tensor,
    train_future_mean: torch.Tensor,
    head_weight: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> dict[str, torch.Tensor]:
    """Nonredundant error, scale-free, and decoder-functional diagnostics.

    Every returned value is per transition. Error-like quantities are oriented
    so smaller means a more faithful/predictable transition. The centering and
    normalization reference is fitted on training proteins only.
    """
    current, future, predicted = current.float(), future.float(), predicted.float()
    center = train_future_mean.to(device=future.device, dtype=future.dtype)
    head = head_weight.detach().to(device=future.device, dtype=future.dtype)
    difference = predicted - future
    squared_sum = difference.square().sum(-1)
    raw_mse = difference.square().mean(-1)
    raw_smooth_l1 = F.smooth_l1_loss(predicted, future, reduction="none").mean(-1)
    normalized_state_mse = squared_sum / ((future - center).square().sum(-1) + eps)
    transition_relative_mse = squared_sum / ((future - current).square().sum(-1) + eps)
    centered_cosine = F.cosine_similarity(predicted - center, future - center, dim=-1, eps=eps)

    teacher_log = F.log_softmax(F.linear(future, head), dim=-1)
    predicted_log = F.log_softmax(F.linear(predicted, head), dim=-1)
    teacher_probability, predicted_probability = teacher_log.exp(), predicted_log.exp()
    midpoint = .5 * (teacher_probability + predicted_probability)
    decoder_js = .5 * (
        (teacher_probability * (teacher_log - torch.log(midpoint + 1e-30))).sum(-1)
        + (predicted_probability * (predicted_log - torch.log(midpoint + 1e-30))).sum(-1)
    )
    faithful_kl = F.kl_div(
        predicted_log, teacher_log, log_target=True, reduction="none"
    ).sum(-1)
    return {
        "raw_smooth_l1": raw_smooth_l1,
        "raw_mse": raw_mse,
        "normalized_state_mse": normalized_state_mse,
        "transition_relative_mse": transition_relative_mse,
        "centered_cosine_error": 1.0 - centered_cosine,
        "decoder_js": decoder_js,
        "faithful_total": raw_smooth_l1 + faithful_kl,
    }
