"""Causal-language adaptation of the V-JEPA 2.1 pretraining mechanism.

The scientific reference is V-JEPA 2.1 arXiv:2603.14482v3 and the official
facebookresearch/vjepa2 implementation at commit
204698b45b3712590f06245fbfba32d3be539812.  Vision-only geometry is translated
to one-dimensional causal reaction sequences; the load-bearing mechanics are
kept: dense L1 prediction, distance-weighted context supervision, four-level
deep self-supervision, a transformer predictor, stop-gradient EMA targets, and
the progressive context-loss schedule.
"""

from __future__ import annotations

import copy
import math
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call


VJEPA21_UPSTREAM_COMMIT = "204698b45b3712590f06245fbfba32d3be539812"
VJEPA21_PAPER = "arXiv:2603.14482v3"


@dataclass(frozen=True)
class DenseVJEPA21Config:
    encoder_dim: int
    encoder_depths: tuple[int, ...] = (6, 11, 17, 22)
    predictor_dim: int = 384
    predictor_depth: int = 24
    predictor_heads: int = 12
    predictor_mlp_ratio: float = 4.0
    num_mask_tokens: int = 10
    short_mask_tokens: int = 8
    short_horizon_fraction: float = 0.15
    long_horizon_fraction: float = 0.70
    context_lambda: float = 0.5
    context_warmup_start_fraction: float = 1.0 / 9.0
    context_warmup_end_fraction: float = 2.0 / 9.0
    ema_start: float = 0.99925
    ema_end: float = 0.99925
    seed: int = 533
    layer_norm_eps: float = 1e-6

    def __post_init__(self) -> None:
        if self.encoder_dim < 1:
            raise ValueError("encoder_dim must be positive")
        if not self.encoder_depths or tuple(sorted(set(self.encoder_depths))) != self.encoder_depths:
            raise ValueError("encoder_depths must be unique and increasing")
        if self.predictor_dim % self.predictor_heads:
            raise ValueError("predictor_dim must be divisible by predictor_heads")
        if self.predictor_depth < 1 or self.predictor_heads < 1:
            raise ValueError("predictor depth and heads must be positive")
        if not 0 < self.short_mask_tokens < self.num_mask_tokens:
            raise ValueError("mask-token allocation must contain both short and long masks")
        if not 0.0 < self.short_horizon_fraction < self.long_horizon_fraction <= 1.0:
            raise ValueError("horizon fractions must satisfy 0 < short < long <= 1")
        if not 0.0 <= self.context_lambda:
            raise ValueError("context_lambda must be nonnegative")
        if not 0.0 <= self.context_warmup_start_fraction < self.context_warmup_end_fraction <= 1.0:
            raise ValueError("invalid progressive context schedule")
        if not 0.0 <= self.ema_start <= self.ema_end < 1.0:
            raise ValueError("EMA coefficients must satisfy 0 <= start <= end < 1")


@dataclass
class CausalSuffixMask:
    boundaries: torch.Tensor
    active_lengths: torch.Tensor
    target_starts: torch.Tensor
    horizons: torch.Tensor
    mask_token_indices: torch.Tensor
    context_mask: torch.Tensor
    future_mask: torch.Tensor
    valid_mask: torch.Tensor

@dataclass
class DenseVJEPA21Output:
    loss: torch.Tensor
    native_loss: torch.Tensor
    jepa_loss: torch.Tensor | None
    mask_loss: torch.Tensor | None
    context_loss: torch.Tensor | None
    context_coefficient: float
    mask_loss_by_depth: dict[int, torch.Tensor]
    context_loss_by_depth: dict[int, torch.Tensor]
    student_scale_by_depth: dict[int, torch.Tensor]
    target_scale_by_depth: dict[int, torch.Tensor]
    mask: CausalSuffixMask | None


class ProgressiveContextSchedule:
    """V-JEPA 2.1's 15k--30k warmup expressed on a planned-step budget."""

    def __init__(
        self,
        value: float,
        total_steps: int,
        start_fraction: float = 1.0 / 9.0,
        end_fraction: float = 2.0 / 9.0,
    ) -> None:
        if total_steps < 1:
            raise ValueError("total_steps must be positive")
        self.maximum = float(value)
        self.total_steps = int(total_steps)
        self.start = max(1, int(round(total_steps * start_fraction)))
        self.end = max(self.start + 1, int(round(total_steps * end_fraction)))
        self.end = min(self.total_steps, self.end)
        if self.end <= self.start:
            self.start = max(0, self.total_steps - 1)
            self.end = self.total_steps

    def value(self, global_step: int) -> float:
        if global_step < self.start:
            return 0.0
        if global_step >= self.end:
            return self.maximum
        return self.maximum * (global_step - self.start) / (self.end - self.start)


class CausalSuffixMaskSampler:
    """Deterministic two-scale causal suffix sampler with resumable call index."""

    def __init__(self, config: DenseVJEPA21Config) -> None:
        self.config = config
        self.sample_step = 0

    def state_dict(self) -> dict[str, int]:
        return {"sample_step": self.sample_step}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        step = int(state["sample_step"])
        if step < 0:
            raise ValueError("sample_step cannot be negative")
        self.sample_step = step

    def sample(
        self, labels: torch.Tensor, attention_mask: torch.Tensor, ignore_index: int,
    ) -> CausalSuffixMask:
        if labels.ndim != 2 or attention_mask.shape != labels.shape:
            raise ValueError("labels and attention_mask must be equal 2D tensors")
        label_mask = labels.ne(ignore_index) & attention_mask.bool()
        if not label_mask.any(dim=1).all():
            raise ValueError("every reaction must contain target labels")
        transitions = label_mask.int().argmax(dim=1)
        active_lengths = attention_mask.long().sum(dim=1)
        # ChemFM targets start with <prostart>.  It is always visible, matching
        # the actual generation prompt; molecular product tokens and EOS form
        # the eligible future region.
        eligible = active_lengths - transitions - 1
        if not eligible.gt(0).all():
            raise ValueError("every reaction must have a product token after <prostart>")

        generator = torch.Generator(device="cpu").manual_seed(
            self.config.seed + self.sample_step
        )
        slots = torch.randint(
            self.config.num_mask_tokens,
            (labels.size(0),),
            generator=generator,
            device="cpu",
        ).to(labels.device)
        self.sample_step += 1
        fractions = torch.where(
            slots < self.config.short_mask_tokens,
            torch.full_like(slots, self.config.short_horizon_fraction, dtype=torch.float32),
            torch.full_like(slots, self.config.long_horizon_fraction, dtype=torch.float32),
        )
        horizons = torch.ceil(eligible.float() * fractions).long()
        horizons = torch.maximum(torch.ones_like(horizons), torch.minimum(horizons, eligible))
        boundaries = active_lengths - horizons
        if not (boundaries > transitions).all():
            raise RuntimeError("causal boundary exposed an invalid target prefix")

        positions = torch.arange(labels.size(1), device=labels.device).unsqueeze(0)
        valid = positions < active_lengths.unsqueeze(1)
        context = valid & (positions < boundaries.unsqueeze(1))
        future = valid & (positions >= boundaries.unsqueeze(1))
        if (context & future).any() or not torch.equal(context | future, valid):
            raise RuntimeError("context/future masks must partition the active sequence")
        return CausalSuffixMask(
            boundaries=boundaries,
            active_lengths=active_lengths,
            target_starts=transitions,
            horizons=horizons,
            mask_token_indices=slots,
            context_mask=context,
            future_mask=future,
            valid_mask=valid,
        )


def context_distance_weights(mask: CausalSuffixMask, dtype: torch.dtype) -> torch.Tensor:
    """Return the 1/sqrt(distance-to-future-boundary) V-JEPA 2.1 weights."""
    positions = torch.arange(mask.context_mask.size(1), device=mask.boundaries.device)
    distances = mask.boundaries.unsqueeze(1) - positions.unsqueeze(0)
    weights = distances.clamp_min(1).to(dtype).rsqrt()
    return weights * mask.context_mask.to(dtype)


def _llama_backbone(model: nn.Module) -> nn.Module:
    causal_lm = model.get_base_model() if hasattr(model, "get_base_model") else model
    backbone = getattr(causal_lm, "model", None)
    if backbone is None or not hasattr(backbone, "layers"):
        raise TypeError("dense V-JEPA requires a Llama-style causal backbone with layers")
    return backbone


class LayerCapture:
    """Capture selected block outputs without retaining every hidden state."""

    def __init__(self, backbone: nn.Module, depths: Sequence[int]) -> None:
        self.backbone = backbone
        self.depths = tuple(depths)
        self.outputs: dict[int, torch.Tensor] = {}
        self.handles: list[Any] = []

    def __enter__(self) -> "LayerCapture":
        for depth in self.depths:
            layer_index = depth - 1
            if not 0 <= layer_index < len(self.backbone.layers):
                raise ValueError(f"encoder depth {depth} is outside the backbone")

            def hook(_module, _inputs, output, *, captured_depth=depth):
                self.outputs[captured_depth] = output[0] if isinstance(output, tuple) else output

            self.handles.append(self.backbone.layers[layer_index].register_forward_hook(hook))
        return self

    def values(self) -> tuple[torch.Tensor, ...]:
        missing = [depth for depth in self.depths if depth not in self.outputs]
        if missing:
            raise RuntimeError(f"selected encoder layers were not executed: {missing}")
        return tuple(self.outputs[depth] for depth in self.depths)

    def __exit__(self, *_exc) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def _rotate_half(values: torch.Tensor) -> torch.Tensor:
    first = values[..., : values.size(-1) // 2]
    second = values[..., values.size(-1) // 2 :]
    return torch.cat((-second, first), dim=-1)


def _apply_rope(values: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    head_dim = values.size(-1)
    if head_dim % 2:
        raise ValueError("rotary attention requires an even head dimension")
    frequencies = 1.0 / (
        10000.0
        ** (torch.arange(0, head_dim, 2, device=values.device, dtype=torch.float32) / head_dim)
    )
    angles = positions.float().unsqueeze(-1) * frequencies
    angles = torch.cat((angles, angles), dim=-1).unsqueeze(1)
    cosine = angles.cos().to(values.dtype)
    sine = angles.sin().to(values.dtype)
    return values * cosine + _rotate_half(values) * sine


class DensePredictorAttention(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(
        self, values: torch.Tensor, positions: torch.Tensor, valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, length, _ = values.shape
        qkv = self.qkv(values).view(batch, length, 3, self.heads, self.head_dim)
        query, key, value = qkv.unbind(dim=2)
        query = _apply_rope(query.transpose(1, 2), positions)
        key = _apply_rope(key.transpose(1, 2), positions)
        value = value.transpose(1, 2)
        allowed_keys = valid_mask[:, None, None, :]
        attended = F.scaled_dot_product_attention(
            query, key, value, attn_mask=allowed_keys, dropout_p=0.0, is_causal=False
        )
        attended = attended.transpose(1, 2).reshape(batch, length, self.dim)
        return self.proj(attended) * valid_mask.unsqueeze(-1).to(values.dtype)


class DensePredictorBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float, eps: float) -> None:
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.norm1 = nn.LayerNorm(dim, eps=eps)
        self.attention = DensePredictorAttention(dim, heads)
        self.norm2 = nn.LayerNorm(dim, eps=eps)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden, bias=True),
            nn.GELU(),
            nn.Linear(hidden, dim, bias=True),
        )

    def forward(
        self, values: torch.Tensor, positions: torch.Tensor, valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        values = values + self.attention(self.norm1(values), positions, valid_mask)
        values = values + self.mlp(self.norm2(values)) * valid_mask.unsqueeze(-1).to(values.dtype)
        return values


class CausalLanguagePredictor(nn.Module):
    """V-JEPA 2.1 multi-level fusion and dense transformer prediction in 1D."""

    def __init__(self, config: DenseVJEPA21Config) -> None:
        super().__init__()
        self.config = config
        levels = len(config.encoder_depths)
        self.fusion = nn.Sequential(
            nn.Linear(levels * config.encoder_dim, config.encoder_dim, bias=True),
            nn.GELU(),
            nn.Linear(config.encoder_dim, config.predictor_dim, bias=True),
        )
        self.mask_tokens = nn.Parameter(
            torch.zeros(config.num_mask_tokens, config.predictor_dim)
        )
        self.blocks = nn.ModuleList([
            DensePredictorBlock(
                config.predictor_dim,
                config.predictor_heads,
                config.predictor_mlp_ratio,
                config.layer_norm_eps,
            )
            for _ in range(config.predictor_depth)
        ])
        self.norm = nn.LayerNorm(config.predictor_dim, eps=config.layer_norm_eps)
        output_dim = levels * config.encoder_dim
        self.future_projection = nn.Linear(config.predictor_dim, output_dim, bias=True)
        self.context_projection = nn.Linear(config.predictor_dim, output_dim, bias=True)
        self._initialize()

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        # Official V-JEPA 2.1 uses zero-initialized mask tokens and rescales
        # residual output matrices according to predictor depth.
        nn.init.zeros_(self.mask_tokens)
        for index, block in enumerate(self.blocks, start=1):
            block.attention.proj.weight.data.div_(math.sqrt(2.0 * index))
            block.mlp[-1].weight.data.div_(math.sqrt(2.0 * index))

    def forward(
        self,
        online_levels: Sequence[torch.Tensor],
        mask: CausalSuffixMask,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if len(online_levels) != len(self.config.encoder_depths):
            raise ValueError("predictor received the wrong number of encoder levels")
        expected = online_levels[0].shape[:2]
        if any(level.shape[:2] != expected for level in online_levels):
            raise ValueError("all encoder levels must align in batch and sequence")
        fused = self.fusion(torch.cat(tuple(online_levels), dim=-1))
        selected_masks = self.mask_tokens[mask.mask_token_indices]
        future_tokens = selected_masks.unsqueeze(1).expand(-1, fused.size(1), -1)
        tokens = torch.where(mask.context_mask.unsqueeze(-1), fused, future_tokens)
        tokens = tokens * mask.valid_mask.unsqueeze(-1).to(tokens.dtype)
        positions = torch.arange(tokens.size(1), device=tokens.device).unsqueeze(0)
        positions = positions.expand(tokens.size(0), -1)
        for block in self.blocks:
            tokens = block(tokens, positions, mask.valid_mask)
        tokens = self.norm(tokens)
        return self.future_projection(tokens), self.context_projection(tokens)


def _masked_token_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    per_token = values.mean(dim=-1)
    denominator = mask.sum().clamp_min(1)
    return (per_token * mask.to(per_token.dtype)).sum() / denominator


class DenseVJEPA21(nn.Module):
    """Dense causal JEPA objective with an EMA view of trainable ChemFM state."""

    def __init__(
        self,
        config: DenseVJEPA21Config,
        *,
        total_steps: int,
        ignore_index: int,
    ) -> None:
        super().__init__()
        self.config = config
        self.ignore_index = int(ignore_index)
        self.online_norms = nn.ModuleList([
            nn.LayerNorm(config.encoder_dim, eps=config.layer_norm_eps)
            for _ in config.encoder_depths
        ])
        self.target_norms = copy.deepcopy(self.online_norms)
        for parameter in self.target_norms.parameters():
            parameter.requires_grad_(False)
        self.predictor = CausalLanguagePredictor(config)
        self.mask_sampler = CausalSuffixMaskSampler(config)
        self.context_schedule = ProgressiveContextSchedule(
            config.context_lambda,
            total_steps,
            config.context_warmup_start_fraction,
            config.context_warmup_end_fraction,
        )
        self.total_steps = int(total_steps)
        self.ema_update_step = 0
        self.ema_parameter_names: list[str] = []
        self.ema_buffer_names: list[str] = []

    @property
    def supervised_depths(self) -> tuple[int, ...]:
        return self.config.encoder_depths

    def initialize_ema(self, model: nn.Module) -> None:
        if self.ema_parameter_names:
            return
        backbone = _llama_backbone(model)
        trainable = [
            (name, parameter)
            for name, parameter in backbone.named_parameters()
            if parameter.requires_grad
        ]
        if not trainable:
            raise ValueError("ChemFM backbone exposes no trainable encoder parameters")
        for index, (name, parameter) in enumerate(trainable):
            buffer_name = f"ema_encoder_{index:04d}"
            self.register_buffer(buffer_name, parameter.detach().clone(), persistent=True)
            self.ema_parameter_names.append(name)
            self.ema_buffer_names.append(buffer_name)

    def _ema_mapping(self) -> dict[str, torch.Tensor]:
        return {
            name: getattr(self, buffer_name)
            for name, buffer_name in zip(self.ema_parameter_names, self.ema_buffer_names)
        }

    def ema_coefficient(self, update_step: int | None = None) -> float:
        step = self.ema_update_step if update_step is None else int(update_step)
        if self.total_steps <= 1:
            return self.config.ema_end
        progress = min(max(step, 0), self.total_steps - 1) / (self.total_steps - 1)
        return self.config.ema_start + progress * (
            self.config.ema_end - self.config.ema_start
        )

    @torch.no_grad()
    def update_ema(self, model: nn.Module) -> float:
        if not self.ema_parameter_names:
            raise RuntimeError("EMA state has not been initialized")
        tau = self.ema_coefficient()
        online = dict(_llama_backbone(model).named_parameters())
        for name, buffer_name in zip(self.ema_parameter_names, self.ema_buffer_names):
            target = getattr(self, buffer_name)
            target.mul_(tau).add_(online[name].detach(), alpha=1.0 - tau)
        for source, target in zip(self.online_norms.parameters(), self.target_norms.parameters()):
            target.mul_(tau).add_(source.detach(), alpha=1.0 - tau)
        self.ema_update_step += 1
        return tau

    @contextmanager
    def _target_eval_mode(self, backbone: nn.Module):
        was_training = backbone.training
        backbone.eval()
        try:
            yield
        finally:
            backbone.train(was_training)

    def _target_levels(
        self,
        backbone: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        if not self.ema_parameter_names:
            raise RuntimeError("EMA state has not been initialized")
        with torch.no_grad(), self._target_eval_mode(backbone), LayerCapture(
            backbone, self.supervised_depths
        ) as capture:
            functional_call(
                backbone,
                self._ema_mapping(),
                (),
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "use_cache": False,
                    "output_hidden_states": False,
                    "return_dict": True,
                },
                strict=False,
            )
            raw = capture.values()
        # The reference target path applies a second parameter-free layer norm
        # to each learned-normalized hierarchical slice before the L1 target.
        return tuple(
            F.layer_norm(norm(level), (self.config.encoder_dim,)).detach()
            for norm, level in zip(self.target_norms, raw)
        )

    def checkpoint_state(self) -> dict[str, Any]:
        return {
            "format": "dense_vjepa2_1_v1",
            "config": asdict(self.config),
            "module": self.state_dict(),
            "ema_parameter_names": list(self.ema_parameter_names),
            "ema_buffer_names": list(self.ema_buffer_names),
            "ema_update_step": self.ema_update_step,
            "mask_sampler": self.mask_sampler.state_dict(),
            "total_steps": self.total_steps,
        }

    def load_checkpoint_state(self, state: Mapping[str, Any], model: nn.Module) -> None:
        if state.get("format") != "dense_vjepa2_1_v1":
            raise ValueError("unsupported dense V-JEPA checkpoint format")
        if state.get("config") != asdict(self.config):
            raise ValueError("resume configuration does not match dense V-JEPA checkpoint")
        if int(state["total_steps"]) != self.total_steps:
            raise ValueError("resume planned-step budget does not match checkpoint")
        self.initialize_ema(model)
        if list(state["ema_parameter_names"]) != self.ema_parameter_names:
            raise ValueError("EMA encoder parameter names changed across resume")
        if list(state["ema_buffer_names"]) != self.ema_buffer_names:
            raise ValueError("EMA buffer layout changed across resume")
        self.load_state_dict(state["module"], strict=True)
        self.ema_update_step = int(state["ema_update_step"])
        self.mask_sampler.load_state_dict(state["mask_sampler"])

    def forward(
        self,
        model: nn.Module,
        batch: Mapping[str, torch.Tensor],
        *,
        jepa_weight: float,
        global_step: int,
    ) -> DenseVJEPA21Output:
        model_inputs = {
            key: batch[key]
            for key in ("input_ids", "attention_mask", "labels")
        }
        if jepa_weight == 0.0:
            native = model(**model_inputs, use_cache=False, return_dict=True)
            return DenseVJEPA21Output(
                loss=native.loss,
                native_loss=native.loss,
                jepa_loss=None,
                mask_loss=None,
                context_loss=None,
                context_coefficient=0.0,
                mask_loss_by_depth={},
                context_loss_by_depth={},
                student_scale_by_depth={},
                target_scale_by_depth={},
                mask=None,
            )

        if not self.ema_parameter_names:
            self.initialize_ema(model)
        backbone = _llama_backbone(model)
        mask = self.mask_sampler.sample(
            batch["labels"], batch["attention_mask"], self.ignore_index
        )
        with LayerCapture(backbone, self.supervised_depths) as capture:
            native = model(**model_inputs, use_cache=False, return_dict=True)
            raw_online = capture.values()
        online_levels = tuple(
            norm(level) for norm, level in zip(self.online_norms, raw_online)
        )
        target_levels = self._target_levels(
            backbone, batch["input_ids"], batch["attention_mask"]
        )
        future_prediction, context_prediction = self.predictor(online_levels, mask)
        split_future = future_prediction.split(self.config.encoder_dim, dim=-1)
        split_context = context_prediction.split(self.config.encoder_dim, dim=-1)
        if len(split_future) != len(self.supervised_depths):
            raise RuntimeError("predictor output does not align with supervised levels")

        weights = context_distance_weights(mask, future_prediction.dtype)
        mask_by_depth: dict[int, torch.Tensor] = {}
        context_by_depth: dict[int, torch.Tensor] = {}
        student_scale: dict[int, torch.Tensor] = {}
        target_scale: dict[int, torch.Tensor] = {}
        for depth, predicted_future, predicted_context, target, online in zip(
            self.supervised_depths,
            split_future,
            split_context,
            target_levels,
            online_levels,
        ):
            mask_by_depth[depth] = _masked_token_mean(
                (predicted_future - target).abs(), mask.future_mask
            )
            weighted_context_error = (predicted_context - target).abs() * weights.unsqueeze(-1)
            context_by_depth[depth] = _masked_token_mean(
                weighted_context_error, mask.context_mask
            )
            student_scale[depth] = _masked_token_mean(online.float().square(), mask.valid_mask).sqrt()
            target_scale[depth] = _masked_token_mean(target.float().square(), mask.valid_mask).sqrt()

        mask_loss = torch.stack(tuple(mask_by_depth.values())).mean()
        context_loss = torch.stack(tuple(context_by_depth.values())).mean()
        context_coefficient = self.context_schedule.value(global_step)
        jepa_loss = mask_loss + context_coefficient * context_loss
        total = native.loss + float(jepa_weight) * jepa_loss
        return DenseVJEPA21Output(
            loss=total,
            native_loss=native.loss,
            jepa_loss=jepa_loss,
            mask_loss=mask_loss,
            context_loss=context_loss,
            context_coefficient=context_coefficient,
            mask_loss_by_depth=mask_by_depth,
            context_loss_by_depth=context_by_depth,
            student_scale_by_depth=student_scale,
            target_scale_by_depth=target_scale,
            mask=mask,
        )


def dense_trainable_parameters(method: DenseVJEPA21) -> Iterable[nn.Parameter]:
    return (parameter for parameter in method.parameters() if parameter.requires_grad)


def component_gradient_norms(
    losses: Mapping[str, torch.Tensor],
    parameter_groups: Mapping[str, Sequence[nn.Parameter]],
) -> dict[str, float]:
    """Measure exact per-component norms without mutating ``.grad`` buffers.

    This is intentionally an occasional feasibility diagnostic: each loss
    requires a separate reverse-mode traversal.  Values describe the gradient
    contribution exactly as weighted in the final dense JEPA objective.
    """
    group_names = tuple(parameter_groups)
    group_sizes = tuple(len(parameter_groups[name]) for name in group_names)
    parameters = tuple(
        parameter
        for name in group_names
        for parameter in parameter_groups[name]
        if parameter.requires_grad
    )
    if sum(group_sizes) != len(parameters):
        raise ValueError("gradient diagnostic groups must contain trainable parameters only")
    metrics: dict[str, float] = {}
    for loss_name, loss in losses.items():
        gradients = torch.autograd.grad(
            loss, parameters, retain_graph=True, allow_unused=True,
        )
        offset = 0
        for group_name, group_size in zip(group_names, group_sizes):
            squared_norm = torch.zeros((), device=loss.device, dtype=torch.float32)
            for gradient in gradients[offset : offset + group_size]:
                if gradient is not None:
                    squared_norm.add_(gradient.detach().float().square().sum())
            metrics[f"{loss_name}_{group_name}_gradient_norm"] = float(
                squared_norm.sqrt()
            )
            offset += group_size
    return metrics
