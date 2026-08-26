"""Frozen-checkpoint SIGReg gradient-response assay for the RTX 4050."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import set_seed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chemfm import (
    MODEL_DIR,
    TOKENIZER_DIR,
    ReactionCollator,
    load_lora_model,
    load_reaction_tokenizer,
)
from geometry_diagnosis import effective_rank
from jepa import CLMJEPA, add_predictor_tokens
from diagnose_sigreg_batch16_rtx4050 import (
    _capture_gradients,
    _cosine,
    _vector_stats,
    relative_sigreg_coefficient,
)
from train import load_adapter_checkpoint, read_rows, validate_serialization_endings

TRAIN_MANIFEST = ROOT / "data" / "clm_jepa_uspto_mit_pilot_1280" / "uspto_mit_train.csv"
OUTPUT = ROOT / "runs" / "diagnostics" / "sigreg_gradient_response.json"
PLOT_SOURCE = ROOT / "runs" / "diagnostics" / "sigreg_gradient_response_source.svg"
PLOT_TARGET = ROOT / "runs" / "diagnostics" / "sigreg_gradient_response_target.svg"

SEED = 533
BATCH_SIZE = 16
PHYSICAL_BATCH = 2
LEJEPA_TRADEOFF = 0.01
JEPA_ACTIVITY_PROBABILITY = 0.5
ACTIVE_OUTER_COEFFICIENT = 2.0
EXPECTED_COSINE_COEFFICIENT = 1.0
REGULARIZER_RELATIVE_COEFFICIENT = relative_sigreg_coefficient(LEJEPA_TRADEOFF)
ACTIVE_REGULARIZER_COEFFICIENT = (
    ACTIVE_OUTER_COEFFICIENT * REGULARIZER_RELATIVE_COEFFICIENT
)
EXPECTED_REGULARIZER_COEFFICIENT = (
    JEPA_ACTIVITY_PROBABILITY * ACTIVE_REGULARIZER_COEFFICIENT
)
PRIMARY_DIRECTION_SEED = 533
SECONDARY_DIRECTION_SEEDS = (917, 1907)
VISREG_SCALE_EPSILON = 1e-6

CHECKPOINTS = (
    {
        "label": "base_chemfm",
        "path": None,
        "historical_objective": "pretrained ChemFM-1B",
        "historical_readout": "none",
    },
    {
        "label": "native_epoch2",
        "path": ROOT / "runs" / "gate5" / "checkpoints" / "native-s533" / "epoch_2",
        "historical_objective": "native NTP",
        "historical_readout": "none",
    },
    {
        "label": "symmetric_jepa_epoch2",
        "path": ROOT / "runs" / "gate4_v2" / "reliable" / "clm_jepa-s533-checkpoints" / "epoch_2",
        "historical_objective": "symmetric cosine cLM-JEPA + NTP",
        "historical_readout": "k=1 [PRED]",
    },
    {
        "label": "target_sg_epoch2",
        "path": ROOT / "runs" / "gate4_rescue" / "target_sg-s533-batched-checkpoints" / "epoch_2",
        "historical_objective": "target-stop-gradient cosine cLM-JEPA + NTP",
        "historical_readout": "k=1 [PRED]",
    },
    {
        "label": "sigreg_batch128_epoch2",
        "path": ROOT / "runs" / "gate4_sigreg_batch128" / "sigreg-k0-b128-s533-checkpoints" / "epoch_2",
        "historical_objective": "symmetric cosine cLM-JEPA + batch-128 SIGReg + NTP",
        "historical_readout": "k=0 source EOS",
    },
)


def fixed_sigreg_parameters(
    dimensions: int,
    *,
    seed: int,
    device: torch.device,
    knots: int = 17,
    t_max: float = 3.0,
    num_slices: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(seed)
    directions = torch.randn(
        dimensions,
        num_slices,
        device=device,
        dtype=torch.float32,
        generator=generator,
    )
    directions = directions / directions.norm(dim=0, keepdim=True).clamp_min(1e-12)
    t = torch.linspace(0.0, t_max, knots, device=device, dtype=torch.float32)
    dt = t_max / (knots - 1)
    quadrature = torch.full_like(t, 2.0 * dt)
    quadrature[[0, -1]] = dt
    normal_cf = torch.exp(-0.5 * t.square())
    weights = quadrature * normal_cf
    return directions, t, normal_cf, weights


def fixed_sigreg_loss(
    representations: torch.Tensor,
    parameters: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """Official LeJEPA Epps-Pulley statistic with caller-frozen slices."""
    if representations.ndim != 3 or representations.size(1) < 2:
        raise ValueError("SIGReg requires [views, samples>=2, dimensions]")
    directions, t, normal_cf, weights = parameters
    values = representations.float()
    projected = values @ directions
    arguments = projected.unsqueeze(-1) * t
    real_error = arguments.cos().mean(dim=1) - normal_cf
    imaginary = arguments.sin().mean(dim=1)
    statistic = (
        ((real_error.square() + imaginary.square()) @ weights)
        * representations.size(1)
    )
    return statistic.mean()


def visreg_scale_loss(representations: torch.Tensor) -> torch.Tensor:
    """VISReg's scale-only term, independently averaged over the two views."""
    if representations.ndim != 3 or representations.size(1) < 2:
        raise ValueError("VISReg scale loss requires [views, samples>=2, dimensions]")
    values = representations.float()
    centered = values - values.mean(dim=1, keepdim=True)
    std = centered.norm(dim=1).div(math.sqrt(values.size(1))) + VISREG_SCALE_EPSILON
    return (1.0 - std).square().mean()


def cosine_jepa_loss(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (1.0 - F.cosine_similarity(source, target, dim=-1)).mean()


def _device_batch(raw: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        name: value.to(device)
        for name, value in raw.items()
        if torch.is_tensor(value)
    }


def _endpoint_output(method, model, batch):
    return method(
        model,
        batch,
        k=0,
        jepa_weight=0.0,
        native_weight=1.0,
        monitor_only=True,
        stop_gradient_target=False,
        sigreg_tradeoff=0.0,
        jepa_ratio=1.0,
        force_jepa_active=True,
    )


def _stack_states(output) -> torch.Tensor:
    if output.source_states is None or output.target_states is None:
        raise RuntimeError("missing JEPA endpoint states")
    return torch.stack((output.source_states, output.target_states))


def _disable_stochastic_training_behavior(model) -> dict[str, int]:
    dropout_modules = 0
    attention_modules = 0
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
            dropout_modules += 1
        if hasattr(module, "attention_dropout"):
            module.attention_dropout = 0.0
            attention_modules += 1
    return {
        "dropout_modules_zeroed": dropout_modules,
        "attention_dropout_fields_zeroed": attention_modules,
    }


def _geometry(values: torch.Tensor) -> dict[str, float]:
    values = values.float().cpu()
    return {
        "variance": float(values.var(dim=0, unbiased=False).mean()),
        "mean_direction_energy": float(
            values.mean(dim=0).square().sum()
            / values.square().sum(dim=1).mean().clamp_min(1e-30)
        ),
        "effective_rank": effective_rank(values),
    }


def _norm(vector: dict[str, torch.Tensor]) -> float:
    return _vector_stats({"vector": vector}, {"vector": 1.0})["norm"]


def _endpoint_losses_and_gradients(
    states: torch.Tensor,
    *,
    direction_seed: int,
) -> tuple[dict[str, float], dict[str, torch.Tensor], str]:
    leaf = states.detach().clone().requires_grad_(True)
    parameters = fixed_sigreg_parameters(
        leaf.size(-1), seed=direction_seed, device=leaf.device
    )
    losses = {
        "cosine": cosine_jepa_loss(leaf[0], leaf[1]),
        "sigreg": fixed_sigreg_loss(leaf, parameters),
        "variance": visreg_scale_loss(leaf),
    }
    gradients: dict[str, torch.Tensor] = {}
    for index, (name, loss) in enumerate(losses.items()):
        gradients[name] = torch.autograd.grad(
            loss,
            leaf,
            retain_graph=index < len(losses) - 1,
        )[0].detach()
    direction_bytes = parameters[0].detach().float().cpu().numpy().tobytes()
    fingerprint = hashlib.sha256(direction_bytes).hexdigest()
    return (
        {name: float(loss.detach()) for name, loss in losses.items()},
        gradients,
        fingerprint,
    )


def _parameter_gradient_from_endpoint_vjp(
    *, model, method, raw_chunks, endpoint_gradient: torch.Tensor
) -> tuple[dict[str, torch.Tensor], float]:
    model.zero_grad(set_to_none=True)
    offset = 0
    max_recompute_error = 0.0
    for raw in raw_chunks:
        batch = _device_batch(raw, model.device)
        output = _endpoint_output(method, model, batch)
        states = _stack_states(output)
        count = states.size(1)
        gradient = endpoint_gradient[:, offset:offset + count].to(states.dtype)
        surrogate = (states * gradient).sum()
        surrogate.backward()
        offset += count
    if offset != endpoint_gradient.size(1):
        raise RuntimeError("endpoint VJP did not consume the fixed batch")
    vector = _capture_gradients(model)
    model.zero_grad(set_to_none=True)
    return vector, max_recompute_error


def _ntp_parameter_gradient(
    *, model, method, raw_chunks
) -> tuple[float, dict[str, torch.Tensor]]:
    model.zero_grad(set_to_none=True)
    losses = []
    for raw in raw_chunks:
        batch = _device_batch(raw, model.device)
        output = _endpoint_output(method, model, batch)
        (output.native_loss / len(raw_chunks)).backward()
        losses.append(float(output.native_loss.detach()))
    vector = _capture_gradients(model)
    model.zero_grad(set_to_none=True)
    return sum(losses) / len(losses), vector


def _endpoint_gradient_metrics(gradients: dict[str, torch.Tensor]) -> dict[str, Any]:
    result = {}
    for view_index, view in enumerate(("source", "target")):
        norms = {
            name: float(gradient[view_index].float().norm())
            for name, gradient in gradients.items()
        }
        result[view] = {
            "raw_norms": norms,
            "weighted_norms_expected": {
                "cosine": norms["cosine"] * EXPECTED_COSINE_COEFFICIENT,
                "sigreg": norms["sigreg"] * EXPECTED_REGULARIZER_COEFFICIENT,
                "variance": norms["variance"] * EXPECTED_REGULARIZER_COEFFICIENT,
            },
            "raw_ratios": {
                "sigreg_to_cosine": norms["sigreg"] / max(norms["cosine"], 1e-30),
                "variance_to_cosine": norms["variance"] / max(norms["cosine"], 1e-30),
            },
            "weighted_ratios_expected": {
                "sigreg_to_cosine": (
                    norms["sigreg"] * EXPECTED_REGULARIZER_COEFFICIENT
                    / max(norms["cosine"] * EXPECTED_COSINE_COEFFICIENT, 1e-30)
                ),
                "variance_to_cosine": (
                    norms["variance"] * EXPECTED_REGULARIZER_COEFFICIENT
                    / max(norms["cosine"] * EXPECTED_COSINE_COEFFICIENT, 1e-30)
                ),
            },
            "cosines": {
                "sigreg_vs_cosine": float(F.cosine_similarity(
                    gradients["sigreg"][view_index].float().flatten(),
                    gradients["cosine"][view_index].float().flatten(),
                    dim=0,
                )),
                "variance_vs_cosine": float(F.cosine_similarity(
                    gradients["variance"][view_index].float().flatten(),
                    gradients["cosine"][view_index].float().flatten(),
                    dim=0,
                )),
            },
        }
    return result


def _parameter_metrics(vectors: dict[str, dict[str, torch.Tensor]]) -> dict[str, Any]:
    norms = {name: _norm(vector) for name, vector in vectors.items()}
    weighted_active = {
        "ntp": norms["ntp"],
        "cosine": ACTIVE_OUTER_COEFFICIENT * norms["cosine"],
        "sigreg": ACTIVE_REGULARIZER_COEFFICIENT * norms["sigreg"],
        "variance": ACTIVE_REGULARIZER_COEFFICIENT * norms["variance"],
    }
    weighted_expected = {
        "ntp": norms["ntp"],
        "cosine": EXPECTED_COSINE_COEFFICIENT * norms["cosine"],
        "sigreg": EXPECTED_REGULARIZER_COEFFICIENT * norms["sigreg"],
        "variance": EXPECTED_REGULARIZER_COEFFICIENT * norms["variance"],
    }
    return {
        "raw_norms": norms,
        "weighted_norms_active_update": weighted_active,
        "weighted_norms_expected_over_dropout": weighted_expected,
        "raw_ratios": {
            "sigreg_to_cosine": norms["sigreg"] / max(norms["cosine"], 1e-30),
            "sigreg_to_ntp": norms["sigreg"] / max(norms["ntp"], 1e-30),
            "variance_to_cosine": norms["variance"] / max(norms["cosine"], 1e-30),
            "variance_to_ntp": norms["variance"] / max(norms["ntp"], 1e-30),
        },
        "weighted_ratios_expected": {
            "sigreg_to_cosine": weighted_expected["sigreg"] / max(weighted_expected["cosine"], 1e-30),
            "sigreg_to_ntp": weighted_expected["sigreg"] / max(weighted_expected["ntp"], 1e-30),
            "variance_to_cosine": weighted_expected["variance"] / max(weighted_expected["cosine"], 1e-30),
            "variance_to_ntp": weighted_expected["variance"] / max(weighted_expected["ntp"], 1e-30),
        },
        "cosines": {
            "sigreg_vs_cosine": _cosine(vectors["sigreg"], vectors["cosine"]),
            "sigreg_vs_ntp": _cosine(vectors["sigreg"], vectors["ntp"]),
            "variance_vs_cosine": _cosine(vectors["variance"], vectors["cosine"]),
            "variance_vs_ntp": _cosine(vectors["variance"], vectors["ntp"]),
            "cosine_vs_ntp": _cosine(vectors["cosine"], vectors["ntp"]),
        },
    }


def _checkpoint_result(*, model, method, raw_chunks, checkpoint: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    model.zero_grad(set_to_none=True)
    states = []
    token_count = 0
    with torch.no_grad():
        for raw in raw_chunks:
            token_count += int(raw["attention_mask"].sum())
            batch = _device_batch(raw, model.device)
            states.append(_stack_states(_endpoint_output(method, model, batch)))
    states = torch.cat(states, dim=1)
    primary_losses, endpoint_gradients, fingerprint = _endpoint_losses_and_gradients(
        states, direction_seed=PRIMARY_DIRECTION_SEED
    )
    ntp_loss, ntp_vector = _ntp_parameter_gradient(
        model=model, method=method, raw_chunks=raw_chunks
    )
    parameter_vectors = {"ntp": ntp_vector}
    for name in ("cosine", "sigreg", "variance"):
        parameter_vectors[name], _ = _parameter_gradient_from_endpoint_vjp(
            model=model,
            method=method,
            raw_chunks=raw_chunks,
            endpoint_gradient=endpoint_gradients[name],
        )

    repeated = {}
    for direction_seed in (PRIMARY_DIRECTION_SEED, *SECONDARY_DIRECTION_SEEDS):
        if direction_seed == PRIMARY_DIRECTION_SEED:
            gradients = endpoint_gradients
        else:
            _, gradients, _ = _endpoint_losses_and_gradients(
                states, direction_seed=direction_seed
            )
        repeated[str(direction_seed)] = _endpoint_gradient_metrics(gradients)

    result = {
        "checkpoint": None if checkpoint["path"] is None else str(checkpoint["path"].resolve()),
        "historical_objective": checkpoint["historical_objective"],
        "historical_readout": checkpoint["historical_readout"],
        "assay_readout": "k=0 final source EOS and final target EOS for every checkpoint",
        "tokens": token_count,
        "losses": {"native_ntp": ntp_loss, **primary_losses},
        "geometry": {
            "source": _geometry(states[0]),
            "target": _geometry(states[1]),
        },
        "parameter_gradients": _parameter_metrics(parameter_vectors),
        "endpoint_gradients": _endpoint_gradient_metrics(endpoint_gradients),
        "repeated_direction_endpoint_gradients": repeated,
        "primary_sigreg_direction_sha256": fingerprint,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_allocated_vram_bytes": int(torch.cuda.max_memory_allocated()),
    }
    del states, endpoint_gradients, parameter_vectors
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    return result


def _write_svg(results: dict[str, Any], view: str, path: Path) -> None:
    width, height = 900, 520
    left, right, top, bottom = 90, 35, 45, 75
    plot_w = width - left - right
    plot_h = height - top - bottom
    rows = []
    for label, result in results.items():
        variance = result["geometry"][view]["variance"]
        ratios = result["endpoint_gradients"][view]["raw_ratios"]
        rows.append((label, variance, ratios["sigreg_to_cosine"], ratios["variance_to_cosine"]))
    x_values = [math.log10(max(row[1], 1e-30)) for row in rows]
    y_values = [math.log10(max(value, 1e-30)) for row in rows for value in row[2:]]
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    x_pad = max((x_max - x_min) * 0.08, 0.1)
    y_pad = max((y_max - y_min) * 0.12, 0.1)
    x_min, x_max = x_min - x_pad, x_max + x_pad
    y_min, y_max = y_min - y_pad, y_max + y_pad

    def xy(x: float, y: float) -> tuple[float, float]:
        px = left + (math.log10(max(x, 1e-30)) - x_min) / (x_max - x_min) * plot_w
        py = top + (y_max - math.log10(max(y, 1e-30))) / (y_max - y_min) * plot_h
        return px, py

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="25" text-anchor="middle" font-family="sans-serif" font-size="18">{html.escape(view.title())} corrective gradient vs representation variance</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="black"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="black"/>',
        f'<text x="{left + plot_w/2}" y="{height - 18}" text-anchor="middle" font-family="sans-serif" font-size="14">representation variance (log scale)</text>',
        f'<text x="20" y="{top + plot_h/2}" text-anchor="middle" transform="rotate(-90 20 {top + plot_h/2})" font-family="sans-serif" font-size="14">raw endpoint gradient norm / cosine norm (log scale)</text>',
    ]
    for tick in range(5):
        x_log = x_min + tick * (x_max - x_min) / 4
        x = left + tick * plot_w / 4
        parts.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_h}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{x:.1f}" y="{top + plot_h + 22}" text-anchor="middle" font-family="sans-serif" font-size="11">1e{x_log:.1f}</text>')
        y_log = y_min + tick * (y_max - y_min) / 4
        y = top + plot_h - tick * plot_h / 4
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" font-family="sans-serif" font-size="11">1e{y_log:.1f}</text>')

    series = ((2, "#2563eb", "SIGReg / cosine"), (3, "#dc2626", "VISReg scale / cosine"))
    sorted_rows = sorted(rows, key=lambda row: row[1])
    for value_index, color, legend in series:
        points = [xy(row[1], row[value_index]) for row in sorted_rows]
        parts.append('<polyline fill="none" stroke="{}" stroke-width="2" points="{}"/>'.format(
            color, " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        ))
        for row, (x, y) in zip(sorted_rows, points):
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{color}"/>')
            if value_index == 2:
                parts.append(f'<text x="{x + 7:.1f}" y="{y - 7:.1f}" font-family="sans-serif" font-size="10">{html.escape(row[0])}</text>')
        legend_y = top + 18 + (value_index - 2) * 22
        parts.append(f'<line x1="{left + plot_w - 190}" y1="{legend_y}" x2="{left + plot_w - 160}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{left + plot_w - 150}" y="{legend_y + 4}" font-family="sans-serif" font-size="12">{legend}</text>')
    parts.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen-checkpoint SIGReg gradient-response assay")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--source-plot", type=Path, default=PLOT_SOURCE)
    parser.add_argument("--target-plot", type=Path, default=PLOT_TARGET)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise EnvironmentError("ChemFM frozen gradient assay requires CUDA")
    for checkpoint in CHECKPOINTS:
        if checkpoint["path"] is not None and not checkpoint["path"].exists():
            raise FileNotFoundError(checkpoint["path"])

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    set_seed(SEED)
    rows = read_rows("uspto_mit_synthesis", path=TRAIN_MANIFEST)
    permutation = torch.randperm(
        len(rows), generator=torch.Generator().manual_seed(SEED)
    ).tolist()
    fixed_indices = permutation[:BATCH_SIZE]
    fixed_rows = [rows[index] for index in fixed_indices]

    tokenizer = load_reaction_tokenizer(TOKENIZER_DIR)
    chemfm_vocab_size = len(tokenizer)
    predictor_ids = add_predictor_tokens(tokenizer)
    collator = ReactionCollator(tokenizer, task="forward")
    validate_serialization_endings(collator, fixed_rows, tokenizer.eos_token_id)
    raw_chunks = [
        collator(fixed_rows[start:start + PHYSICAL_BATCH])
        for start in range(0, BATCH_SIZE, PHYSICAL_BATCH)
    ]
    model = load_lora_model(
        MODEL_DIR, tokenizer, chemfm_vocab_size=chemfm_vocab_size
    ).cuda()
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.enable_input_require_grads()
    model.train()
    stochastic_controls = _disable_stochastic_training_behavior(model)
    method = CLMJEPA(
        predictor_ids,
        tokenizer.eos_token_id,
        tokenizer.pad_token_id,
        sigreg_seed=SEED,
    )
    base_trainable_state = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }

    output = {
        "scope": "frozen-checkpoint gradient-response assay; no optimizer steps",
        "seed": SEED,
        "manifest": str(TRAIN_MANIFEST.resolve()),
        "fixed_examples": [
            {
                "manifest_index_zero_based": index,
                "source_sha256": hashlib.sha256(row["src"].encode()).hexdigest(),
                "target_sha256": hashlib.sha256(row["tgt"].encode()).hexdigest(),
                "source": row["src"],
                "target": row["tgt"],
            }
            for index, row in zip(fixed_indices, fixed_rows)
        ],
        "configuration": {
            "examples": BATCH_SIZE,
            "physical_recomputation_batch": PHYSICAL_BATCH,
            "assay_readout": "k=0 final source EOS; final target EOS",
            "stochastic_behavior": "model.train for checkpointing, but all Dropout p and attention_dropout fields set to zero",
            **stochastic_controls,
            "sigreg": "LeJEPA Epps-Pulley; 17 knots [0,3]; 1024 fixed unit-Gaussian slices; views evaluated independently then averaged",
            "primary_direction_seed": PRIMARY_DIRECTION_SEED,
            "secondary_direction_seeds": list(SECONDARY_DIRECTION_SEEDS),
            "visreg_scale": "mean_views,dimensions (1 - (||z-centered(z)||_2/sqrt(N) + 1e-6))^2",
            "vicreg_not_used_for_g_variance": "official VICReg instead uses mean ReLU(1-sqrt(var+1e-4)); VISReg scale was selected to test the VISReg claim directly",
            "lejepa_tradeoff": LEJEPA_TRADEOFF,
            "regularizer_relative_coefficient": REGULARIZER_RELATIVE_COEFFICIENT,
            "active_outer_cosine_coefficient": ACTIVE_OUTER_COEFFICIENT,
            "active_regularizer_coefficient": ACTIVE_REGULARIZER_COEFFICIENT,
            "expected_cosine_coefficient": EXPECTED_COSINE_COEFFICIENT,
            "expected_regularizer_coefficient": EXPECTED_REGULARIZER_COEFFICIENT,
            "variance_weighting_for_assay": "same hypothetical regularizer slot as SIGReg for coefficient-normalized comparison only",
        },
        "checkpoints": {},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for checkpoint in CHECKPOINTS:
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if parameter.requires_grad:
                    parameter.copy_(base_trainable_state[name].to(parameter.device))
        if checkpoint["path"] is not None:
            load_adapter_checkpoint(model, checkpoint["path"].resolve())
        torch.cuda.reset_peak_memory_stats()
        output["checkpoints"][checkpoint["label"]] = _checkpoint_result(
            model=model,
            method=method,
            raw_chunks=raw_chunks,
            checkpoint=checkpoint,
        )
        args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({
            "checkpoint": checkpoint["label"],
            "geometry": output["checkpoints"][checkpoint["label"]]["geometry"],
            "raw_ratios": output["checkpoints"][checkpoint["label"]]["parameter_gradients"]["raw_ratios"],
        }))

    _write_svg(output["checkpoints"], "source", args.source_plot)
    _write_svg(output["checkpoints"], "target", args.target_plot)
    output["plots"] = {
        "source": str(args.source_plot.resolve()),
        "target": str(args.target_plot.resolve()),
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "plots": output["plots"]}, indent=2))


if __name__ == "__main__":
    main()
