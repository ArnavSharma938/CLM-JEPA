"""Numerically explicit geometry primitives for the frozen STP mechanism audit.

All functions in this module are model-independent and operate in FP32/FP64.
They deliberately expose the quantities in the STP theorem rather than using
the training objectives as proxies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F


EPS = 1e-12


def _fp32(x: torch.Tensor) -> torch.Tensor:
    return x if x.dtype == torch.float32 else x.float()


def chord_coordinates(
    h_s: torch.Tensor, h_r: torch.Tensor, h_t: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return alpha, perpendicular deviation q, and normalized tube radius."""
    h_s, h_r, h_t = map(_fp32, (h_s, h_r, h_t))
    chord = h_t - h_s
    u = h_r - h_s
    denom = chord.square().sum(dim=-1)
    alpha = (u * chord).sum(dim=-1) / denom.clamp_min(EPS)
    q = u - alpha.unsqueeze(-1) * chord
    rho = q.norm(dim=-1) / denom.sqrt().clamp_min(EPS)
    invalid = denom <= EPS
    alpha = alpha.masked_fill(invalid, torch.nan)
    rho = rho.masked_fill(invalid, torch.nan)
    q = q.masked_fill(invalid.unsqueeze(-1), torch.nan)
    return alpha, q, rho


def tube_scale_space(
    path: torch.Tensor,
    *,
    thresholds: Sequence[float] = (0.05, 0.10, 0.20, 0.50),
) -> list[dict[str, float]]:
    """Exhaustively summarize every interior point for every outer span length.

    The reduction is performed span-by-span, avoiding materialization of the
    O(n^3) collection of triples while still evaluating every triple.
    """
    path = _fp32(path)
    n = int(path.shape[0])
    # Translation-invariant Gram identities reduce the exhaustive computation
    # from O(n^3 * hidden_size) to O(n^2 * hidden_size + n^3) without changing
    # the set of evaluated triples.
    centered = path - path[:1]
    # The O(n^2*d) Gram multiplication stays on the input device.  Moving the
    # tiny Gram matrix once to NumPy avoids hundreds of small CUDA kernels and
    # synchronizations in the span-by-span scalar reductions.
    gram = (centered @ centered.T).detach().cpu().numpy()
    diagonal = np.diag(gram)
    out: list[dict[str, float]] = []
    for length in range(2, n):
        starts = np.arange(n - length)[:, None]
        ends = starts + length
        interiors = starts + np.arange(1, length)[None, :]
        chord_sq = (
            diagonal[ends] + diagonal[starts] - 2 * gram[starts, ends]
        )
        u_sq = (
            diagonal[interiors] + diagonal[starts] - 2 * gram[interiors, starts]
        )
        u_dot_chord = (
            gram[interiors, ends] - gram[interiors, starts]
            - gram[starts, ends] + diagonal[starts]
        )
        safe_chord = np.maximum(chord_sq, EPS)
        alpha = u_dot_chord / safe_chord
        q_sq = np.maximum(u_sq - np.square(u_dot_chord) / safe_chord, 0)
        rho = np.sqrt(q_sq) / np.maximum(np.sqrt(safe_chord), EPS)
        invalid_chords = chord_sq <= EPS
        rho = np.where(invalid_chords, np.nan, rho)
        alpha = np.where(invalid_chords, np.nan, alpha)
        valid = np.isfinite(rho)
        valid_count = int(valid.sum())
        if valid_count == 0:
            continue
        zero_chords = int(rho.size - valid_count)
        rho = rho[valid]
        alpha = alpha[valid]
        quantiles = np.quantile(rho, [.5, .9, .95])
        statistics = [
            float(rho.mean()), float(np.sqrt(np.square(rho).mean())), float(rho.max()),
            *map(float, quantiles),
            float(((alpha < 0) | (alpha > 1)).mean()),
            *(float((rho > threshold).mean()) for threshold in thresholds),
        ]
        row: dict[str, float] = {
            "span_length": length,
            "triples": int(rho.size),
            "zero_chords": zero_chords,
            "mean": statistics[0], "rms": statistics[1],
            "maximum": statistics[2], "p50": statistics[3],
            "p90": statistics[4], "p95": statistics[5],
            "monotonicity_violation": statistics[6],
        }
        for offset, threshold in enumerate(thresholds, start=7):
            row[f"fraction_gt_{threshold:g}"] = statistics[offset]
        out.append(row)
    return out


def estimate_piecewise_change_point(rows: Sequence[dict], value: str = "rms") -> dict:
    """Fit two continuous lines to log(metric) and return the SSE breakpoint."""
    clean = [(float(r["span_length"]), float(r[value])) for r in rows if r.get(value, 0) > 0]
    if len(clean) < 9:
        return {"breakpoint": math.nan, "sse": math.nan, "n": len(clean)}
    x = np.asarray([p[0] for p in clean], dtype=np.float64)
    y = np.log(np.asarray([p[1] for p in clean], dtype=np.float64))
    best = None
    for index in range(4, len(x) - 4 + 1):
        knot = x[index]
        design = np.column_stack([np.ones_like(x), x, np.maximum(x - knot, 0.0)])
        coef, *_ = np.linalg.lstsq(design, y, rcond=None)
        residual = y - design @ coef
        candidate = (float(residual @ residual), float(knot), coef)
        if best is None or candidate[0] < best[0]:
            best = candidate
    assert best is not None
    return {
        "breakpoint": best[1], "sse": best[0], "n": len(clean),
        "intercept": float(best[2][0]), "slope_before": float(best[2][1]),
        "slope_after": float(best[2][1] + best[2][2]),
    }


def cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(_fp32(a), _fp32(b), dim=-1, eps=1e-8)


def tangent_autocorrelation(path: torch.Tensor, max_lag: int = 32) -> list[dict]:
    velocity = (_fp32(path)[1:] - _fp32(path)[:-1]).detach().cpu().numpy()
    velocity = velocity / np.maximum(np.linalg.norm(velocity, axis=-1, keepdims=True), 1e-8)
    out = []
    for lag in range(1, min(max_lag, len(velocity) - 1) + 1):
        values = np.sum(velocity[:-lag] * velocity[lag:], axis=-1)
        out.append({
            "lag": lag, "n": int(values.size), "mean": float(values.mean()),
            "median": float(np.median(values)),
        })
    return out


def multiscale_turning(path: torch.Tensor, max_scale: int = 32) -> list[dict]:
    path = _fp32(path)
    out = []
    for scale in range(1, min(max_scale, (len(path) - 1) // 2) + 1):
        before = path[scale:-scale] - path[:-2 * scale]
        after = path[2 * scale:] - path[scale:-scale]
        c = cosine(before, after).clamp(-1, 1)
        angles = torch.acos(c)
        for position, angle in enumerate(angles, start=scale):
            out.append({"position": position, "scale": scale, "angle": float(angle)})
    return out


def acceleration_decomposition(path: torch.Tensor) -> dict[str, torch.Tensor]:
    path = _fp32(path)
    v = path[1:-1] - path[:-2]
    a = path[2:] - 2 * path[1:-1] + path[:-2]
    coefficient = (a * v).sum(-1) / v.square().sum(-1).clamp_min(EPS)
    parallel = coefficient.unsqueeze(-1) * v
    normal = a - parallel
    speed = v.norm(dim=-1)
    return {
        "speed": speed,
        "acceleration_parallel": parallel.norm(dim=-1),
        "acceleration_normal": normal.norm(dim=-1),
        "normalized_normal": normal.norm(dim=-1) / speed.clamp_min(EPS),
        "velocity": v,
        "acceleration": a,
    }


def optimal_ray_residual(h_s: torch.Tensor, h_r: torch.Tensor, h_t: torch.Tensor) -> torch.Tensor:
    v = _fp32(h_r) - _fp32(h_s)
    future = _fp32(h_t) - _fp32(h_r)
    alpha = (future * v).sum(-1) / v.square().sum(-1).clamp_min(EPS)
    residual = future - alpha.unsqueeze(-1) * v
    return residual.norm(dim=-1) / future.norm(dim=-1).clamp_min(EPS)


def categorical_fisher_squared(delta_logits: torch.Tensor, probabilities: torch.Tensor) -> torch.Tensor:
    """u^T G u from delta_logits=Wu and categorical probabilities."""
    delta_logits, probabilities = map(_fp32, (delta_logits, probabilities))
    mean = (probabilities * delta_logits).sum(-1)
    return (probabilities * delta_logits.square()).sum(-1) - mean.square()


def fisher_rao_distance(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    p, q = map(_fp32, (p, q))
    coefficient = (p.clamp_min(0).sqrt() * q.clamp_min(0).sqrt()).sum(-1)
    return 2.0 * torch.acos(coefficient.clamp(-1.0, 1.0))


def fisher_rao_triangle_excess(p_s: torch.Tensor, p_r: torch.Tensor, p_t: torch.Tensor) -> torch.Tensor:
    return fisher_rao_distance(p_s, p_r) + fisher_rao_distance(p_r, p_t) - fisher_rao_distance(p_s, p_t)


def fisher_rao_path_efficiency(probabilities: torch.Tensor) -> torch.Tensor:
    endpoint = fisher_rao_distance(probabilities[0], probabilities[-1])
    length = fisher_rao_distance(probabilities[:-1], probabilities[1:]).sum()
    return endpoint / length.clamp_min(EPS)


def gold_logprob_gradient(
    hidden: torch.Tensor, lm_weight: torch.Tensor, gold_token: torch.Tensor | int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Analytic gradient of log p(gold) with respect to final hidden state."""
    hidden = _fp32(hidden)
    weight = _fp32(lm_weight)
    logits = hidden @ weight.T
    probabilities = logits.softmax(-1)
    gold = torch.as_tensor(gold_token, device=hidden.device, dtype=torch.long)
    gradient = weight[gold] - probabilities @ weight
    return gradient, logits, probabilities


def predictive_sensitivity(gradient: torch.Tensor, component: torch.Tensor) -> dict[str, torch.Tensor]:
    gradient, component = map(_fp32, (gradient, component))
    signed = (gradient * component).sum(-1)
    normalized = signed / (gradient.norm(dim=-1) * component.norm(dim=-1)).clamp_min(EPS)
    return {"signed": signed, "cosine": normalized}


def logits_summary(logits: torch.Tensor, gold: int, topk: int = 10) -> dict:
    logits = _fp32(logits)
    logp = logits.log_softmax(-1)
    gold_value = logp[gold]
    rank = int((logits > logits[gold]).sum().item()) + 1
    competitor = torch.cat([logits[:gold], logits[gold + 1:]]).max()
    probabilities = logits.softmax(-1)
    entropy = -(probabilities * logp).sum()
    ids = torch.topk(logits, min(topk, logits.numel())).indices.tolist()
    return {
        "gold_log_probability": float(gold_value), "gold_rank": rank,
        "gold_margin": float(logits[gold] - competitor), "entropy": float(entropy),
        "topk": ids,
    }


def curvature_removal_intervention(
    hidden: torch.Tensor,
    perpendicular: torch.Tensor,
    lm_weight: torch.Tensor,
    gold: int,
    gamma: float,
    restore_norm: bool,
) -> dict:
    hidden, perpendicular, weight = map(_fp32, (hidden, perpendicular, lm_weight))
    changed = hidden - gamma * perpendicular
    if restore_norm:
        changed = changed * (hidden.norm() / changed.norm().clamp_min(EPS))
    return logits_summary(changed @ weight.T, gold)


def released_objective_anatomy(before: torch.Tensor, patch: torch.Tensor, after: torch.Tensor) -> dict:
    before, patch, after = map(_fp32, (before, patch, after))
    complement = before + after
    denominator = before.norm(dim=-1) + after.norm(dim=-1)
    return {
        "loss": 1.0 - cosine(patch, complement),
        "cos_patch_before": cosine(patch, before),
        "cos_patch_after": cosine(patch, after),
        "cos_before_after": cosine(before, after),
        "cancellation_ratio": complement.norm(dim=-1) / denominator.clamp_min(EPS),
    }


def matched_geodesic_displacement(
    native: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    treatment: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> dict[str, torch.Tensor]:
    n_s, n_r, n_t = map(_fp32, native)
    s_s, s_r, s_t = map(_fp32, treatment)
    alpha, q, native_rho = chord_coordinates(n_s, n_r, n_t)
    _, _, treatment_rho = chord_coordinates(s_s, s_r, s_t)
    delta_s, delta_r, delta_t = s_s - n_s, s_r - n_r, s_t - n_t
    native_chord, treatment_chord = n_t - n_s, s_t - s_s
    return {
        "native_rho": native_rho, "treatment_rho": treatment_rho,
        "delta_rho": treatment_rho - native_rho,
        "correction_cosine": cosine(delta_r, -q),
        "endpoint_middle_ratio": (
            delta_s.norm(dim=-1) + delta_t.norm(dim=-1)
        ) / (2 * delta_r.norm(dim=-1)).clamp_min(EPS),
        "chord_cosine": cosine(native_chord, treatment_chord),
        "chord_norm_ratio": treatment_chord.norm(dim=-1) / native_chord.norm(dim=-1).clamp_min(EPS),
        "middle_displacement": delta_r.norm(dim=-1),
        "endpoint_displacement": .5 * (delta_s.norm(dim=-1) + delta_t.norm(dim=-1)),
        "native_alpha": alpha,
    }


def local_tangent_acceleration(
    query_state: torch.Tensor,
    velocity: torch.Tensor,
    acceleration: torch.Tensor,
    neighbors: torch.Tensor,
    tangent_dim: int,
) -> dict[str, torch.Tensor]:
    """Decompose acceleration using a local PCA tangent basis."""
    query_state, velocity, acceleration, neighbors = map(
        _fp32, (query_state, velocity, acceleration, neighbors)
    )
    centered = neighbors - neighbors.mean(dim=0, keepdim=True)
    # Right singular vectors span ambient directions; full_matrices=False keeps
    # the decomposition bounded by the neighbor count.
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    basis = vh[: min(tangent_dim, vh.shape[0])]
    tangent_a = (acceleration @ basis.T) @ basis
    normal_a = acceleration - tangent_a
    tangent_v = (velocity @ basis.T) @ basis
    along = (tangent_a * tangent_v).sum() / tangent_v.square().sum().clamp_min(EPS)
    geo = tangent_a - along * tangent_v
    return {
        "tangent_acceleration": tangent_a.norm(),
        "normal_acceleration": normal_a.norm(),
        "geodesic_violation": geo.norm(),
        "projected_velocity": tangent_v.norm(),
    }


@dataclass(frozen=True)
class NeighborIndex:
    states: torch.Tensor
    reaction_ids: np.ndarray
    segments: np.ndarray
    mean: torch.Tensor | None = None
    scale: torch.Tensor | None = None

    def neighbors(
        self, query: torch.Tensor, reaction_id: str, segment: str, k: int,
        *, same_segment: bool, whitened: bool,
    ) -> torch.Tensor:
        eligible_np = self.reaction_ids != reaction_id
        if same_segment:
            eligible_np &= self.segments == segment
        eligible = torch.from_numpy(eligible_np).to(self.states.device)
        candidates = self.states[eligible]
        if len(candidates) < k:
            raise ValueError(f"only {len(candidates)} eligible neighbors for k={k}")
        if whitened:
            if self.mean is None or self.scale is None:
                raise ValueError("whitened search requires mean and scale")
            distance = (((candidates - self.mean) / self.scale) - ((query - self.mean) / self.scale)).square().sum(-1)
        else:
            distance = (candidates - query).square().sum(-1)
        selected = torch.topk(distance, k, largest=False).indices
        return candidates[selected]
