"""Paired-replicate inference for the locked causal pilot."""
from __future__ import annotations

import itertools
from typing import Callable, Sequence

import numpy as np
from scipy.stats import rankdata, spearmanr


def exact_sign_flip_pvalue(differences: Sequence[float]) -> float:
    values = np.asarray(differences, dtype=float)
    observed = abs(values.mean())
    null = [
        abs(np.mean(values * np.asarray(signs)))
        for signs in itertools.product((-1.0, 1.0), repeat=len(values))
    ]
    return float(np.mean(np.asarray(null) >= observed - 1e-15))


def paired_bootstrap_ci(differences: Sequence[float], seed: int = 20260915,
                        draws: int = 10_000) -> tuple[float, float]:
    values = np.asarray(differences, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    means = values[indices].mean(1)
    return tuple(float(value) for value in np.quantile(means, [.025, .975]))


def bh_adjust(pvalues: Sequence[float]) -> list[float]:
    values = np.asarray(pvalues, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 1.0
    for reverse_rank in range(len(values) - 1, -1, -1):
        index = order[reverse_rank]
        rank = reverse_rank + 1
        running = min(running, values[index] * len(values) / rank)
        adjusted[index] = running
    return adjusted.tolist()


def paired_summary(native: Sequence[float], nextlat: Sequence[float]) -> dict:
    native = np.asarray(native, dtype=float)
    nextlat = np.asarray(nextlat, dtype=float)
    if native.shape != nextlat.shape or native.ndim != 1:
        raise ValueError("paired vectors must have the same one-dimensional shape")
    difference = nextlat - native
    return {
        "native_mean": float(native.mean()),
        "nextlat_mean": float(nextlat.mean()),
        "nextlat_minus_native": float(difference.mean()),
        "ci95": paired_bootstrap_ci(difference),
        "exact_sign_flip_p": exact_sign_flip_pvalue(difference),
        "native_by_replicate": native.tolist(),
        "nextlat_by_replicate": nextlat.tolist(),
        "replicate_differences": difference.tolist(),
    }


def exact_spearman_permutation_p(first: Sequence[float], second: Sequence[float]) -> float:
    first, second = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    if len(first) > 9:
        raise ValueError("exact permutation is intentionally limited to at most 9 pairs")
    left = rankdata(first).astype(float)
    right = rankdata(second).astype(float)
    left -= left.mean()
    right -= right.mean()
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator == 0:
        return float("nan")
    observed = abs(float(left @ right / denominator))
    extreme, total = 0, 0
    for permutation in itertools.permutations(right.tolist()):
        statistic = abs(float(left @ np.asarray(permutation) / denominator))
        extreme += statistic >= observed - 1e-15
        total += 1
    return extreme / total


def spearman_bootstrap_ci(first: Sequence[float], second: Sequence[float],
                          seed: int = 20260915, draws: int = 10_000):
    first, second = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(draws):
        indices = rng.integers(0, len(first), len(first))
        statistic = float(spearmanr(first[indices], second[indices]).statistic)
        if np.isfinite(statistic):
            values.append(statistic)
    if not values:
        return [float("nan"), float("nan")]
    return [float(value) for value in np.quantile(values, [.025, .975])]


def hierarchical_paired_ci(native: Sequence[Sequence[float]],
                           nextlat: Sequence[Sequence[float]],
                           seed: int = 20260915, draws: int = 10_000):
    if len(native) != len(nextlat):
        raise ValueError("replicate counts differ")
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(draws):
        replicate_indices = rng.integers(0, len(native), len(native))
        replicate_effects = []
        for replicate in replicate_indices:
            left = np.asarray(native[replicate], dtype=float)
            right = np.asarray(nextlat[replicate], dtype=float)
            if left.shape != right.shape:
                raise ValueError("nested paired units differ")
            indices = rng.integers(0, len(left), len(left))
            replicate_effects.append(float(np.mean(right[indices] - left[indices])))
        samples.append(np.mean(replicate_effects))
    return tuple(float(value) for value in np.quantile(samples, [.025, .975]))
