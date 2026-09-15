"""Protein/cluster-level nonparametric inference helpers."""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


def cluster_bootstrap_mean(values, clusters, *, draws: int = 5000, seed: int = 0):
    values = np.asarray(values, dtype=float)
    clusters = np.asarray(clusters)
    unique = np.unique(clusters)
    grouped = np.asarray([values[clusters == key].mean() for key in unique])
    rng = np.random.default_rng(seed)
    samples = np.empty(draws)
    for index in range(draws):
        chosen = rng.integers(0, len(unique), size=len(unique))
        samples[index] = grouped[chosen].mean()
    estimate = grouped.mean()
    low, high = np.quantile(samples, [0.025, 0.975])
    return float(estimate), (float(low), float(high))


def sign_flip_pvalue(values, *, draws: int = 20000, seed: int = 0) -> float:
    values = np.asarray(values, dtype=float)
    observed = abs(values.mean())
    rng = np.random.default_rng(seed)
    null = np.abs((rng.choice((-1.0, 1.0), size=(draws, len(values))) * values).mean(axis=1))
    return float((1 + np.count_nonzero(null >= observed)) / (draws + 1))


def cluster_spearman(x, y, clusters, *, draws: int = 5000, seed: int = 0):
    """Spearman correlation with whole clusters as bootstrap/permutation units."""
    x, y, clusters = np.asarray(x, float), np.asarray(y, float), np.asarray(clusters)
    unique = np.unique(clusters)
    grouped = {key: (x[clusters == key].mean(), y[clusters == key].mean()) for key in unique}
    gx = np.asarray([grouped[key][0] for key in unique])
    gy = np.asarray([grouped[key][1] for key in unique])
    observed = float(spearmanr(gx, gy).statistic)
    rng = np.random.default_rng(seed)
    boot, null = [], []
    for _ in range(draws):
        indices = rng.integers(0, len(unique), len(unique))
        if np.unique(gx[indices]).size > 1 and np.unique(gy[indices]).size > 1:
            boot.append(float(spearmanr(gx[indices], gy[indices]).statistic))
        null.append(float(spearmanr(gx, gy[rng.permutation(len(gy))]).statistic))
    ci = tuple(float(value) for value in np.quantile(boot, [.025, .975]))
    pvalue = float((1 + np.count_nonzero(np.abs(null) >= abs(observed))) / (draws + 1))
    return observed, ci, pvalue
