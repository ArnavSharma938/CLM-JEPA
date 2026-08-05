from __future__ import annotations

import math
import random
from collections import defaultdict

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def _minimum_cost_derangement(costs: list[list[int]]) -> tuple[list[int], int]:
    """Solve the exact minimum-cost derangement assignment in O(n^3)."""
    count = len(costs)
    if count < 2 or any(len(row) != count for row in costs):
        raise ValueError("matched derangement requires at least two identities")
    max_off_diagonal = max(
        costs[left][right]
        for left in range(count)
        for right in range(count)
        if left != right
    )
    forbidden = count * max_off_diagonal + 1
    matrix = [
        [forbidden if left == right else costs[left][right] for right in range(count)]
        for left in range(count)
    ]
    rows, columns = linear_sum_assignment(matrix)
    assignment = [0] * count
    for row, column in zip(rows.tolist(), columns.tolist()):
        assignment[row] = column
    if any(left == right for left, right in enumerate(assignment)):
        raise RuntimeError("minimum-cost assignment unexpectedly contains a fixed point")
    total = sum(costs[left][right] for left, right in enumerate(assignment))
    return assignment, total


def identity_mappings(identities: list[str], token_lengths: dict[str, int], heavy_atoms: dict[str, int], seed: int):
    unique = sorted(set(identities))
    rng = random.Random(seed)
    random_order = unique.copy()
    while True:
        rng.shuffle(random_order)
        if all(left != right for left, right in zip(unique, random_order)):
            break
    random_map = dict(zip(unique, random_order))

    costs = [
        [
            abs(token_lengths[left] - token_lengths[right])
            + abs(heavy_atoms[left] - heavy_atoms[right])
            for right in unique
        ]
        for left in unique
    ]
    assignment, matched_cost = _minimum_cost_derangement(costs)
    matched_map = {
        identity: unique[assignment[index]] for index, identity in enumerate(unique)
    }
    return random_map, matched_map, matched_cost


def effective_rank(values: torch.Tensor) -> float:
    centered = values - values.mean(dim=0, keepdim=True)
    singular_values = torch.linalg.svdvals(centered)
    energy = singular_values.square()
    probabilities = energy / energy.sum().clamp_min(torch.finfo(energy.dtype).eps)
    entropy = -(probabilities * probabilities.clamp_min(1e-30).log()).sum()
    return float(entropy.exp())


def ridge_explained_variance(
    sources: torch.Tensor,
    targets: torch.Tensor,
    identities: list[str],
    heldout_identities: set[str],
    alpha: float = 1.0,
) -> float:
    train = torch.tensor(
        [identity not in heldout_identities for identity in identities], device=sources.device
    )
    test = ~train
    x_train = sources[train].float()
    y_train = targets[train].float()
    x_test = sources[test].float()
    y_test = targets[test].float()
    x_mean = x_train.mean(dim=0, keepdim=True)
    y_mean = y_train.mean(dim=0, keepdim=True)
    x_train = x_train - x_mean
    kernel = x_train @ x_train.T
    coefficients = torch.linalg.solve(
        kernel + alpha * torch.eye(len(kernel), dtype=kernel.dtype, device=kernel.device),
        y_train - y_mean,
    )
    predictions = (x_test - x_mean) @ x_train.T @ coefficients + y_mean
    residual = (y_test - predictions).square().sum()
    baseline = (y_test - y_mean).square().sum().clamp_min(1e-30)
    return float(1.0 - residual / baseline)


def relationship_metrics(
    sources: torch.Tensor,
    targets: torch.Tensor,
    identities: list[str],
    token_lengths: dict[str, int],
    heavy_atoms: dict[str, int],
    seed: int = 533,
    centered: bool = False,
) -> dict:
    unique = sorted(set(identities))
    heldout_count = max(2, math.ceil(0.2 * len(unique)))
    heldout = set(unique[-heldout_count:])
    train_mask = torch.tensor(
        [identity not in heldout for identity in identities], device=sources.device
    )
    raw_sources = sources.float()
    raw_targets = targets.float()
    if centered:
        raw_sources = raw_sources - raw_sources[train_mask].mean(dim=0, keepdim=True)
        raw_targets = raw_targets - raw_targets[train_mask].mean(dim=0, keepdim=True)
    normalized_sources = F.normalize(raw_sources, dim=-1)
    normalized_targets = F.normalize(raw_targets, dim=-1)

    groups: dict[str, list[int]] = defaultdict(list)
    for index, identity in enumerate(identities):
        groups[identity].append(index)
    prototypes = {
        identity: F.normalize(normalized_targets[indices].mean(dim=0), dim=0)
        for identity, indices in groups.items()
    }
    correct = torch.stack([
        torch.dot(normalized_sources[index], prototypes[identity])
        for index, identity in enumerate(identities)
    ])
    random_map, matched_map, matched_cost = identity_mappings(
        identities, token_lengths, heavy_atoms, seed
    )
    random_scores = torch.stack([
        torch.dot(normalized_sources[index], prototypes[random_map[identity]])
        for index, identity in enumerate(identities)
    ])
    matched_scores = torch.stack([
        torch.dot(normalized_sources[index], prototypes[matched_map[identity]])
        for index, identity in enumerate(identities)
    ])

    reciprocal_ranks = []
    hits = []
    candidate_counts = []
    for index, identity in enumerate(identities):
        negatives = sorted(
            (other for other in unique if other != identity),
            key=lambda other: (
                abs(token_lengths[identity] - token_lengths[other])
                + abs(heavy_atoms[identity] - heavy_atoms[other]),
                other,
            ),
        )[:3]
        candidates = [identity] + negatives
        scores = torch.stack([
            torch.dot(normalized_sources[index], prototypes[candidate]) for candidate in candidates
        ])
        rank = int((scores > scores[0]).sum()) + 1
        reciprocal_ranks.append(1.0 / rank)
        hits.append(rank == 1)
        candidate_counts.append(len(candidates))

    target_variance = float(raw_targets.var(dim=0, unbiased=False).mean())
    mean_direction_energy = float(
        raw_targets.mean(dim=0).square().sum()
        / raw_targets.square().sum(dim=1).mean().clamp_min(1e-30)
    )
    ridge = ridge_explained_variance(raw_sources, raw_targets, identities, heldout)
    top1 = sum(hits) / len(hits)
    chance = sum(1.0 / count for count in candidate_counts) / len(candidate_counts)
    random_margin = float((correct - random_scores).mean())
    matched_margin = float((correct - matched_scores).mean())
    result = {
        "centered_normalized_rescue": centered,
        "correct_cosine": float(correct.mean()),
        "random_cosine": float(random_scores.mean()),
        "matched_shuffle_cosine": float(matched_scores.mean()),
        "correct_minus_random": random_margin,
        "correct_minus_matched": matched_margin,
        "matched_assignment_cost": matched_cost,
        "target_variance": target_variance,
        "target_effective_rank": effective_rank(raw_targets),
        "target_mean_direction_energy": mean_direction_energy,
        "ridge_explained_variance": ridge,
        "retrieval_top1": top1,
        "retrieval_mrr": sum(reciprocal_ranks) / len(reciprocal_ranks),
        "retrieval_chance_top1": chance,
        "heldout_identity_count": len(heldout),
    }
    result["retains_pair_signal"] = bool(
        random_margin > 0.0
        and matched_margin > 0.0
        and top1 > chance
        and result["target_effective_rank"] > 1.5
    )
    return result
