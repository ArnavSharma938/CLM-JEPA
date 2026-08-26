from __future__ import annotations

import math
import random
from collections import defaultdict

import torch
import torch.nn.functional as F
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from scipy.optimize import linear_sum_assignment

from chemfm import canonicalize


def canonical_set(smiles: str) -> str:
    parts = [canonicalize(part) for part in smiles.split(".")]
    return ".".join(sorted(parts)) if parts and all(parts) else ""


def _identity(smiles: str, task: str, *, stereo: bool = True) -> str:
    if task == "retro":
        return canonical_set(smiles)
    molecule = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(molecule, isomericSmiles=stereo) if molecule is not None else ""


def rank_augmented_candidates(
    augmentations: list[list[str]], task: str, n_best: int
) -> list[str]:
    """Reproduce ChemFM score.py reciprocal-rank aggregation across R-SMILES views."""
    scores: dict[str, float] = {}
    first_seen: dict[str, int] = {}
    encounter = 0
    for predictions in augmentations:
        ranked_unique: list[str] = []
        seen: set[str] = set()
        for prediction in predictions:
            identity = _identity(prediction, task)
            if not identity or identity in seen:
                continue
            seen.add(identity)
            ranked_unique.append(identity)
        for rank, identity in enumerate(ranked_unique):
            if identity not in first_seen:
                first_seen[identity] = encounter
                encounter += 1
            scores[identity] = scores.get(identity, 0.0) + 1.0 / (rank + 1.0)
    ranked = sorted(scores, key=lambda value: (-scores[value], first_seen[value]))[:n_best]
    return ranked + [""] * (n_best - len(ranked))


def score_candidates(
    rows: list[dict[str, str]], candidates: list[list[str]], task: str
) -> tuple[dict[str, float], list[dict]]:
    if len(rows) != len(candidates):
        raise ValueError("one candidate list is required per example")
    records = []
    for row, predictions in zip(rows, candidates):
        target = _identity(row["tgt"], task)
        normalized = [_identity(value, task) for value in predictions]
        valid = [value for value in normalized if value]
        records.append({
            "src": row["src"], "target": target, "candidates": normalized,
            "valid": [bool(value) for value in normalized],
            "exact": [bool(value) and value == target for value in normalized],
            "unique_valid": len(set(valid)),
        })
    n = max(1, len(records))
    metrics = {
        f"exact_top{k}": sum(any(r["exact"][:k]) for r in records) / n
        for k in (1, 3, 5, 10)
    }
    metrics.update({
        "valid_rate": sum(sum(r["valid"]) for r in records) / max(1, sum(len(r["valid"]) for r in records)),
        "duplicate_rate": 1.0 - sum(r["unique_valid"] for r in records) / max(1, sum(sum(r["valid"]) for r in records)),
        "mean_unique_valid": sum(r["unique_valid"] for r in records) / n,
    })
    if task == "forward":
        connectivity = []
        stereo_wrong = []
        for row, record in zip(rows, records):
            target = _identity(row["tgt"], task, stereo=False)
            top = _identity(record["candidates"][0], task, stereo=False) if record["candidates"] else ""
            connection = bool(top) and top == target
            connectivity.append(connection)
            stereo_wrong.append(connection and not record["exact"][0])
        metrics["stereo_exact_top1"] = metrics["exact_top1"]
        metrics["connectivity_correct_stereo_wrong"] = sum(stereo_wrong) / n
    if task == "metabolism":
        metrics.update(_metabolism_metrics(rows, records))
    return metrics, records


def _metabolism_metrics(rows: list[dict[str, str]], records: list[dict]) -> dict[str, float]:
    targets = defaultdict(set)
    outputs = defaultdict(list)
    for row, record in zip(rows, records):
        targets[canonicalize(row["src"])].add(canonicalize(row["tgt"]))
        outputs[canonicalize(row["src"])].extend(record["candidates"])
    result: dict[str, float] = {}
    parents = list(targets)
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    for k in (5, 10, 20):
        recovered = 0
        parent_any = parent_half = parent_all = 0
        total_output = valid_output = 0
        historical = 0
        for parent in parents:
            predicted = [value for value in outputs[parent][:k] if value]
            known = targets[parent]
            hit = known.intersection(predicted)
            recovered += len(hit)
            parent_any += bool(hit)
            parent_half += len(hit) >= (len(known) + 1) // 2
            parent_all += len(hit) == len(known)
            total_output += min(k, len(outputs[parent]))
            valid_output += len(predicted)
            known_fp = [generator.GetFingerprint(Chem.MolFromSmiles(x)) for x in known]
            for value in predicted:
                fp = generator.GetFingerprint(Chem.MolFromSmiles(value))
                historical += any(DataStructs.TanimotoSimilarity(fp, ref) == 1.0 for ref in known_fp)
        denom_known = max(1, sum(map(len, targets.values())))
        denom_parent = max(1, len(parents))
        result.update({
            f"parent_any_at{k}": parent_any / denom_parent,
            f"parent_half_at{k}": parent_half / denom_parent,
            f"parent_all_at{k}": parent_all / denom_parent,
            f"recall_at{k}": recovered / denom_known,
            f"lower_bound_precision_at{k}": recovered / max(1, total_output),
            f"historical_tanimoto1_at{k}": historical / max(1, total_output),
            f"average_output_size_at{k}": total_output / denom_parent,
            f"valid_prediction_rate_at{k}": valid_output / max(1, total_output),
        })
    return result


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


def identity_mappings(
    identities: list[str], token_lengths: dict[str, int], heavy_atoms: dict[str, int], seed: int
):
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
    # The nonzero squared singular values are the eigenvalues of the smaller
    # sample Gram matrix. Endpoint panels have D=2048 and N<=1280, so this is
    # exact up to floating-point roundoff and avoids the much larger SVD.
    energy = torch.linalg.eigvalsh(centered @ centered.T).clamp_min(0.0)
    probabilities = energy / energy.sum().clamp_min(torch.finfo(energy.dtype).eps)
    entropy = -(probabilities * probabilities.clamp_min(1e-30).log()).sum()
    return float(entropy.exp())


def pca_structure(values: torch.Tensor) -> dict[str, float | list[float]]:
    centered = values.float() - values.float().mean(dim=0, keepdim=True)
    energy = torch.linalg.eigvalsh(centered @ centered.T).clamp_min(0.0).flip(0)
    fractions = energy / energy.sum().clamp_min(torch.finfo(energy.dtype).eps)
    top = fractions[: min(8, len(fractions))]
    return {
        "top8_explained_variance_fraction": [float(value) for value in top],
        "top1_explained_variance_fraction": float(fractions[0]),
        "top2_cumulative_explained_variance_fraction": float(fractions[:2].sum()),
        "top8_cumulative_explained_variance_fraction": float(fractions[:8].sum()),
    }


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
    x_train, y_train = sources[train].float(), targets[train].float()
    x_test, y_test = sources[test].float(), targets[test].float()
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
    raw_sources, raw_targets = sources.float(), targets.float()
    if centered:
        raw_sources = raw_sources - raw_sources[train_mask].mean(dim=0, keepdim=True)
        raw_targets = raw_targets - raw_targets[train_mask].mean(dim=0, keepdim=True)
    normalized_sources = F.normalize(raw_sources, dim=-1)
    normalized_targets = F.normalize(raw_targets, dim=-1)
    identity_to_index = {identity: index for index, identity in enumerate(unique)}
    group_indices = torch.tensor(
        [identity_to_index[identity] for identity in identities],
        device=sources.device,
    )
    prototype_sums = torch.zeros(
        len(unique), normalized_targets.size(1),
        device=normalized_targets.device, dtype=normalized_targets.dtype,
    )
    prototype_sums.index_add_(0, group_indices, normalized_targets)
    prototypes = F.normalize(prototype_sums, dim=-1)
    scores = normalized_sources @ prototypes.T
    row_indices = torch.arange(len(identities), device=sources.device)
    correct = scores[row_indices, group_indices]
    random_map, matched_map, matched_cost = identity_mappings(
        identities, token_lengths, heavy_atoms, seed
    )
    random_indices = torch.tensor(
        [identity_to_index[random_map[identity]] for identity in identities],
        device=sources.device,
    )
    matched_indices = torch.tensor(
        [identity_to_index[matched_map[identity]] for identity in identities],
        device=sources.device,
    )
    random_scores = scores[row_indices, random_indices]
    matched_scores = scores[row_indices, matched_indices]
    candidate_indices = []
    for identity in identities:
        negatives = sorted(
            (other for other in unique if other != identity),
            key=lambda other: (
                abs(token_lengths[identity] - token_lengths[other])
                + abs(heavy_atoms[identity] - heavy_atoms[other]),
                other,
            ),
        )[:3]
        candidate_indices.append([
            identity_to_index[candidate] for candidate in [identity] + negatives
        ])
    candidate_tensor = torch.tensor(candidate_indices, device=sources.device)
    candidate_scores = scores.gather(1, candidate_tensor)
    ranks = (candidate_scores > candidate_scores[:, :1]).sum(dim=1) + 1
    reciprocal_ranks = 1.0 / ranks.float()
    hits = ranks.eq(1)
    candidate_counts = torch.tensor(
        [len(values) for values in candidate_indices], device=sources.device,
    )
    random_margin = float((correct - random_scores).mean())
    matched_margin = float((correct - matched_scores).mean())
    top1 = float(hits.float().mean())
    chance = float((1.0 / candidate_counts.float()).mean())
    result = {
        "centered_normalized_rescue": centered,
        "correct_cosine": float(correct.mean()),
        "random_cosine": float(random_scores.mean()),
        "matched_shuffle_cosine": float(matched_scores.mean()),
        "correct_minus_random": random_margin,
        "correct_minus_matched": matched_margin,
        "matched_assignment_cost": matched_cost,
        "target_variance": float(raw_targets.var(dim=0, unbiased=False).mean()),
        "target_effective_rank": effective_rank(raw_targets),
        "target_mean_direction_energy": float(
            raw_targets.mean(dim=0).square().sum()
            / raw_targets.square().sum(dim=1).mean().clamp_min(1e-30)
        ),
        "ridge_explained_variance": ridge_explained_variance(
            raw_sources, raw_targets, identities, heldout
        ),
        "retrieval_top1": top1,
        "retrieval_mrr": float(reciprocal_ranks.mean()),
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
