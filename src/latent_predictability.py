"""Frozen latent-predictability, decoder-coupling, and view-invariance tools.

This module deliberately contains no model training code.  The only learned
objects are small, frozen-data probes.  Splits are made at reaction level
before token positions or alternate serializations are expanded.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem, rdBase
from transformers.cache_utils import DynamicCache


SPLIT_SALT = "latent-decoder-audit-v1"
SPLIT_COUNTS = {"train": 640, "validation": 192, "test": 192}
HORIZONS = (1, 2, 4, 8)
LAYERS = ("layer_6", "layer_16", "layer_21", "final_post_norm")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def locked_reaction_split(
    reaction_ids: Sequence[str], counts: Mapping[str, int] = SPLIT_COUNTS,
) -> dict[str, str]:
    """Assign exactly the locked counts by a stable hash ordering."""
    unique = sorted(set(reaction_ids))
    if len(unique) != len(reaction_ids):
        raise ValueError("reaction identities must be unique before splitting")
    if sum(counts.values()) != len(unique):
        raise ValueError("split counts must consume every reaction")
    ordered = sorted(
        unique,
        key=lambda value: hashlib.sha256(
            f"{SPLIT_SALT}|{value}".encode("utf-8")
        ).hexdigest(),
    )
    result: dict[str, str] = {}
    start = 0
    for name in ("train", "validation", "test"):
        stop = start + int(counts[name])
        result.update({identity: name for identity in ordered[start:stop]})
        start = stop
    return result


def chemical_pair_id(source: str, target: str) -> str:
    """Namespace-stable identity for canonical directional chemistry."""
    return hashlib.sha256(f"{source}>>{target}".encode("utf-8")).hexdigest()


def read_confirmation_identifiers(path: Path) -> tuple[set[str], set[str]]:
    pairs, official = set(), set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            pair = row.get("chemical_pair_id")
            if pair is None and "canonical_source" in row and "canonical_target" in row:
                pair = chemical_pair_id(row["canonical_source"], row["canonical_target"])
            if pair is None:
                raise ValueError(f"missing chemical_pair_id/canonical chemistry in {path}")
            pairs.add(str(pair))
            if "reaction_identity" in row:
                official.add(str(row["reaction_identity"]))
    return pairs, official


def assert_disjoint_confirmation(
    audit_chemical_pair_ids: Iterable[str], confirmation_manifest: Path,
    audit_official_reaction_ids: Iterable[str] | None = None,
) -> dict[str, object]:
    confirmation_pairs, confirmation_official = read_confirmation_identifiers(confirmation_manifest)
    audit_pairs = set(audit_chemical_pair_ids)
    pair_overlap = sorted(audit_pairs & confirmation_pairs)
    official_overlap = sorted(set(audit_official_reaction_ids or ()) & confirmation_official)
    if pair_overlap or official_overlap:
        raise RuntimeError(
            "confirmation/audit leakage: "
            f"{len(pair_overlap)} chemical pairs and {len(official_overlap)} same-namespace official identities overlap"
        )
    return {
        "confirmation_manifest": str(confirmation_manifest.resolve()),
        "confirmation_sha256": sha256_file(confirmation_manifest),
        "confirmation_reactions": len(confirmation_pairs),
        "chemical_pair_overlap": 0,
        "official_identity_overlap": 0,
    }


def history_positions(
    segment_positions: Sequence[int], horizon: int, history: int = 3,
) -> list[tuple[list[int], int]]:
    """Return causal history indices and a future index within one segment."""
    if horizon not in HORIZONS:
        raise ValueError(f"unsupported horizon {horizon}")
    positions = list(map(int, segment_positions))
    return [
        (positions[offset - history:offset + 1], positions[offset + horizon])
        for offset in range(history, len(positions) - horizon)
    ]


def forecast_matrices(
    records: Sequence[dict], layer: str, segment: str, horizon: int,
    mode: str, history: int = 3,
) -> tuple[torch.Tensor, torch.Tensor, list[dict]]:
    """Expand cached reactions into causal probe matrices.

    The state cache uses ``states[layer, token, hidden]`` and never includes a
    future state in X.  Metadata is retained for reaction-clustered reductions.
    """
    if mode not in {"current", "history"}:
        raise ValueError("mode must be current or history")
    xs, ys, metadata = [], [], []
    for record in records:
        states = record["states"][layer].float()
        positions = record[f"{segment}_indices"]
        token_metadata = record.get("token_metadata", {})
        for past, future in history_positions(positions, horizon, history):
            x = states[past[-1]] if mode == "current" else states[past].reshape(-1)
            xs.append(x)
            ys.append(states[future])
            info = dict(token_metadata.get(str(future), {}))
            current_info = token_metadata.get(str(past[-1]), {})
            intervening = [
                token_metadata.get(str(index), {})
                for index in positions
                if past[-1] < index < future
            ]
            center_indices = [
                int(index) for index, value in token_metadata.items()
                if "reaction_center" in value.get("events", [])
            ]
            metadata.append({
                "reaction_identity": record["reaction_identity"],
                "segment": segment,
                "horizon": horizon,
                "current_index": past[-1],
                "future_index": future,
                "sequence_length": int(states.shape[0]),
                "gold_id": int(record["input_ids"][min(future + 1, len(record["input_ids"]) - 1)]),
                **info,
                "current_token_class": current_info.get("token_class"),
                "current_segment_rank": current_info.get("segment_rank"),
                "segment_length": len(positions),
                "current_events": current_info.get("events", []),
                "atom_to_atom": (
                    current_info.get("token_class") == "atom"
                    and info.get("token_class") == "atom"
                ),
                "event_to_next_event": (
                    bool(current_info.get("events")) and bool(info.get("events"))
                    and not any(value.get("events") for value in intervening)
                ),
                "component_boundary": current_info.get("component") != info.get("component"),
                "around_reaction_center": any(abs(future - index) <= 2 for index in center_indices),
            })
    if not xs:
        hidden = int(records[0]["states"][layer].shape[-1]) if records else 0
        width = hidden if mode == "current" else hidden * (history + 1)
        return torch.empty(0, width), torch.empty(0, hidden), []
    return torch.stack(xs), torch.stack(ys), metadata


def reaction_balanced_indices(
    metadata: Sequence[Mapping[str, object]], cap: int, seed: int,
) -> torch.Tensor:
    """Round-robin deterministic position sample so long reactions cannot dominate."""
    by_reaction: dict[str, list[int]] = {}
    for index, row in enumerate(metadata):
        by_reaction.setdefault(str(row["reaction_identity"]), []).append(index)
    rng = random.Random(seed)
    for values in by_reaction.values():
        rng.shuffle(values)
    keys = sorted(by_reaction)
    rng.shuffle(keys)
    selected = []
    depth = 0
    while len(selected) < min(cap, len(metadata)):
        added = False
        for key in keys:
            if depth < len(by_reaction[key]):
                selected.append(by_reaction[key][depth])
                added = True
                if len(selected) == cap:
                    break
        if not added:
            break
        depth += 1
    return torch.tensor(selected, dtype=torch.long)


def shuffled_reaction_targets(
    targets: torch.Tensor, metadata: Sequence[Mapping[str, object]], seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Derange future states across reactions while approximately matching position/length.

    Returns shuffled targets and donor indices. No row is allowed to receive a
    state from its own reaction. Ordering is deterministic for a fixed seed.
    """
    if len(targets) != len(metadata):
        raise ValueError("target/metadata length mismatch")
    if len({str(row["reaction_identity"]) for row in metadata}) < 2:
        raise ValueError("derangement needs at least two reactions")
    rng = random.Random(seed)
    by_reaction: dict[str, list[int]] = {}
    for index, row in enumerate(metadata):
        by_reaction.setdefault(str(row["reaction_identity"]), []).append(index)
    reactions = sorted(by_reaction)
    rng.shuffle(reactions)
    donor_reaction = {
        reaction: reactions[(index + 1) % len(reactions)]
        for index, reaction in enumerate(reactions)
    }
    donors = torch.empty(len(metadata), dtype=torch.long)
    for recipient, row in enumerate(metadata):
        candidates = by_reaction[donor_reaction[str(row["reaction_identity"])]]
        donors[recipient] = min(
            candidates,
            key=lambda donor: (
                metadata[donor].get("token_class") != row.get("token_class"),
                abs(int(metadata[donor].get("sequence_length", 0)) - int(row.get("sequence_length", 0))),
                abs(int(metadata[donor].get("future_index", 0)) - int(row.get("future_index", 0))),
                donor,
            ),
        )
    return targets.index_select(0, donors), donors


@dataclass
class TargetBasis:
    mean: torch.Tensor
    components: torch.Tensor
    variance_coverage: float

    @classmethod
    def fit(cls, targets: torch.Tensor, rank: int = 256, seed: int = 20260904) -> "TargetBasis":
        values = targets.float()
        mean = values.mean(0)
        centered = values - mean
        q = max(1, min(rank, centered.shape[0] - 1, centered.shape[1]))
        # PCA through the smaller Gram matrix is deterministic and considerably
        # cheaper than a full 2048x2048 decomposition for these probes.
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            _, singular, vectors = torch.pca_lowrank(centered, q=q, center=False, niter=4)
        total = centered.square().sum().clamp_min(1e-30)
        coverage = float(singular.square().sum() / total)
        return cls(mean.cpu(), vectors.T.contiguous().cpu(), coverage)

    def encode(self, targets: torch.Tensor) -> torch.Tensor:
        return (targets.float() - self.mean) @ self.components.T

    def decode(self, scores: torch.Tensor) -> torch.Tensor:
        return self.mean + scores.float() @ self.components


class RidgeProbe(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.linear(values)


class ResidualMLPProbe(nn.Module):
    """A frozen ridge prediction plus a small nonlinear residual branch."""

    def __init__(self, ridge: RidgeProbe, input_size: int, output_size: int, width: int = 128):
        super().__init__()
        self.ridge = copy.deepcopy(ridge).requires_grad_(False)
        self.residual = nn.Sequential(
            nn.Linear(input_size, width), nn.GELU(), nn.Linear(width, output_size),
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.ridge(values) + self.residual(values)


@dataclass
class Standardizer:
    mean: torch.Tensor
    scale: torch.Tensor

    @classmethod
    def fit(cls, values: torch.Tensor) -> "Standardizer":
        values = values.float()
        return cls(values.mean(0), values.std(0, unbiased=False).clamp_min(1e-5))

    def __call__(self, values: torch.Tensor) -> torch.Tensor:
        return (values.float() - self.mean) / self.scale


def fit_probe(
    model: nn.Module, train_x: torch.Tensor, train_y: torch.Tensor,
    validation_x: torch.Tensor, validation_y: torch.Tensor, *,
    weight_decay: float, epochs: int, batch_size: int, seed: int,
) -> tuple[nn.Module, dict[str, object]]:
    """Fit a probe with validation-only early stopping."""
    torch.manual_seed(seed)
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=weight_decay)
    generator = torch.Generator().manual_seed(seed)
    best_loss, best_state, stale = math.inf, None, 0
    trace = []
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(train_x), generator=generator)
        for start in range(0, len(order), batch_size):
            chosen = order[start:start + batch_size]
            x = train_x[chosen].to(device)
            y = train_y[chosen].to(device)
            loss = F.mse_loss(model(x).float(), y.float())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            values = []
            for start in range(0, len(validation_x), batch_size):
                prediction = model(validation_x[start:start + batch_size].to(device)).float().cpu()
                values.append(F.mse_loss(prediction, validation_y[start:start + batch_size].float(), reduction="sum"))
            score = float(sum(values) / max(1, validation_y.numel()))
        trace.append(score)
        if score < best_loss - 1e-8:
            best_loss = score
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= 3:
                break
    if best_state is None:
        raise RuntimeError("probe did not produce a finite validation result")
    model.load_state_dict(best_state)
    model.eval()
    return model, {"best_validation_mse": best_loss, "epochs": len(trace), "trace": trace}


def latent_metrics(
    target: torch.Tensor, prediction: torch.Tensor, training_mean: torch.Tensor,
    cluster_ids: Sequence[str] | None = None,
) -> dict[str, float]:
    target, prediction = target.float(), prediction.float()
    center = training_mean.float()
    residual = prediction - target
    sse = residual.square().sum()
    sst = (target - center).square().sum().clamp_min(1e-30)
    mse = residual.square().mean()
    centered_cos = F.cosine_similarity(prediction - center, target - center, dim=-1)
    result = {
        "mse": float(mse),
        "normalized_mse": float(sse / sst),
        "r2": float(1.0 - sse / sst),
        "cosine": float(F.cosine_similarity(prediction, target, dim=-1).mean()),
        "centered_cosine": float(torch.nanmean(centered_cos)),
        "n": int(len(target)),
    }
    if cluster_ids is not None:
        if len(cluster_ids) != len(target):
            raise ValueError("cluster identity count does not match positions")
        grouped = []
        for identity in sorted(set(cluster_ids)):
            chosen = torch.tensor([value == identity for value in cluster_ids])
            grouped.append(latent_metrics(target[chosen], prediction[chosen], center))
        for name in ("mse", "normalized_mse", "r2", "cosine", "centered_cosine"):
            result[f"reaction_mean_{name}"] = float(np.mean([row[name] for row in grouped]))
        result["reactions"] = len(grouped)
    return result


def decoder_distribution_metrics(
    true_logits: torch.Tensor, predicted_logits: torch.Tensor,
    gold_ids: torch.Tensor, topk: Sequence[int] = (5, 10),
) -> dict[str, torch.Tensor]:
    """Per-position functional metrics; KL direction is true || predicted."""
    true_logp = F.log_softmax(true_logits.float(), dim=-1)
    pred_logp = F.log_softmax(predicted_logits.float(), dim=-1)
    true_p, pred_p = true_logp.exp(), pred_logp.exp()
    midpoint = 0.5 * (true_p + pred_p)
    log_midpoint = midpoint.clamp_min(1e-30).log()
    kl = (true_p * (true_logp - pred_logp)).sum(-1)
    js = 0.5 * (
        (true_p * (true_logp - log_midpoint)).sum(-1)
        + (pred_p * (pred_logp - log_midpoint)).sum(-1)
    )
    row = torch.arange(len(gold_ids), device=gold_ids.device)
    gold_logits = predicted_logits.float()[row, gold_ids]
    masked = predicted_logits.float().clone()
    masked[row, gold_ids] = -torch.inf
    result = {
        "kl_true_predicted": kl,
        "js": js,
        "gold_log_probability": pred_logp[row, gold_ids],
        "gold_probability": pred_p[row, gold_ids],
        "gold_rank": (predicted_logits.float() > gold_logits[:, None]).sum(-1) + 1,
        "gold_margin": gold_logits - masked.max(-1).values,
        "top1_agreement": predicted_logits.argmax(-1).eq(true_logits.argmax(-1)),
    }
    for k in topk:
        k = min(int(k), true_logits.shape[-1])
        true_top = true_logits.topk(k, dim=-1).indices
        pred_top = predicted_logits.topk(k, dim=-1).indices
        intersection = (true_top[:, :, None] == pred_top[:, None, :]).any(-1).sum(-1)
        result[f"top{k}_overlap"] = intersection.float() / k
    return result


def invariance_metrics(values: torch.Tensor) -> dict[str, float]:
    """Measure view variation against identity-centroid variation.

    ``values`` has shape [identity, view, hidden].
    """
    values = values.float()
    centroids = values.mean(1)
    within = (values - centroids[:, None]).square().sum(-1).mean()
    grand = centroids.mean(0)
    between = (centroids - grand).square().sum(-1).mean().clamp_min(1e-30)
    normalized = F.normalize(values, dim=-1)
    cosine = torch.einsum("ivd,iwd->ivw", normalized, normalized)
    view_n = values.shape[1]
    off_diagonal = ~torch.eye(view_n, dtype=torch.bool, device=values.device)
    return {
        "within_variability": float(within),
        "between_variability": float(between),
        "within_between_ratio": float(within / between),
        "matched_view_cosine": float(cosine[:, off_diagonal].mean()),
        "identities": int(values.shape[0]),
        "views": int(view_n),
    }


def deterministic_random_smiles(smiles: str, seeds: Sequence[int]) -> list[str]:
    """Return canonical plus deterministic valid randomized serializations."""
    components = smiles.split(".")
    molecules = [Chem.MolFromSmiles(value) for value in components]
    if any(value is None for value in molecules):
        raise ValueError(f"invalid SMILES: {smiles}")
    canonical = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=True)
    output = [canonical]
    for seed in seeds:
        rdBase.SeedRandomNumberGenerator(int(seed))
        randomized = [
            Chem.MolToSmiles(mol, doRandom=True, isomericSmiles=True, canonical=False)
            for mol in molecules
        ]
        rng = random.Random(seed)
        rng.shuffle(randomized)
        value = ".".join(randomized)
        if Chem.MolToSmiles(Chem.MolFromSmiles(value), isomericSmiles=True) != canonical:
            raise RuntimeError("randomized SMILES changed chemical identity")
        output.append(value)
    return output


def canonical_atom_correspondence(canonical_smiles: str, view_smiles: str) -> dict[int, int]:
    """Map atom indices in a view molecule to canonical-molecule atom indices."""
    canonical = Chem.MolFromSmiles(canonical_smiles)
    view = Chem.MolFromSmiles(view_smiles)
    if canonical is None or view is None or canonical.GetNumAtoms() != view.GetNumAtoms():
        raise ValueError("view and canonical SMILES must encode the same atom count")
    match = view.GetSubstructMatch(canonical, useChirality=True)
    if not match:
        match = view.GetSubstructMatch(canonical, useChirality=False)
    if len(match) != canonical.GetNumAtoms():
        raise ValueError("could not establish full graph correspondence")
    return {int(view_atom): int(canonical_atom) for canonical_atom, view_atom in enumerate(match)}


def clone_cropped_cache(cache: DynamicCache, length: int) -> DynamicCache:
    result = DynamicCache()
    result.key_cache = [([] if isinstance(value, list) else value[..., :length, :].clone()) for value in cache.key_cache]
    result.value_cache = [([] if isinstance(value, list) else value[..., :length, :].clone()) for value in cache.value_cache]
    result._seen_tokens = length
    return result


@torch.inference_mode()
def suffix_replay_one_position(
    llama, layer_index: int, layer_input: torch.Tensor,
    predicted_state: torch.Tensor, position: int,
) -> torch.Tensor:
    """Run substituted states through the remaining decoder using cached prefix K/V.

    ``layer_input`` is [batch, sequence, hidden] at the input to
    ``llama.layers[layer_index]``.  ``predicted_state`` may contain multiple
    alternatives for the same batch/position.  This implementation is exact
    for a causal decoder and avoids repeating the unchanged transformer prefix.
    """
    if layer_input.shape[0] != 1:
        raise ValueError("baseline cache construction currently expects batch size one")
    sequence = layer_input.shape[1]
    device = layer_input.device
    causal = torch.full((1, 1, sequence, sequence), torch.finfo(layer_input.dtype).min, device=device, dtype=layer_input.dtype)
    causal = torch.triu(causal, diagonal=1)
    position_ids = torch.arange(sequence, device=device).unsqueeze(0)
    cache = DynamicCache()
    hidden = layer_input
    for block in llama.layers[layer_index:]:
        hidden = block(
            hidden, attention_mask=causal, position_ids=position_ids,
            past_key_value=cache, use_cache=True,
            cache_position=torch.arange(sequence, device=device),
        )[0]
    alternatives = predicted_state.reshape(-1, 1, predicted_state.shape[-1])
    outputs = []
    one_mask = torch.zeros((1, 1, 1, position + 1), device=device, dtype=layer_input.dtype)
    one_position = torch.tensor([[position]], device=device)
    cache_position = torch.tensor([position], device=device)
    for alternative in alternatives:
        local_cache = clone_cropped_cache(cache, position)
        hidden = alternative.unsqueeze(0) if alternative.ndim == 2 else alternative
        for block in llama.layers[layer_index:]:
            hidden = block(
                hidden, attention_mask=one_mask, position_ids=one_position,
                past_key_value=local_cache, use_cache=True,
                cache_position=cache_position,
            )[0]
        outputs.append(llama.norm(hidden)[:, 0])
    return torch.cat(outputs, dim=0)
