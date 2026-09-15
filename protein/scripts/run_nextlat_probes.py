#!/usr/bin/env python
"""Frozen A/B/C/C-shuffle and state-sufficiency probes for RITA-M."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.special import softmax
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.modeling import load_rita
from src.stats import cluster_bootstrap_mean, cluster_spearman


def iter_cache(cache_dir: Path):
    for path in sorted(cache_dir.glob("shard_*.pt")):
        yield from torch.load(path, map_location="cpu", weights_only=False)


def stable_positions(identifier: str, length: int, maximum: int, seed: int) -> np.ndarray:
    eligible = np.arange(0, length - 1)
    if eligible.size <= maximum:
        return eligible
    digest = hashlib.sha256(f"{seed}\0{identifier}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    return np.sort(rng.choice(eligible, maximum, replace=False))


def collect(cache_dir: Path, split_manifest: Path, maximum: int = 24, seed: int = 20260914):
    assignments = {
        row["id"]: (row["split"], row["cluster_id"])
        for row in map(json.loads, split_manifest.read_text().splitlines())
    }
    result = {split: [] for split in ("train", "validation", "test")}
    for item in iter_cache(cache_dir):
        split, cluster = assignments[item["id"]]
        positions = stable_positions(item["id"], len(item["sequence"]), maximum, seed)
        ids = item["input_ids"].numpy()
        for position in positions:
            result[split].append({
                "id": item["id"], "cluster": cluster, "position": int(position),
                "length": len(item["sequence"]), "current": item["final"][position].numpy(),
                "future": item["final"][position + 1].numpy(),
                "intermediate": item["intermediate"][position].numpy(),
                "next_id": int(ids[position + 1]),
                "decoder_gold": int(ids[position + 2]) if position + 2 < len(item["sequence"]) else -1,
            })
    return result


def arrays(rows: list[dict], embedding: np.ndarray):
    current = np.stack([row["current"] for row in rows]).astype(np.float32)
    future = np.stack([row["future"] for row in rows]).astype(np.float32)
    intermediate = np.stack([row["intermediate"] for row in rows]).astype(np.float32)
    next_ids = np.asarray([row["next_id"] for row in rows])
    return current, future, intermediate, embedding[next_ids]


def shuffled_embeddings(rows: list[dict], embeddings: np.ndarray, seed: int):
    """Cross-protein shuffle within length-64 and relative-position-decile bins."""
    rng = np.random.default_rng(seed)
    bins: dict[tuple[int, int], list[int]] = {}
    for i, row in enumerate(rows):
        key = (row["length"] // 64, min(9, int(10 * row["position"] / row["length"])))
        bins.setdefault(key, []).append(i)
    donor = np.full(len(rows), -1, dtype=int)
    for indices in bins.values():
        if len(indices) < 2:
            continue
        order = np.asarray(indices)
        for shift in range(1, len(order)):
            rolled = np.roll(order, shift)
            if all(rows[a]["id"] != rows[b]["id"] for a, b in zip(order, rolled)):
                donor[order] = rolled
                break
    unresolved = np.flatnonzero(donor < 0)
    all_indices = np.arange(len(rows))
    for index in unresolved:
        candidates = all_indices[[rows[j]["id"] != rows[index]["id"] for j in all_indices]]
        distance = np.asarray([
            abs(rows[j]["length"] - rows[index]["length"])
            + abs(rows[j]["position"] / rows[j]["length"] - rows[index]["position"] / rows[index]["length"]) * 64
            for j in candidates
        ])
        best = candidates[np.flatnonzero(distance == distance.min())]
        donor[index] = int(rng.choice(best))
    ids = np.asarray([rows[j]["next_id"] for j in donor])
    return embeddings[ids], donor


class PCATransform:
    def __init__(self, width: int, seed: int):
        self.width, self.seed = width, seed

    def fit(self, values: np.ndarray):
        self.mean = values.mean(0)
        self.scale = values.std(0)
        self.scale[self.scale < 1e-6] = 1.0
        normalized = (values - self.mean) / self.scale
        width = min(self.width, normalized.shape[0] - 1, normalized.shape[1])
        self.pca = PCA(width, svd_solver="randomized", random_state=self.seed).fit(normalized)
        return self

    def transform(self, values):
        return self.pca.transform((values - self.mean) / self.scale).astype(np.float32)

    def inverse(self, values):
        return self.pca.inverse_transform(values) * self.scale + self.mean


def centered_cosine(true, predicted, center):
    a, b = true - center, predicted - center
    return np.sum(a * b, -1) / np.maximum(1e-12, np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))


def metrics(true, predicted, train_mean, head, gold):
    residual = np.sum((true - predicted) ** 2)
    total = np.sum((true - train_mean) ** 2)
    true_logits, pred_logits = true @ head.T, predicted @ head.T
    true_p, pred_p = softmax(true_logits, -1), softmax(pred_logits, -1)
    midpoint = (true_p + pred_p) / 2
    js = .5 * np.sum(true_p * (np.log(true_p + 1e-30) - np.log(midpoint + 1e-30)), -1)
    js += .5 * np.sum(pred_p * (np.log(pred_p + 1e-30) - np.log(midpoint + 1e-30)), -1)
    valid = gold >= 0
    gold_logits = pred_logits[np.arange(len(gold))[valid], gold[valid]]
    ranks = 1 + np.sum(pred_logits[valid] > gold_logits[:, None], axis=1)
    masked = pred_logits[valid].copy()
    masked[np.arange(valid.sum()), gold[valid]] = -np.inf
    margins = gold_logits - masked.max(-1)
    return {
        "r2": float(1 - residual / total), "normalized_mse": float(residual / total),
        "centered_cosine": float(np.nanmean(centered_cosine(true, predicted, train_mean))),
        "decoder_js": float(js.mean()),
        "decoder_top1_agreement": float(np.mean(true_logits.argmax(-1) == pred_logits.argmax(-1))),
        "decoder_gold_probability": float(np.mean(softmax(pred_logits[valid], -1)[np.arange(valid.sum()), gold[valid]])),
        "decoder_gold_rank": float(np.mean(ranks)), "decoder_gold_margin": float(np.mean(margins)),
    }


def fit_ridge(train_x, train_y, test_x, alpha=1.0):
    model = Ridge(alpha=alpha, fit_intercept=True).fit(train_x, train_y)
    return model.predict(test_x).astype(np.float32), model


def protein_level(rows, true, predicted, head):
    by_id = {}
    true_logits, predicted_logits = true @ head.T, predicted @ head.T
    next_ids = np.asarray([row["next_id"] for row in rows])
    native_logp = np.log(softmax(np.stack([row["current"] for row in rows]) @ head.T, -1) + 1e-30)
    error = np.mean((true - predicted) ** 2, -1)
    decoder_disagreement = np.mean((softmax(true_logits, -1) - softmax(predicted_logits, -1)) ** 2, -1)
    for index, row in enumerate(rows):
        cell = by_id.setdefault(row["id"], {"cluster_id": row["cluster"], "error": [],
            "decoder_disagreement": [], "native_ntp_loss": [], "native_correct_probability": []})
        cell["error"].append(float(error[index]))
        cell["decoder_disagreement"].append(float(decoder_disagreement[index]))
        cell["native_ntp_loss"].append(float(-native_logp[index, next_ids[index]]))
        cell["native_correct_probability"].append(float(np.exp(native_logp[index, next_ids[index]])))
    records = [{"id": identifier, "cluster_id": cell["cluster_id"], **{
        key: float(np.mean(value)) for key, value in cell.items() if key != "cluster_id"
    }} for identifier, cell in by_id.items()]
    correlations = {}
    for outcome in ("native_ntp_loss", "native_correct_probability"):
        rho, ci, pvalue = cluster_spearman(
            [row["error"] for row in records], [row[outcome] for row in records],
            [row["cluster_id"] for row in records], seed=20260914,
        )
        correlations[f"transition_error_vs_{outcome}"] = {"spearman": rho, "ci95": ci, "permutation_p": pvalue}
    return records, correlations


class MLP(torch.nn.Module):
    def __init__(self, input_width, output_width, hidden_width=256):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_width, hidden_width), torch.nn.GELU(),
            torch.nn.Linear(hidden_width, output_width),
        )

    def forward(self, values):
        return self.network(values)


def fit_mlp(train_x, train_y, val_x, val_y, test_x, seed, device):
    torch.manual_seed(seed)
    model = MLP(train_x.shape[1], train_y.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    tensors = [torch.from_numpy(value) for value in (train_x, train_y, val_x, val_y)]
    tx, ty, vx, vy = [value.to(device) for value in tensors]
    generator = torch.Generator().manual_seed(seed)
    best, best_state, patience = float("inf"), None, 0
    for epoch in range(30):
        model.train()
        for indices in torch.randperm(len(tx), generator=generator).split(512):
            prediction = model(tx[indices])
            loss = torch.nn.functional.mse_loss(prediction, ty[indices])
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad():
            validation = float(torch.nn.functional.mse_loss(model(vx), vy))
        if validation < best - 1e-6:
            best, patience = validation, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            patience += 1
            if patience == 4:
                break
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        predicted = model(torch.from_numpy(test_x).to(device)).cpu().numpy()
    return predicted.astype(np.float32), {"epochs": epoch + 1, "best_validation_mse": best}


def run(cache_dir: Path, split_manifest: Path, output: Path, maximum: int):
    loaded = load_rita()
    embedding = loaded.model.get_input_embeddings().weight.detach().float().cpu().numpy()
    head = loaded.model.get_output_embeddings().weight.detach().float().cpu().numpy()
    rows = collect(cache_dir, split_manifest, maximum)
    values, shuffled = {}, {}
    for split in rows:
        current, future, intermediate, next_embedding = arrays(rows[split], embedding)
        shuffle_embedding, donors = shuffled_embeddings(rows[split], embedding, 20260914)
        values[split] = {"current": current, "future": future, "intermediate": intermediate,
                         "embedding": next_embedding, "shuffle_embedding": shuffle_embedding}
        shuffled[split] = donors
    target_transform = PCATransform(256, 20260914).fit(values["train"]["future"])
    target = {split: target_transform.transform(values[split]["future"]) for split in values}
    conditions = {
        "A_current_state": lambda v: v["current"],
        "B_realized_residue": lambda v: v["embedding"],
        "C_oracle": lambda v: np.concatenate((v["current"], v["embedding"]), -1),
        "C_shuffle": lambda v: np.concatenate((v["current"], v["shuffle_embedding"]), -1),
        "state_final_only": lambda v: v["current"],
        "state_final_plus_layer12": lambda v: np.concatenate((v["current"], v["intermediate"]), -1),
    }
    # C and C-shuffle deliberately share one fitted input transform.
    transforms, transformed = {}, {}
    for name, function in conditions.items():
        key = "C_shared" if name in ("C_oracle", "C_shuffle") else name
        if key not in transforms:
            fit_name = "C_oracle" if key == "C_shared" else name
            transforms[key] = PCATransform(256, 20260914).fit(conditions[fit_name](values["train"]))
        transformed[name] = {split: transforms[key].transform(function(values[split])) for split in values}
    results = {}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gold = np.asarray([row["decoder_gold"] for row in rows["test"]])
    for name in conditions:
        results[name] = {}
        ridge_scores, _ridge_model = fit_ridge(transformed[name]["train"], target["train"], transformed[name]["test"])
        ridge_full = target_transform.inverse(ridge_scores)
        results[name]["ridge"] = metrics(
            values["test"]["future"], ridge_full, values["train"]["future"].mean(0), head, gold
        )
        protein_records, correlations = protein_level(rows["test"], values["test"]["future"], ridge_full, head)
        results[name]["ridge"]["protein_records"] = protein_records
        results[name]["ridge"]["behavior_correlations"] = correlations
        mlp_scores, training = fit_mlp(
            transformed[name]["train"], target["train"], transformed[name]["validation"],
            target["validation"], transformed[name]["test"], 20260914, device,
        )
        mlp_full = target_transform.inverse(mlp_scores)
        results[name]["mlp"] = metrics(
            values["test"]["future"], mlp_full, values["train"]["future"].mean(0), head, gold
        )
        results[name]["mlp"]["training"] = training
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "results": results,
        "counts": {split: len(rows[split]) for split in rows},
        "proteins": {split: len({row['id'] for row in rows[split]}) for split in rows},
        "positions_per_protein_cap": maximum,
        "input_pca_width": 256, "target_pca_width": 256,
        "target_train_variance_covered": float(target_transform.pca.explained_variance_ratio_.sum()),
        "shuffle_self_protein_matches": {split: int(sum(
            rows[split][i]["id"] == rows[split][j]["id"] for i, j in enumerate(shuffled[split])
        )) for split in rows},
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cache_dir", type=Path)
    parser.add_argument("split_manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--positions-per-protein", type=int, default=24)
    args = parser.parse_args()
    run(args.cache_dir, args.split_manifest, args.output, args.positions_per_protein)


if __name__ == "__main__":
    main()
