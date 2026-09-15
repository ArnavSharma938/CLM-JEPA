#!/usr/bin/env python
"""Amended full-1024D NextLat probes and conditional layer-12 residual test."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.special import softmax
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from run_nextlat_probes import PCATransform, arrays, collect, shuffled_embeddings
from src.modeling import load_rita
from src.residual_probe import fit_conditional_residual_ridge
from src.stats import cluster_bootstrap_mean, sign_flip_pvalue


def state_metrics(true, predicted, train_mean, head):
    residual = np.sum((true - predicted) ** 2); total = np.sum((true - train_mean) ** 2)
    true_logits, predicted_logits = true @ head.T, predicted @ head.T
    true_log = np.log(softmax(true_logits, -1) + 1e-30)
    predicted_log = np.log(softmax(predicted_logits, -1) + 1e-30)
    true_p, predicted_p = np.exp(true_log), np.exp(predicted_log)
    midpoint = .5 * (true_p + predicted_p)
    js = .5 * np.sum(true_p * (true_log - np.log(midpoint + 1e-30)), -1)
    js += .5 * np.sum(predicted_p * (predicted_log - np.log(midpoint + 1e-30)), -1)
    kl = np.sum(true_p * (true_log - predicted_log), -1)
    a, b = true - train_mean, predicted - train_mean
    cosine = np.sum(a * b, -1) / np.maximum(1e-12, np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))
    return {"r2": float(1 - residual / total), "normalized_mse": float(residual / total),
        "centered_cosine": float(np.mean(cosine)), "decoder_js": float(np.mean(js)),
        "decoder_kl_teacher_student": float(np.mean(kl)),
        "decoder_top1_agreement": float(np.mean(true_logits.argmax(-1) == predicted_logits.argmax(-1)))}


def protein_metrics(rows, true, predicted, head):
    true_p = softmax(true @ head.T, -1); predicted_p = softmax(predicted @ head.T, -1)
    kl = np.sum(true_p * np.log((true_p + 1e-30) / (predicted_p + 1e-30)), -1)
    midpoint = .5 * (true_p + predicted_p)
    js = .5 * np.sum(true_p * np.log((true_p + 1e-30) / (midpoint + 1e-30)), -1)
    js += .5 * np.sum(predicted_p * np.log((predicted_p + 1e-30) / (midpoint + 1e-30)), -1)
    mse = np.mean((true - predicted) ** 2, -1)
    agreement = ((true @ head.T).argmax(-1) == (predicted @ head.T).argmax(-1)).astype(float)
    cells = {}
    for index, row in enumerate(rows):
        cell = cells.setdefault(row["id"], {"cluster_id": row["cluster"], "mse": [], "decoder_js": [],
                                                 "decoder_kl": [], "top1": []})
        cell["mse"].append(mse[index]); cell["decoder_js"].append(js[index])
        cell["decoder_kl"].append(kl[index]); cell["top1"].append(agreement[index])
    return {identifier: {"cluster_id": cell["cluster_id"],
        **{name: float(np.mean(value)) for name, value in cell.items() if name != "cluster_id"}}
        for identifier, cell in cells.items()}


def paired_difference(left, right):
    common = sorted(left.keys() & right.keys()); result = {"paired_proteins": len(common)}
    for metric in ("mse", "decoder_js", "decoder_kl", "top1"):
        differences = [left[key][metric] - right[key][metric] for key in common]
        clusters = [left[key]["cluster_id"] for key in common]
        mean, ci = cluster_bootstrap_mean(differences, clusters, seed=20260914)
        result[metric] = {"mean_difference": mean, "ci95": ci,
                          "sign_flip_p": sign_flip_pvalue(differences, seed=20260914)}
    return result


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("cache_dir", type=Path)
    parser.add_argument("split_manifest", type=Path); parser.add_argument("output", type=Path)
    parser.add_argument("--positions-per-protein", type=int, default=24); args = parser.parse_args()
    loaded = load_rita(); embedding = loaded.model.get_input_embeddings().weight.detach().float().cpu().numpy()
    head = loaded.model.get_output_embeddings().weight.detach().float().cpu().numpy()
    rows = collect(args.cache_dir, args.split_manifest, args.positions_per_protein)
    values = {}; donors = {}
    for split in rows:
        current, future, intermediate, next_embedding = arrays(rows[split], embedding)
        shuffled, donor = shuffled_embeddings(rows[split], embedding, 20260914)
        values[split] = {"current": current, "future": future, "intermediate": intermediate,
                         "embedding": next_embedding, "shuffle_embedding": shuffled}
        donors[split] = donor
    conditions = {
        "A_current_state": lambda value: value["current"],
        "B_realized_residue": lambda value: value["embedding"],
        "C_oracle": lambda value: np.concatenate((value["current"], value["embedding"]), -1),
        "C_shuffle": lambda value: np.concatenate((value["current"], value["shuffle_embedding"]), -1),
    }
    transforms = {
        "A_current_state": PCATransform(256, 20260914).fit(conditions["A_current_state"](values["train"])),
        "B_realized_residue": PCATransform(256, 20260914).fit(conditions["B_realized_residue"](values["train"])),
        "C_shared": PCATransform(256, 20260914).fit(conditions["C_oracle"](values["train"])),
    }
    transformed = {}
    for name, function in conditions.items():
        transform = transforms["C_shared"] if name in ("C_oracle", "C_shuffle") else transforms[name]
        transformed[name] = {split: transform.transform(function(values[split])) for split in values}
    train_mean = values["train"]["future"].mean(0)
    result, predictions, per_protein = {}, {}, {}
    for name in conditions:
        ridge = Ridge(alpha=1.0, fit_intercept=True).fit(transformed[name]["train"], values["train"]["future"])
        predictions[name] = ridge.predict(transformed[name]["test"]).astype(np.float32)
        result[name] = state_metrics(values["test"]["future"], predictions[name], train_mean, head)
        per_protein[name] = protein_metrics(rows["test"], values["test"]["future"], predictions[name], head)

    # Quantify how much the old PCA256 target representation itself discarded.
    target_pca = PCATransform(256, 20260914).fit(values["train"]["future"])
    reconstructed = target_pca.inverse(target_pca.transform(values["test"]["future"]))
    ceiling = state_metrics(values["test"]["future"], reconstructed, train_mean, head)
    ceiling["train_variance_covered"] = float(target_pca.pca.explained_variance_ratio_.sum())

    # Direct conditional-information test: residualize layer-12 features against
    # final-state features, then ask whether only the unique remainder predicts
    # the full-state future residual.
    layer_transform = PCATransform(256, 20260914).fit(values["train"]["intermediate"])
    layer_features = {split: layer_transform.transform(values[split]["intermediate"]) for split in values}
    conditional = fit_conditional_residual_ridge(
        transformed["A_current_state"]["train"], layer_features["train"], values["train"]["future"],
        transformed["A_current_state"]["test"], layer_features["test"], alpha=1.0,
    )
    base_metrics = state_metrics(values["test"]["future"], conditional["base_test"], train_mean, head)
    combined_metrics = state_metrics(values["test"]["future"], conditional["combined_test"], train_mean, head)
    true_residual = values["test"]["future"] - conditional["base_test"]
    correction_residual = true_residual - conditional["correction_test"]
    residual_r2 = 1.0 - float(np.sum(correction_residual ** 2) / np.sum(true_residual ** 2))
    base_protein = protein_metrics(rows["test"], values["test"]["future"], conditional["base_test"], head)
    combined_protein = protein_metrics(rows["test"], values["test"]["future"], conditional["combined_test"], head)
    state_sufficiency = {"base_final_only": base_metrics, "plus_unique_layer12_residual": combined_metrics,
        "incremental_full_state_r2": combined_metrics["r2"] - base_metrics["r2"],
        "residual_target_r2": residual_r2,
        "unique_layer_feature_rms_train": conditional["unique_train_rms"],
        "unique_layer_feature_rms_test": conditional["unique_test_rms"],
        "paired_plus_unique_minus_base": paired_difference(combined_protein, base_protein)}

    output = {"amendment": True, "original_results_preserved": True,
        "target": "full 1024-dimensional final h[t+1]", "input_pca_width": 256,
        "counts": {split: len(rows[split]) for split in rows},
        "proteins": {split: len({row['id'] for row in rows[split]}) for split in rows},
        "full_target_results": result,
        "paired_C_minus_A": paired_difference(per_protein["C_oracle"], per_protein["A_current_state"]),
        "paired_C_minus_C_shuffle": paired_difference(per_protein["C_oracle"], per_protein["C_shuffle"]),
        "old_target_pca256_reconstruction_ceiling": ceiling,
        "conditional_layer12": state_sufficiency,
        "shuffle_self_protein_matches": {split: int(sum(rows[split][i]["id"] == rows[split][j]["id"]
            for i, j in enumerate(donors[split]))) for split in rows},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__": main()
