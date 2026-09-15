#!/usr/bin/env python
"""Condition layer-12 residual information on the full 1024-D final state."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from run_full_target_and_residual_probes import paired_difference, protein_metrics, state_metrics
from run_nextlat_probes import PCATransform, arrays, collect
from src.modeling import load_rita
from src.residual_probe import fit_conditional_residual_ridge


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cache_dir", type=Path); parser.add_argument("split_manifest", type=Path)
    parser.add_argument("output", type=Path); parser.add_argument("--positions-per-protein", type=int, default=24)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--residualizer-alpha", type=float, default=1.0)
    args = parser.parse_args()

    loaded = load_rita()
    embedding = loaded.model.get_input_embeddings().weight.detach().float().cpu().numpy()
    head = loaded.model.get_output_embeddings().weight.detach().float().cpu().numpy()
    rows = collect(args.cache_dir, args.split_manifest, args.positions_per_protein)
    values = {}
    for split in rows:
        current, future, intermediate, _ = arrays(rows[split], embedding)
        values[split] = {"current": current, "future": future, "intermediate": intermediate}

    # Conditioning features retain every coordinate of final h_t. Scaling is
    # train-fitted only and does not reduce dimensionality.
    final_scaler = StandardScaler().fit(values["train"]["current"])
    # Float64 avoids the ill-conditioned FP32 normal-equation solve observed
    # for the highly collinear full hidden-state coordinates.
    final = {split: final_scaler.transform(values[split]["current"]).astype(np.float64) for split in values}
    layer_transform = PCATransform(256, 20260914).fit(values["train"]["intermediate"])
    layer = {split: layer_transform.transform(values[split]["intermediate"]).astype(np.float64) for split in values}
    fit = fit_conditional_residual_ridge(
        final["train"], layer["train"], values["train"]["future"],
        final["test"], layer["test"], alpha=args.alpha,
        residualizer_alpha=args.residualizer_alpha,
    )
    train_mean = values["train"]["future"].mean(0)
    base = state_metrics(values["test"]["future"], fit["base_test"], train_mean, head)
    combined = state_metrics(values["test"]["future"], fit["combined_test"], train_mean, head)
    base_protein = protein_metrics(rows["test"], values["test"]["future"], fit["base_test"], head)
    combined_protein = protein_metrics(rows["test"], values["test"]["future"], fit["combined_test"], head)
    result = {
        "conditioning": "all 1024 standardized coordinates of final h_t; no final-state PCA",
        "extra_representation": "layer 12 (zero-based layer 11), train PCA256, then ridge-residualized on full final h_t",
        "target": "full 1024-dimensional final h[t+1]",
        "alpha": args.alpha, "residualizer_alpha": args.residualizer_alpha,
        "counts": {split: len(rows[split]) for split in rows},
        "base_full_final": base, "plus_unique_layer12": combined,
        "incremental_full_state_r2": combined["r2"] - base["r2"],
        "residual_error_reduction": 1.0 - combined["normalized_mse"] / base["normalized_mse"],
        "decoder_js_improvement": base["decoder_js"] - combined["decoder_js"],
        "decoder_kl_improvement": base["decoder_kl_teacher_student"] - combined["decoder_kl_teacher_student"],
        "decoder_top1_change": combined["decoder_top1_agreement"] - base["decoder_top1_agreement"],
        "unique_layer_feature_rms_train": fit["unique_train_rms"],
        "unique_layer_feature_rms_test": fit["unique_test_rms"],
        "paired_plus_unique_minus_base": paired_difference(combined_protein, base_protein),
        "homology_separated": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
