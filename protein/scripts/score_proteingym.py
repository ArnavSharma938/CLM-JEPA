#!/usr/bin/env python
"""Score the locked ProteinGym panel with official RITA fitness and frozen surrogates."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.geometry import adjacent_tangent_correlation, local_curvature, path_efficiency, tube_radius
from src.modeling import load_rita
from src.nextlat import FaithfulNextLatPredictor
from src.stats import sign_flip_pvalue
from src.tokenization import collate_encoded, encode_canonical


def score_direction(model, tokenizer, sequences, predictor, collect_surrogates, batch_size):
    scores, surrogates = [], []
    for start in range(0, len(sequences), batch_size):
        chosen = sequences[start:start + batch_size]
        encoded = [encode_canonical(tokenizer, sequence) for sequence in chosen]
        batch = {key: value.cuda() for key, value in collate_encoded(encoded).items() if key != "residue_mask"}
        with torch.inference_mode(): outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        for row, sequence in enumerate(chosen):
            length = len(sequence)
            logits = outputs.logits[row, :length].float()
            targets = batch["input_ids"][row, 1:length + 1]
            scores.append(float(-F.cross_entropy(logits, targets)))
            if collect_surrogates:
                states = outputs.hidden_states[row, :length].float()
                current, future = states[:-1], states[1:]
                next_ids = batch["input_ids"][row, 1:length]
                prediction = predictor(current, model.get_input_embeddings()(next_ids).float())
                error = float(F.smooth_l1_loss(prediction, future))
                segment = states
                surrogates.append({"nextlat_error": error,
                    "curvature": float(local_curvature(segment).mean()),
                    "tangent_c1": float(adjacent_tangent_correlation(segment).mean()),
                    "tube_radius": float(tube_radius(segment).mean()),
                    "path_efficiency": float(path_efficiency(segment))})
    return scores, surrogates


def bootstrap_assays(values, draws=10000, seed=20260914):
    rng = np.random.default_rng(seed); values = np.asarray(values, float)
    samples = np.asarray([rng.choice(values, len(values), replace=True).mean() for _ in range(draws)])
    return float(values.mean()), [float(x) for x in np.quantile(samples, [.025, .975])]


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("panel", type=Path)
    parser.add_argument("data_dir", type=Path); parser.add_argument("predictor", type=Path)
    parser.add_argument("output", type=Path); parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    specification = json.loads(args.panel.read_text())
    loaded = load_rita(); model, tokenizer = loaded.model, loaded.tokenizer
    payload = torch.load(args.predictor, map_location="cuda", weights_only=True)
    predictor = FaithfulNextLatPredictor(payload["hidden_size"]).cuda().eval()
    predictor.load_state_dict(payload["state_dict"])
    records, assay_stats = [], []
    for assay in specification["assays"]:
        table = pd.read_csv(args.data_dir / assay["DMS_filename"])
        sequences = table.mutated_sequence.astype(str).tolist()
        forward, surrogate = score_direction(model, tokenizer, sequences, predictor, True, args.batch_size)
        reverse, _ = score_direction(model, tokenizer, [sequence[::-1] for sequence in sequences], predictor, False, args.batch_size)
        current = []
        for index, row in table.reset_index(drop=True).iterrows():
            current.append({"DMS_id": assay["DMS_id"], "mutant": row.mutant,
                "DMS_score": float(row.DMS_score), "rita_official_fitness": forward[index] + reverse[index],
                "rita_forward_fitness": forward[index], **surrogate[index]})
        records.extend(current)
        statistics = {"DMS_id": assay["DMS_id"], "category": assay["coarse_selection_type"], "variants": len(current)}
        for metric in ("rita_official_fitness", "rita_forward_fitness", "nextlat_error", "curvature",
                       "tangent_c1", "tube_radius", "path_efficiency"):
            statistics[f"spearman_{metric}"] = float(spearmanr(
                [row[metric] for row in current], [row["DMS_score"] for row in current]).statistic)
        # Coupling within an assay between each surrogate and native RITA behavior.
        statistics["spearman_nextlat_error_vs_rita_fitness"] = float(spearmanr(
            [row["nextlat_error"] for row in current], [row["rita_official_fitness"] for row in current]).statistic)
        assay_stats.append(statistics)
    macro = {}
    for key in assay_stats[0]:
        if not key.startswith("spearman_"): continue
        mean, ci = bootstrap_assays([row[key] for row in assay_stats])
        macro[key] = {"macro_mean": mean, "assay_bootstrap_ci95": ci,
                      "sign_flip_p": sign_flip_pvalue([row[key] for row in assay_stats], seed=20260914)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"assays": assay_stats, "macro": macro, "records": records,
        "fitness_definition": "sum of forward and reverse negative mean CE; first residue omitted; EOS included",
        "backbone_updated": False}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__": main()
