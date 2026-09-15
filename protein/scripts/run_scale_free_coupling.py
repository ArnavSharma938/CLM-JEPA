#!/usr/bin/env python
"""Amended scale-free NextLat coupling on UniRef, ProteinGym, and generations."""
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
from src.modeling import load_rita
from src.nextlat import FaithfulNextLatPredictor, transition_diagnostics
from src.stats import cluster_spearman, sign_flip_pvalue
from src.tokenization import collate_encoded, encode_canonical

METRICS = (
    "raw_smooth_l1", "raw_mse", "normalized_state_mse",
    "transition_relative_mse", "centered_cosine_error", "decoder_js",
    "faithful_total",
)


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def iter_cache(cache_dir: Path):
    for path in sorted(cache_dir.glob("shard_*.pt")):
        yield from torch.load(path, map_location="cpu", weights_only=False)


def train_future_mean(cache_dir: Path, assignments: dict[str, str]):
    total = None; count = 0
    for item in iter_cache(cache_dir):
        if assignments[item["id"]] != "train":
            continue
        future = item["final"][1:].double()
        total = future.sum(0) if total is None else total + future.sum(0)
        count += len(future)
    if not count:
        raise ValueError("no train transitions found")
    return (total / count).float(), count


def mean_diagnostics(current, future, next_ids, predictor, embedding, head, center):
    prediction = predictor(current, embedding[next_ids].float())
    values = transition_diagnostics(current, future, prediction, center, head)
    return {name: float(values[name].mean().cpu()) for name in METRICS}


@torch.inference_mode()
def uniref_records(cache_dir, assignments, predictor, embedding, head, center, device):
    records = []
    for item in iter_cache(cache_dir):
        if assignments[item["id"]] != "test":
            continue
        states = item["final"].to(device=device, dtype=torch.float32)
        ids = item["input_ids"].to(device)
        current, future, next_ids = states[:-1], states[1:], ids[1:len(states)]
        metric = mean_diagnostics(current, future, next_ids, predictor, embedding, head, center)
        logits = F.linear(current, head.float()); chosen = F.log_softmax(logits, -1).gather(1, next_ids[:, None]).squeeze(1)
        records.append({"id": item["id"], "cluster_id": item["cluster_id"], **metric,
            "native_ntp_loss": float(-chosen.mean()),
            "native_correct_probability": float(chosen.exp().mean())})
    return records


def uniref_correlations(records):
    result = {}
    clusters = [row["cluster_id"] for row in records]
    for metric in METRICS:
        for outcome in ("native_ntp_loss", "native_correct_probability"):
            rho, ci, p = cluster_spearman([row[metric] for row in records],
                [row[outcome] for row in records], clusters, draws=2000, seed=20260914)
            result[f"{metric}_vs_{outcome}"] = {"spearman": rho, "ci95": ci, "permutation_p": p}
    return result


@torch.inference_mode()
def sequence_metrics(model, tokenizer, predictor, sequences, center, batch_size):
    embedding = model.get_input_embeddings().weight; head = model.get_output_embeddings().weight
    result = []
    for start in range(0, len(sequences), batch_size):
        chosen = sequences[start:start + batch_size]
        encoded = [encode_canonical(tokenizer, sequence) for sequence in chosen]
        batch = {key: value.to(head.device) for key, value in collate_encoded(encoded).items() if key != "residue_mask"}
        outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        for row, sequence in enumerate(chosen):
            length = len(sequence); states = outputs.hidden_states[row, :length].float()
            result.append(mean_diagnostics(states[:-1], states[1:],
                batch["input_ids"][row, 1:length], predictor, embedding, head, center))
    return result


def assay_bootstrap(values, draws=10000, seed=20260914):
    values = np.asarray(values, float); rng = np.random.default_rng(seed)
    samples = [rng.choice(values, len(values), replace=True).mean() for _ in range(draws)]
    return float(values.mean()), [float(x) for x in np.quantile(samples, [.025, .975])]


def proteingym_analysis(panel_path, data_dir, original_scores, model, tokenizer, predictor, center, batch_size):
    panel = json.loads(panel_path.read_text(encoding="utf-8"))
    original = json.loads(original_scores.read_text(encoding="utf-8"))
    lookup = {(row["DMS_id"], row["mutant"]): row for row in original["records"]}
    assay_stats = []; record_count = 0
    for assay in panel["assays"]:
        table = pd.read_csv(data_dir / assay["DMS_filename"]).reset_index(drop=True)
        sequences = table.mutated_sequence.astype(str).tolist()
        diagnostics = sequence_metrics(model, tokenizer, predictor, sequences, center, batch_size)
        records = []
        for index, row in table.iterrows():
            old = lookup[(assay["DMS_id"], row.mutant)]
            records.append({"DMS_score": float(row.DMS_score),
                "rita_official_fitness": float(old["rita_official_fitness"]), **diagnostics[index]})
        cell = {"DMS_id": assay["DMS_id"], "category": assay["coarse_selection_type"], "variants": len(records)}
        for metric in METRICS:
            cell[f"spearman_{metric}_vs_DMS"] = float(spearmanr(
                [row[metric] for row in records], [row["DMS_score"] for row in records]).statistic)
            cell[f"spearman_{metric}_vs_RITA_fitness"] = float(spearmanr(
                [row[metric] for row in records], [row["rita_official_fitness"] for row in records]).statistic)
        assay_stats.append(cell); record_count += len(records)
    macro = {}
    for key in assay_stats[0]:
        if not key.startswith("spearman_"):
            continue
        values = [row[key] for row in assay_stats]; mean, ci = assay_bootstrap(values)
        macro[key] = {"macro_mean": mean, "assay_bootstrap_ci95": ci,
                      "sign_flip_p": sign_flip_pvalue(values, seed=20260914)}
    return {"assays": assay_stats, "macro": macro, "variants": record_count,
            "statistical_unit": "assay (within-assay Spearman, macro-averaged)"}


def independent_group_difference(records, metric, seed=20260914):
    better = np.asarray([row[metric] for row in records if row["quality_group"] == "better"])
    worse = np.asarray([row[metric] for row in records if row["quality_group"] == "worse"])
    rng = np.random.default_rng(seed)
    boot = [rng.choice(better, len(better)).mean() - rng.choice(worse, len(worse)).mean() for _ in range(5000)]
    observed = float(better.mean() - worse.mean()); pooled = np.r_[better, worse]; null = []
    for _ in range(10000):
        shuffled = rng.permutation(pooled); null.append(shuffled[:len(better)].mean() - shuffled[len(better):].mean())
    return {"better_minus_worse": observed,
        "ci95": [float(x) for x in np.quantile(boot, [.025, .975])],
        "permutation_p": float((1 + np.count_nonzero(np.abs(null) >= abs(observed))) / 10001)}


def generation_analysis(replay_path, model, tokenizer, predictor, center, batch_size):
    replay = json.loads(replay_path.read_text(encoding="utf-8")); old = replay["records"]
    diagnostics = sequence_metrics(model, tokenizer, predictor, [row["sequence"] for row in old], center, batch_size)
    records = [{"id": row["id"], "quality_composite": row["quality_composite"],
                "quality_group": row["quality_group"], **diagnostics[index]} for index, row in enumerate(old)]
    return {"sequences": len(records), "quality_groups_reused_unchanged": True,
            "comparisons": {metric: independent_group_difference(records, metric) for metric in METRICS}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cache_dir", type=Path); parser.add_argument("split_manifest", type=Path)
    parser.add_argument("panel", type=Path); parser.add_argument("proteingym_data", type=Path)
    parser.add_argument("original_proteingym_scores", type=Path); parser.add_argument("generation_replay", type=Path)
    parser.add_argument("predictor", type=Path); parser.add_argument("output", type=Path)
    parser.add_argument("--batch-size", type=int, default=8); args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"; loaded = load_rita(device=device)
    model, tokenizer = loaded.model, loaded.tokenizer
    payload = torch.load(args.predictor, map_location=device, weights_only=True)
    predictor = FaithfulNextLatPredictor(payload["hidden_size"]).to(device).eval(); predictor.load_state_dict(payload["state_dict"])
    assignments = {row["id"]: row["split"] for row in read_jsonl(args.split_manifest)}
    center_cpu, center_count = train_future_mean(args.cache_dir, assignments); center = center_cpu.to(device)
    embedding = model.get_input_embeddings().weight; head = model.get_output_embeddings().weight
    uni = uniref_records(args.cache_dir, assignments, predictor, embedding, head, center, device)
    output = {"amendment": True, "predictor_objective": payload.get("objective"),
        "metric_orientation": "all seven metrics are errors: lower means more predictable",
        "normalization_reference": {"future_state_mean": "all transitions in homology-separated UniRef train proteins",
                                    "transitions": center_count},
        "uniref": {"proteins": len(uni), "statistical_unit": "homology cluster",
                   "correlations": uniref_correlations(uni)},
        "proteingym": proteingym_analysis(args.panel, args.proteingym_data, args.original_proteingym_scores,
                                           model, tokenizer, predictor, center, args.batch_size),
        "generation_replay": generation_analysis(args.generation_replay, model, tokenizer, predictor, center, args.batch_size),
        "backbone_updated": False, "generation_regenerated": False,
        "original_results_overwritten": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__": main()
