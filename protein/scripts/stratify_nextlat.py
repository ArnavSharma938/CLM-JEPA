#!/usr/bin/env python
"""Biological stratification of a warmed faithful frozen NextLat predictor."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.modeling import load_rita
from src.nextlat import FaithfulNextLatPredictor
from src.stats import cluster_bootstrap_mean, sign_flip_pvalue


def iter_cache(path):
    for shard in sorted(path.glob("shard_*.pt")):
        yield from torch.load(shard, map_location="cpu", weights_only=False)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("cache", type=Path)
    parser.add_argument("predictor", type=Path); parser.add_argument("output", type=Path); args = parser.parse_args()
    loaded = load_rita(); embedding = loaded.model.get_input_embeddings().weight.detach().float().cpu()
    payload = torch.load(args.predictor, map_location="cuda", weights_only=True)
    predictor = FaithfulNextLatPredictor(payload["hidden_size"]).cuda().eval(); predictor.load_state_dict(payload["state_dict"])
    records = []
    for item in iter_cache(args.cache):
        states = item["final"].float().cuda(); ids = item["input_ids"]
        with torch.inference_mode(): predicted = predictor(states[:-1], embedding[ids[1:len(states)]].cuda())
        error = F.smooth_l1_loss(predicted, states[1:], reduction="none").mean(-1).cpu().numpy()
        annotations = item.get("annotations", {}); buckets = {}
        if "ss3" in annotations:
            labels, valid = annotations["ss3"], annotations["structure_valid"]
            names = {0: "helix", 1: "beta", 2: "coil"}
            for position in range(len(error)):
                target = position + 1
                if valid[target] and labels[target] in names:
                    buckets.setdefault(names[labels[target]], []).append(error[position])
                    transition = labels[target - 1] != labels[target] or (target + 1 < len(labels) and labels[target] != labels[target + 1])
                    buckets.setdefault("ss_transition" if transition else "ss_stable", []).append(error[position])
        if "long_range_contact_density" in annotations:
            density = np.asarray([np.nan if x is None else x for x in annotations["long_range_contact_density"]])
            finite = density[np.isfinite(density)]
            if len(finite):
                threshold = np.median(finite)
                for position in range(len(error)):
                    target = position + 1
                    if np.isfinite(density[target]): buckets.setdefault(
                        "contact_high" if density[target] > threshold else "contact_low", []).append(error[position])
        for name, values in buckets.items():
            if len(values) >= 3: records.append({"id": item["id"], "cluster_id": item["cluster_id"],
                "stratum": name, "positions": len(values), "smooth_l1": float(np.mean(values))})
    summaries = {}
    for name in sorted({row["stratum"] for row in records}):
        chosen = [row for row in records if row["stratum"] == name]
        mean, ci = cluster_bootstrap_mean([row["smooth_l1"] for row in chosen], [row["cluster_id"] for row in chosen], seed=20260914)
        summaries[name] = {"proteins": len(chosen), "mean": mean, "ci95": ci}
    by = {(row["id"], row["stratum"]): row for row in records}; comparisons = {}
    for left, right in (("helix", "coil"), ("beta", "coil"), ("ss_transition", "ss_stable"), ("contact_high", "contact_low")):
        common = sorted({row["id"] for row in records if row["stratum"] == left} &
                        {row["id"] for row in records if row["stratum"] == right})
        differences = [by[(key, left)]["smooth_l1"] - by[(key, right)]["smooth_l1"] for key in common]
        mean, ci = cluster_bootstrap_mean(differences, common, seed=20260914)
        comparisons[f"{left}_minus_{right}"] = {"paired_proteins": len(common), "mean_difference": mean,
            "ci95": ci, "sign_flip_p": sign_flip_pvalue(differences, seed=20260914)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"summaries": summaries, "paired_comparisons": comparisons,
        "records": records, "predictor_only_warmup": True, "backbone_updated": False}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__": main()
