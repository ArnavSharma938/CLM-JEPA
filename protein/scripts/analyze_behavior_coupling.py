#!/usr/bin/env python
"""Protein-level native behavior coupling and paired reversal analyses."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.special import softmax

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.modeling import load_rita
from src.stats import cluster_bootstrap_mean, cluster_spearman, sign_flip_pvalue


def iter_cache(cache_dir):
    for path in sorted(cache_dir.glob("shard_*.pt")):
        yield from torch.load(path, map_location="cpu", weights_only=False)


def read_geometry(path):
    return {row["id"]: row for row in json.loads(path.read_text())["records"]}


def native_records(cache_dir, geometry_path):
    geometry = read_geometry(geometry_path)
    head = load_rita().model.get_output_embeddings().weight.detach().float().cpu().numpy()
    records = []
    for item in iter_cache(cache_dir):
        states = item["final"].float().numpy()
        ids = item["input_ids"].numpy()
        # Biological next-residue decisions only; terminal EOS excluded.
        logits = states[:-1] @ head.T
        targets = ids[1:len(states)]
        logp = np.log(softmax(logits, -1) + 1e-30)
        chosen = logp[np.arange(len(targets)), targets]
        records.append({**geometry[item["id"]], "native_ntp_loss": float(-chosen.mean()),
                        "native_correct_probability": float(np.exp(chosen).mean())})
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("forward_cache", type=Path); parser.add_argument("forward_geometry", type=Path)
    parser.add_argument("reverse_geometry", type=Path); parser.add_argument("output", type=Path)
    parser.add_argument("--reverse-forward-geometry", type=Path,
                        help="matching forward subset when behavior geometry uses another pool")
    args = parser.parse_args()
    records = native_records(args.forward_cache, args.forward_geometry)
    correlations = {}
    geometry_metrics = ("curvature_mean", "tangent_c1", "tube_4_8", "tube_9_24", "tube_25_64",
                        "efficiency_4_8", "efficiency_9_24", "efficiency_25_64")
    for metric in geometry_metrics:
        for outcome in ("native_ntp_loss", "native_correct_probability"):
            rho, ci, pvalue = cluster_spearman([row[metric] for row in records],
                [row[outcome] for row in records], [row["cluster_id"] for row in records],
                draws=1000, seed=20260914)
            correlations[f"{metric}_vs_{outcome}"] = {"spearman": rho, "ci95": ci, "permutation_p": pvalue}
    forward = read_geometry(args.reverse_forward_geometry or args.forward_geometry)
    reverse_raw = read_geometry(args.reverse_geometry)
    reverse = {key.removeprefix("reverse__"): value for key, value in reverse_raw.items()}
    paired = {}
    common = sorted(forward.keys() & reverse.keys())
    for metric in geometry_metrics:
        differences = [forward[key][metric] - reverse[key][metric] for key in common]
        mean, ci = cluster_bootstrap_mean(differences, common, seed=20260914)
        paired[metric] = {"forward_minus_reverse": mean, "ci95": ci,
                          "sign_flip_p": sign_flip_pvalue(differences, seed=20260914)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"protein_records": records, "correlations": correlations,
        "reverse_paired_proteins": len(common), "forward_reverse": paired}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__": main()
