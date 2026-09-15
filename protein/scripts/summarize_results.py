#!/usr/bin/env python
"""Create compact protein-level estimates used by the diagnostic report."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.stats import cluster_bootstrap_mean, sign_flip_pvalue


def load(path): return json.loads(path.read_text())


def main():
    geometry = load(ROOT / "runs/uniref/geometry.json")["records"]
    probes = load(ROOT / "runs/uniref/nextlat_probes.json")
    output = {"native_geometry": {}, "probe_paired_differences": {}}
    metrics = [key for key in geometry[0] if key.startswith("tube_") or key.startswith("efficiency_")]
    metrics = ["curvature_mean", "tangent_c1"] + metrics
    for metric in metrics:
        mean, ci = cluster_bootstrap_mean([row[metric] for row in geometry],
            [row["cluster_id"] for row in geometry], seed=20260914)
        output["native_geometry"][metric] = {"mean": mean, "ci95": ci}
    conditions = probes["results"]
    pairs = (("C_oracle", "A_current_state"), ("C_oracle", "C_shuffle"),
             ("state_final_plus_layer12", "state_final_only"))
    for left, right in pairs:
        lrows = {row["id"]: row for row in conditions[left]["ridge"]["protein_records"]}
        rrows = {row["id"]: row for row in conditions[right]["ridge"]["protein_records"]}
        common = sorted(lrows.keys() & rrows.keys()); cell = {"paired_proteins": len(common)}
        for metric in ("error", "decoder_disagreement"):
            differences = [lrows[key][metric] - rrows[key][metric] for key in common]
            mean, ci = cluster_bootstrap_mean(differences, [lrows[key]["cluster_id"] for key in common], seed=20260914)
            cell[metric] = {"mean_difference": mean, "ci95": ci,
                            "sign_flip_p": sign_flip_pvalue(differences, seed=20260914)}
        output["probe_paired_differences"][f"{left}_minus_{right}"] = cell
    important = [
        ROOT / "data/diagnostic_pool.jsonl", ROOT / "data/tape_structural_manifest.jsonl",
        ROOT / "data/proteingym_panel.json", ROOT / "data/proteingym_panel/manifest.json",
        ROOT / "runs/uniref/nextlat_probes.json", ROOT / "runs/uniref/stp_gradient.json",
        ROOT / "runs/uniref/nextlat_gradient.json", ROOT / "runs/proteingym/scores.json",
        ROOT / "runs/generation/native_replay.json",
    ]
    output["artifact_sha256"] = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                  for path in important}
    target = ROOT / "runs/summary.json"; target.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__": main()
