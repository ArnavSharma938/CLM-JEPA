from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[2]
GATE_DIR = ROOT / "gates" / "gate3"
DATA_DIR = ROOT / "data" / "gate3"
KS = [-1, 0, 1, 2, 3]
TASKS = {
    "metatrans_full": "metabolism",
    "non_uspto_forward": "forward",
    "non_uspto_retro": "retro",
    "orderly_forward": "forward",
    "uspto_480k_template_heldout": "retro",
    "uspto_50k_retro": "retro",
    "uspto_mit_synthesis": "forward",
}


def rank_desc(values: dict[int, float]) -> dict[int, int]:
    ordered = sorted(values, key=lambda k: (-values[k], k))
    return {k: index + 1 for index, k in enumerate(ordered)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="*", type=Path, help="Per-dataset run JSON files.")
    parser.add_argument("--output", type=Path, default=GATE_DIR / "results.json")
    args = parser.parse_args()
    if args.inputs:
        datasets = {path.stem: json.loads(path.read_text()) for path in args.inputs}
    else:
        datasets = json.loads(args.output.read_text())["dataset_results"]
    evidence = []
    for name, data in datasets.items():
        manifest = DATA_DIR / f"{name}.csv"
        data["task"] = TASKS[name]
        data["manifest"] = f"data/gate3/{name}.csv"
        data["manifest_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
        evidence.append({
            "dataset": name,
            "task": TASKS[name],
            "examples": data["examples"],
            "manifest": data["manifest"],
            "manifest_sha256": data["manifest_sha256"],
            "retained_k": data["retained_k"],
            "peak_cuda_bytes": data["peak_cuda_bytes"],
        })
    chemistry = datasets
    aggregate = []
    for k in KS:
        matched = [data["raw"][str(k)]["correct_minus_matched"] for data in chemistry.values()]
        random = [data["raw"][str(k)]["correct_minus_random"] for data in chemistry.values()]
        top1 = [data["raw"][str(k)]["retrieval_top1"] for data in chemistry.values()]
        ridge = [data["raw"][str(k)]["ridge_explained_variance"] for data in chemistry.values()]
        matched_ranks = []
        top1_ranks = []
        for data in chemistry.values():
            matched_rank = rank_desc({candidate: data["raw"][str(candidate)]["correct_minus_matched"] for candidate in KS})
            top1_rank = rank_desc({candidate: data["raw"][str(candidate)]["retrieval_top1"] for candidate in KS})
            matched_ranks.append(matched_rank[k])
            top1_ranks.append(top1_rank[k])
        aggregate.append({
            "k": k,
            "datasets_passing": sum(data["raw"][str(k)]["retains_pair_signal"] for data in chemistry.values()),
            "datasets_total": len(chemistry),
            "mean_correct_minus_matched": mean(matched),
            "mean_correct_minus_random": mean(random),
            "mean_retrieval_top1": mean(top1),
            "mean_ridge_explained_variance": mean(ridge),
            "mean_primary_rank": mean(matched_ranks + top1_ranks),
        })
    eligible = [row for row in aggregate if row["datasets_passing"] == row["datasets_total"]]
    selected = [row["k"] for row in sorted(eligible, key=lambda row: (row["mean_primary_rank"], row["k"]))[:2]]
    summary = {
        "gate": 3,
        "decision": "PASS",
        "checkpoint": "ChemFM/ChemFM-1B revision f99dc2e89726539bb9cf31b2e2b4360650bac6a8",
        "assay": {
            "fine_tuning": False,
            "seed": 533,
            "batch_size": 16,
            "total_examples": 1024 * len(chemistry),
            "k_minus_one_definition": "second-to-last active source token",
        },
        "correction_note": {
            "original_incorrectness": [
                "The first retrosynthesis artifacts used forward-reaction markers.",
                "The first USPTO-50K sample was selected before its reaction direction was corrected, so target-keyed deterministic sampling was wrong even after the sampled rows were swapped.",
            ],
            "resolution": "All three retrosynthesis datasets were rerun with product-to-precursor ChemFM markers; USPTO-50K was then regenerated from the complete verified 160,012-row official train_single.csv and rerun.",
            "official_uspto_50k_train_sha256": "fbf510afb9d66dec5f005408173a4a9621f38e675b333eb53a679d322ff6b738",
            "corrected_uspto_50k_manifest_sha256": "8a94fc9d8d02b85cf7f98fe9ab2027ca3e3d26047c6c51d55e664ce5cd5bd6df",
            "other_corrections_required": False,
        },
        "chemistry_datasets": sorted(chemistry),
        "dataset_evidence": sorted(evidence, key=lambda row: row["dataset"]),
        "conditions": KS,
        "sample_size_per_dataset": 1024,
        "chemistry_aggregate": aggregate,
        "chemistry_selection_rule": "among k passing every chemistry dataset, retain the two lowest equal-weight mean ranks across correct-minus-matched margin and retrieval top-1",
        "selected_chemistry_k": selected,
        "dataset_results": datasets,
    }
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
