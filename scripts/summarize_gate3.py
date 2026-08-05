from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "artifacts" / "gate3" / "multi"
KS = [-1, 0, 1, 2, 3]


def rank_desc(values: dict[int, float]) -> dict[int, int]:
    ordered = sorted(values, key=lambda k: (-values[k], k))
    return {k: index + 1 for index, k in enumerate(ordered)}


def main() -> None:
    paths = sorted(RESULTS.glob("*.json"))
    datasets = {path.stem: json.loads(path.read_text()) for path in paths if path.stem != "summary"}
    chemistry = {name: data for name, data in datasets.items() if name != "nl_rx_synth_llama"}
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
    llama = datasets["nl_rx_synth_llama"]
    summary = {
        "chemistry_datasets": sorted(chemistry),
        "conditions": KS,
        "sample_size_per_dataset": 1024,
        "chemistry_aggregate": aggregate,
        "chemistry_selection_rule": "among k passing every chemistry dataset, retain the two lowest equal-weight mean ranks across correct-minus-matched margin and retrieval top-1",
        "selected_chemistry_k": selected,
        "selected_nl_rx_synth_k": llama["retained_k"][:2],
        "nl_rx_synth_pass_by_k": {k: llama["raw"][str(k)]["retains_pair_signal"] for k in KS},
    }
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (RESULTS / "chemistry_aggregate.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate[0]))
        writer.writeheader()
        writer.writerows(aggregate)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
