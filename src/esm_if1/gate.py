from __future__ import annotations

import argparse
import json
from pathlib import Path


def coverage_gate(ft: dict[str, float], dtft: dict[str, float], lora: dict[str, float]) -> dict:
    # Convert NLL to a higher-is-better performance quantity.
    values = {
        "heldout_likelihood": (-ft["heldout_nll"], -dtft["heldout_nll"], -lora["heldout_nll"]),
        "stability_ranking": (
            ft["domain_spearman_mean"],
            dtft["domain_spearman_mean"],
            lora["domain_spearman_mean"],
        ),
    }
    outcomes = {}
    for name, (p_ft, p_dtft, p_lora) in values.items():
        total = p_ft - p_lora
        coverage = p_ft - p_dtft
        lora_gap = p_dtft - p_lora
        fraction = coverage / total if total != 0 else None
        outcomes[name] = {
            "G_total": total,
            "G_coverage": coverage,
            "G_LoRA": lora_gap,
            "coverage_fraction": fraction,
        }
    stop = all(
        outcome["G_total"] > 0
        and outcome["G_coverage"] > 0
        and outcome["coverage_fraction"] is not None
        and outcome["coverage_fraction"] > 0.5
        for outcome in outcomes.values()
    )
    return {
        "outcomes": outcomes,
        "stop_h1_h4": stop,
        "registered_statement": "target coverage is currently the dominant adaptation bottleneck" if stop else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ft", required=True, type=Path)
    parser.add_argument("--dtft", required=True, type=Path)
    parser.add_argument("--lora", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    load = lambda path: json.loads(path.read_text(encoding="utf-8"))
    result = coverage_gate(load(args.ft), load(args.dtft), load(args.lora))
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
