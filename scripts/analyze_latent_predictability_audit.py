#!/usr/bin/env python
"""Falsification-oriented analysis of the frozen latent/decoder audit.

All treatment contrasts are same-seed and reaction-paired.  The script keeps
development-beam analyses separate from the locked probe test split and never
reads the untouched confirmation outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.stats import rankdata


METRIC_DIRECTION = {
    "r2": 1, "cosine": 1, "centered_cosine": 1,
    "normalized_mse": -1, "mse": -1,
    "kl_true_predicted": -1, "js": -1,
    "gold_log_probability": 1, "gold_probability": 1,
    "gold_rank": -1, "gold_margin": 1, "top1_agreement": 1,
    "top5_overlap": 1, "top10_overlap": 1,
    "within_between_ratio": -1, "unit_normalized_within_variability": -1,
    "matched_view_cosine": 1, "centered_linear_cka": 1,
    "cross_view_identity_retrieval": 1,
}


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def checkpoint_parts(key: str) -> tuple[str, int]:
    seed = int(key.rsplit("_s", 1)[1])
    if key.startswith("native_"):
        return "native", seed
    if key.startswith("released_"):
        return "released", seed
    if key.startswith("paper_"):
        return "paper", seed
    raise ValueError(key)


def native_key(seed: int) -> str:
    return f"native_r8_s{seed}"


def bootstrap(values: Iterable[float], seed: int = 20260904, draws: int = 10000) -> dict:
    x = np.asarray(list(values), dtype=np.float64)
    x = x[np.isfinite(x)]
    if not len(x):
        return {"n": 0, "mean": None, "ci95": [None, None], "p_signflip": None}
    rng = np.random.default_rng(seed)
    if len(x) == 1:
        interval = [float(x[0]), float(x[0])]
        p_value = 1.0
    else:
        indices = rng.integers(0, len(x), size=(draws, len(x)))
        means = x[indices].mean(1)
        interval = np.quantile(means, [0.025, 0.975]).tolist()
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=(draws, len(x)))
        null = (signs * x).mean(1)
        p_value = float((1 + np.sum(np.abs(null) >= abs(x.mean()))) / (draws + 1))
    return {
        "n": int(len(x)), "mean": float(x.mean()),
        "ci95": [float(value) for value in interval], "p_signflip": p_value,
    }


def hierarchical_bootstrap(seed_values: dict[int, list[float]], draws: int = 10000) -> dict:
    clean = {seed: np.asarray(values, dtype=np.float64) for seed, values in seed_values.items() if values}
    if not clean:
        return {"seeds": 0, "reactions": 0, "mean_seed_effect": None, "ci95": [None, None]}
    effects = {seed: float(values.mean()) for seed, values in clean.items()}
    seeds = sorted(clean)
    rng = np.random.default_rng(20260904)
    sampled = []
    for _ in range(draws):
        chosen = rng.choice(seeds, size=len(seeds), replace=True)
        sampled.append(float(np.mean([
            rng.choice(clean[int(seed)], size=len(clean[int(seed)]), replace=True).mean()
            for seed in chosen
        ])))
    return {
        "seeds": len(seeds), "reactions": int(sum(len(value) for value in clean.values())),
        "seed_effects": effects, "mean_seed_effect": float(np.mean(list(effects.values()))),
        "ci95": [float(value) for value in np.quantile(sampled, [0.025, 0.975])],
        "min_seed_effect": float(min(effects.values())), "max_seed_effect": float(max(effects.values())),
    }


def spearman(x: Iterable[float], y: Iterable[float]) -> float | None:
    pairs = [(a, b) for a, b in zip(x, y) if np.isfinite(a) and np.isfinite(b)]
    if len(pairs) < 4:
        return None
    left, right = map(np.asarray, zip(*pairs))
    if np.ptp(left) == 0 or np.ptp(right) == 0:
        return None
    return float(np.corrcoef(rankdata(left), rankdata(right))[0, 1])


def holm(rows: list[dict], field: str = "p_signflip") -> None:
    valid = [(index, row[field]) for index, row in enumerate(rows) if row.get(field) is not None]
    valid.sort(key=lambda pair: pair[1])
    running = 0.0
    count = len(valid)
    for rank, (index, value) in enumerate(valid):
        adjusted = min(1.0, float(value) * (count - rank))
        running = max(running, adjusted)
        rows[index]["p_holm"] = running


def probe_analysis(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    absolute, index = [], {}
    for row in rows:
        cell = (row["layer"], row["segment"], row["horizon"], row["mode"])
        index[(row["checkpoint"], *cell)] = row
        formulation, seed = checkpoint_parts(row["checkpoint"])
        for probe in ("constant", "ridge", "residual_mlp"):
            for support, metrics in row["metrics"][probe].items():
                for metric in ("normalized_mse", "r2", "cosine", "centered_cosine"):
                    if metric in metrics:
                        absolute.append({
                            "checkpoint": row["checkpoint"], "formulation": formulation, "seed": seed,
                            "layer": row["layer"], "segment": row["segment"],
                            "horizon": row["horizon"], "mode": row["mode"],
                            "probe": probe, "support": support, "metric": metric,
                            "value": metrics[metric], "n_positions": metrics.get("n"),
                            "n_reactions": metrics.get("reactions"),
                        })
        absolute.append({
            "checkpoint": row["checkpoint"], "formulation": formulation, "seed": seed,
            "layer": row["layer"], "segment": row["segment"], "horizon": row["horizon"],
            "mode": row["mode"], "probe": "residual_mlp-minus-ridge", "support": "arbitrary",
            "metric": "r2", "value": row["metrics"]["nonlinear_r2_improvement"],
        })

    grouped: dict[tuple, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for (checkpoint, *cell), treatment in index.items():
        formulation, seed = checkpoint_parts(checkpoint)
        if formulation == "native" or (native_key(seed), *cell) not in index:
            continue
        control = index[(native_key(seed), *cell)]
        for probe in ("ridge", "residual_mlp"):
            shared_supports = set(treatment["reaction_metrics"][probe]) & set(control["reaction_metrics"][probe])
            for support in shared_supports:
                tr = treatment["reaction_metrics"][probe][support]
                co = control["reaction_metrics"][probe][support]
                for metric in ("normalized_mse", "r2", "cosine", "centered_cosine"):
                    deltas = [tr[key][metric] - co[key][metric] for key in sorted(set(tr) & set(co))]
                    grouped[(formulation, *cell, probe, support, metric)][seed].extend(deltas)
    contrasts = []
    for key, by_seed in sorted(grouped.items(), key=str):
        formulation, layer, segment, horizon, mode, probe, support, metric = key
        summary = hierarchical_bootstrap(by_seed)
        row = {
            "formulation": formulation, "layer": layer, "segment": segment,
            "horizon": horizon, "mode": mode, "probe": probe, "support": support,
            "metric": metric, "direction_adjusted_mean": (
                summary["mean_seed_effect"] * METRIC_DIRECTION[metric]
                if summary["mean_seed_effect"] is not None else None
            ), **summary,
        }
        contrasts.append(row)
    return absolute, contrasts


def decoder_analysis(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    absolute, index = [], {}
    for row in rows:
        cell = (row["layer"], row["segment"], row["horizon"], row["mode"], row["probe"])
        index[(row["checkpoint"], *cell)] = row
        formulation, seed = checkpoint_parts(row["checkpoint"])
        supports = row.get("supports") or {"arbitrary": {metric: row[metric] for metric in METRIC_DIRECTION if metric in row}}
        for support, metrics in supports.items():
            for metric, value in metrics.items():
                absolute.append({
                    "checkpoint": row["checkpoint"], "formulation": formulation, "seed": seed,
                    "layer": row["layer"], "segment": row["segment"], "horizon": row["horizon"],
                    "mode": row["mode"], "probe": row["probe"], "support": support,
                    "metric": metric, "value": value, "n_positions": row["n"],
                })
    grouped: dict[tuple, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for (checkpoint, *cell), treatment in index.items():
        formulation, seed = checkpoint_parts(checkpoint)
        if formulation == "native" or (native_key(seed), *cell) not in index:
            continue
        control = index[(native_key(seed), *cell)]
        tr_supports = treatment.get("reaction_metrics", {})
        co_supports = control.get("reaction_metrics", {})
        # Compatibility with early flat reaction-metric artifacts.
        if tr_supports and all("kl_true_predicted" in value for value in tr_supports.values()):
            tr_supports = {"arbitrary": tr_supports}
            co_supports = {"arbitrary": co_supports}
        for support in set(tr_supports) & set(co_supports):
            tr, co = tr_supports[support], co_supports[support]
            for metric in METRIC_DIRECTION:
                common = sorted(set(tr) & set(co))
                if not common or metric not in tr[common[0]]:
                    continue
                grouped[(formulation, *cell, support, metric)][seed].extend(
                    tr[key][metric] - co[key][metric] for key in common
                )
    contrasts = []
    for key, by_seed in sorted(grouped.items(), key=str):
        formulation, layer, segment, horizon, mode, probe, support, metric = key
        summary = hierarchical_bootstrap(by_seed)
        contrasts.append({
            "formulation": formulation, "layer": layer, "segment": segment,
            "horizon": horizon, "mode": mode, "probe": probe, "support": support,
            "metric": metric, "direction_adjusted_mean": summary["mean_seed_effect"] * METRIC_DIRECTION[metric],
            **summary,
        })
    return absolute, contrasts


def invariance_analysis(rows: list[dict]) -> list[dict]:
    index = {(row["checkpoint"], row["layer"], row["segment"], row["object"], row["split"]): row for row in rows}
    grouped: dict[tuple, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    metrics = [name for name in METRIC_DIRECTION if name in {key for row in rows for key in row}]
    for (checkpoint, layer, segment, obj, split), treatment in index.items():
        formulation, seed = checkpoint_parts(checkpoint)
        control = index.get((native_key(seed), layer, segment, obj, split))
        if formulation == "native" or control is None:
            continue
        for metric in metrics:
            grouped[(formulation, layer, segment, obj, split, metric)][seed].append(
                treatment[metric] - control[metric]
            )
    output = []
    for key, by_seed in sorted(grouped.items(), key=str):
        formulation, layer, segment, obj, split, metric = key
        effects = {seed: values[0] for seed, values in by_seed.items()}
        output.append({
            "formulation": formulation, "layer": layer, "segment": segment,
            "object": obj, "split": split, "metric": metric,
            "seed_effects": effects, "mean_seed_effect": float(np.mean(list(effects.values()))),
            "direction_adjusted_mean": float(np.mean(list(effects.values()))) * METRIC_DIRECTION[metric],
            "seeds": len(effects),
        })
    return output


def candidate_analysis(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    cells = defaultdict(dict)
    for row in rows:
        key = (row["checkpoint"], row["reaction_identity"], row["view"], row["horizon"], row["mode"], row["probe"])
        # Keep the explicit seed-1301 promoted candidate separate from ordinary highest-wrong.
        cells[key][row["role"]] = row
    metrics = ("latent_normalized_mse", "latent_r2", "decoder_js", "decoder_kl_true_predicted",
               "decoder_gold_log_probability", "decoder_gold_rank", "decoder_gold_margin", "decoder_top1_agreement")
    separations = []
    for key, roles in cells.items():
        if "gold" not in roles or "highest_wrong" not in roles:
            continue
        checkpoint, reaction, view, horizon, mode, probe = key
        formulation, seed = checkpoint_parts(checkpoint)
        for metric in metrics:
            separations.append({
                "checkpoint": checkpoint, "formulation": formulation, "seed": seed,
                "reaction_identity": reaction, "view": view, "horizon": horizon,
                "mode": mode, "probe": probe, "metric": metric,
                "wrong_minus_gold": roles["highest_wrong"][metric] - roles["gold"][metric],
            })
    summary = []
    grouped = defaultdict(list)
    for row in separations:
        grouped[(row["checkpoint"], row["horizon"], row["mode"], row["probe"], row["metric"])].append(row["wrong_minus_gold"])
    for key, values in sorted(grouped.items(), key=str):
        checkpoint, horizon, mode, probe, metric = key
        formulation, seed = checkpoint_parts(checkpoint)
        summary.append({"checkpoint": checkpoint, "formulation": formulation, "seed": seed,
                        "horizon": horizon, "mode": mode, "probe": probe, "metric": metric,
                        **bootstrap(values)})

    sep_index = {(row["checkpoint"], row["reaction_identity"], row["view"], row["horizon"], row["mode"], row["probe"], row["metric"]): row["wrong_minus_gold"] for row in separations}
    treatment = defaultdict(lambda: defaultdict(list))
    for key, value in sep_index.items():
        checkpoint, reaction, view, horizon, mode, probe, metric = key
        formulation, seed = checkpoint_parts(checkpoint)
        native = (native_key(seed), reaction, view, horizon, mode, probe, metric)
        if formulation != "native" and native in sep_index:
            treatment[(formulation, horizon, mode, probe, metric)][seed].append(value - sep_index[native])
    contrasts = []
    for key, by_seed in sorted(treatment.items(), key=str):
        formulation, horizon, mode, probe, metric = key
        contrasts.append({"formulation": formulation, "horizon": horizon, "mode": mode,
                          "probe": probe, "metric": metric, **hierarchical_bootstrap(by_seed)})
    return summary, contrasts


def development_coupling(directory: Path) -> tuple[list[dict], list[dict]]:
    rows = [row for path in sorted(directory.glob("*.jsonl")) for row in read_jsonl(path)] if directory.exists() else []
    correlations = []
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["checkpoint"], row["layer"], row["segment"], row["object"])].append(row)
    for key, values in sorted(grouped.items(), key=str):
        checkpoint, layer, segment, obj = key
        formulation, seed = checkpoint_parts(checkpoint)
        for geometry in ("within_between_ratio", "matched_view_cosine", "centered_linear_cka"):
            for outcome in ("candidate_jaccard", "aggregate_gold_score", "aggregate_correct",
                            "cross_view_aggregation_failure", "within_view_ranking_failure"):
                correlations.append({
                    "checkpoint": checkpoint, "formulation": formulation, "seed": seed,
                    "layer": layer, "segment": segment, "object": obj,
                    "geometry": geometry, "outcome": outcome, "n": len(values),
                    "spearman": spearman(
                        [float(row[geometry]) for row in values],
                        [float(row[outcome]) for row in values],
                    ),
                })
    natural = []
    native_rows = {(row["reaction_identity"], row["layer"], row["segment"], row["object"]): row
                   for row in rows if row["checkpoint"] == "native_r8_s1301"}
    for row in rows:
        if row["checkpoint"] != "released_r8_l0.02_s1301" or not row["cross_view_aggregation_failure"]:
            continue
        key = (row["reaction_identity"], row["layer"], row["segment"], row["object"])
        control = native_rows.get(key)
        if control and control["aggregate_correct"] and not row["aggregate_correct"]:
            natural.append({
                "reaction_identity": row["reaction_identity"], "layer": row["layer"],
                "segment": row["segment"], "object": row["object"],
                **{f"delta_{metric}": row[metric] - control[metric] for metric in
                   ("within_between_ratio", "matched_view_cosine", "centered_linear_cka", "candidate_jaccard", "aggregate_gold_score")},
            })
    return correlations, natural


def joint_analysis(probes: list[dict], decoder: list[dict], invariance_reactions: list[dict]) -> list[dict]:
    probe_rows = [row for row in probes if row["layer"] == "final_post_norm" and row["segment"] == "product"]
    inv = {(row["checkpoint"], row["split"], row["reaction_identity"]): row
           for row in invariance_reactions if row["layer"] == "final_post_norm" and row["segment"] == "product" and row["object"] == "atom"}
    decoder_index = {}
    for row in decoder:
        if row["layer"] != "final_post_norm" or row["segment"] != "product" or row["probe"] != "ridge":
            continue
        reactions = row.get("reaction_metrics", {})
        if reactions and "arbitrary" in reactions:
            reactions = reactions["arbitrary"]
        for reaction, metrics in reactions.items():
            decoder_index[(row["checkpoint"], row["horizon"], row["mode"], reaction)] = metrics
    output = []
    for row in probe_rows:
        checkpoint, horizon, mode = row["checkpoint"], row["horizon"], row["mode"]
        validation = row["validation_reaction_metrics"]["ridge"]
        val_pairs = [(metrics["r2"], inv[(checkpoint, "validation", reaction)]["within_between_ratio"])
                     for reaction, metrics in validation.items() if (checkpoint, "validation", reaction) in inv]
        if not val_pairs:
            continue
        r2_cut = float(np.quantile([x[0] for x in val_pairs], 2 / 3))
        inv_cut = float(np.quantile([x[1] for x in val_pairs], 1 / 3))
        test = row["reaction_metrics"]["ridge"]["arbitrary"]
        joined = []
        for reaction, metrics in test.items():
            inv_row = inv.get((checkpoint, "test", reaction))
            dec = decoder_index.get((checkpoint, horizon, mode, reaction))
            if inv_row is None or dec is None:
                continue
            joined.append((metrics["r2"], inv_row["within_between_ratio"], dec))
        if not joined:
            continue
        joint = [value for value in joined if value[0] >= r2_cut and value[1] <= inv_cut]
        rest = [value for value in joined if not (value[0] >= r2_cut and value[1] <= inv_cut)]
        formulation, seed = checkpoint_parts(checkpoint)
        output.append({
            "checkpoint": checkpoint, "formulation": formulation, "seed": seed,
            "horizon": horizon, "mode": mode, "validation_r2_top_tertile_cut": r2_cut,
            "validation_invariance_bottom_tertile_cut": inv_cut,
            "test_reactions": len(joined), "joint_reactions": len(joint),
            "joint_fraction": len(joint) / len(joined),
            "joint_decoder_js": float(np.mean([value[2]["js"] for value in joint])) if joint else None,
            "other_decoder_js": float(np.mean([value[2]["js"] for value in rest])) if rest else None,
            "joint_decoder_top1_agreement": float(np.mean([value[2]["top1_agreement"] for value in joint])) if joint else None,
            "other_decoder_top1_agreement": float(np.mean([value[2]["top1_agreement"] for value in rest])) if rest else None,
            "r2_decoder_js_spearman": spearman([x[0] for x in joined], [x[2]["js"] for x in joined]),
            "invariance_decoder_js_spearman": spearman([x[1] for x in joined], [x[2]["js"] for x in joined]),
        })
    return output


def svg_lines(path: Path, title: str, x_label: str, y_label: str, series: dict[str, list[tuple[float, float]]]) -> None:
    width, height, margin = 760, 480, 70
    points = [(x, y) for values in series.values() for x, y in values if np.isfinite(y)]
    if not points:
        return
    xs, ys = zip(*points)
    xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
    if xmin == xmax: xmax += 1
    if ymin == ymax: ymax += 1
    pad = 0.08 * (ymax - ymin); ymin -= pad; ymax += pad
    sx = lambda x: margin + (x - xmin) / (xmax - xmin) * (width - 2 * margin)
    sy = lambda y: height - margin - (y - ymin) / (ymax - ymin) * (height - 2 * margin)
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]
    body = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
            '<rect width="100%" height="100%" fill="white"/>',
            f'<text x="{width/2}" y="28" text-anchor="middle" font-size="18">{title}</text>',
            f'<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="black"/>',
            f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="black"/>',
            f'<text x="{width/2}" y="{height-18}" text-anchor="middle">{x_label}</text>',
            f'<text x="18" y="{height/2}" transform="rotate(-90 18 {height/2})" text-anchor="middle">{y_label}</text>']
    for tick in np.linspace(xmin, xmax, 5):
        body.append(f'<text x="{sx(tick):.1f}" y="{height-margin+20}" text-anchor="middle" font-size="11">{tick:g}</text>')
    for tick in np.linspace(ymin, ymax, 5):
        body.append(f'<text x="{margin-8}" y="{sy(tick)+4:.1f}" text-anchor="end" font-size="11">{tick:.3g}</text>')
    for index, (name, values) in enumerate(sorted(series.items())):
        color = colors[index % len(colors)]
        ordered = sorted((x, y) for x, y in values if np.isfinite(y))
        coords = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in ordered)
        body.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2"/>')
        body.append(f'<text x="{width-margin-120}" y="{margin+18*index}" fill="{color}" font-size="12">{name}</text>')
    body.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(body), encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("runs/latent_predictability_audit"))
    args = parser.parse_args()
    root = args.input.resolve(); raw = root / "raw"; tables = root / "tables"; plots = root / "plots"
    probes = read_jsonl(raw / "probe_metrics.jsonl")
    decoder = read_jsonl(raw / "decoder_metrics.jsonl")
    invariance = read_jsonl(raw / "invariance.jsonl")
    invariance_reactions = read_jsonl(raw / "invariance_reaction.jsonl")
    candidates = read_jsonl(raw / "candidate_predictability.jsonl")
    probe_absolute, probe_contrasts = probe_analysis(probes)
    decoder_absolute, decoder_contrasts = decoder_analysis(decoder)
    invariance_contrasts = invariance_analysis(invariance)
    candidate_summary, candidate_contrasts = candidate_analysis(candidates)
    generation_coupling, seed1301 = development_coupling(raw / "development_invariance")
    joint = joint_analysis(probes, decoder, invariance_reactions)
    for name, rows in {
        "probe_absolute.csv": probe_absolute, "probe_paired_contrasts.csv": probe_contrasts,
        "decoder_absolute.csv": decoder_absolute, "decoder_paired_contrasts.csv": decoder_contrasts,
        "invariance_absolute.csv": invariance, "invariance_paired_contrasts.csv": invariance_contrasts,
        "candidate_gold_wrong.csv": candidate_summary, "candidate_paired_contrasts.csv": candidate_contrasts,
        "generation_invariance_coupling.csv": generation_coupling,
        "seed1301_aggregation_invariance.csv": seed1301, "joint_predictability_invariance.csv": joint,
    }.items():
        write_csv(tables / name, rows)
    predict_series = defaultdict(list)
    for row in probe_absolute:
        if (row["layer"], row["segment"], row["mode"], row["probe"], row["support"], row["metric"]) == ("final_post_norm", "product", "current", "ridge", "arbitrary", "r2"):
            predict_series[row["formulation"]].append((row["horizon"], row["value"]))
    predict_series = {key: [(x, float(np.mean([y for xx, y in values if xx == x]))) for x in sorted(set(xx for xx, _ in values))] for key, values in predict_series.items()}
    svg_lines(plots / "product_predictable_fraction.svg", "Future latent predictability", "horizon", "test R²", predict_series)
    decoder_series = defaultdict(list)
    for row in decoder_absolute:
        if (row["layer"], row["segment"], row["mode"], row["probe"], row["support"], row["metric"]) == ("final_post_norm", "product", "current", "ridge", "arbitrary", "js"):
            decoder_series[row["formulation"]].append((row["horizon"], row["value"]))
    decoder_series = {key: [(x, float(np.mean([y for xx, y in values if xx == x]))) for x in sorted(set(xx for xx, _ in values))] for key, values in decoder_series.items()}
    svg_lines(plots / "product_decoder_js.svg", "Decoder error from predicted states", "horizon", "JS divergence", decoder_series)
    output = {
        "type": "latent_predictability_decoder_coupling_analysis",
        "confirmation_outcomes_consumed": False,
        "counts": {"probe_cells": len(probes), "decoder_cells": len(decoder), "candidate_rows": len(candidates),
                   "invariance_cells": len(invariance), "development_coupling_cells": len(generation_coupling)},
        "primary_definitions": {
            "predictable_fraction": "test R2 in original 2048-dimensional state space relative to the train-target mean",
            "functional_error": "KL(true-state decoder distribution || predicted-state decoder distribution) and JS",
            "invariance": "graph-aligned within-identity view variance / between-identity centroid variance",
            "treatment_effect": "same-seed, same-reaction STP minus Native; intervals use hierarchical seed/reaction bootstrap",
            "joint_subset": "test reactions crossing validation-locked top-R2 and bottom-invariance tertiles",
        },
        "probe_paired_contrasts": probe_contrasts,
        "decoder_paired_contrasts": decoder_contrasts,
        "invariance_paired_contrasts": invariance_contrasts,
        "candidate_paired_contrasts": candidate_contrasts,
        "generation_invariance_coupling": generation_coupling,
        "seed1301_aggregation_invariance": seed1301,
        "joint_predictability_invariance": joint,
    }
    write_json(root / "analysis.json", output)
    print(json.dumps({"stage": "analysis_complete", "counts": output["counts"]}, sort_keys=True))


if __name__ == "__main__":
    main()
