#!/usr/bin/env python
"""Reduce Geodesic Mechanism Audit raw streams into compact tables and plots."""

from __future__ import annotations

import argparse
import gzip
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "runs" / "geodesic_mechanism_audit"
RNG_SEED = 20260902


def records(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


def checkpoint_fields(key: str) -> dict:
    formulation = "native" if key.startswith("native") else "released" if key.startswith("released") else "paper"
    rank = 128 if "_r128_" in key else 8
    seed = int(key.rsplit("_s", 1)[1])
    value = 0.0
    if "_l" in key:
        value = float(key.split("_l", 1)[1].split("_s", 1)[0])
    return {"formulation": formulation, "rank": rank, "seed": seed, "lambda": value}


def treatment_pairs(keys):
    keys = set(keys)
    for key in sorted(keys):
        fields = checkpoint_fields(key)
        if fields["formulation"] == "native":
            continue
        native = f"native_r{fields['rank']}_s{fields['seed']}"
        if native in keys:
            yield native, key


def bootstrap_ci(values, replicates=10000):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return [math.nan, math.nan]
    rng = np.random.default_rng(RNG_SEED)
    samples = rng.choice(values, size=(replicates, len(values)), replace=True).mean(1)
    return np.quantile(samples, [.025, .975]).tolist()


def grouped_stream(path: Path, keys: tuple[str, ...], metrics: tuple[str, ...]):
    groups = defaultdict(lambda: {"n": 0, **{metric: 0.0 for metric in metrics}, **{f"sq_{metric}": 0.0 for metric in metrics}})
    for row in records(path):
        group = groups[tuple(row[key] for key in keys)]
        group["n"] += 1
        for metric in metrics:
            value = float(row[metric])
            if math.isfinite(value):
                group[metric] += value
                group[f"sq_{metric}"] += value * value
    output = []
    for key, group in groups.items():
        row = dict(zip(keys, key))
        row["n"] = group["n"]
        for metric in metrics:
            mean = group[metric] / max(1, group["n"])
            variance = max(0.0, group[f"sq_{metric}"] / max(1, group["n"]) - mean * mean)
            row[metric] = mean
            row[f"sd_{metric}"] = math.sqrt(variance)
        output.append(row)
    return output


def summarize_interventions(path: Path):
    groups = defaultdict(list)
    metrics = ("delta_gold_log_probability", "delta_gold_margin", "delta_gold_rank", "delta_entropy")
    for row in records(path):
        for effect in row["interventions"]:
            key = (row["checkpoint"], row["segment"], effect["gamma"], effect["norm_restored"])
            groups[key].append([float(effect[metric]) for metric in metrics])
    output = []
    for key, values in groups.items():
        data = np.asarray(values)
        row = dict(zip(("checkpoint", "segment", "gamma", "norm_restored"), key))
        row["n"] = len(data)
        for column, metric in enumerate(metrics):
            row[metric] = float(data[:, column].mean())
            row[f"sd_{metric}"] = float(data[:, column].std(ddof=1))
        row["fraction_logprob_harmed"] = float((data[:, 0] < 0).mean())
        row["fraction_margin_harmed"] = float((data[:, 1] < 0).mean())
        output.append(row)
    return output


def paired_tube(root: Path):
    frame = pd.DataFrame(records(root / "raw" / "tube_scale_space_aggregate.jsonl.gz"))
    keys = frame.checkpoint.unique().tolist()
    rows = []
    indexed = frame.set_index(["checkpoint", "layer", "segment", "span_length"])
    for native, treatment in treatment_pairs(keys):
        common = frame[frame.checkpoint == treatment][["layer", "segment", "span_length"]]
        for item in common.itertuples(index=False):
            index_n = (native, item.layer, item.segment, item.span_length)
            index_t = (treatment, item.layer, item.segment, item.span_length)
            if index_n not in indexed.index or index_t not in indexed.index:
                continue
            n, t = indexed.loc[index_n], indexed.loc[index_t]
            rows.append({
                "native": native, "treatment": treatment,
                **checkpoint_fields(treatment), "layer": item.layer,
                "segment": item.segment, "span_length": int(item.span_length),
                **{f"delta_{metric}": float(t[metric] - n[metric]) for metric in ("rms", "maximum", "p95", "monotonicity_violation")},
            })
    return frame, pd.DataFrame(rows)


def summarize_candidates(root: Path):
    values = {}
    natural = []
    for row in records(root / "raw" / "gold_wrong_candidate_geometry.jsonl.gz"):
        if row["layer"] != "final_post_norm" or row["role"] not in {"gold", "highest_wrong", "seed1301_promoted_wrong"}:
            continue
        tube_mean = float(np.mean([item["rms"] for item in row["tube_scale"]])) if row["tube_scale"] else math.nan
        metrics = {
            "tube_rms_integral": tube_mean,
            "paper_loss": row["paper_loss"], "released_loss": row["released_loss"],
            "euclidean_inefficiency": 1 - row["euclidean_path_efficiency"],
            "fisher_inefficiency": 1 - row["fisher_path_efficiency"],
            "fisher_local_curvature": row["fisher_local_curvature"],
            "normal_acceleration": row["normalized_normal_acceleration"],
        }
        index = (row["checkpoint"], row["panel_index"], row["view"], row["role"])
        values.setdefault(index, metrics)
    paired = []
    for (checkpoint, panel, view, role), wrong in values.items():
        if role != "highest_wrong":
            continue
        gold = values.get((checkpoint, panel, view, "gold"))
        if gold is None:
            continue
        paired.append({
            "checkpoint": checkpoint, "panel_index": panel, "view": view,
            **checkpoint_fields(checkpoint),
            **{f"wrong_minus_gold_{metric}": wrong[metric] - gold[metric] for metric in wrong},
        })
    # The promoted wrong candidate is evaluated under both seed-1301 models.
    for (checkpoint, panel, view, role), wrong in values.items():
        if role != "seed1301_promoted_wrong":
            continue
        gold = values.get((checkpoint, panel, view, "gold"))
        if gold:
            natural.append({
                "checkpoint": checkpoint, "panel_index": panel, "view": view,
                **{f"wrong_minus_gold_{metric}": wrong[metric] - gold[metric] for metric in wrong},
            })
    return pd.DataFrame(paired), pd.DataFrame(natural)


def make_plots(root: Path, tube: pd.DataFrame, tube_delta: pd.DataFrame, summaries: dict):
    import matplotlib.pyplot as plt

    plot_dir = root / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    selected = tube[(tube.layer == "final_post_norm") & tube.checkpoint.isin([
        "native_r8_s533", "released_r8_l0.02_s533", "paper_r8_l0.02_s533",
    ])]
    labels = {"native_r8_s533": "Native", "released_r8_l0.02_s533": "Released STP", "paper_r8_l0.02_s533": "Paper STP"}
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for axis, segment in zip(axes, ("source", "product", "cross")):
        for checkpoint, group in selected[selected.segment == segment].groupby("checkpoint"):
            axis.plot(group.span_length, group.rms, label=labels[checkpoint])
        axis.set_title(segment)
        axis.set_xlabel("outer span length L")
        axis.grid(alpha=.25)
    axes[0].set_ylabel("RMS normalized tube radius")
    axes[-1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(plot_dir / "tube_scale_space_final.png", dpi=180)
    plt.close(fig)

    final_delta = tube_delta[(tube_delta.layer == "final_post_norm") & (tube_delta.segment == "product")]
    fig, axis = plt.subplots(figsize=(7.5, 4.2))
    for treatment, group in final_delta.groupby("treatment"):
        if "l0.02" not in treatment or "r8" not in treatment:
            continue
        axis.plot(group.span_length, group.delta_rms, alpha=.7, label=treatment)
    axis.axhline(0, color="black", linewidth=.8)
    axis.set(xlabel="outer span length L", ylabel="STP - Native RMS tube radius")
    axis.grid(alpha=.25); axis.legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(plot_dir / "tube_treatment_delta.png", dpi=180); plt.close(fig)

    candidate = pd.DataFrame(summaries["gold_wrong"])
    if not candidate.empty:
        metric = "wrong_minus_gold_fisher_inefficiency"
        means = candidate.groupby("checkpoint")[metric].mean().sort_values()
        fig, axis = plt.subplots(figsize=(9, 4.5))
        means.plot.bar(ax=axis)
        axis.axhline(0, color="black", linewidth=.8)
        axis.set_ylabel("wrong - gold Fisher inefficiency")
        fig.tight_layout(); fig.savefig(plot_dir / "gold_wrong_fisher_separation.png", dpi=180); plt.close(fig)


def run(args):
    root = args.root.resolve()
    tube, tube_delta = paired_tube(root)
    intervention = summarize_interventions(root / "raw" / "signal_noise_interventions.jsonl.gz")
    anatomy = grouped_stream(
        root / "raw" / "released_objective_anatomy.jsonl.gz",
        ("checkpoint", "span_length"),
        ("loss", "cos_patch_before", "cos_patch_after", "cos_before_after", "cancellation_ratio"),
    )
    matched = grouped_stream(
        root / "raw" / "matched_native_stp_displacement.jsonl.gz",
        ("native", "treatment", "layer", "segment", "span_length"),
        ("delta_rho", "correction_cosine", "endpoint_middle_ratio", "chord_cosine", "chord_norm_ratio"),
    )
    intrinsic = grouped_stream(
        root / "raw" / "intrinsic_manifold_decomposition.jsonl.gz",
        ("checkpoint", "layer", "search_metric", "same_segment", "neighbors", "tangent_dim"),
        ("geodesic_violation", "normal_acceleration", "geodesic_over_acceleration", "normal_over_acceleration"),
    )
    cones = grouped_stream(
        root / "raw" / "inference_cones.jsonl.gz",
        ("checkpoint", "kind", "horizon"),
        ("weighted_angle_from_mean", "weighted_angle_from_gold", "perpendicular_variance", "mean_axis_gold_cosine", "fisher_dispersion"),
    )
    gold_wrong, natural = summarize_candidates(root)
    summaries = {
        "tube_treatment_effects": tube_delta.to_dict("records"),
        "interventions": intervention, "released_anatomy": anatomy,
        "matched_displacement": matched, "intrinsic": intrinsic, "cones": cones,
        "gold_wrong": gold_wrong.to_dict("records"),
        "seed1301_natural_experiment": natural.to_dict("records"),
    }
    (root / "analysis").mkdir(parents=True, exist_ok=True)
    for name, value in summaries.items():
        (root / "analysis" / f"{name}.json").write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    compact = {
        "tube_delta_final_product": tube_delta[(tube_delta.layer == "final_post_norm") & (tube_delta.segment == "product")].to_dict("records"),
        "interventions": intervention, "intrinsic": intrinsic, "cones": cones,
        "gold_wrong_means": gold_wrong.groupby("checkpoint").mean(numeric_only=True).reset_index().to_dict("records") if not gold_wrong.empty else [],
        "seed1301_natural_means": natural.groupby("checkpoint").mean(numeric_only=True).reset_index().to_dict("records") if not natural.empty else [],
    }
    (root / "analysis.json").write_text(json.dumps(compact, indent=2) + "\n", encoding="utf-8")
    make_plots(root, tube, tube_delta, summaries)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    run(parser.parse_args())
