#!/usr/bin/env python
"""Reduce Geodesic Mechanism Audit raw streams into compact tables and plots."""

from __future__ import annotations

import argparse
import gzip
import json
import math
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
DEFAULT_ROOT = ROOT / "runs" / "geodesic_mechanism_audit"
RNG_SEED = 20260902

from src.chemfm import TOKENIZER_DIR, load_reaction_tokenizer
from src.frozen_geometry import DEFAULT_PANEL
from scripts.run_geodesic_audit import build_gold_examples


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
    groups = defaultdict(lambda: {
        "n": 0, **{metric: 0.0 for metric in metrics},
        **{f"sq_{metric}": 0.0 for metric in metrics},
        **{f"n_{metric}": 0 for metric in metrics},
    })
    for row in records(path):
        group = groups[tuple(row[key] for key in keys)]
        group["n"] += 1
        for metric in metrics:
            value = float(row[metric])
            if math.isfinite(value):
                group[metric] += value
                group[f"sq_{metric}"] += value * value
                group[f"n_{metric}"] += 1
    output = []
    for key, group in groups.items():
        row = dict(zip(keys, key))
        row["n"] = group["n"]
        for metric in metrics:
            metric_n = group[f"n_{metric}"]
            mean = group[metric] / max(1, metric_n)
            variance = max(0.0, group[f"sq_{metric}"] / max(1, metric_n) - mean * mean)
            row[metric] = mean
            row[f"sd_{metric}"] = math.sqrt(variance)
            row[f"n_{metric}"] = metric_n
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


def summarize_multiscale_turning(path: Path):
    groups = defaultdict(lambda: {"n": 0, "sum": 0.0, "sq": 0.0})
    for row in records(path):
        for scale in row["scales"]:
            values = np.asarray(scale["angles"], dtype=float)
            values = values[np.isfinite(values)]
            group = groups[(row["checkpoint"], row["layer"], row["segment"], int(scale["scale"]))]
            group["n"] += len(values); group["sum"] += float(values.sum())
            group["sq"] += float(values @ values)
    output = []
    for key, group in groups.items():
        mean = group["sum"] / max(1, group["n"])
        output.append({
            **dict(zip(("checkpoint", "layer", "segment", "scale"), key)),
            "n": group["n"], "mean_angle": mean,
            "sd_mean_angle": math.sqrt(max(0.0, group["sq"] / max(1, group["n"]) - mean * mean)),
        })
    return output


def sensitivity_event_map():
    tokenizer = load_reaction_tokenizer(TOKENIZER_DIR)
    result = {}
    for example in build_gold_examples(tokenizer, DEFAULT_PANEL):
        for segment in ("source", "target"):
            tokens = [token for token in example.tokens if token.segment == segment]
            for local, token in enumerate(tokens, start=1):
                result[(example.panel_index, "product" if segment == "target" else segment, local)] = sorted(token.events)
    return result


def summarize_sensitivity(path: Path):
    groups = defaultdict(lambda: defaultdict(float))
    event_groups = defaultdict(lambda: defaultdict(float))
    correlation_groups = defaultdict(lambda: defaultdict(float))
    event_map = sensitivity_event_map()

    def add(group, parallel, perpendicular, perp_cos, rho, ray, fisher):
        group["n"] += 1
        group["parallel_signed"] += parallel
        group["perpendicular_signed"] += perpendicular
        group["parallel_absolute"] += abs(parallel)
        group["perpendicular_absolute"] += abs(perpendicular)
        group["perpendicular_cosine_absolute"] += abs(perp_cos)
        group["perpendicular_cosine_gt_.1"] += abs(perp_cos) > .1
        group["perpendicular_cosine_gt_.25"] += abs(perp_cos) > .25
        group["tube_radius"] += rho
        group["ray_residual"] += ray
        group["fisher_triangle_excess"] += fisher

    for row in records(path):
        length = int(row["span_length"])
        span_bin = "2" if length == 2 else "3_8" if length <= 8 else "9_24" if length <= 24 else "25_plus"
        group = groups[(row["checkpoint"], row["segment"], span_bin)]
        parallel = float(row["parallel_signed_sensitivity"])
        perpendicular = float(row["perpendicular_signed_sensitivity"])
        perp_cos = float(row["perpendicular_cosine_sensitivity"])
        rho = float(row["rho"]); ray = float(row["ray_residual"])
        fisher = float(row["fisher_triangle_excess"])
        add(group, parallel, perpendicular, perp_cos, rho, ray, fisher)
        correlation = correlation_groups[(row["checkpoint"], row["segment"])]
        correlation["n"] += 1; correlation["x"] += rho; correlation["y"] += fisher
        correlation["xx"] += rho * rho; correlation["yy"] += fisher * fisher
        correlation["xy"] += rho * fisher
        events = event_map.get((int(row["panel_index"]), row["segment"], int(row["r"])), [])
        labels = events or ["ordinary"]
        if events:
            labels = [*events, "any_event"]
        for label in labels:
            add(event_groups[(row["checkpoint"], row["segment"], label)], parallel, perpendicular, perp_cos, rho, ray, fisher)

    def finish(source, names):
        output = []
        for key, group in source.items():
            n = group.pop("n")
            row = dict(zip(names, key)); row["n"] = int(n)
            row.update({name: value / n for name, value in group.items()})
            row["absolute_perpendicular_to_parallel_ratio"] = row["perpendicular_absolute"] / max(row["parallel_absolute"], 1e-12)
            output.append(row)
        return output

    correlations = []
    for (checkpoint, segment), group in correlation_groups.items():
        n = group["n"]
        covariance = group["xy"] / n - (group["x"] / n) * (group["y"] / n)
        variance_x = max(0.0, group["xx"] / n - (group["x"] / n) ** 2)
        variance_y = max(0.0, group["yy"] / n - (group["y"] / n) ** 2)
        correlations.append({
            "checkpoint": checkpoint, "segment": segment, "n": int(n),
            "pearson_tube_fisher": covariance / max(math.sqrt(variance_x * variance_y), 1e-12),
        })
    return (
        finish(groups, ("checkpoint", "segment", "span_bin")),
        finish(event_groups, ("checkpoint", "segment", "event_category")),
        correlations,
    )


def intrinsic_treatment_summary(rows):
    frame = pd.DataFrame(rows)
    if frame.empty:
        return []
    keys = frame.checkpoint.unique().tolist()
    index_columns = ["layer", "segment", "search_metric", "same_segment", "neighbors", "tangent_dim"]
    output = []
    for native, treatment in treatment_pairs(keys):
        n = frame[frame.checkpoint == native].set_index(index_columns)
        t = frame[frame.checkpoint == treatment].set_index(index_columns)
        common = n.index.intersection(t.index)
        for layer in sorted({index[0] for index in common}):
            for segment in sorted({index[1] for index in common if index[0] == layer}):
                layer_index = [index for index in common if index[0] == layer and index[1] == segment]
                row = {
                    "native": native, "treatment": treatment, "layer": layer,
                    "segment": segment, "robustness_settings": len(layer_index),
                }
                for metric in ("geodesic_violation", "normal_acceleration", "geodesic_over_acceleration", "normal_over_acceleration"):
                    delta = t.loc[layer_index, metric].to_numpy() - n.loc[layer_index, metric].to_numpy()
                    row[f"delta_{metric}_mean"] = float(delta.mean())
                    row[f"delta_{metric}_min"] = float(delta.min())
                    row[f"delta_{metric}_max"] = float(delta.max())
                    row[f"delta_{metric}_fraction_negative"] = float((delta < 0).mean())
                output.append(row)
    return output


def paired_reduced_summary(rows, index_columns, metrics):
    frame = pd.DataFrame(rows)
    if frame.empty:
        return []
    output = []
    for native, treatment in treatment_pairs(frame.checkpoint.unique()):
        n = frame[frame.checkpoint == native].set_index(index_columns)
        t = frame[frame.checkpoint == treatment].set_index(index_columns)
        for index in n.index.intersection(t.index):
            nrow, trow = n.loc[index], t.loc[index]
            if not isinstance(index, tuple):
                index = (index,)
            output.append({
                "native": native, "treatment": treatment,
                **dict(zip(index_columns, index)),
                **{f"delta_{metric}": float(trow[metric] - nrow[metric]) for metric in metrics},
            })
    return output


def paired_tube(root: Path):
    aggregate_path = root / "raw" / "tube_scale_space_aggregate.jsonl.gz"
    # A long analysis may be resumed in checkpoint shards, in which case the
    # per-reaction stream is authoritative and a monolithic end-of-run aggregate
    # may not exist.  Rebuild it deterministically rather than rerunning model or
    # geometry computation.
    aggregate = grouped_stream(
        root / "raw" / "tube_scale_space_by_reaction.jsonl.gz",
        ("checkpoint", "layer", "segment", "span_length"),
        ("mean", "rms", "maximum", "p90", "p95", "monotonicity_violation",
         "fraction_gt_0.05", "fraction_gt_0.1", "fraction_gt_0.2", "fraction_gt_0.5"),
    )
    aggregate_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(aggregate_path, "wt", encoding="utf-8", compresslevel=6) as handle:
        for row in aggregate:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    frame = pd.DataFrame(aggregate)
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


def trajectory_treatment_uncertainty(root: Path):
    metrics = (
        "speed", "tangential_acceleration", "normal_acceleration",
        "normalized_normal_acceleration", "euclidean_path_efficiency",
        "fisher_path_efficiency", "fisher_local_curvature",
    )
    values = {}
    keys = set()
    for row in records(root / "raw" / "trajectory_metrics.jsonl.gz"):
        keys.add(row["checkpoint"])
        values[(row["checkpoint"], int(row["panel_index"]), row["layer"], row["segment"])] = row
    output = []
    for native, treatment in treatment_pairs(keys):
        fields = checkpoint_fields(treatment)
        for layer in ("embedding", "layer_6", "layer_16", "layer_21", "final_post_norm"):
            for segment in ("source", "product", "cross"):
                matched = []
                for panel in range(256):
                    n = values.get((native, panel, layer, segment))
                    t = values.get((treatment, panel, layer, segment))
                    if n is not None and t is not None:
                        matched.append((n, t))
                if not matched:
                    continue
                row = {
                    "native": native, "treatment": treatment, **fields,
                    "layer": layer, "segment": segment, "reactions": len(matched),
                }
                for metric in metrics:
                    differences = np.asarray([
                        float(t.get(metric, math.nan)) - float(n.get(metric, math.nan))
                        for n, t in matched
                    ])
                    differences = differences[np.isfinite(differences)]
                    if not len(differences):
                        continue
                    row[f"delta_{metric}"] = float(differences.mean())
                    row[f"delta_{metric}_ci95"] = bootstrap_ci(differences)
                    row[f"paired_dz_{metric}"] = float(
                        differences.mean() / differences.std(ddof=1)
                    ) if len(differences) > 1 and differences.std(ddof=1) > 0 else math.nan
                output.append(row)
    return output


def tube_reaction_uncertainty(root: Path):
    selected_lengths = {2, 4, 8, 16, 32, 64}
    values = {}
    keys = set()
    for row in records(root / "raw" / "tube_scale_space_by_reaction.jsonl.gz"):
        if row["checkpoint"] not in {
            "native_r8_s533", "native_r8_s917", "native_r8_s1301",
            "released_r8_l0.02_s533", "released_r8_l0.02_s917", "released_r8_l0.02_s1301",
            "paper_r8_l0.02_s533", "paper_r8_l0.02_s917",
        } or row["layer"] != "final_post_norm" or row["span_length"] not in selected_lengths:
            continue
        index = (row["checkpoint"], row["panel_index"], row["segment"], row["span_length"])
        values[index] = (row["rms"], row["maximum"], row["p95"])
        keys.add(row["checkpoint"])
    output = []
    for native, treatment in treatment_pairs(keys):
        for segment in ("source", "product", "cross"):
            for length in sorted(selected_lengths):
                effects = []
                for panel in range(256):
                    n = values.get((native, panel, segment, length))
                    t = values.get((treatment, panel, segment, length))
                    if n is not None and t is not None:
                        effects.append([t[i] - n[i] for i in range(3)])
                if not effects:
                    continue
                effects = np.asarray(effects)
                row = {"native": native, "treatment": treatment, "segment": segment, "span_length": length, "reactions": len(effects)}
                for index, metric in enumerate(("rms", "maximum", "p95")):
                    row[f"delta_{metric}"] = float(effects[:, index].mean())
                    row[f"delta_{metric}_ci95"] = bootstrap_ci(effects[:, index])
                    sd = effects[:, index].std(ddof=1)
                    row[f"paired_dz_{metric}"] = float(effects[:, index].mean() / sd) if sd > 0 else math.nan
                output.append(row)
    return output


def individual_persistence_scales(root: Path):
    from src.geodesic_audit import estimate_piecewise_change_point

    groups = defaultdict(list)
    current_key = None
    curve = []

    def finish(key, values):
        if key is not None:
            estimate = estimate_piecewise_change_point(values)
            if math.isfinite(estimate["breakpoint"]):
                groups[key[:3]].append(estimate["breakpoint"])

    for row in records(root / "raw" / "tube_scale_space_by_reaction.jsonl.gz"):
        key = (row["checkpoint"], row["layer"], row["segment"], row["panel_index"])
        if key != current_key:
            finish(current_key, curve)
            current_key, curve = key, []
        curve.append(row)
    finish(current_key, curve)
    output = []
    rng = np.random.default_rng(RNG_SEED)
    for (checkpoint, layer, segment), values in groups.items():
        values = np.asarray(values, dtype=float)
        bootstrap = []
        for start in range(0, 10_000, 250):
            samples = rng.choice(values, size=(min(250, 10_000 - start), len(values)), replace=True)
            bootstrap.extend(np.median(samples, axis=1).tolist())
        output.append({
            "checkpoint": checkpoint, "layer": layer, "segment": segment,
            "reaction_n": len(values), "median_breakpoint": float(np.median(values)),
            "mean_breakpoint": float(values.mean()),
            "q25_q75": np.quantile(values, [.25, .75]).tolist(),
            "median_bootstrap_ci95": np.quantile(bootstrap, [.025, .975]).tolist(),
        })
    return output


def summarize_candidates(root: Path):
    values = {}
    natural = []
    for row in records(root / "raw" / "gold_wrong_candidate_geometry.jsonl.gz"):
        if row["layer"] != "final_post_norm" or row["role"] not in {"gold", "highest_wrong", "seed1301_promoted_wrong"}:
            continue
        tube_mean = float(np.mean(row["tube_scale"]["rms"])) if row["tube_scale"]["rms"] else math.nan
        metrics = {
            "tube_rms_integral": tube_mean,
            "paper_loss": row["paper_loss"], "released_loss": row["released_loss"],
            "euclidean_inefficiency": 1 - row["euclidean_path_efficiency"],
            "fisher_inefficiency": 1 - row["fisher_path_efficiency"],
            "fisher_local_curvature": row["fisher_local_curvature"],
            "normal_acceleration": row["normalized_normal_acceleration"],
            "_model_aggregate_correct": bool(row["model_aggregate_correct"]),
            "_gold_aggregate_rank": row["aggregate_rank"],
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
            "model_aggregate_correct": wrong["_model_aggregate_correct"],
            "gold_aggregate_rank": wrong["_gold_aggregate_rank"],
            **{
                f"wrong_minus_gold_{metric}": wrong[metric] - gold[metric]
                for metric in wrong if not metric.startswith("_")
            },
        })
    # The promoted wrong candidate is evaluated under both seed-1301 models.
    for (checkpoint, panel, view, role), wrong in values.items():
        if role != "seed1301_promoted_wrong":
            continue
        gold = values.get((checkpoint, panel, view, "gold"))
        if gold:
            natural.append({
                "checkpoint": checkpoint, "panel_index": panel, "view": view,
                **{
                    f"wrong_minus_gold_{metric}": wrong[metric] - gold[metric]
                    for metric in wrong if not metric.startswith("_")
                },
            })
    return pd.DataFrame(paired), pd.DataFrame(natural)


def candidate_checkpoint_summary(frame: pd.DataFrame):
    if frame.empty:
        return []
    metrics = [column for column in frame if column.startswith("wrong_minus_gold_")]
    output = []
    strata = [("all", frame)]
    if "model_aggregate_correct" in frame:
        strata.extend([
            ("model_correct", frame[frame.model_aggregate_correct]),
            ("model_wrong", frame[~frame.model_aggregate_correct]),
        ])
    for stratum, stratum_frame in strata:
        for checkpoint, group in stratum_frame.groupby("checkpoint"):
            reaction = group.groupby("panel_index")[metrics].mean()
            row = {
                "checkpoint": checkpoint, "stratum": stratum,
                "reactions": len(reaction), **checkpoint_fields(checkpoint),
            }
            for metric in metrics:
                values = reaction[metric].to_numpy()
                row[metric] = float(np.mean(values))
                row[f"{metric}_ci95"] = bootstrap_ci(values)
                sd = values.std(ddof=1)
                row[f"paired_dz_{metric}"] = float(values.mean() / sd) if len(values) > 1 and sd > 0 else math.nan
                row[f"{metric}_fraction_wrong_more_geodesic"] = float((values < 0).mean())
            output.append(row)
    return output


def seed1301_change_summary(frame: pd.DataFrame):
    if frame.empty:
        return []
    metrics = [column for column in frame if column.startswith("wrong_minus_gold_")]
    native = frame[frame.checkpoint == "native_r8_s1301"].set_index(["panel_index", "view"])
    released = frame[frame.checkpoint == "released_r8_l0.02_s1301"].set_index(["panel_index", "view"])
    common = native.index.intersection(released.index)
    output = []
    for metric in metrics:
        delta = released.loc[common, metric].to_numpy() - native.loc[common, metric].to_numpy()
        reaction_values = pd.DataFrame({
            "panel_index": [index[0] for index in common], "value": delta,
        }).groupby("panel_index").value.mean().to_numpy()
        output.append({
            "metric": metric, "reaction_n": len(reaction_values),
            "released_change_in_wrong_minus_gold": float(reaction_values.mean()),
            "ci95": bootstrap_ci(reaction_values),
            "fraction_shift_toward_wrong_more_geodesic": float((reaction_values < 0).mean()),
        })
    return output


def summarize_candidate_intrinsic(root: Path):
    path = root / "raw" / "candidate_intrinsic_geometry.jsonl.gz"
    if not path.exists():
        return [], []
    metrics = (
        "geodesic_violation", "normal_acceleration",
        "geodesic_over_acceleration", "normal_over_acceleration",
    )
    # Average the three fixed query positions within each trajectory before
    # comparing the wrong and gold paths for the same reaction/view.
    trajectory = defaultdict(lambda: {"n": 0, **{metric: 0.0 for metric in metrics}})
    for row in records(path):
        role = row["role"]
        setting = (
            row["search_metric"], int(row["neighbors"]), int(row["tangent_dim"]),
        )
        key = (
            row["checkpoint"], int(row["panel_index"]), int(row["view"]),
            role, setting,
        )
        group = trajectory[key]; group["n"] += 1
        for metric in metrics:
            group[metric] += float(row[metric])
    averaged = {
        key: {metric: group[metric] / group["n"] for metric in metrics}
        for key, group in trajectory.items()
    }
    paired = []
    for (checkpoint, panel, view, role, setting), wrong in averaged.items():
        if role not in {"highest_wrong", "seed1301_promoted_wrong"}:
            continue
        gold = averaged.get((checkpoint, panel, view, "gold", setting))
        if gold is None:
            continue
        paired.append({
            "checkpoint": checkpoint, "panel_index": panel, "view": view,
            "role": role, "search_metric": setting[0],
            "neighbors": setting[1], "tangent_dim": setting[2],
            **{
                f"wrong_minus_gold_{metric}": wrong[metric] - gold[metric]
                for metric in metrics
            },
        })
    frame = pd.DataFrame(paired)
    if frame.empty:
        return [], []
    central = frame[
        (frame.role == "highest_wrong") & (frame.search_metric == "euclidean")
        & (frame.neighbors == 64) & (frame.tangent_dim == 16)
    ]
    central_summary = candidate_checkpoint_summary(central)
    natural = frame[frame.role == "seed1301_promoted_wrong"]
    natural_summary = []
    for setting, group in natural.groupby(["search_metric", "neighbors", "tangent_dim"]):
        native = group[group.checkpoint == "native_r8_s1301"].set_index(["panel_index", "view"])
        released = group[group.checkpoint == "released_r8_l0.02_s1301"].set_index(["panel_index", "view"])
        common = native.index.intersection(released.index)
        for metric in (column for column in group if column.startswith("wrong_minus_gold_")):
            delta = released.loc[common, metric] - native.loc[common, metric]
            reaction = pd.DataFrame({
                "panel_index": [index[0] for index in common], "value": delta.to_numpy(),
            }).groupby("panel_index").value.mean().to_numpy()
            natural_summary.append({
                "search_metric": setting[0], "neighbors": int(setting[1]),
                "tangent_dim": int(setting[2]), "metric": metric,
                "reaction_n": len(reaction), "released_change": float(reaction.mean()),
                "ci95": bootstrap_ci(reaction),
            })
    return central_summary, natural_summary


def summarize_final_operations(root: Path):
    path = root / "raw" / "final_operation_geometry.jsonl.gz"
    if not path.exists():
        return [], []
    scalar_metrics = (
        "local_curvature_mean", "speed", "tangential_acceleration",
        "normal_acceleration", "normalized_normal_acceleration",
        "euclidean_path_efficiency",
    )
    rows = []
    values = {}
    for row in records(path):
        flat = {metric: float(row[metric]) for metric in scalar_metrics}
        for length, tube in row["tube_selected"].items():
            for metric, value in tube.items():
                flat[f"tube_{metric}_L{length}"] = float(value)
        values[(row["checkpoint"], int(row["panel_index"]), row["segment"], row["layer"])] = flat
    all_metrics = sorted({metric for value in values.values() for metric in value})
    effects = []
    for checkpoint in sorted({key[0] for key in values}):
        for segment in ("source", "product", "cross"):
            for first, second, operation in (
                ("layer_21", "post_last_pre_norm", "last_block"),
                ("post_last_pre_norm", "final_post_norm", "final_rmsnorm"),
            ):
                paired = []
                for panel in range(64):
                    left = values.get((checkpoint, panel, segment, first))
                    right = values.get((checkpoint, panel, segment, second))
                    if left is not None and right is not None:
                        paired.append((left, right))
                if not paired:
                    continue
                result = {
                    "checkpoint": checkpoint, "segment": segment,
                    "operation": operation, "reactions": len(paired),
                }
                for metric in all_metrics:
                    delta = np.asarray([
                        right[metric] - left[metric]
                        for left, right in paired if metric in left and metric in right
                    ])
                    if len(delta):
                        result[f"delta_{metric}"] = float(delta.mean())
                        result[f"delta_{metric}_ci95"] = bootstrap_ci(delta)
                effects.append(result)
    grouped = defaultdict(list)
    for (checkpoint, _, segment, layer), value in values.items():
        grouped[(checkpoint, segment, layer)].append(value)
    for (checkpoint, segment, layer), group in grouped.items():
        row = {"checkpoint": checkpoint, "segment": segment, "layer": layer, "reactions": len(group)}
        for metric in all_metrics:
            data = [value[metric] for value in group if metric in value]
            if data:
                row[metric] = float(np.mean(data))
        rows.append(row)
    return rows, effects


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

    trajectory = pd.DataFrame(summaries["trajectory_treatment_uncertainty"])
    required = {"delta_euclidean_path_efficiency", "delta_fisher_path_efficiency"}
    if not trajectory.empty and required.issubset(trajectory.columns):
        selected_trajectory = trajectory[
            (trajectory.layer == "final_post_norm") & trajectory.treatment.str.contains("_r8_l0.02_")
        ]
        fig, axis = plt.subplots(figsize=(6.2, 5.2))
        for treatment, group in selected_trajectory.groupby("treatment"):
            axis.scatter(
                group.delta_euclidean_path_efficiency,
                group.delta_fisher_path_efficiency,
                label=treatment, s=32,
            )
        axis.axhline(0, color="black", linewidth=.7); axis.axvline(0, color="black", linewidth=.7)
        axis.set(xlabel="STP - Native Euclidean path efficiency",
                 ylabel="STP - Native Fisher path efficiency")
        axis.grid(alpha=.25); axis.legend(fontsize=7, frameon=False)
        fig.tight_layout(); fig.savefig(plot_dir / "euclidean_vs_fisher_change.png", dpi=180); plt.close(fig)

    event = pd.DataFrame(summaries["event_signal_sensitivity"])
    if not event.empty:
        chosen = event[
            event.checkpoint.isin(["native_r8_s533", "released_r8_l0.02_s533", "paper_r8_l0.02_s533"])
            & event.event_category.isin(["ordinary", "any_event", "ring_closure", "branch", "stereochemistry", "motif_completion"])
            & (event.segment == "product")
        ].copy()
        if not chosen.empty:
            chosen["label"] = chosen.checkpoint + ":" + chosen.event_category
            fig, axis = plt.subplots(figsize=(10, 4.8))
            chosen.set_index("label").absolute_perpendicular_to_parallel_ratio.plot.bar(ax=axis)
            axis.set_ylabel("mean |perpendicular sensitivity| / |parallel sensitivity|")
            axis.tick_params(axis="x", labelsize=7)
            fig.tight_layout(); fig.savefig(plot_dir / "perpendicular_signal_by_event.png", dpi=180); plt.close(fig)

    intrinsic = pd.DataFrame(summaries["intrinsic_treatment_summary"])
    if not intrinsic.empty:
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
        image = None
        for axis, segment in zip(axes, ("source", "product")):
            pivot = intrinsic[intrinsic.segment == segment].pivot_table(
                index="treatment", columns="layer",
                values="delta_geodesic_violation_fraction_negative", aggfunc="mean",
            )
            image = axis.imshow(pivot.to_numpy(), vmin=0, vmax=1, aspect="auto", cmap="coolwarm")
            axis.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=30, ha="right")
            axis.set_yticks(range(len(pivot.index)), pivot.index, fontsize=7)
            axis.set_title(segment)
        fig.colorbar(image, ax=axes, label="fraction of settings with reduced violation")
        fig.tight_layout(); fig.savefig(plot_dir / "intrinsic_robustness.png", dpi=180); plt.close(fig)

    anatomy = pd.DataFrame(summaries["released_anatomy_treatment"])
    if not anatomy.empty:
        chosen = anatomy[anatomy.treatment.str.contains("released_r8_l0.02")]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        for treatment, group in chosen.groupby("treatment"):
            group = group.sort_values("span_length")
            axes[0].plot(group.span_length, group.delta_loss, label=treatment)
            axes[1].plot(group.span_length, group.delta_cancellation_ratio, label=treatment)
        axes[0].set_ylabel("Released loss: STP - Native")
        axes[1].set_ylabel("Complement cancellation ratio: STP - Native")
        for axis in axes:
            axis.set_xlabel("patch length"); axis.axhline(0, color="black", linewidth=.7); axis.grid(alpha=.25)
        axes[1].legend(fontsize=7, frameon=False)
        fig.tight_layout(); fig.savefig(plot_dir / "released_objective_anatomy.png", dpi=180); plt.close(fig)

    cones = pd.DataFrame(summaries["cone_treatment_summary"])
    if not cones.empty:
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
        for treatment, group in cones.groupby("treatment"):
            for kind, part in group.groupby("kind"):
                axes[0].plot(part.horizon, part.delta_mean_axis_gold_cosine, label=f"{treatment}:{kind}")
                axes[1].plot(part.horizon, part.delta_fisher_dispersion, label=f"{treatment}:{kind}")
        axes[0].set_ylabel("STP - Native cone-axis/gold cosine")
        axes[1].set_ylabel("STP - Native Fisher dispersion")
        for axis in axes:
            axis.set_xlabel("rollout horizon"); axis.axhline(0, color="black", linewidth=.7); axis.grid(alpha=.25)
        axes[1].legend(fontsize=6, frameon=False)
        fig.tight_layout(); fig.savefig(plot_dir / "inference_cone_changes.png", dpi=180); plt.close(fig)

    natural = pd.DataFrame(summaries["seed1301_change_summary"])
    if not natural.empty:
        fig, axis = plt.subplots(figsize=(9, 4.5))
        natural.set_index("metric").released_change_in_wrong_minus_gold.plot.bar(ax=axis)
        axis.axhline(0, color="black", linewidth=.7)
        axis.set_ylabel("Released STP change in wrong - gold geometry")
        axis.tick_params(axis="x", labelsize=7)
        fig.tight_layout(); fig.savefig(plot_dir / "seed1301_natural_experiment.png", dpi=180); plt.close(fig)

    operations = pd.DataFrame(summaries["final_operation_effects"])
    if not operations.empty:
        chosen = operations[
            operations.checkpoint.isin(["native_r8_s533", "released_r8_l0.02_s533", "paper_r8_l0.02_s533"])
            & (operations.segment == "product")
        ]
        if not chosen.empty:
            labels = chosen.checkpoint + ":" + chosen.operation
            fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
            axes[0].bar(labels, chosen.delta_local_curvature_mean)
            axes[1].bar(labels, chosen.delta_normalized_normal_acceleration)
            axes[0].set_ylabel("operation-induced local-curvature change")
            axes[1].set_ylabel("operation-induced normalized-normal-acceleration change")
            for axis in axes:
                axis.axhline(0, color="black", linewidth=.7)
                axis.tick_params(axis="x", rotation=60, labelsize=7)
            fig.tight_layout(); fig.savefig(plot_dir / "last_block_vs_rmsnorm.png", dpi=180); plt.close(fig)

    persistence = pd.DataFrame(summaries["tangent_persistence"])
    if not persistence.empty:
        chosen = persistence[
            (persistence.layer == "final_post_norm") & (persistence.segment == "product")
            & persistence.checkpoint.isin(["native_r8_s533", "released_r8_l0.02_s533", "paper_r8_l0.02_s533"])
        ]
        fig, axis = plt.subplots(figsize=(7.5, 4.5))
        for checkpoint, group in chosen.groupby("checkpoint"):
            axis.plot(group.lag, group["mean"], label=checkpoint)
        axis.axhline(0, color="black", linewidth=.7)
        axis.set(xlabel="transition lag k", ylabel="tangent autocorrelation C(k)")
        axis.grid(alpha=.25); axis.legend(fontsize=7, frameon=False)
        fig.tight_layout(); fig.savefig(plot_dir / "tangent_persistence_final_product.png", dpi=180); plt.close(fig)

    turning = pd.DataFrame(summaries["multiscale_turning_treatment"])
    if not turning.empty:
        chosen = turning[
            (turning.layer == "final_post_norm") & (turning.segment == "product")
            & turning.treatment.str.contains("_r8_l0.02_")
        ]
        fig, axis = plt.subplots(figsize=(7.5, 4.5))
        for treatment, group in chosen.groupby("treatment"):
            axis.plot(group.scale, group.delta_mean_angle, label=treatment)
        axis.axhline(0, color="black", linewidth=.7)
        axis.set(xlabel="turning scale k", ylabel="STP - Native mean turning angle")
        axis.grid(alpha=.25); axis.legend(fontsize=7, frameon=False)
        fig.tight_layout(); fig.savefig(plot_dir / "multiscale_turning_change.png", dpi=180); plt.close(fig)


def run(args):
    root = args.root.resolve()
    tube, tube_delta = paired_tube(root)
    tube_uncertainty = tube_reaction_uncertainty(root)
    trajectory_uncertainty = trajectory_treatment_uncertainty(root)
    persistence_scales = individual_persistence_scales(root)
    persistence = grouped_stream(
        root / "raw" / "tangent_persistence.jsonl.gz",
        ("checkpoint", "layer", "segment", "lag"), ("mean", "median"),
    )
    persistence_treatment = paired_reduced_summary(
        persistence, ["layer", "segment", "lag"], ["mean", "median"],
    )
    turning = summarize_multiscale_turning(root / "raw" / "multiscale_turning.jsonl.gz")
    turning_treatment = paired_reduced_summary(
        turning, ["layer", "segment", "scale"], ["mean_angle"],
    )
    intervention = summarize_interventions(root / "raw" / "signal_noise_interventions.jsonl.gz")
    sensitivity, event_sensitivity, hidden_fisher = summarize_sensitivity(
        root / "raw" / "signal_noise_interventions.jsonl.gz"
    )
    anatomy = grouped_stream(
        root / "raw" / "released_objective_anatomy.jsonl.gz",
        ("checkpoint", "span_length"),
        ("loss", "cos_patch_before", "cos_patch_after", "cos_before_after", "cancellation_ratio"),
    )
    anatomy_treatment = paired_reduced_summary(
        anatomy, ["span_length"],
        ["loss", "cos_patch_before", "cos_patch_after", "cos_before_after", "cancellation_ratio"],
    )
    matched = grouped_stream(
        root / "raw" / "matched_native_stp_displacement.jsonl.gz",
        ("native", "treatment", "layer", "segment", "span_length"),
        ("delta_rho", "correction_cosine", "endpoint_middle_ratio", "chord_cosine", "chord_norm_ratio"),
    )
    intrinsic = grouped_stream(
        root / "raw" / "intrinsic_manifold_decomposition.jsonl.gz",
        ("checkpoint", "layer", "segment", "search_metric", "same_segment", "neighbors", "tangent_dim"),
        ("geodesic_violation", "normal_acceleration", "geodesic_over_acceleration", "normal_over_acceleration"),
    )
    intrinsic_treatment = intrinsic_treatment_summary(intrinsic)
    cones = grouped_stream(
        root / "raw" / "inference_cones.jsonl.gz",
        ("checkpoint", "kind", "horizon"),
        ("weighted_angle_from_mean", "weighted_angle_from_gold", "perpendicular_variance", "mean_axis_gold_cosine", "fisher_dispersion"),
    )
    cone_treatment = paired_reduced_summary(
        cones, ["kind", "horizon"],
        ["weighted_angle_from_mean", "weighted_angle_from_gold", "perpendicular_variance", "mean_axis_gold_cosine", "fisher_dispersion"],
    )
    gold_wrong, natural = summarize_candidates(root)
    gold_wrong_summary = candidate_checkpoint_summary(gold_wrong)
    natural_change = seed1301_change_summary(natural)
    candidate_intrinsic, candidate_intrinsic_natural = summarize_candidate_intrinsic(root)
    final_operations, final_operation_effects = summarize_final_operations(root)
    summaries = {
        "tube_treatment_effects": tube_delta.to_dict("records"),
        "tube_reaction_uncertainty": tube_uncertainty,
        "trajectory_treatment_uncertainty": trajectory_uncertainty,
        "individual_persistence_scales": persistence_scales,
        "tangent_persistence": persistence,
        "tangent_persistence_treatment": persistence_treatment,
        "multiscale_turning": turning,
        "multiscale_turning_treatment": turning_treatment,
        "interventions": intervention, "signal_sensitivity": sensitivity,
        "event_signal_sensitivity": event_sensitivity,
        "hidden_fisher_correlations": hidden_fisher,
        "released_anatomy": anatomy, "released_anatomy_treatment": anatomy_treatment,
        "matched_displacement": matched, "intrinsic": intrinsic,
        "intrinsic_treatment_summary": intrinsic_treatment, "cones": cones,
        "cone_treatment_summary": cone_treatment,
        "gold_wrong": gold_wrong.to_dict("records"),
        "gold_wrong_checkpoint_summary": gold_wrong_summary,
        "candidate_intrinsic_checkpoint_summary": candidate_intrinsic,
        "candidate_intrinsic_seed1301_robustness": candidate_intrinsic_natural,
        "final_operation_geometry": final_operations,
        "final_operation_effects": final_operation_effects,
        "seed1301_natural_experiment": natural.to_dict("records"),
        "seed1301_change_summary": natural_change,
    }
    (root / "analysis").mkdir(parents=True, exist_ok=True)
    for name, value in summaries.items():
        (root / "analysis" / f"{name}.json").write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    compact = {
        "tube_delta_final_product": tube_delta[(tube_delta.layer == "final_post_norm") & (tube_delta.segment == "product")].to_dict("records"),
        "tube_reaction_uncertainty": tube_uncertainty,
        "trajectory_treatment_uncertainty": trajectory_uncertainty,
        "individual_persistence_scales": persistence_scales,
        "tangent_persistence_treatment": persistence_treatment,
        "multiscale_turning_treatment": turning_treatment,
        "interventions": intervention, "signal_sensitivity": sensitivity,
        "event_signal_sensitivity": event_sensitivity,
        "hidden_fisher_correlations": hidden_fisher,
        "intrinsic": intrinsic, "intrinsic_treatment_summary": intrinsic_treatment,
        "cones": cones, "cone_treatment_summary": cone_treatment,
        "gold_wrong_checkpoint_summary": gold_wrong_summary,
        "candidate_intrinsic_checkpoint_summary": candidate_intrinsic,
        "candidate_intrinsic_seed1301_robustness": candidate_intrinsic_natural,
        "final_operation_effects": final_operation_effects,
        "seed1301_natural_means": natural.groupby("checkpoint").mean(numeric_only=True).reset_index().to_dict("records") if not natural.empty else [],
        "seed1301_change_summary": natural_change,
    }
    (root / "analysis.json").write_text(json.dumps(compact, indent=2) + "\n", encoding="utf-8")
    make_plots(root, tube, tube_delta, summaries)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    run(parser.parse_args())
