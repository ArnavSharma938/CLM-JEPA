"""Analyze the preregistered frozen STP checkpoint representation matrix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from frozen_geometry import EVENT_TYPES  # noqa: E402
from stp_representation_analysis import checkpoint_specs  # noqa: E402


REACTION_METRICS = (
    "released_fixed_loss", "paper_fixed_loss",
    "source_activation_norm", "target_activation_norm",
    "source_transition_norm", "target_transition_norm",
    "source_curvature", "target_curvature",
    "source_path_efficiency", "target_path_efficiency",
)


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def bootstrap(values: np.ndarray, seed: int, samples: int = 5000) -> dict:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"n": 0, "mean": math.nan, "ci95": [math.nan, math.nan], "dz": math.nan}
    rng = np.random.default_rng(seed)
    means = np.empty(samples)
    for start in range(0, samples, 500):
        end = min(samples, start + 500)
        means[start:end] = values[rng.integers(0, len(values), size=(end - start, len(values)))].mean(1)
    sd = values.std(ddof=1) if len(values) > 1 else math.nan
    return {
        "n": len(values), "mean": float(values.mean()),
        "ci95": np.quantile(means, [.025, .975]).tolist(),
        "dz": float(values.mean() / sd) if sd and np.isfinite(sd) else math.nan,
    }


def bh(records: list[dict], p_key: str = "p") -> None:
    valid = sorted(
        ((index, row[p_key]) for index, row in enumerate(records) if np.isfinite(row.get(p_key, math.nan))),
        key=lambda item: item[1],
    )
    running = 1.0
    adjusted = [1.0] * len(valid)
    for reverse in range(len(valid) - 1, -1, -1):
        running = min(running, valid[reverse][1] * len(valid) / (reverse + 1))
        adjusted[reverse] = running
    for (index, _), value in zip(valid, adjusted):
        records[index]["q_bh"] = float(value)


def sign_flip_p(values: np.ndarray, seed: int, permutations: int = 10000) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values) or np.all(values == 0):
        return 1.0
    rng = np.random.default_rng(seed)
    observed = abs(values.mean())
    extreme = 0
    done = 0
    while done < permutations:
        count = min(1000, permutations - done)
        signs = rng.integers(0, 2, size=(count, len(values)), dtype=np.int8) * 2 - 1
        extreme += int((np.abs((signs * values).mean(1)) >= observed - 1e-15).sum())
        done += count
    return (extreme + 1) / (permutations + 1)


def generation_lookup() -> dict[str, dict]:
    matrix = load_json(ROOT / "runs/stp_matrix/a6000/analysis.json")["stages"]
    completion = load_json(ROOT / "runs/stp_completion/a6000/analysis.json")["conditions"]
    result = {}

    def ingest(prefix: str, comparisons: dict):
        for seed, value in comparisons.items():
            result[f"{prefix}_s{seed}"] = value

    ingest("released_r8_l0.02", matrix["A_rank8_released"]["comparisons"])
    ingest("released_r128_l0.02", matrix["B_rank128_released"]["comparisons"])
    ingest("paper_r8_l0.02", completion["paper_r8_l0.02"]["comparisons"])
    ingest("paper_r8_l0.08", completion["paper_r8_l0.08"]["comparisons"])
    ingest("paper_r8_l0.12", completion["paper_r8_l0.12"]["comparisons"])
    ingest("paper_r128_l0.02", completion["paper_r128_l0.02"]["comparisons"])
    for label in ("0.005", "0.08"):
        ingest(f"released_r8_l{label}", matrix["D_lambda"]["lambdas"][label]["comparisons"])
    return result


def cluster_event_delta(treatment: np.lib.npyio.NpzFile, native: np.lib.npyio.NpzFile, metric: str, category: str, layer: int):
    event_key, control_key = (
        ("local_event", "local_control") if metric == "local"
        else ("semi_event", "semi_control")
    )
    categories = treatment["category"]
    reactions = treatment["reaction_identity"]
    mask = categories == category
    delta = (
        treatment[event_key][:, layer] - treatment[control_key][:, layer]
        - native[event_key][:, layer] + native[control_key][:, layer]
    )
    return np.asarray([
        np.nanmean(delta[mask & (reactions == identity)])
        for identity in np.unique(reactions[mask])
        if np.isfinite(delta[mask & (reactions == identity)]).any()
    ])


def rank_biserial(values: np.ndarray, outcomes: np.ndarray) -> dict:
    wins = values[outcomes == 1]
    losses = values[outcomes == -1]
    if not len(wins) or not len(losses):
        return {"wins": len(wins), "losses": len(losses), "win_mean": math.nan, "loss_mean": math.nan, "auc": math.nan}
    combined = np.concatenate((wins, losses))
    ranks = stats.rankdata(combined)
    u = ranks[:len(wins)].sum() - len(wins) * (len(wins) + 1) / 2
    return {
        "wins": len(wins), "losses": len(losses),
        "win_mean": float(wins.mean()), "loss_mean": float(losses.mean()),
        "win_minus_loss": float(wins.mean() - losses.mean()),
        "auc": float(u / (len(wins) * len(losses))),
    }


def config_key(spec) -> str:
    return f"{spec.formulation}_r{spec.rank}_l{spec.stp_lambda:g}"


def analyze(input_dir: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    specs = checkpoint_specs()
    spec_by_key = {spec.key: spec for spec in specs}
    generations = generation_lookup()
    native_key = {(spec.rank, spec.seed): spec.key for spec in specs if spec.is_native}
    reaction = {spec.key: np.load(input_dir / "checkpoints" / spec.key / "reaction_geometry.npz") for spec in specs}
    event = {spec.key: np.load(input_dir / "checkpoints" / spec.key / "event_geometry.npz") for spec in specs}
    representation = {
        spec.key: load_json(input_dir / "checkpoints" / spec.key / "representation_summary.json")
        for spec in specs
    }

    checkpoint_effects = []
    event_effects = []
    reaction_generation_links = []
    for treatment in (spec for spec in specs if not spec.is_native):
        native = spec_by_key[native_key[(treatment.rank, treatment.seed)]]
        record = {
            "key": treatment.key, "configuration": config_key(treatment),
            "rank": treatment.rank, "formulation": treatment.formulation,
            "lambda": treatment.stp_lambda, "seed": treatment.seed,
            "native_key": native.key,
            "generation": {
                metric: generations[treatment.key][metric]
                for metric in ("top1", "top3", "top5", "top10")
            },
            "layers": [],
        }
        outcomes = np.asarray(generations[treatment.key]["per_reaction_top1_difference"][:256])
        for layer in range(23):
            layer_record = {"layer": layer, "reaction_geometry": {}, "representation": {}}
            for metric_index, metric in enumerate(REACTION_METRICS):
                values = reaction[treatment.key][metric][:, layer] - reaction[native.key][metric][:, layer]
                layer_record["reaction_geometry"][metric] = bootstrap(
                    values, 71_000 + treatment.seed + metric_index * 101 + layer,
                )
                if layer in (0, 6, 11, 16, 21, 22):
                    reaction_generation_links.append({
                        "key": treatment.key, "configuration": config_key(treatment),
                        "metric": metric, "layer": layer,
                        **rank_biserial(values, outcomes),
                    })
            tr = representation[treatment.key]["representation"]["layers"][layer]
            nr = representation[native.key]["representation"]["layers"][layer]
            for family in (
                "source_pooled_spectrum", "target_pooled_spectrum",
                "source_transition_spectrum", "target_transition_spectrum",
                "relationship",
            ):
                layer_record["representation"][family] = {
                    metric: float(tr[family][metric] - nr[family][metric])
                    for metric in tr[family]
                }
            layer_record["native_drift"] = representation[treatment.key]["native_drift"]["layers"][layer]
            record["layers"].append(layer_record)
            for metric_index, metric in enumerate(("local", "semi")):
                for category_index, category in enumerate(EVENT_TYPES):
                    values = cluster_event_delta(event[treatment.key], event[native.key], metric, category, layer)
                    summary = bootstrap(values, 91_000 + treatment.seed + metric_index * 1000 + category_index * 100 + layer)
                    event_effects.append({
                        "key": treatment.key, "configuration": config_key(treatment),
                        "rank": treatment.rank, "formulation": treatment.formulation,
                        "lambda": treatment.stp_lambda, "seed": treatment.seed,
                        "metric": metric, "category": category, "layer": layer,
                        **summary,
                        "p": sign_flip_p(values, 121_000 + treatment.seed + metric_index * 1000 + category_index * 100 + layer),
                    })
        checkpoint_effects.append(record)
    bh(event_effects)

    grouped = defaultdict(list)
    for record in checkpoint_effects:
        grouped[record["configuration"]].append(record)
    configuration_summary = []
    for configuration, records in sorted(grouped.items()):
        final_values = {}
        for metric in REACTION_METRICS:
            values = [record["layers"][-1]["reaction_geometry"][metric]["mean"] for record in records]
            final_values[metric] = {"seed_values": values, "mean": float(np.mean(values)), "range": [float(min(values)), float(max(values))]}
        for family, metric in (
            ("source_pooled_spectrum", "effective_rank"),
            ("target_pooled_spectrum", "effective_rank"),
            ("source_transition_spectrum", "effective_rank"),
            ("target_transition_spectrum", "effective_rank"),
            ("relationship", "pairing_gap"),
            ("relationship", "retrieval_top1"),
        ):
            values = [record["layers"][-1]["representation"][family][metric] for record in records]
            final_values[f"{family}.{metric}"] = {"seed_values": values, "mean": float(np.mean(values)), "range": [float(min(values)), float(max(values))]}
        for segment in ("source", "target"):
            for metric in ("centered_linear_cka", "aligned_cosine", "relative_rms_displacement", "displacement_effective_rank"):
                values = [record["layers"][-1]["native_drift"][segment][metric] for record in records]
                final_values[f"{segment}_drift.{metric}"] = {"seed_values": values, "mean": float(np.mean(values)), "range": [float(min(values)), float(max(values))]}
        generation_values = [record["generation"]["top1"]["absolute_difference"] for record in records]
        configuration_summary.append({
            "configuration": configuration, "seeds": [record["seed"] for record in records],
            "top1_effect_seed_values": generation_values,
            "top1_effect_mean": float(np.mean(generation_values)),
            "final_layer_effects": final_values,
        })

    # Across-checkpoint associations are descriptive because configurations and
    # seeds are neither independent nor numerous.
    association_metrics = (
        "released_fixed_loss", "paper_fixed_loss", "source_curvature", "target_curvature",
        "source_path_efficiency", "target_path_efficiency",
    )
    generation_associations = []
    y = np.asarray([row["generation"]["top1"]["absolute_difference"] for row in checkpoint_effects])
    for metric in association_metrics:
        x = np.asarray([row["layers"][-1]["reaction_geometry"][metric]["mean"] for row in checkpoint_effects])
        pearson = stats.pearsonr(x, y)
        spearman = stats.spearmanr(x, y)
        leave_config_out = []
        configs = np.asarray([row["configuration"] for row in checkpoint_effects])
        for omitted in sorted(set(configs)):
            keep = configs != omitted
            if keep.sum() >= 4:
                leave_config_out.append(float(stats.spearmanr(x[keep], y[keep]).statistic))
        generation_associations.append({
            "metric": metric, "n": len(x), "pearson_r": float(pearson.statistic),
            "pearson_p": float(pearson.pvalue), "spearman_rho": float(spearman.statistic),
            "spearman_p": float(spearman.pvalue),
            "leave_one_configuration_out_spearman_range": [min(leave_config_out), max(leave_config_out)],
        })

    result = {
        "type": "stp_representation_analysis",
        "input_manifest": load_json(input_dir / "manifest.json"),
        "checkpoint_effects": checkpoint_effects,
        "configuration_summary": configuration_summary,
        "event_effects": event_effects,
        "reaction_generation_links": reaction_generation_links,
        "generation_associations": generation_associations,
        "multiplicity": "BH across all 17*23*5*2 treatment/event/layer tests",
    }
    analysis_path = output_dir / "analysis.json"
    analysis_path.write_text(json.dumps(result, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    write_tables(result, output_dir)
    write_plots(result, output_dir)
    manifest = []
    for path in sorted(output_dir.iterdir()):
        # A manifest cannot truthfully hash the file that contains itself.
        if path.is_file() and path.name != "artifact_manifest.json":
            manifest.append({"name": path.name, "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    (output_dir / "artifact_manifest.json").write_text(json.dumps({"artifacts": manifest}, indent=2) + "\n", encoding="utf-8")
    return result


def write_tables(result: dict, output_dir: Path) -> None:
    with (output_dir / "configuration_final_layer.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["configuration", "seeds", "top1_effect_mean", *REACTION_METRICS,
                  "source_effective_rank", "target_effective_rank", "source_cka", "target_cka",
                  "source_displacement", "target_displacement"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in result["configuration_summary"]:
            values = row["final_layer_effects"]
            writer.writerow({
                "configuration": row["configuration"], "seeds": ",".join(map(str, row["seeds"])),
                "top1_effect_mean": row["top1_effect_mean"],
                **{metric: values[metric]["mean"] for metric in REACTION_METRICS},
                "source_effective_rank": values["source_pooled_spectrum.effective_rank"]["mean"],
                "target_effective_rank": values["target_pooled_spectrum.effective_rank"]["mean"],
                "source_cka": values["source_drift.centered_linear_cka"]["mean"],
                "target_cka": values["target_drift.centered_linear_cka"]["mean"],
                "source_displacement": values["source_drift.relative_rms_displacement"]["mean"],
                "target_displacement": values["target_drift.relative_rms_displacement"]["mean"],
            })
    with (output_dir / "event_treatment_effects.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["key", "configuration", "rank", "formulation", "lambda", "seed", "metric", "category", "layer", "n", "mean", "ci95_low", "ci95_high", "dz", "p", "q_bh"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in result["event_effects"]:
            writer.writerow({**{key: row[key] for key in fields if key in row}, "ci95_low": row["ci95"][0], "ci95_high": row["ci95"][1]})


def write_plots(result: dict, output_dir: Path) -> None:
    configs = result["configuration_summary"]
    labels = [row["configuration"].replace("released_", "rel ").replace("paper_", "paper ") for row in configs]
    panels = []
    for metric, title in (
        ("released_fixed_loss", "Released objective change"),
        ("paper_fixed_loss", "Paper objective change"),
        ("target_path_efficiency", "Target path-efficiency change"),
    ):
        panels.append({"title": title, "values": [row["final_layer_effects"][metric]["mean"] for row in configs]})
    _svg_bar_panels(output_dir / "configuration_geometry.svg", labels, panels)

    selected = {"released_r8_l0.02", "paper_r8_l0.02", "released_r8_l0.08", "paper_r8_l0.12", "released_r128_l0.02", "paper_r128_l0.02"}
    grouped = defaultdict(list)
    for row in result["checkpoint_effects"]:
        if row["configuration"] in selected: grouped[row["configuration"]].append(row)
    line_panels = []
    for getter, title in (
        (lambda r,l: r["layers"][l]["reaction_geometry"]["released_fixed_loss"]["mean"], "Released objective change by depth"),
        (lambda r,l: r["layers"][l]["reaction_geometry"]["paper_fixed_loss"]["mean"], "Paper objective change by depth"),
        (lambda r,l: r["layers"][l]["reaction_geometry"]["target_path_efficiency"]["mean"], "Target path-efficiency change by depth"),
        (lambda r,l: r["layers"][l]["native_drift"]["target"]["centered_linear_cka"], "Target Native/STP CKA by depth"),
    ):
        line_panels.append({
            "title": title,
            "series": [{"label": label, "values": np.mean([[getter(row, layer) for layer in range(23)] for row in rows], axis=0).tolist()} for label, rows in grouped.items()],
        })
    _svg_line_panels(output_dir / "layerwise_geometry_and_drift.svg", line_panels)

    final = [row for row in result["event_effects"] if row["layer"] == 22]
    config_names = sorted(set(row["configuration"] for row in final))
    matrices = []
    for metric in ("local", "semi"):
        matrices.append({
            "title": f"Final-layer STP-minus-Native change in {metric} event effect",
            "values": np.asarray([[np.mean([row["mean"] for row in final if row["configuration"] == config and row["metric"] == metric and row["category"] == category]) for category in EVENT_TYPES] for config in config_names]),
        })
    _svg_heatmaps(output_dir / "event_geometry_treatment_heatmap.svg", config_names, list(EVENT_TYPES), matrices)


def _svg_bar_panels(path: Path, labels: list[str], panels: list[dict]) -> None:
    width, panel_h = 1200, 280
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{panel_h * len(panels)}">', '<rect width="100%" height="100%" fill="white"/>']
    for panel_i, panel in enumerate(panels):
        y0 = panel_i * panel_h + 30; left, right, height = 90, width - 20, 170
        values = np.asarray(panel["values"]); scale = max(abs(values.min()), abs(values.max()), 1e-12)
        zero = y0 + height / 2
        parts += [f'<text x="{left}" y="{y0-10}" font-family="sans-serif" font-size="14">{html.escape(panel["title"])}</text>', f'<line x1="{left}" y1="{zero}" x2="{right}" y2="{zero}" stroke="#555"/>']
        step = (right-left)/len(labels)
        for i,(label,value) in enumerate(zip(labels,values)):
            x=left+i*step+step*.18; bar_w=step*.64; y=zero-(value/scale)*(height*.44)
            parts.append(f'<rect x="{x:.1f}" y="{min(y,zero):.1f}" width="{bar_w:.1f}" height="{abs(y-zero):.1f}" fill="#4472c4"/>')
            parts.append(f'<text transform="translate({x+bar_w/2:.1f},{y0+height+12}) rotate(45)" font-family="sans-serif" font-size="8">{html.escape(label)}</text>')
    parts.append('</svg>'); path.write_text('\n'.join(parts), encoding='utf-8')


def _svg_line_panels(path: Path, panels: list[dict]) -> None:
    width, panel_h = 1100, 260; colors=("#3366cc","#dc3912","#109618","#990099","#ff9900","#0099c6")
    parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{panel_h*len(panels)}">','<rect width="100%" height="100%" fill="white"/>']
    for pi,panel in enumerate(panels):
        top=pi*panel_h+30; left,right,bottom=70,width-250,top+180
        allv=np.asarray([s["values"] for s in panel["series"]]); lo=float(np.nanmin(allv)); hi=float(np.nanmax(allv)); pad=max((hi-lo)*.08,1e-9); lo-=pad; hi+=pad
        parts.append(f'<text x="{left}" y="{top-10}" font-family="sans-serif" font-size="14">{html.escape(panel["title"])}</text>')
        parts.append(f'<rect x="{left}" y="{top}" width="{right-left}" height="{bottom-top}" fill="none" stroke="#aaa"/>')
        for si,series in enumerate(panel["series"]):
            points=[]
            for i,v in enumerate(series["values"]): points.append(f'{left+i*(right-left)/22:.1f},{bottom-(v-lo)/(hi-lo)*(bottom-top):.1f}')
            color=colors[si%len(colors)]; parts.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="1.5"/>')
            parts.append(f'<text x="{right+10}" y="{top+12+si*16}" font-family="sans-serif" font-size="9" fill="{color}">{html.escape(series["label"])}</text>')
        parts.append(f'<text x="{left}" y="{bottom+16}" font-family="sans-serif" font-size="9">0</text><text x="{right-10}" y="{bottom+16}" font-family="sans-serif" font-size="9">22</text>')
    parts.append('</svg>'); path.write_text('\n'.join(parts), encoding='utf-8')


def _svg_heatmaps(path: Path, rows: list[str], columns: list[str], panels: list[dict]) -> None:
    width, panel_w, cell_h, top = 1400, 650, 24, 90; height=top+cell_h*len(rows)+80
    parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">','<rect width="100%" height="100%" fill="white"/>']
    for pi,panel in enumerate(panels):
        x0=pi*panel_w+180; values=panel["values"]; scale=max(float(np.max(np.abs(values))),1e-12); cell_w=75
        parts.append(f'<text x="{x0}" y="25" font-family="sans-serif" font-size="13">{html.escape(panel["title"])}</text>')
        for ci,column in enumerate(columns): parts.append(f'<text transform="translate({x0+ci*cell_w+cell_w/2},{top-8}) rotate(-35)" font-family="sans-serif" font-size="9">{html.escape(column)}</text>')
        for ri,row in enumerate(rows):
            if pi==0: parts.append(f'<text x="{x0-8}" y="{top+ri*cell_h+16}" text-anchor="end" font-family="sans-serif" font-size="8">{html.escape(row)}</text>')
            for ci,value in enumerate(values[ri]):
                fraction=min(1,abs(float(value))/scale); base=(220,55,55) if value>0 else (55,100,210); rgb=tuple(round(255-(255-c)*fraction) for c in base)
                parts.append(f'<rect x="{x0+ci*cell_w}" y="{top+ri*cell_h}" width="{cell_w}" height="{cell_h}" fill="rgb{rgb}" stroke="white"/>')
    parts.append('</svg>'); path.write_text('\n'.join(parts), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "runs/stp_representation/frozen_all_checkpoints")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/stp_representation/analysis")
    args = parser.parse_args()
    result = analyze(args.input.resolve(), args.output.resolve())
    print(json.dumps({"stage": "complete", "configurations": len(result["configuration_summary"]), "treatments": len(result["checkpoint_effects"]), "output": str(args.output.resolve())}))


if __name__ == "__main__":
    main()
