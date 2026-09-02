#!/usr/bin/env python
"""Extract and analyze frozen ChemFM states for the Geodesic Mechanism Audit.

The expensive transformer pass is cached once per checkpoint.  All geometry is
then recomputable from compact BF16 state shards and FP32 LM-head weights.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.chemfm import MODEL_DIR, TOKENIZER_DIR, load_lora_model, load_reaction_tokenizer
from src.frozen_geometry import DEFAULT_PANEL, annotate_example, read_panel
from src.geodesic_audit import (
    acceleration_decomposition,
    categorical_fisher_squared,
    chord_coordinates,
    cosine,
    curvature_removal_intervention,
    estimate_piecewise_change_point,
    fisher_rao_distance,
    fisher_rao_path_efficiency,
    fisher_rao_triangle_excess,
    gold_logprob_gradient,
    local_tangent_acceleration,
    matched_geodesic_displacement,
    multiscale_turning,
    optimal_ray_residual,
    predictive_sensitivity,
    released_objective_anatomy,
    tangent_autocorrelation,
    tube_scale_space,
)
from src.jepa import add_predictor_tokens
from src.stp_representation_analysis import CheckpointSpec, checkpoint_specs, validate_checkpoint_specs
from src.train import load_adapter_checkpoint


DEFAULT_OUTPUT = ROOT / "runs" / "geodesic_mechanism_audit"
PRIMARY_KEYS = {
    *(f"native_r8_s{s}" for s in (533, 917, 1301)),
    *(f"released_r8_l0.02_s{s}" for s in (533, 917, 1301)),
    *(f"paper_r8_l0.02_s{s}" for s in (533, 917)),
}
DEPTHS = (0, 6, 16, 21, 22)
DEPTH_LABELS = ("embedding", "layer_6", "layer_16", "layer_21", "final_post_norm")
BOOTSTRAP_SEED = 20260902


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()


def json_default(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=json_default) + "\n", encoding="utf-8")


def write_jsonl_gz(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with gzip.open(path, "wt", encoding="utf-8", compresslevel=6) as handle:
        for row in rows:
            handle.write(json.dumps(row, default=json_default, separators=(",", ":")) + "\n")
            count += 1
    return count


class JsonlGzipWriter:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = gzip.open(path, "wt", encoding="utf-8", compresslevel=5)
        self.count = 0

    def write(self, row: dict) -> None:
        self.handle.write(json.dumps(row, default=json_default, separators=(",", ":")) + "\n")
        self.count += 1

    def close(self) -> None:
        self.handle.close()


def build_gold_examples(tokenizer, panel: Path) -> list:
    return [
        annotate_example(tokenizer, row, index, "prespecified_256")
        for index, row in enumerate(read_panel(panel))
    ]


def find_llama(model):
    candidates = [
        model,
        getattr(model, "base_model", None),
        getattr(getattr(model, "base_model", None), "model", None),
        getattr(getattr(getattr(model, "base_model", None), "model", None), "model", None),
    ]
    for candidate in candidates:
        if candidate is not None and hasattr(candidate, "layers") and hasattr(candidate, "norm"):
            return candidate
    for module in model.modules():
        if hasattr(module, "layers") and hasattr(module, "norm") and hasattr(module, "embed_tokens"):
            return module
    raise RuntimeError("could not locate the underlying LlamaModel")


class SelectedStateCapture:
    """Hooks exact states without `output_hidden_states=True` materialization."""

    def __init__(self, model):
        llama = find_llama(model)
        self.values: dict[str, torch.Tensor] = {}
        self.handles = []

        def capture(name):
            def hook(_module, _inputs, output):
                value = output[0] if isinstance(output, tuple) else output
                self.values[name] = value
            return hook

        self.handles.append(llama.embed_tokens.register_forward_hook(capture("embedding")))
        self.handles.append(llama.layers[5].register_forward_hook(capture("layer_6")))
        self.handles.append(llama.layers[15].register_forward_hook(capture("layer_16")))
        self.handles.append(llama.layers[20].register_forward_hook(capture("layer_21")))
        self.handles.append(llama.layers[-1].register_forward_hook(capture("post_last_pre_norm")))
        self.handles.append(llama.norm.register_forward_hook(capture("final_post_norm")))

    def clear(self):
        self.values.clear()

    def close(self):
        for handle in self.handles:
            handle.remove()


def spec_limit(spec: CheckpointSpec) -> int:
    return 256 if spec.key in PRIMARY_KEYS else 64


def load_model_for_spec(spec: CheckpointSpec, tokenizer, chemfm_vocab_size: int, device: str):
    model = load_lora_model(
        MODEL_DIR, tokenizer, chemfm_vocab_size=chemfm_vocab_size,
        attention_dropout=0.0, attn_implementation="sdpa",
        lora_rank=spec.rank, lora_alpha=spec.alpha,
    ).to(device)
    load_adapter_checkpoint(model, ROOT / spec.checkpoint)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def extract_gold(args) -> None:
    output = args.output.resolve()
    cache = output / "cache" / "gold_states"
    cache.mkdir(parents=True, exist_ok=True)
    tokenizer = load_reaction_tokenizer(TOKENIZER_DIR)
    chemfm_vocab_size = len(tokenizer)
    add_predictor_tokens(tokenizer)
    examples = build_gold_examples(tokenizer, args.panel.resolve())
    specs = checkpoint_specs()
    validate_checkpoint_specs(specs)
    if args.keys:
        wanted = set(args.keys.split(","))
        specs = [spec for spec in specs if spec.key in wanted]
        missing = wanted - {spec.key for spec in specs}
        if missing:
            raise ValueError(f"unknown checkpoint keys: {sorted(missing)}")
    metadata = {
        "type": "geodesic_audit_gold_state_cache", "git_commit": git_commit(),
        "panel": str(args.panel.resolve()), "panel_sha256": sha256(args.panel.resolve()),
        "device": args.device, "torch": torch.__version__, "python": platform.python_version(),
        "depths": dict(zip(DEPTH_LABELS, DEPTHS)), "checkpoints": [],
    }
    for index, spec in enumerate(specs, 1):
        destination = cache / f"{spec.key}.pt"
        if destination.exists() and not args.overwrite:
            print(json.dumps({"stage": "cache_reused", "key": spec.key}), flush=True)
            continue
        model = load_model_for_spec(spec, tokenizer, chemfm_vocab_size, args.device)
        capture = SelectedStateCapture(model)
        lm_weight = model.get_output_embeddings().weight.detach().float().cpu()
        records = []
        limit = spec_limit(spec)
        ordered = sorted(examples[:limit], key=lambda example: len(example.input_ids))
        start = time.perf_counter()
        if str(args.device).startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            for batch_start in range(0, len(ordered), args.batch_size):
                batch = ordered[batch_start:batch_start + args.batch_size]
                maximum = max(len(example.input_ids) for example in batch)
                ids = torch.zeros((len(batch), maximum), dtype=torch.long, device=args.device)
                mask = torch.zeros_like(ids, dtype=torch.bool)
                for row, example in enumerate(batch):
                    length = len(example.input_ids)
                    ids[row, :length] = torch.tensor(example.input_ids, device=args.device)
                    mask[row, :length] = True
                capture.clear()
                model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True)
                required = (*DEPTH_LABELS, "post_last_pre_norm")
                if any(name not in capture.values for name in required):
                    raise RuntimeError(f"missing capture: {set(required) - set(capture.values)}")
                for row, example in enumerate(batch):
                    length = len(example.input_ids)
                    source = [token.index for token in example.tokens if token.segment == "source"]
                    target = [token.index for token in example.tokens if token.segment == "target"]
                    records.append({
                        "panel_index": example.panel_index,
                        "reaction_identity": example.reaction_identity,
                        "source": example.source, "target": example.target,
                        "input_ids": torch.tensor(example.input_ids, dtype=torch.int32),
                        "source_indices": torch.tensor(source, dtype=torch.int16),
                        "target_indices": torch.tensor(target, dtype=torch.int16),
                        "states": torch.stack([
                            capture.values[name][row, :length].detach().to(torch.bfloat16).cpu()
                            for name in DEPTH_LABELS
                        ]),
                        "post_last_pre_norm": capture.values["post_last_pre_norm"][row, :length].detach().to(torch.bfloat16).cpu(),
                    })
                del ids, mask
        seconds = time.perf_counter() - start
        records.sort(key=lambda record: record["panel_index"])
        payload = {
            "spec": spec.__dict__, "depth_labels": DEPTH_LABELS,
            "records": records, "lm_head": lm_weight,
            "checkpoint_sha256": sha256(ROOT / spec.checkpoint / "USPTO-MIT-Synthesis" / "adapter_model.safetensors"),
        }
        torch.save(payload, destination)
        record = {
            "key": spec.key, "examples": limit, "seconds": seconds,
            "examples_per_second": limit / seconds,
            "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()) if str(args.device).startswith("cuda") else 0,
            "cache_bytes": destination.stat().st_size,
        }
        metadata["checkpoints"].append(record)
        print(json.dumps({"stage": "checkpoint_complete", "index": index, "total": len(specs), **record}), flush=True)
        capture.close()
        del capture, model, lm_weight, records, payload
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    write_json(output / "gold_extraction_metadata.json", metadata)


def trajectory_from_record(record: dict, layer_index: int, segment: str, device: str):
    states = record["states"][layer_index].float().to(device)
    source = record["source_indices"].long().tolist()
    target = record["target_indices"].long().tolist()
    if segment == "source":
        positions = [source[0] - 1, *source]
        return states[positions], positions
    if segment == "product":
        positions = [target[0] - 1, *target]
        return states[positions], positions
    if segment == "cross":
        source_path = states[[source[0] - 1, *source]]
        target_path = states[target] - states[target[0] - 1] + states[source[-1]]
        return torch.cat((source_path, target_path)), [source[0] - 1, *source, *target]
    raise ValueError(segment)


def aggregate_tube_rows(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["checkpoint"], row["layer"], row["segment"], row["span_length"])].append(row)
    output = []
    for key, values in groups.items():
        checkpoint, layer, segment, length = key
        result = {"checkpoint": checkpoint, "layer": layer, "segment": segment, "span_length": length, "reactions": len(values)}
        for metric in ("mean", "rms", "maximum", "p90", "p95", "monotonicity_violation", "fraction_gt_0.05", "fraction_gt_0.1", "fraction_gt_0.2", "fraction_gt_0.5"):
            data = np.asarray([value[metric] for value in values], dtype=np.float64)
            result[metric] = float(np.mean(data))
            result[f"{metric}_median"] = float(np.median(data))
        output.append(result)
    return sorted(output, key=lambda x: (x["checkpoint"], x["layer"], x["segment"], x["span_length"]))


def fisher_inner(first: torch.Tensor, second: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    first_mean = (p * first).sum(-1)
    second_mean = (p * second).sum(-1)
    return (p * first * second).sum(-1) - first_mean * second_mean


def functional_metrics(path: torch.Tensor, weight: torch.Tensor) -> dict:
    logits = path @ weight.T
    probabilities = logits.softmax(-1)
    euclidean_efficiency = float((path[-1] - path[0]).norm() / (path[1:] - path[:-1]).norm(dim=-1).sum().clamp_min(1e-12))
    fr_efficiency = float(fisher_rao_path_efficiency(probabilities))
    if len(path) > 2:
        before = logits[1:-1] - logits[:-2]
        after = logits[2:] - logits[1:-1]
        p = probabilities[1:-1]
        numerator = fisher_inner(before, after, p)
        denominator = (categorical_fisher_squared(before, p) * categorical_fisher_squared(after, p)).clamp_min(1e-20).sqrt()
        fisher_curvature = float((1 - numerator / denominator).mean())
    else:
        fisher_curvature = math.nan
    return {
        "euclidean_path_efficiency": euclidean_efficiency,
        "fisher_path_efficiency": fr_efficiency,
        "fisher_local_curvature": fisher_curvature,
        "probabilities": probabilities,
        "logits": logits,
    }


def deterministic_starts(n: int, length: int, maximum: int, salt: int) -> list[int]:
    count = n - length
    if count <= maximum:
        return list(range(count))
    rng = np.random.default_rng(BOOTSTRAP_SEED + salt * 1_000_003 + length)
    return sorted(rng.choice(count, maximum, replace=False).tolist())


def analyze_gold(args) -> None:
    output = args.output.resolve()
    cache_paths = sorted((output / "cache" / "gold_states").glob("*.pt"))
    if not cache_paths:
        raise FileNotFoundError("no gold-state caches; run extract-gold first")
    raw = output / "raw"
    writers = {
        "tube_reaction_rows": JsonlGzipWriter(raw / "tube_scale_space_by_reaction.jsonl.gz"),
        "trajectory_rows": JsonlGzipWriter(raw / "trajectory_metrics.jsonl.gz"),
        "persistence_rows": JsonlGzipWriter(raw / "tangent_persistence.jsonl.gz"),
        "turning_rows": JsonlGzipWriter(raw / "multiscale_turning.jsonl.gz"),
        "intervention_rows": JsonlGzipWriter(raw / "signal_noise_interventions.jsonl.gz"),
        "anatomy_rows": JsonlGzipWriter(raw / "released_objective_anatomy.jsonl.gz"),
    }
    tube_accumulators = defaultdict(lambda: defaultdict(float))
    start = time.perf_counter()
    for cache_index, path in enumerate(cache_paths, 1):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        spec = payload["spec"]
        key = spec["key"]
        weight = payload["lm_head"].float().to(args.device)
        for record in payload["records"]:
            for layer_index, label in enumerate(payload["depth_labels"]):
                for segment in ("source", "product", "cross"):
                    path_states, positions = trajectory_from_record(record, layer_index, segment, args.device)
                    tube = tube_scale_space(path_states)
                    for row in tube:
                        tube_row = {
                            "checkpoint": key, "rank": spec["rank"], "formulation": spec["formulation"],
                            "lambda": spec["stp_lambda"], "seed": spec["seed"],
                            "panel_index": record["panel_index"], "reaction_identity": record["reaction_identity"],
                            "layer": label, "segment": segment, **row,
                        }
                        writers["tube_reaction_rows"].write(tube_row)
                        group = tube_accumulators[(key, label, segment, row["span_length"])]
                        group["reactions"] += 1
                        for metric in ("mean", "rms", "maximum", "p90", "p95", "monotonicity_violation", "fraction_gt_0.05", "fraction_gt_0.1", "fraction_gt_0.2", "fraction_gt_0.5"):
                            group[metric] += row[metric]
                    acceleration = acceleration_decomposition(path_states)
                    trajectory = {
                        "checkpoint": key, "rank": spec["rank"], "formulation": spec["formulation"],
                        "lambda": spec["stp_lambda"], "seed": spec["seed"],
                        "panel_index": record["panel_index"], "reaction_identity": record["reaction_identity"],
                        "layer": label, "segment": segment, "tokens": len(path_states),
                        "speed": float(acceleration["speed"].mean()),
                        "tangential_acceleration": float(acceleration["acceleration_parallel"].mean()),
                        "normal_acceleration": float(acceleration["acceleration_normal"].mean()),
                        "normalized_normal_acceleration": float(acceleration["normalized_normal"].mean()),
                    }
                    if label == "final_post_norm":
                        functional = functional_metrics(path_states, weight)
                        trajectory.update({key_: value for key_, value in functional.items() if not torch.is_tensor(value)})
                    writers["trajectory_rows"].write(trajectory)
                    for row in tangent_autocorrelation(path_states):
                        writers["persistence_rows"].write({"checkpoint": key, "panel_index": record["panel_index"], "layer": label, "segment": segment, **row})
                    # Full (position,scale) heatmaps are retained for the first 64
                    # reactions; other rows contribute to reduced summaries.
                    if record["panel_index"] < 64:
                        for row in multiscale_turning(path_states):
                            writers["turning_rows"].write({"checkpoint": key, "panel_index": record["panel_index"], "layer": label, "segment": segment, **row})

                    if label != "final_post_norm" or segment == "cross":
                        continue
                    functional = functional_metrics(path_states, weight)
                    probabilities = functional["probabilities"]
                    ids = record["input_ids"].long().to(args.device)
                    maximum_length = len(path_states) - 1
                    for length in range(2, maximum_length + 1):
                        for s in deterministic_starts(len(path_states), length, 8, record["panel_index"]):
                            t = s + length
                            r = (s + t) // 2
                            alpha, q, rho = chord_coordinates(path_states[s], path_states[r], path_states[t])
                            u = path_states[r] - path_states[s]
                            chord = path_states[t] - path_states[s]
                            parallel = alpha * chord
                            original_position = positions[r]
                            if original_position + 1 >= len(ids):
                                continue
                            gold = int(ids[original_position + 1])
                            gradient, logits, _ = gold_logprob_gradient(path_states[r], weight, gold)
                            para_s = predictive_sensitivity(gradient, parallel)
                            perp_s = predictive_sensitivity(gradient, q)
                            base_topk = torch.topk(logits, min(10, logits.numel())).indices.tolist()
                            base_logp = float(logits.log_softmax(-1)[gold])
                            base_rank = int((logits > logits[gold]).sum()) + 1
                            competitor = torch.cat((logits[:gold], logits[gold + 1:])).max()
                            base_margin = float(logits[gold] - competitor)
                            base_entropy = float(-(logits.softmax(-1) * logits.log_softmax(-1)).sum())
                            fr_excess = float(fisher_rao_triangle_excess(probabilities[s], probabilities[r], probabilities[t]))
                            ray = float(optimal_ray_residual(path_states[s], path_states[r], path_states[t]))
                            common = {
                                "checkpoint": key, "panel_index": record["panel_index"], "reaction_identity": record["reaction_identity"],
                                "segment": segment, "span_length": length, "s": s, "r": r, "t": t,
                                "rho": float(rho), "alpha": float(alpha), "ray_residual": ray,
                                "fisher_triangle_excess": fr_excess,
                                "parallel_signed_sensitivity": float(para_s["signed"]),
                                "parallel_cosine_sensitivity": float(para_s["cosine"]),
                                "perpendicular_signed_sensitivity": float(perp_s["signed"]),
                                "perpendicular_cosine_sensitivity": float(perp_s["cosine"]),
                                "perpendicular_norm": float(q.norm()), "parallel_norm": float(parallel.norm()),
                                "base_gold_log_probability": base_logp, "base_gold_rank": base_rank,
                                "base_gold_margin": base_margin, "base_entropy": base_entropy,
                            }
                            for gamma in (.1, .25, .5):
                                for restore in (False, True):
                                    changed = curvature_removal_intervention(path_states[r], q, weight, gold, gamma, restore)
                                    writers["intervention_rows"].write({
                                        **common, "gamma": gamma, "norm_restored": restore,
                                        "delta_gold_log_probability": changed["gold_log_probability"] - base_logp,
                                        "delta_gold_rank": changed["gold_rank"] - base_rank,
                                        "delta_gold_margin": changed["gold_margin"] - base_margin,
                                        "delta_entropy": changed["entropy"] - base_entropy,
                                        "topk_changed": changed["topk"] != base_topk,
                                    })
                # Released-objective anatomy uses the exact framing-excluded path.
                if label == "final_post_norm":
                    cross, _ = trajectory_from_record(record, layer_index, "cross", args.device)
                    total = cross[-1] - cross[0]
                    for length in range(1, len(cross)):
                        for s in deterministic_starts(len(cross), length, 8, record["panel_index"] + 17):
                            t = s + length
                            if s == 0 and t == len(cross) - 1:
                                continue
                            patch = cross[t] - cross[s]
                            before = cross[s] - cross[0]
                            after = cross[-1] - cross[t]
                            values = released_objective_anatomy(before[None], patch[None], after[None])
                            writers["anatomy_rows"].write({
                                "checkpoint": key, "panel_index": record["panel_index"], "span_length": t - s,
                                **{name: float(value[0]) for name, value in values.items()},
                            })
        print(json.dumps({"stage": "gold_analysis_checkpoint", "index": cache_index, "total": len(cache_paths), "key": key}), flush=True)
        del payload, weight
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    aggregate = []
    for (checkpoint, layer, segment, length), values in tube_accumulators.items():
        count = values["reactions"]
        row = {"checkpoint": checkpoint, "layer": layer, "segment": segment, "span_length": length, "reactions": int(count)}
        for metric, total in values.items():
            if metric != "reactions":
                row[metric] = total / count
        aggregate.append(row)
    aggregate.sort(key=lambda x: (x["checkpoint"], x["layer"], x["segment"], x["span_length"]))
    change_points = []
    groups = defaultdict(list)
    for row in aggregate:
        groups[(row["checkpoint"], row["layer"], row["segment"])].append(row)
    for (checkpoint, layer, segment), values in groups.items():
        change_points.append({"checkpoint": checkpoint, "layer": layer, "segment": segment, **estimate_piecewise_change_point(values)})
    for writer in writers.values():
        writer.close()
    counts = {name: writer.count for name, writer in writers.items()}
    counts["tube_aggregate_rows"] = write_jsonl_gz(raw / "tube_scale_space_aggregate.jsonl.gz", aggregate)
    write_json(output / "analysis" / "tube_change_points.json", change_points)
    write_json(output / "gold_analysis_metadata.json", {"git_commit": git_commit(), "seconds": time.perf_counter() - start, "counts": counts})


def paired_checkpoint_keys(specs: list[CheckpointSpec]) -> list[tuple[str, str]]:
    available = {spec.key for spec in specs}
    pairs = []
    for treatment in specs:
        if treatment.formulation == "native":
            continue
        native = f"native_r{treatment.rank}_s{treatment.seed}"
        if native in available:
            pairs.append((native, treatment.key))
    return pairs


def analyze_matched(args) -> None:
    cache = args.output.resolve() / "cache" / "gold_states"
    specs = checkpoint_specs()
    rows = []
    for native_key, treatment_key in paired_checkpoint_keys(specs):
        native_path, treatment_path = cache / f"{native_key}.pt", cache / f"{treatment_key}.pt"
        if not native_path.exists() or not treatment_path.exists():
            continue
        native = torch.load(native_path, map_location="cpu", weights_only=False)
        treatment = torch.load(treatment_path, map_location="cpu", weights_only=False)
        native_records = {r["panel_index"]: r for r in native["records"]}
        for tr in treatment["records"]:
            nr = native_records[tr["panel_index"]]
            limit = min(len(nr["states"]), len(tr["states"]))
            for layer_index in range(limit):
                label = native["depth_labels"][layer_index]
                for segment in ("source", "product", "cross"):
                    n_path, _ = trajectory_from_record(nr, layer_index, segment, args.device)
                    t_path, _ = trajectory_from_record(tr, layer_index, segment, args.device)
                    n = min(len(n_path), len(t_path))
                    for length in range(2, n):
                        for s in deterministic_starts(n, length, 8, nr["panel_index"]):
                            t = s + length
                            r = (s + t) // 2
                            result = matched_geodesic_displacement(
                                (n_path[s], n_path[r], n_path[t]),
                                (t_path[s], t_path[r], t_path[t]),
                            )
                            rows.append({
                                "native": native_key, "treatment": treatment_key,
                                "panel_index": nr["panel_index"], "reaction_identity": nr["reaction_identity"],
                                "layer": label, "segment": segment, "span_length": length,
                                **{name: float(value) for name, value in result.items()},
                            })
        print(json.dumps({"stage": "matched_pair_complete", "native": native_key, "treatment": treatment_key}), flush=True)
    count = write_jsonl_gz(args.output.resolve() / "raw" / "matched_native_stp_displacement.jsonl.gz", rows)
    write_json(args.output.resolve() / "matched_analysis_metadata.json", {"git_commit": git_commit(), "rows": count})


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("extract-gold", "analyze-gold", "analyze-matched"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--keys", default="")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.command == "extract-gold":
        extract_gold(arguments)
    elif arguments.command == "analyze-gold":
        analyze_gold(arguments)
    else:
        analyze_matched(arguments)
