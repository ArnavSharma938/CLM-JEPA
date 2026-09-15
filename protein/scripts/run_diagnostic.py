#!/usr/bin/env python
"""Frozen RITA-M audit, extraction, and core geometry stages."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.constants import *
from src.geometry import adjacent_tangent_correlation, attenuate_orthogonal, local_curvature, path_efficiency, tube_radius
from src.modeling import capture_layer, load_rita, tied_weight_audit
from src.stats import cluster_bootstrap_mean, sign_flip_pvalue
from src.stp import released_stp_loss
from src.tokenization import collate_encoded, encode_canonical


def json_dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def environment() -> dict:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT.parent, text=True
        ).strip()
    except Exception:
        commit = None
    return {
        "python": sys.version, "platform": platform.platform(),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "repository_commit": commit,
    }


def audit(output: Path) -> None:
    loaded = load_rita(device="cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = loaded.model, loaded.tokenizer
    sample = encode_canonical(tokenizer, CANONICAL_AA)
    config = model.config.to_dict()
    result = {
        "model_id": RITA_MODEL_ID, "model_revision": RITA_REVISION,
        "tokenizer_revision": RITA_TOKENIZER_REVISION,
        "canonical_sequence": CANONICAL_AA, "canonical_ids": sample.input_ids,
        "canonical_tokens": tokenizer.convert_ids_to_tokens(sample.input_ids),
        "residue_token_ratio": len(sample.residue_positions) / len(CANONICAL_AA),
        "encoded_tokens_per_residue_including_eos": len(sample.input_ids) / len(CANONICAL_AA),
        "tokenizer_attributes": {
            "bos_token_id": tokenizer.bos_token_id, "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id, "special_tokens_map": tokenizer.special_tokens_map,
        },
        "model_config": {key: config.get(key) for key in (
            "vocab_size", "d_model", "num_layers", "num_heads", "d_feedforward",
            "max_seq_len", "eos_token_id", "pad_token_id", "tie_word_embeddings",
        )},
        "weights": tied_weight_audit(model),
        "official_fitness_contract": {
            "directions": ["forward", "reversed"],
            "aggregation": "sum of negative mean cross-entropies",
            "shift": "encode(direction)[:-1] predicts encode(direction)[1:]",
            "first_residue_scored": False, "terminal_eos_scored": True,
            "source": RITA_FITNESS_SOURCE,
        },
        "environment": environment(),
    }
    json_dump(output, result)


def read_manifest(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def reverse_manifest(source: Path, output: Path, limit: int = 128) -> None:
    rows = sorted(read_manifest(source), key=lambda row: row["id"])[:limit]
    reversed_rows = []
    per_residue = {"ss3", "structure_valid", "contact_valid", "long_range_contact_density"}
    for row in rows:
        changed = dict(row)
        changed["id"] = "reverse__" + row["id"]
        changed["paired_forward_id"] = row["id"]
        changed["sequence"] = row["sequence"][::-1]
        for key in per_residue:
            if key in changed:
                changed[key] = list(reversed(changed[key]))
        reversed_rows.append(changed)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in reversed_rows),
        encoding="utf-8", newline="\n",
    )


@torch.inference_mode()
def extract(
    manifest: Path, out_dir: Path, batch_size: int = 4,
    intermediate_layer: int = 11, limit: int | None = None,
) -> None:
    rows = read_manifest(manifest)
    if limit is not None:
        rows = rows[:limit]
    loaded = load_rita()
    model, tokenizer = loaded.model, loaded.tokenizer
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    index_rows = []
    # Length sorting only changes batch packing; IDs retain manifest provenance.
    rows = sorted(rows, key=lambda row: (row["length"], row["id"]))
    for shard_index in range(0, len(rows), batch_size):
        batch_rows = rows[shard_index: shard_index + batch_size]
        encoded = [encode_canonical(tokenizer, row["sequence"]) for row in batch_rows]
        batch = {key: value.cuda() for key, value in collate_encoded(encoded).items() if key != "residue_mask"}
        with capture_layer(model, intermediate_layer) as captured:
            outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        if len(captured) != 1:
            raise RuntimeError("intermediate layer hook did not fire exactly once")
        final = outputs.hidden_states
        payload = []
        for local, row in enumerate(batch_rows):
            length = len(row["sequence"])
            payload.append({
                "id": row["id"], "sequence": row["sequence"], "split": row.get("split"),
                "cluster_id": row.get("cluster_id", row["id"]),
                "annotations": {key: value for key, value in row.items() if key not in {
                    "id", "sequence", "split", "cluster_id", "description", "length"
                }},
                "input_ids": batch["input_ids"][local, :length + 1].cpu(),
                "final": final[local, :length].half().cpu(),
                "intermediate": captured[0][local, :length].half().cpu(),
            })
        name = f"shard_{shard_index // batch_size:05d}.pt"
        torch.save(payload, out_dir / name)
        for local, row in enumerate(batch_rows):
            index_rows.append({"id": row["id"], "shard": name, "offset": local})
    json_dump(out_dir / "manifest.json", {
        "source_manifest": str(manifest), "source_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "model_revision": RITA_REVISION, "intermediate_layer_zero_based": intermediate_layer,
        "count": len(rows), "runtime_seconds": time.perf_counter() - start,
        "peak_vram_bytes": torch.cuda.max_memory_allocated(), "environment": environment(),
        "index": index_rows,
    })


def iter_cache(cache_dir: Path):
    for path in sorted(cache_dir.glob("shard_*.pt")):
        yield from torch.load(path, map_location="cpu", weights_only=False)


def geometry(cache_dir: Path, output: Path, reverse: bool = False) -> None:
    records = []
    span_bins = ((4, 8), (9, 24), (25, 64))
    for item in iter_cache(cache_dir):
        states = item["final"].float()
        if reverse:
            # Reverse control requires a separate forward extraction; reversing
            # cached states would be invalid and is intentionally prohibited.
            raise ValueError("extract reversed sequences through the model before geometry")
        curvature = local_curvature(states)
        record = {
            "id": item["id"], "cluster_id": item["cluster_id"], "split": item["split"],
            "length": states.shape[0], "curvature_mean": float(curvature.mean()),
            "tangent_c1": float(adjacent_tangent_correlation(states).mean()),
        }
        for low, high in span_bins:
            radii, efficiencies = [], []
            for start in range(0, states.shape[0] - low, max(1, high // 2)):
                end = min(start + high, states.shape[0] - 1)
                if end - start < low:
                    continue
                segment = states[start:end + 1]
                radii.append(float(tube_radius(segment).mean()))
                efficiencies.append(float(path_efficiency(segment)))
            record[f"tube_{low}_{high}"] = float(np.mean(radii)) if radii else None
            record[f"efficiency_{low}_{high}"] = float(np.mean(efficiencies)) if efficiencies else None
        records.append(record)
    json_dump(output, {"records": records, "span_bins": span_bins, "unit": "protein"})


def _local_window_metrics(states: torch.Tensor, radius: int = 4):
    curvature = local_curvature(states)
    result = []
    for center in range(1, states.shape[0] - 1):
        start, end = max(0, center - radius), min(states.shape[0], center + radius + 1)
        segment = states[start:end]
        result.append({
            "position": center,
            "curvature": float(curvature[center - 1]),
            "tangent_c1": float(1.0 - curvature[center - 1]),
            "tube_local": float(tube_radius(segment).mean()) if segment.shape[0] >= 3 else None,
            "efficiency_local": float(path_efficiency(segment)),
        })
    return result


def stratify(cache_dir: Path, output: Path) -> None:
    protein_strata = []
    for item in iter_cache(cache_dir):
        states = item["final"].float()
        local = _local_window_metrics(states)
        annotations = item.get("annotations", {})
        buckets: dict[str, list[dict]] = {}
        if "ss3" in annotations:
            labels = annotations["ss3"]
            valid = annotations["structure_valid"]
            names = {0: "helix", 1: "beta", 2: "coil"}
            for row in local:
                p = row["position"]
                if not valid[p] or labels[p] not in names:
                    continue
                buckets.setdefault(names[labels[p]], []).append(row)
                transition = labels[p - 1] != labels[p] or labels[p] != labels[p + 1]
                buckets.setdefault("ss_transition" if transition else "ss_stable", []).append(row)
        if "long_range_contact_density" in annotations:
            values = np.asarray([
                np.nan if value is None else value
                for value in annotations["long_range_contact_density"]
            ], dtype=float)
            valid_values = values[np.isfinite(values)]
            if valid_values.size:
                threshold = float(np.median(valid_values))
                for row in local:
                    p = row["position"]
                    if np.isfinite(values[p]):
                        name = "contact_high" if values[p] > threshold else "contact_low"
                        buckets.setdefault(name, []).append(row)
        for name, rows in buckets.items():
            if len(rows) < 3:
                continue
            protein_strata.append({
                "id": item["id"], "cluster_id": item["cluster_id"], "stratum": name,
                "positions": len(rows),
                **{metric: float(np.mean([row[metric] for row in rows])) for metric in (
                    "curvature", "tangent_c1", "tube_local", "efficiency_local"
                )},
            })
    summaries = {}
    for stratum in sorted({row["stratum"] for row in protein_strata}):
        chosen = [row for row in protein_strata if row["stratum"] == stratum]
        summaries[stratum] = {"proteins": len(chosen)}
        for metric in ("curvature", "tangent_c1", "tube_local", "efficiency_local"):
            estimate, interval = cluster_bootstrap_mean(
                [row[metric] for row in chosen], [row["cluster_id"] for row in chosen], seed=20260914
            )
            summaries[stratum][metric] = {"mean": estimate, "ci95": interval}
    comparisons = {}
    pairs = (
        ("helix", "coil"), ("beta", "coil"),
        ("ss_transition", "ss_stable"), ("contact_high", "contact_low"),
    )
    p_cells = []
    by_key = {(row["id"], row["stratum"]): row for row in protein_strata}
    for left, right in pairs:
        common = sorted(
            {row["id"] for row in protein_strata if row["stratum"] == left}
            & {row["id"] for row in protein_strata if row["stratum"] == right}
        )
        name = f"{left}_minus_{right}"
        comparisons[name] = {"paired_proteins": len(common)}
        for metric in ("curvature", "tangent_c1", "tube_local", "efficiency_local"):
            differences = [by_key[(identifier, left)][metric] - by_key[(identifier, right)][metric] for identifier in common]
            mean, interval = cluster_bootstrap_mean(differences, common, seed=20260914)
            pvalue = sign_flip_pvalue(differences, seed=20260914)
            comparisons[name][metric] = {"mean_difference": mean, "ci95": interval, "sign_flip_p": pvalue}
            p_cells.append((pvalue, name, metric))
    # Benjamini-Hochberg across the disclosed family of 16 stratum contrasts.
    ordered = sorted(enumerate(p_cells), key=lambda pair: pair[1][0])
    adjusted = [1.0] * len(p_cells)
    running = 1.0
    for rank_index in range(len(ordered) - 1, -1, -1):
        original_index, (pvalue, _name, _metric) = ordered[rank_index]
        rank = rank_index + 1
        running = min(running, pvalue * len(ordered) / rank)
        adjusted[original_index] = min(1.0, running)
    for qvalue, (_pvalue, name, metric) in zip(adjusted, p_cells):
        comparisons[name][metric]["bh_q"] = qvalue
    json_dump(output, {"protein_strata": protein_strata, "summaries": summaries, "paired_comparisons": comparisons})


def _margin(logits: torch.Tensor, gold: int) -> torch.Tensor:
    competitor = torch.cat((logits[:gold], logits[gold + 1:])).max()
    return logits[gold] - competitor


@torch.inference_mode()
def orthogonal_audit(cache_dir: Path, output: Path, fraction: float = 0.10) -> None:
    loaded = load_rita()
    head = loaded.model.get_output_embeddings().weight.detach().float().cpu()
    records = []
    for item in iter_cache(cache_dir):
        states = item["final"].float()
        ids = item["input_ids"]
        changes_logp, changes_margin = [], []
        # At most 24 evenly spread decisions per protein; proteins are the unit.
        positions = np.linspace(8, states.shape[0] - 10, num=min(24, max(0, states.shape[0] - 17)), dtype=int)
        for position in np.unique(positions):
            start, end = position - 8, position + 9
            segment = states[start:end]
            changed = attenuate_orthogonal(segment, fraction=fraction, restore_norm=True)[8]
            original_logits = states[position] @ head.T
            changed_logits = changed @ head.T
            gold = int(ids[position + 1])
            original_logp = torch.log_softmax(original_logits, -1)[gold]
            changed_logp = torch.log_softmax(changed_logits, -1)[gold]
            changes_logp.append(float(changed_logp - original_logp))
            changes_margin.append(float(_margin(changed_logits, gold) - _margin(original_logits, gold)))
        if changes_logp:
            records.append({
                "id": item["id"], "cluster_id": item["cluster_id"],
                "n": len(changes_logp), "delta_gold_logp": float(np.mean(changes_logp)),
                "delta_gold_margin": float(np.mean(changes_margin)),
                "fraction_logp_harmed": float(np.mean(np.asarray(changes_logp) < 0)),
                "fraction_margin_harmed": float(np.mean(np.asarray(changes_margin) < 0)),
            })
    summary = {}
    for metric in ("delta_gold_logp", "delta_gold_margin"):
        values = [row[metric] for row in records]
        clusters = [row["cluster_id"] for row in records]
        mean, interval = cluster_bootstrap_mean(values, clusters, seed=20260914)
        summary[metric] = {"mean": mean, "ci95": interval, "sign_flip_p": sign_flip_pvalue(values, seed=20260914)}
    json_dump(output, {"fraction_attenuated": fraction, "records": records, "summary": summary})


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="stage", required=True)
    p = sub.add_parser("audit"); p.add_argument("--output", type=Path, default=ROOT / "runs/audit/audit.json")
    p = sub.add_parser("extract"); p.add_argument("manifest", type=Path); p.add_argument("out_dir", type=Path); p.add_argument("--batch-size", type=int, default=4); p.add_argument("--limit", type=int)
    p = sub.add_parser("geometry"); p.add_argument("cache_dir", type=Path); p.add_argument("output", type=Path)
    p = sub.add_parser("reverse-manifest"); p.add_argument("source", type=Path); p.add_argument("output", type=Path); p.add_argument("--limit", type=int, default=128)
    p = sub.add_parser("stratify"); p.add_argument("cache_dir", type=Path); p.add_argument("output", type=Path)
    p = sub.add_parser("orthogonal"); p.add_argument("cache_dir", type=Path); p.add_argument("output", type=Path); p.add_argument("--fraction", type=float, default=.10)
    args = parser.parse_args()
    if args.stage == "audit": audit(args.output)
    elif args.stage == "extract": extract(args.manifest, args.out_dir, args.batch_size, limit=args.limit)
    elif args.stage == "geometry": geometry(args.cache_dir, args.output)
    elif args.stage == "reverse-manifest": reverse_manifest(args.source, args.output, args.limit)
    elif args.stage == "stratify": stratify(args.cache_dir, args.output)
    elif args.stage == "orthogonal": orthogonal_audit(args.cache_dir, args.output, args.fraction)


if __name__ == "__main__":
    main()
