#!/usr/bin/env python
"""Execute the preregistered frozen latent-predictability audit.

Stages are intentionally resumable.  Nothing in this runner trains or changes
ChemFM; ``fit-probes`` trains only diagnostic predictors over frozen tensors.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chemfm import MODEL_DIR, TOKENIZER_DIR, load_lora_model, load_reaction_tokenizer  # noqa: E402
from frozen_geometry import MOTIF_QUERIES, annotate_example, _atom_spans  # noqa: E402
from jepa import add_predictor_tokens  # noqa: E402
from latent_predictability import (  # noqa: E402
    HORIZONS, LAYERS, ResidualMLPProbe, RidgeProbe, Standardizer, TargetBasis,
    assert_disjoint_confirmation, canonical_atom_correspondence,
    chemical_pair_id,
    decoder_distribution_metrics, deterministic_random_smiles, fit_probe,
    forecast_matrices, invariance_metrics, latent_metrics, locked_reaction_split,
    reaction_balanced_indices, sha256_file, suffix_replay_one_position,
    shuffled_reaction_targets,
)
from stp_representation_analysis import checkpoint_specs  # noqa: E402
from train import load_adapter_checkpoint  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from run_geodesic_audit import SelectedStateCapture, _candidate_workload, find_llama  # noqa: E402


DEFAULT_PANEL = ROOT / "data/clm_jepa_uspto_mit_validation_1024/uspto_mit_validation_1024.csv"
DEFAULT_OUTPUT = ROOT / "runs/latent_predictability_audit"
DEFAULT_SPLIT = ROOT / "data/clm_jepa_uspto_mit_latent_audit/splits.json"
PRIMARY_KEYS = {
    *(f"native_r8_s{seed}" for seed in (533, 917, 1301)),
    *(f"released_r8_l0.02_s{seed}" for seed in (533, 917, 1301)),
    *(f"paper_r8_l0.02_s{seed}" for seed in (533, 917)),
}
VIEW_SEEDS = (202609041, 202609042, 202609043, 202609044)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True) + "\n")


def read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1024:
        raise RuntimeError(f"locked probe panel must contain 1024 reactions, got {len(rows)}")
    identities = [row["reaction_identity"] for row in rows]
    if len(set(identities)) != len(rows):
        raise RuntimeError("probe panel contains duplicate reactions")
    return rows


def load_split_manifest(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("type") != "latent_predictability_reaction_splits":
        raise ValueError("wrong split-manifest type")
    return value


def recheck_confirmation(args) -> None:
    split = load_split_manifest(args.split_manifest)
    assert_disjoint_confirmation(
        [row["chemical_pair_id"] for row in split["records"]],
        args.confirmation_manifest,
    )


def lock_splits(args) -> None:
    rows = read_rows(args.panel)
    assignment = locked_reaction_split([row["reaction_identity"] for row in rows])
    pair_by_identity = {
        row["reaction_identity"]: chemical_pair_id(row["source_identity"], row["target_identity"])
        for row in rows
    }
    exclusion = assert_disjoint_confirmation(pair_by_identity.values(), args.confirmation_manifest)
    records = [
        {"reaction_identity": key, "chemical_pair_id": pair_by_identity[key], "split": assignment[key]}
        for key in sorted(assignment)
    ]
    value = {
        "type": "latent_predictability_reaction_splits",
        "salt": "latent-decoder-audit-v1",
        "panel": str(args.panel.resolve()),
        "panel_sha256": sha256_file(args.panel),
        "counts": {name: sum(row["split"] == name for row in records) for name in ("train", "validation", "test")},
        "confirmation_exclusion": exclusion,
        "records": records,
    }
    if args.split_manifest.exists():
        existing = json.loads(args.split_manifest.read_text(encoding="utf-8"))
        if existing != value:
            raise ValueError("locked latent-audit split changed")
    write_json(args.split_manifest, value)
    print(json.dumps({"stage": "splits_locked", "counts": value["counts"]}), flush=True)


def selected_specs(keys: str | None = None):
    specs = [spec for spec in checkpoint_specs() if spec.key in PRIMARY_KEYS]
    if keys:
        wanted = {item.strip() for item in keys.split(",") if item.strip()}
        specs = [spec for spec in specs if spec.key in wanted]
        if wanted != {spec.key for spec in specs}:
            raise ValueError(f"unknown/non-primary keys: {sorted(wanted - {spec.key for spec in specs})}")
    for spec in specs:
        config_path = ROOT / spec.checkpoint / "USPTO-MIT-Synthesis/adapter_config.json"
        if not config_path.exists():
            raise FileNotFoundError(config_path)
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if int(config["r"]) != 8 or int(config["lora_alpha"]) != 8:
            raise ValueError(f"rank/alpha mismatch for {spec.key}")
    return sorted(specs, key=lambda spec: spec.key)


def token_metadata(example) -> dict[str, dict]:
    return {
        str(token.index): {
            "token_class": token.token_class,
            "component": token.component,
            "segment_rank": token.segment_rank,
            "events": sorted(token.events),
            "event_details": dict(token.details),
            "around_reaction_center": "reaction_center" in token.events,
        }
        for token in example.tokens if token.segment != "marker"
    }


def build_examples(tokenizer, rows: list[dict], split_assignment: dict[str, str]) -> list[tuple[object, str]]:
    result = []
    for index, row in enumerate(rows):
        adapted = {
            "reaction_identity": row["reaction_identity"],
            "canonical_source": row["source"],
            "canonical_target": row["target"],
        }
        result.append((annotate_example(tokenizer, adapted, index, "latent_probe_1024"), split_assignment[row["reaction_identity"]]))
    return result


def extract_records(model, capture, examples, device: str, batch_size: int) -> list[dict]:
    records = []
    ordered = sorted(examples, key=lambda item: len(item[0].input_ids))
    with torch.inference_mode():
        for start in range(0, len(ordered), batch_size):
            batch = ordered[start:start + batch_size]
            maximum = max(len(example.input_ids) for example, _ in batch)
            ids = torch.zeros((len(batch), maximum), dtype=torch.long, device=device)
            mask = torch.zeros_like(ids, dtype=torch.bool)
            for row_index, (example, _) in enumerate(batch):
                length = len(example.input_ids)
                ids[row_index, :length] = torch.tensor(example.input_ids, device=device)
                mask[row_index, :length] = True
            capture.clear()
            model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True)
            for row_index, (example, split) in enumerate(batch):
                length = len(example.input_ids)
                records.append({
                    "panel_index": example.panel_index,
                    "reaction_identity": example.reaction_identity,
                    "split": split,
                    "source": example.source,
                    "target": example.target,
                    "input_ids": torch.tensor(example.input_ids, dtype=torch.int32),
                    "source_indices": [token.index for token in example.tokens if token.segment == "source"],
                    "product_indices": [token.index for token in example.tokens if token.segment == "target"],
                    "token_metadata": token_metadata(example),
                    "reaction_center_metadata": example.reaction_center_metadata,
                    "states": {
                        name: capture.values[name][row_index, :length].to(torch.bfloat16).cpu()
                        for name in LAYERS
                    },
                })
    return sorted(records, key=lambda row: row["panel_index"])


def extract(args) -> None:
    split_manifest = load_split_manifest(args.split_manifest)
    assignment = {row["reaction_identity"]: row["split"] for row in split_manifest["records"]}
    # Recheck at every expensive entrypoint, not only when the split file is made.
    assert_disjoint_confirmation(
        [row["chemical_pair_id"] for row in split_manifest["records"]],
        args.confirmation_manifest,
    )
    rows = read_rows(args.panel)
    tokenizer = load_reaction_tokenizer(TOKENIZER_DIR)
    chemfm_vocab = len(tokenizer)
    add_predictor_tokens(tokenizer)
    examples = build_examples(tokenizer, rows, assignment)
    specs = selected_specs(args.keys)
    model = load_lora_model(
        MODEL_DIR, tokenizer, chemfm_vocab_size=chemfm_vocab,
        attention_dropout=0.0, attn_implementation="sdpa", lora_rank=8, lora_alpha=8,
    ).to(args.device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    capture = SelectedStateCapture(model)
    metadata = []
    for index, spec in enumerate(specs, 1):
        destination = args.output / "cache/canonical" / f"{spec.key}.pt"
        if destination.exists() and not args.overwrite:
            print(json.dumps({"stage": "cache_reused", "key": spec.key}), flush=True)
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        load_adapter_checkpoint(model, ROOT / spec.checkpoint)
        before = time.perf_counter()
        if args.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        records = extract_records(model, capture, examples, args.device, args.batch_size)
        payload = {
            "type": "latent_predictability_state_cache",
            "spec": spec.__dict__, "layers": LAYERS, "records": records,
            "checkpoint_sha256": sha256_file(ROOT / spec.checkpoint / "USPTO-MIT-Synthesis/adapter_model.safetensors"),
            "panel_sha256": sha256_file(args.panel),
        }
        torch.save(payload, destination)
        row = {
            "key": spec.key, "seconds": time.perf_counter() - before,
            "bytes": destination.stat().st_size,
            "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()) if args.device.startswith("cuda") else 0,
        }
        metadata.append(row)
        print(json.dumps({"stage": "checkpoint_extracted", "index": index, "total": len(specs), **row}), flush=True)
        del records, payload
        gc.collect()
    capture.close()
    write_json(args.output / "extraction.json", {
        "type": "latent_predictability_extraction", "torch": torch.__version__,
        "python": platform.python_version(), "checkpoints": metadata,
    })


def predict_batches(model, values: torch.Tensor, device: str, batch_size: int) -> torch.Tensor:
    model.eval()
    output = []
    with torch.inference_mode():
        for start in range(0, len(values), batch_size):
            output.append(model(values[start:start + batch_size].to(device)).float().cpu())
    return torch.cat(output) if output else torch.empty((0, 0))


def load_probe_predictions(path: Path, values: torch.Tensor, kind: str, device: str, batch_size: int):
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    standardizer, basis = artifact["standardizer"], artifact["basis"]
    standardized = standardizer(values)
    ridge = RidgeProbe(artifact["input_size"], artifact["output_size"])
    ridge.load_state_dict(artifact["ridge_state"])
    if kind == "ridge":
        model = ridge.to(device)
    elif kind == "residual_mlp":
        model = ResidualMLPProbe(
            ridge, artifact["input_size"], artifact["output_size"], artifact["mlp_width"],
        )
        model.load_state_dict(artifact["mlp_state"])
        model = model.to(device)
    elif kind == "constant":
        return basis.mean.expand(len(values), -1).clone(), artifact
    else:
        raise ValueError(kind)
    return basis.decode(predict_batches(model, standardized, device, batch_size)), artifact


def support_flags(metadata: list[dict]) -> dict[str, np.ndarray]:
    return {
        "arbitrary": np.ones(len(metadata), dtype=bool),
        "common_k8_eligible": np.asarray([
            int(row.get("current_segment_rank", -1)) <= int(row.get("segment_length", 0)) - 9
            for row in metadata
        ]),
        "atom_to_atom": np.asarray([bool(row.get("atom_to_atom")) for row in metadata]),
        "event_to_next_event": np.asarray([bool(row.get("event_to_next_event")) for row in metadata]),
        "component_boundary": np.asarray([bool(row.get("component_boundary")) for row in metadata]),
        "reaction_center_window": np.asarray([bool(row.get("around_reaction_center")) for row in metadata]),
    }


def fit_probes(args) -> None:
    assert_disjoint_confirmation(
        [row["chemical_pair_id"] for row in load_split_manifest(args.split_manifest)["records"]],
        args.confirmation_manifest,
    )
    result_path = args.output / "raw/probe_metrics.jsonl"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    completed = set()
    if args.resume and result_path.exists():
        completed = {
            json.loads(line)["key"] for line in result_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    handle = result_path.open("a" if args.resume else "w", encoding="utf-8")
    device = args.device
    for spec in selected_specs(args.keys):
        payload = torch.load(args.output / "cache/canonical" / f"{spec.key}.pt", map_location="cpu", weights_only=False)
        records_by_split = {
            split: [row for row in payload["records"] if row["split"] == split]
            for split in ("train", "validation", "test")
        }
        basis_cache = {}
        for layer in LAYERS:
            for segment in ("source", "product"):
                for horizon in HORIZONS:
                    for mode in ("current", "history"):
                        key = f"{spec.key}__{layer}__{segment}__k{horizon}__{mode}"
                        if key in completed:
                            print(json.dumps({"stage": "probe_reused", "key": key}), flush=True)
                            continue
                        matrices = {
                            split: forecast_matrices(records_by_split[split], layer, segment, horizon, mode)
                            for split in records_by_split
                        }
                        if min(len(matrices[name][0]) for name in matrices) == 0:
                            continue
                        tx, ty, tm = matrices["train"]
                        chosen = reaction_balanced_indices(tm, args.train_positions, args.probe_seed)
                        tx, ty = tx[chosen], ty[chosen]
                        vx, vy, _ = matrices["validation"]
                        ex, ey, em = matrices["test"]
                        standardizer = Standardizer.fit(tx)
                        txs, vxs, exs = standardizer(tx), standardizer(vx), standardizer(ex)
                        basis_key = (layer, segment, horizon)
                        if basis_key not in basis_cache:
                            basis_cache[basis_key] = TargetBasis.fit(
                                ty, args.pca_rank, seed=args.probe_seed + horizon
                            )
                        basis = basis_cache[basis_key]
                        tys, vys = basis.encode(ty), basis.encode(vy)
                        ridge = RidgeProbe(txs.shape[1], tys.shape[1]).to(device)
                        ridge, ridge_fit = fit_probe(
                            ridge, txs, tys, vxs, vys, weight_decay=args.ridge,
                            epochs=args.epochs, batch_size=args.probe_batch_size, seed=args.probe_seed,
                        )
                        ridge_scores = predict_batches(ridge, exs, device, args.probe_batch_size)
                        ridge_prediction = basis.decode(ridge_scores)
                        mlp = ResidualMLPProbe(ridge, txs.shape[1], tys.shape[1], args.mlp_width).to(device)
                        mlp, mlp_fit = fit_probe(
                            mlp, txs, tys, vxs, vys, weight_decay=args.mlp_decay,
                            epochs=args.epochs, batch_size=args.probe_batch_size, seed=args.probe_seed,
                        )
                        mlp_prediction = basis.decode(predict_batches(mlp, exs, device, args.probe_batch_size))
                        mlp_seed_robustness = []
                        if layer == "final_post_norm" and segment == "product":
                            for extra_seed in (args.probe_seed + 1, args.probe_seed + 2):
                                replicate = ResidualMLPProbe(
                                    ridge, txs.shape[1], tys.shape[1], args.mlp_width
                                ).to(device)
                                replicate, replicate_fit = fit_probe(
                                    replicate, txs, tys, vxs, vys,
                                    weight_decay=args.mlp_decay, epochs=args.epochs,
                                    batch_size=args.probe_batch_size, seed=extra_seed,
                                )
                                replicate_prediction = basis.decode(
                                    predict_batches(replicate, exs, device, args.probe_batch_size)
                                )
                                mlp_seed_robustness.append({
                                    "seed": extra_seed,
                                    "fit": replicate_fit,
                                    "metrics": latent_metrics(ey, replicate_prediction, basis.mean),
                                })
                                del replicate, replicate_prediction
                        constant = basis.mean.expand_as(ey)
                        predictions = {"constant": constant, "ridge": ridge_prediction, "residual_mlp": mlp_prediction}
                        flags = support_flags(em)
                        shuffled_y, shuffled_donors = shuffled_reaction_targets(
                            ey, em, args.probe_seed + horizon,
                        )
                        metrics = {}
                        for name, prediction in predictions.items():
                            metrics[name] = {}
                            for support, mask_np in flags.items():
                                mask = torch.from_numpy(mask_np)
                                if mask.any():
                                    chosen_ids = [
                                        em[index]["reaction_identity"] for index in np.flatnonzero(mask_np)
                                    ]
                                    metrics[name][support] = latent_metrics(
                                        ey[mask], prediction[mask], basis.mean, chosen_ids
                                    )
                            metrics[name]["shuffled_reaction"] = latent_metrics(
                                shuffled_y, prediction, basis.mean,
                            )
                        metrics["nonlinear_r2_improvement"] = (
                            metrics["residual_mlp"]["arbitrary"]["r2"] - metrics["ridge"]["arbitrary"]["r2"]
                        )
                        artifact = {
                            "key": key, "spec": spec.__dict__, "layer": layer, "segment": segment,
                            "horizon": horizon, "mode": mode, "basis": basis,
                            "standardizer": standardizer, "ridge_state": {k: v.cpu() for k,v in ridge.state_dict().items()},
                            "mlp_state": {k: v.cpu() for k,v in mlp.state_dict().items()},
                            "input_size": txs.shape[1], "output_size": tys.shape[1], "mlp_width": args.mlp_width,
                        }
                        probe_path = args.output / "probes" / f"{key}.pt"
                        probe_path.parent.mkdir(parents=True, exist_ok=True)
                        torch.save(artifact, probe_path)
                        record = {
                            "key": key, "checkpoint": spec.key, "layer": layer, "segment": segment,
                            "horizon": horizon, "mode": mode, "pca_variance_coverage": basis.variance_coverage,
                            "train_positions": len(tx), "validation_positions": len(vx), "test_positions": len(ex),
                            "ridge_fit": ridge_fit, "mlp_fit": mlp_fit, "metrics": metrics,
                            "mlp_seed_robustness": mlp_seed_robustness,
                            "shuffled_self_matches": int(sum(
                                em[i]["reaction_identity"] == em[int(donor)]["reaction_identity"]
                                for i, donor in enumerate(shuffled_donors.tolist())
                            )),
                        }
                        handle.write(json.dumps(record) + "\n")
                        handle.flush()
                        print(json.dumps({"stage": "probe_complete", "key": key}), flush=True)
                        del tx, ty, vx, vy, ex, ey, ridge, mlp, predictions
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
    handle.close()


def graph_token_atoms(example, segment: str, canonical_smiles: str) -> dict[int, list[int]]:
    """Map serialized token indices to canonical graph atoms via RDKit correspondence."""
    value = example.source if segment == "source" else example.target
    correspondence = canonical_atom_correspondence(canonical_smiles, value)
    spans = _atom_spans(value)
    segment_tokens = [token for token in example.tokens if token.segment == ("source" if segment == "source" else "target")]
    segment_start = min(token.start for token in segment_tokens)
    result = {}
    for view_atom, (start, end) in enumerate(spans):
        for token in segment_tokens:
            relative_start, relative_end = token.start - segment_start, token.end - segment_start
            if relative_start < end and relative_end > start:
                result.setdefault(token.index, []).append(correspondence[view_atom])
    return result


def extract_views(args) -> None:
    """Extract deterministic cross-view states and graph-alignment metadata."""
    split_manifest = load_split_manifest(args.split_manifest)
    assignment = {row["reaction_identity"]: row["split"] for row in split_manifest["records"]}
    assert_disjoint_confirmation(
        [row["chemical_pair_id"] for row in split_manifest["records"]],
        args.confirmation_manifest,
    )
    rows = [row for row in read_rows(args.panel) if assignment[row["reaction_identity"]] == "test"]
    tokenizer = load_reaction_tokenizer(TOKENIZER_DIR)
    chemfm_vocab = len(tokenizer)
    add_predictor_tokens(tokenizer)
    expanded = []
    for panel_index, row in enumerate(rows):
        sources = deterministic_random_smiles(row["source_identity"], VIEW_SEEDS)
        targets = deterministic_random_smiles(row["target_identity"], VIEW_SEEDS)
        for view, (source, target) in enumerate(zip(sources, targets)):
            adapted = {"reaction_identity": row["reaction_identity"], "canonical_source": source, "canonical_target": target}
            example = annotate_example(tokenizer, adapted, panel_index * 5 + view, "latent_cross_view")
            expanded.append((example, "test", panel_index, view, row["source_identity"], row["target_identity"]))
    specs = selected_specs(args.keys)
    model = load_lora_model(MODEL_DIR, tokenizer, chemfm_vocab_size=chemfm_vocab, attention_dropout=0.0, attn_implementation="sdpa", lora_rank=8, lora_alpha=8).to(args.device).eval()
    for parameter in model.parameters(): parameter.requires_grad_(False)
    capture = SelectedStateCapture(model)
    for spec in specs:
        load_adapter_checkpoint(model, ROOT / spec.checkpoint)
        simple = [(item[0], item[1]) for item in expanded]
        records = extract_records(model, capture, simple, args.device, args.batch_size)
        for record, item in zip(sorted(records, key=lambda r:r["panel_index"]), sorted(expanded, key=lambda x:x[0].panel_index)):
            example, _, identity_index, view, canonical_source, canonical_target = item
            record.update({
                "identity_index": identity_index, "view": view,
                "canonical_source": canonical_source, "canonical_product": canonical_target,
                "source_atom_map": graph_token_atoms(example, "source", canonical_source),
                "product_atom_map": graph_token_atoms(example, "product", canonical_target),
            })
        destination = args.output / "cache/views" / f"{spec.key}.pt"
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"type":"latent_cross_view_cache", "spec":spec.__dict__, "records":records}, destination)
        print(json.dumps({"stage":"views_extracted", "key":spec.key, "records":len(records)}), flush=True)
    capture.close()


def extract_candidates(args) -> None:
    """Replay the already archived beams; never regenerate or rescore candidates."""
    split_manifest = load_split_manifest(args.split_manifest)
    assert_disjoint_confirmation([row["chemical_pair_id"] for row in split_manifest["records"]], args.confirmation_manifest)
    tokenizer = load_reaction_tokenizer(TOKENIZER_DIR)
    chemfm_vocab = len(tokenizer); add_predictor_tokens(tokenizer)
    model = load_lora_model(MODEL_DIR, tokenizer, chemfm_vocab_size=chemfm_vocab, attention_dropout=0.0, attn_implementation="sdpa", lora_rank=8, lora_alpha=8).to(args.device).eval()
    for parameter in model.parameters(): parameter.requires_grad_(False)
    capture = SelectedStateCapture(model)
    for spec in selected_specs(args.keys):
        workload = [
            row for row in _candidate_workload(tokenizer, spec.key)
            if row["role"] == "gold" or row["valid_candidate"]
        ]
        examples = []
        for index, row in enumerate(workload):
            adapted = {"reaction_identity": row["reaction_identity"], "canonical_source": row["source"], "canonical_target": row["candidate"]}
            examples.append((annotate_example(tokenizer, adapted, index, "archived_candidate"), "beam"))
        load_adapter_checkpoint(model, ROOT / spec.checkpoint)
        records = extract_records(model, capture, examples, args.device, args.batch_size)
        for record, beam in zip(records, workload):
            record["beam_metadata"] = {key:value for key,value in beam.items() if key not in {"input_ids","source_positions","target_positions"}}
        destination = args.output / "cache/candidates" / f"{spec.key}.pt"
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"type":"latent_candidate_cache", "spec":spec.__dict__, "records":records}, destination)
        print(json.dumps({"stage":"candidates_extracted", "key":spec.key, "records":len(records)}), flush=True)
    capture.close()


def analyze_views(args) -> None:
    """Reduce graph-aligned atom states into within/between invariance metrics."""
    recheck_confirmation(args)
    output_path = args.output / "raw/invariance.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for spec in selected_specs(args.keys):
            payload = torch.load(args.output / "cache/views" / f"{spec.key}.pt", map_location="cpu", weights_only=False)
            records = payload["records"]
            for layer in LAYERS:
                for segment in ("source", "product"):
                    map_key = f"{segment}_atom_map"
                    canonical_key = "canonical_source" if segment == "source" else "canonical_product"
                    for object_type in ("atom", "motif", "component"):
                        objects: dict[tuple, dict[int, list[torch.Tensor]]] = {}
                        for record in records:
                            states = record["states"][layer].float()
                            token_atoms = {int(key): set(map(int, atoms)) for key, atoms in record[map_key].items()}
                            canonical = Chem.MolFromSmiles(record[canonical_key])
                            if object_type == "atom":
                                supports = [("atom", atom, {atom}) for atom in range(canonical.GetNumAtoms())]
                            elif object_type == "motif":
                                supports = [
                                    ("motif", name, tuple(match), set(match))
                                    for name, query in MOTIF_QUERIES.items()
                                    for match in canonical.GetSubstructMatches(query, uniquify=True)
                                ]
                            else:
                                supports = [("component", tuple(sorted(fragment)), set(fragment)) for fragment in Chem.GetMolFrags(canonical)]
                            for support in supports:
                                atom_set = support[-1]
                                state_rows = [states[token] for token, atoms in token_atoms.items() if atoms & atom_set]
                                if state_rows:
                                    key = (record["identity_index"], *support[:-1])
                                    objects.setdefault(key, {}).setdefault(record["view"], []).append(torch.stack(state_rows).mean(0))
                        aligned = [
                            torch.stack([torch.stack(views[view]).mean(0) for view in range(5)])
                            for views in objects.values() if set(views) == set(range(5))
                        ]
                        if not aligned:
                            continue
                        values = torch.stack(aligned)
                        result = {
                            "checkpoint": spec.key, "layer": layer, "segment": segment,
                            "object": object_type, **invariance_metrics(values),
                        }
                        handle.write(json.dumps(result) + "\n")
                        print(json.dumps({"stage":"invariance_complete", "checkpoint":spec.key, "layer":layer, "segment":segment, "object":object_type}), flush=True)


def score_decoder(args) -> None:
    """Score decoder preservation, using exact one-token suffix replay below final."""
    recheck_confirmation(args)
    output_path = args.output / "raw/decoder_metrics.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer = load_reaction_tokenizer(TOKENIZER_DIR)
    chemfm_vocab = len(tokenizer); add_predictor_tokens(tokenizer)
    model = load_lora_model(MODEL_DIR, tokenizer, chemfm_vocab_size=chemfm_vocab, attention_dropout=0.0, attn_implementation="sdpa", lora_rank=8, lora_alpha=8).to(args.device).eval()
    for parameter in model.parameters(): parameter.requires_grad_(False)
    llama = find_llama(model)
    layer_numbers = {"layer_6": 6, "layer_16": 16, "layer_21": 21}
    with output_path.open("w", encoding="utf-8") as handle:
        for spec in selected_specs(args.keys):
            load_adapter_checkpoint(model, ROOT / spec.checkpoint)
            payload = torch.load(args.output / "cache/canonical" / f"{spec.key}.pt", map_location="cpu", weights_only=False)
            test_records = [row for row in payload["records"] if row["split"] == "test"]
            record_map = {row["reaction_identity"]: row for row in test_records}
            for layer in LAYERS:
                if args.layers and layer not in args.layers.split(","):
                    continue
                for segment in ("source", "product"):
                    for horizon in HORIZONS:
                        for mode in ("current", "history"):
                            x, y, metadata = forecast_matrices(test_records, layer, segment, horizon, mode)
                            if not len(x): continue
                            sample = reaction_balanced_indices(metadata, args.decoder_positions, args.probe_seed)
                            x, y = x[sample], y[sample]
                            metadata = [metadata[index] for index in sample.tolist()]
                            gold = torch.tensor([row["gold_id"] for row in metadata], device=args.device)
                            for kind in ("constant", "ridge", "residual_mlp"):
                                key = f"{spec.key}__{layer}__{segment}__k{horizon}__{mode}"
                                prediction, _ = load_probe_predictions(args.output / "probes" / f"{key}.pt", x, kind, args.device, args.probe_batch_size)
                                if layer == "final_post_norm":
                                    true_final = y.to(args.device)
                                    predicted_final = prediction.to(args.device)
                                else:
                                    true_rows, predicted_rows = [], []
                                    for truth, predicted, meta in zip(y, prediction, metadata):
                                        record = record_map[meta["reaction_identity"]]
                                        layer_input = record["states"][layer].to(args.device).unsqueeze(0)
                                        replayed = suffix_replay_one_position(
                                            llama, layer_numbers[layer], layer_input,
                                            torch.stack((truth, predicted)).to(args.device), int(meta["future_index"]),
                                        )
                                        reference = record["states"]["final_post_norm"][int(meta["future_index"])].float().to(args.device)
                                        if not torch.allclose(replayed[0].float(), reference, rtol=2e-2, atol=2e-2):
                                            raise RuntimeError("true-state suffix replay failed full-forward parity")
                                        true_rows.append(replayed[0]); predicted_rows.append(replayed[1])
                                    true_final, predicted_final = torch.stack(true_rows), torch.stack(predicted_rows)
                                weight = model.get_output_embeddings().weight
                                metrics = decoder_distribution_metrics(true_final @ weight.T, predicted_final @ weight.T, gold)
                                record = {
                                    "checkpoint":spec.key, "layer":layer, "segment":segment,
                                    "horizon":horizon, "mode":mode, "probe":kind, "n":len(x),
                                    **{name:float(value.float().mean()) for name,value in metrics.items()},
                                }
                                handle.write(json.dumps(record) + "\n"); handle.flush()
                                print(json.dumps({"stage":"decoder_complete", "key":key, "probe":kind}), flush=True)


def score_candidates(args) -> None:
    """Apply locked validation probes to archived candidate trajectories."""
    recheck_confirmation(args)
    output_path = args.output / "raw/candidate_predictability.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer = load_reaction_tokenizer(TOKENIZER_DIR)
    chemfm_vocab = len(tokenizer); add_predictor_tokens(tokenizer)
    model = load_lora_model(MODEL_DIR, tokenizer, chemfm_vocab_size=chemfm_vocab, attention_dropout=0.0, attn_implementation="sdpa", lora_rank=8, lora_alpha=8).to(args.device).eval()
    for parameter in model.parameters(): parameter.requires_grad_(False)
    with output_path.open("w", encoding="utf-8") as handle:
        for spec in selected_specs(args.keys):
            load_adapter_checkpoint(model, ROOT / spec.checkpoint)
            payload = torch.load(args.output / "cache/candidates" / f"{spec.key}.pt", map_location="cpu", weights_only=False)
            records = payload["records"]
            for horizon in HORIZONS:
                for mode in ("current", "history"):
                    for record in records:
                        x, y, metadata = forecast_matrices([record], "final_post_norm", "product", horizon, mode)
                        if not len(x): continue
                        beam = record["beam_metadata"]
                        key = f"{spec.key}__final_post_norm__product__k{horizon}__{mode}"
                        for kind in ("constant", "ridge", "residual_mlp"):
                            prediction, artifact = load_probe_predictions(args.output / "probes" / f"{key}.pt", x, kind, args.device, args.probe_batch_size)
                            latent = latent_metrics(y, prediction, artifact["basis"].mean)
                            weight = model.get_output_embeddings().weight
                            gold = torch.tensor([row["gold_id"] for row in metadata], device=args.device)
                            functional = decoder_distribution_metrics(y.to(args.device) @ weight.T, prediction.to(args.device) @ weight.T, gold)
                            row = {
                                "checkpoint":spec.key, "reaction_identity":record["reaction_identity"],
                                "panel_index":beam["panel_index"], "view":beam["view"], "role":beam["role"],
                                "candidate":beam["candidate"], "horizon":horizon, "mode":mode, "probe":kind,
                                **{f"latent_{name}":value for name,value in latent.items()},
                                **{f"decoder_{name}":float(value.float().mean()) for name,value in functional.items()},
                            }
                            handle.write(json.dumps(row) + "\n")
                    handle.flush()
                print(json.dumps({"stage":"candidate_scores_complete", "checkpoint":spec.key, "horizon":horizon}), flush=True)


def summarize(args) -> None:
    rows = [json.loads(line) for line in (args.output / "raw/probe_metrics.jsonl").read_text(encoding="utf-8").splitlines() if line]
    def optional_rows(name):
        path = args.output / "raw" / name
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line] if path.exists() else []
    decoder = optional_rows("decoder_metrics.jsonl")
    invariance = optional_rows("invariance.jsonl")
    candidates = optional_rows("candidate_predictability.jsonl")
    candidate_groups = {}
    for row in candidates:
        key = (row["checkpoint"], row["horizon"], row["mode"], row["probe"], row["role"])
        candidate_groups.setdefault(key, []).append(row)
    candidate_summary = [
        {
            "checkpoint": key[0], "horizon": key[1], "mode": key[2], "probe": key[3], "role": key[4],
            "n": len(values),
            "latent_normalized_mse": float(np.mean([row["latent_normalized_mse"] for row in values])),
            "decoder_js": float(np.mean([row["decoder_js"] for row in values])),
            "decoder_kl": float(np.mean([row["decoder_kl_true_predicted"] for row in values])),
        }
        for key, values in sorted(candidate_groups.items())
    ]
    summary = {
        "type": "latent_predictability_audit_summary",
        "cells": len(rows),
        "primary": [row for row in rows if row["segment"] == "product"],
        "decoder": decoder,
        "cross_view_invariance": invariance,
        "candidate_role_summary": candidate_summary,
        "notes": {
            "predictable_fraction": "untruncated R2 relative to the train-target mean",
            "kl_direction": "true decoder distribution || predicted-state distribution",
            "confirmation_outcomes_consumed": False,
        },
    }
    write_json(args.output / "analysis.json", summary)
    plot_dir = args.output / "plots"; plot_dir.mkdir(parents=True, exist_ok=True)
    write_json(plot_dir / "predictability_plot_data.json", [
        {
            "checkpoint":row["checkpoint"], "layer":row["layer"], "horizon":row["horizon"],
            "mode":row["mode"], "ridge_r2":row["metrics"]["ridge"]["arbitrary"]["r2"],
            "mlp_r2":row["metrics"]["residual_mlp"]["arbitrary"]["r2"],
        }
        for row in rows if row["segment"] == "product"
    ])
    write_json(plot_dir / "invariance_plot_data.json", invariance)
    write_json(plot_dir / "decoder_plot_data.json", decoder)
    print(json.dumps({"stage":"summary_complete", "cells":len(rows)}), flush=True)


def parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    common.add_argument("--confirmation-manifest", type=Path, required=True)
    common.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT)
    common.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    common.add_argument("--device", default="cuda")
    common.add_argument("--keys")
    common.add_argument("--batch-size", type=int, default=8)
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="command", required=True)
    sub.add_parser("lock-splits", parents=[common]).set_defaults(function=lock_splits)
    p = sub.add_parser("extract", parents=[common]); p.add_argument("--overwrite", action="store_true"); p.set_defaults(function=extract)
    p = sub.add_parser("extract-views", parents=[common]); p.set_defaults(function=extract_views)
    p = sub.add_parser("extract-candidates", parents=[common]); p.set_defaults(function=extract_candidates)
    p = sub.add_parser("analyze-views", parents=[common]); p.set_defaults(function=analyze_views)
    p = sub.add_parser("fit-probes", parents=[common])
    p.add_argument("--pca-rank", type=int, default=256)
    p.add_argument("--train-positions", type=int, default=16384)
    p.add_argument("--probe-seed", type=int, default=20260904)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--probe-batch-size", type=int, default=512)
    p.add_argument("--ridge", type=float, default=1e-3)
    p.add_argument("--mlp-decay", type=float, default=1e-4)
    p.add_argument("--mlp-width", type=int, default=128)
    p.add_argument("--resume", action="store_true")
    p.set_defaults(function=fit_probes)
    p = sub.add_parser("score-decoder", parents=[common])
    p.add_argument("--layers", default="layer_6,layer_16,layer_21,final_post_norm")
    p.add_argument("--decoder-positions", type=int, default=4096)
    p.add_argument("--probe-seed", type=int, default=20260904)
    p.add_argument("--probe-batch-size", type=int, default=512)
    p.set_defaults(function=score_decoder)
    p = sub.add_parser("score-candidates", parents=[common])
    p.add_argument("--probe-batch-size", type=int, default=512)
    p.set_defaults(function=score_candidates)
    sub.add_parser("summarize", parents=[common]).set_defaults(function=summarize)
    return root


def main() -> None:
    args = parser().parse_args()
    args.output = args.output.resolve()
    args.split_manifest = args.split_manifest.resolve()
    args.function(args)


if __name__ == "__main__":
    main()
