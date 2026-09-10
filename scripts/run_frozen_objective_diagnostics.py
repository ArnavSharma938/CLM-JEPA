"""Frozen gradient-coupling and graph-target certificates for Native ChemFM.

This runner never constructs an optimizer and never calls generation.  Its two
stages are intentionally resumable: ``gradient`` writes one result per seed,
while ``extract`` writes the single five-view hidden-state cache consumed by
``certificate``.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from chemfm import (  # noqa: E402
    MODEL_DIR, TOKENIZER_DIR, ReactionCollator, add_predictor_tokens,
    load_adapter_checkpoint, load_lora_model, load_reaction_tokenizer,
)
from faithful_nextlat import FaithfulNextLatObjective  # noqa: E402
from frozen_geometry import MOTIF_QUERIES, annotate_example  # noqa: E402
from latent_predictability import (  # noqa: E402
    RidgeProbe, Standardizer, TargetBasis, canonical_atom_correspondence,
    fit_probe, latent_metrics, sha256_file,
)
from run_latent_predictability_audit import (  # noqa: E402
    SelectedStateCapture, graph_token_atoms,
)
from train import read_rows  # noqa: E402

OUT = ROOT / "runs" / "frozen_objective_diagnostics"
PILOT = ROOT / "data/clm_jepa_uspto_mit_pilot_1280/uspto_mit_train.csv"
GRAD_SEED = 20260909
PROBE_SEED = 20260910
NATIVE = {
    2027: ROOT / "runs/decoder_projected/confirmation/seed_2027/native/training/checkpoints/epoch_4",
    3163: ROOT / "runs/decoder_projected/confirmation/seed_3163/native/training/checkpoints/epoch_4",
}
AUX = {
    533: ROOT / "runs/faithful_nextlat/seed_533/training/checkpoints/epoch_4/auxiliary_training_state.pt",
    917: ROOT / "runs/faithful_nextlat/seed_917/training/checkpoints/epoch_4/auxiliary_training_state.pt",
}
AUX_SOURCE_SEED = {2027: 533, 3163: 917}


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def stable_hash(values) -> str:
    return hashlib.sha256(json.dumps(values, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def group_name(name: str) -> str:
    if "embed_tokens" in name or "lm_head" in name:
        return "embeddings_lm_head"
    if any(f".{key}." in name for key in ("q_proj", "k_proj", "v_proj", "o_proj")):
        return "attention_lora"
    if any(f".{key}." in name for key in ("gate_proj", "up_proj", "down_proj")):
        return "mlp_lora"
    return "other_trainable"


def layer_name(name: str) -> str:
    match = re.search(r"\.layers\.(\d+)\.", name)
    return f"layer_{int(match.group(1)):02d}" if match else group_name(name)


def vector_stats(left: list[torch.Tensor], right: list[torch.Tensor]) -> dict:
    dot = l2 = r2 = absdot = negative = 0.0
    left_conflict2 = right_conflict2 = 0.0
    conflict_coordinates = active_coordinates = coordinates = 0
    for a, b in zip(left, right):
        a, b = a.double().reshape(-1), b.double().reshape(-1)
        product = a * b
        conflict = product < 0
        active = (a != 0) & (b != 0)
        dot += float(product.sum())
        l2 += float(a.square().sum()); r2 += float(b.square().sum())
        absdot += float(product.abs().sum())
        negative += float((-product[conflict]).sum())
        left_conflict2 += float(a[conflict].square().sum())
        right_conflict2 += float(b[conflict].square().sum())
        conflict_coordinates += int(conflict.sum()); active_coordinates += int(active.sum())
        coordinates += product.numel()
    ln, rn = math.sqrt(l2), math.sqrt(r2)
    cosine = dot / max(ln * rn, 1e-30)
    return {
        "ntp_norm": ln, "aux_norm": rn, "aux_to_ntp_norm_ratio": rn / max(ln, 1e-30),
        "cosine": cosine, "dot": dot,
        "negative_dot_magnitude": negative,
        "negative_dot_fraction_of_absolute_dot": negative / max(absdot, 1e-30),
        "conflicting_coordinate_fraction_active": conflict_coordinates / max(active_coordinates, 1),
        "ntp_norm_fraction_on_conflicting_coordinates": math.sqrt(left_conflict2) / max(ln, 1e-30),
        "aux_norm_fraction_on_conflicting_coordinates": math.sqrt(right_conflict2) / max(rn, 1e-30),
        "pcgrad_aux_projection_fraction": max(0.0, -dot) / max(ln * rn, 1e-30),
        "coordinates": coordinates, "jointly_active_coordinates": active_coordinates,
    }


def selected_training_batches(tokenizer, rows: list[dict], count: int = 8, size: int = 4):
    collator = ReactionCollator(tokenizer)
    first = {}
    for row in rows:
        first.setdefault(row["group_id"], row)
    candidates = list(first.values())
    lengths = {row["group_id"]: len(collator([row])["input_ids"][0]) for row in candidates}
    candidates.sort(key=lambda row: (lengths[row["group_id"]], hashlib.sha256(
        f"gradient-batch-v1|{row['group_id']}".encode()).hexdigest()))
    # Evenly cover the empirical length distribution, then batch neighboring
    # lengths to retain the physical batch size without wasteful padding.
    indices = np.linspace(0, len(candidates) - 1, count * size).round().astype(int)
    picked = [candidates[int(index)] for index in indices]
    return [picked[start:start + size] for start in range(0, len(picked), size)], lengths


def run_gradient(seed: int, batch_count: int = 8, auxiliary_seed: int | None = None) -> None:
    auxiliary_seed = AUX_SOURCE_SEED[seed] if auxiliary_seed is None else auxiliary_seed
    default_pair = auxiliary_seed == AUX_SOURCE_SEED[seed]
    destination = OUT / "gradient" / (
        f"seed_{seed}.json" if default_pair else f"native_{seed}_aux_{auxiliary_seed}.json"
    )
    if destination.exists():
        print(json.dumps({"stage": "gradient_reused", "seed": seed}), flush=True)
        return
    tokenizer = load_reaction_tokenizer(TOKENIZER_DIR)
    native_vocab = len(tokenizer); add_predictor_tokens(tokenizer)
    rows = read_rows("uspto_mit_synthesis", path=PILOT)
    batches, lengths = selected_training_batches(tokenizer, rows, batch_count)
    collator = ReactionCollator(tokenizer)
    model = load_lora_model(
        MODEL_DIR, tokenizer, chemfm_vocab_size=native_vocab,
        attn_implementation="sdpa", lora_rank=8, lora_alpha=8,
    ).cuda().train()
    load_adapter_checkpoint(model, NATIVE[seed])
    method = FaithfulNextLatObjective(
        model.get_output_embeddings().weight[:native_vocab],
        product_start_token_id=tokenizer.convert_tokens_to_ids("<prostart>"),
        eos_token_id=tokenizer.eos_token_id, native_vocab_size=native_vocab,
        resume_state=AUX[auxiliary_seed],
    ).cuda().train()
    parameters = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    aggregate = {"ntp": [torch.zeros_like(p, device="cpu", dtype=torch.float32) for _, p in parameters],
                 "aux": [torch.zeros_like(p, device="cpu", dtype=torch.float32) for _, p in parameters]}
    batch_results = []
    torch.cuda.reset_peak_memory_stats(); started = time.perf_counter()
    for batch_index, batch_rows in enumerate(batches):
        torch.manual_seed(GRAD_SEED + seed * 100 + batch_index)
        inputs = {key: value.cuda() for key, value in collator(batch_rows).items() if torch.is_tensor(value)}
        output = method(model, inputs)
        ntp = torch.autograd.grad(output.native_loss, [p for _, p in parameters], retain_graph=True, allow_unused=True)
        aux = torch.autograd.grad(output.jepa_loss, [p for _, p in parameters], allow_unused=True)
        ntp = [torch.zeros_like(p, device="cpu", dtype=torch.float32) if g is None else g.detach().float().cpu()
               for g, (_, p) in zip(ntp, parameters)]
        aux = [torch.zeros_like(p, device="cpu", dtype=torch.float32) if g is None else g.detach().float().cpu()
               for g, (_, p) in zip(aux, parameters)]
        for index in range(len(parameters)):
            aggregate["ntp"][index].add_(ntp[index]); aggregate["aux"][index].add_(aux[index])
        groups = {}
        for label, selector in (
            ("global", lambda _: True),
            ("embeddings_lm_head", lambda n: group_name(n) == "embeddings_lm_head"),
            ("attention_lora", lambda n: group_name(n) == "attention_lora"),
            ("mlp_lora", lambda n: group_name(n) == "mlp_lora"),
        ):
            chosen = [i for i, (name, _) in enumerate(parameters) if selector(name)]
            groups[label] = vector_stats([ntp[i] for i in chosen], [aux[i] for i in chosen])
        batch_results.append({
            "batch": batch_index,
            "group_ids": [row["group_id"] for row in batch_rows],
            "sequence_lengths": [lengths[row["group_id"]] for row in batch_rows],
            "ntp_loss": float(output.native_loss.detach()), "aux_loss": float(output.jepa_loss.detach()),
            "latent_loss": float(output.lz.detach()), "kl_loss": float(output.kl.detach()),
            "groups": groups,
        })
        del output, ntp, aux, inputs
    for key in aggregate:
        for value in aggregate[key]: value.div_(len(batches))
    summaries = {}
    labels = {"global": [i for i in range(len(parameters))]}
    for label in ("embeddings_lm_head", "attention_lora", "mlp_lora", "other_trainable"):
        labels[label] = [i for i, (name, _) in enumerate(parameters) if group_name(name) == label]
    for layer in sorted({layer_name(name) for name, _ in parameters if ".layers." in name}):
        labels[layer] = [i for i, (name, _) in enumerate(parameters) if layer_name(name) == layer]
    for label, chosen in labels.items():
        if chosen:
            summaries[label] = vector_stats(
                [aggregate["ntp"][i] for i in chosen], [aggregate["aux"][i] for i in chosen]
            )
    result = {
        "type": "frozen_ntp_aux_gradient_coupling", "seed": seed,
        "native_checkpoint": str(NATIVE[seed].resolve()),
        "native_adapter_sha256": sha256_file(NATIVE[seed] / "USPTO-MIT-Synthesis/adapter_model.safetensors"),
        "auxiliary_source_seed": auxiliary_seed, "auxiliary_state": str(AUX[auxiliary_seed].resolve()),
        "auxiliary_sha256": sha256_file(AUX[auxiliary_seed]), "train_manifest": str(PILOT.resolve()),
        "train_manifest_sha256": sha256_file(PILOT), "selection_seed": GRAD_SEED,
        "selection_hash": stable_hash([[row["group_id"] for row in batch] for batch in batches]),
        "batches": len(batches), "physical_batch_size": 4, "optimizer_steps": 0,
        "parameter_tensors": len(parameters), "trainable_parameters": sum(p.numel() for _, p in parameters),
        "aggregate_mean_gradient": summaries, "per_batch": batch_results,
        "seconds": time.perf_counter() - started,
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()),
    }
    write_json(destination, result)
    print(json.dumps({"stage": "gradient_complete", "seed": seed, "seconds": result["seconds"]}), flush=True)


def reaction_split(groups: list[str]) -> dict[str, str]:
    ordered = sorted(groups, key=lambda value: hashlib.sha256(f"frozen-target-v1|{value}".encode()).hexdigest())
    counts = {"train": 160, "validation": 48, "test": 48}
    result, start = {}, 0
    for split in ("train", "validation", "test"):
        for value in ordered[start:start + counts[split]]: result[value] = split
        start += counts[split]
    return result


def extract_views() -> None:
    destination = OUT / "certificate" / "native_2027_five_view.pt"
    if destination.exists():
        print(json.dumps({"stage": "view_cache_reused", "path": str(destination)}), flush=True)
        return
    rows = read_rows("uspto_mit_synthesis", path=PILOT)
    groups = sorted({row["group_id"] for row in rows})
    assignment = reaction_split(groups)
    tokenizer = load_reaction_tokenizer(TOKENIZER_DIR)
    native_vocab = len(tokenizer); add_predictor_tokens(tokenizer)
    examples = []
    for index, row in enumerate(rows):
        adapted = {"reaction_identity": row["group_id"], "canonical_source": row["source"], "canonical_target": row["target"]}
        example = annotate_example(
            tokenizer, adapted, index, "existing_pilot_r_smiles",
            infer_reaction_center=False,
        )
        examples.append((example, assignment[row["group_id"]], int(row["augmentation_index"]), row))
    model = load_lora_model(
        MODEL_DIR, tokenizer, chemfm_vocab_size=native_vocab, attention_dropout=0.0,
        attn_implementation="sdpa", lora_rank=8, lora_alpha=8,
    ).cuda().eval()
    for parameter in model.parameters(): parameter.requires_grad_(False)
    load_adapter_checkpoint(model, NATIVE[2027])
    capture = SelectedStateCapture(model)
    records = []
    ordered = sorted(examples, key=lambda item: len(item[0].input_ids))
    torch.cuda.reset_peak_memory_stats(); started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(ordered), 8):
            chunk = ordered[start:start + 8]
            maximum = max(len(item[0].input_ids) for item in chunk)
            ids = torch.zeros((len(chunk), maximum), dtype=torch.long, device="cuda")
            mask = torch.zeros_like(ids, dtype=torch.bool)
            for row_index, (example, _, _, _) in enumerate(chunk):
                length = len(example.input_ids); ids[row_index, :length] = torch.tensor(example.input_ids, device="cuda"); mask[row_index, :length] = True
            capture.clear(); model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True)
            for row_index, (example, split, view, row) in enumerate(chunk):
                length = len(example.input_ids)
                atom_map = graph_token_atoms(example, "product", row["target"])
                records.append({
                    "panel_index": example.panel_index, "group_id": row["group_id"], "split": split,
                    "view": view, "source": row["source"], "target": row["target"],
                    "input_ids": torch.tensor(example.input_ids, dtype=torch.int32),
                    "product_indices": [token.index for token in example.tokens if token.segment == "target"],
                    "product_atom_map": atom_map,
                    "states": capture.values["final_post_norm"][row_index, :length].to(torch.bfloat16).cpu(),
                    "reaction_center_status": example.reaction_center_metadata.get("status"),
                })
            if start % 160 == 0: print(json.dumps({"stage": "view_extract_progress", "rows": len(records)}), flush=True)
    capture.close()
    payload = {
        "type": "native_existing_five_view_final_state_cache", "records": sorted(records, key=lambda row: row["panel_index"]),
        "native_checkpoint": str(NATIVE[2027].resolve()),
        "native_adapter_sha256": sha256_file(NATIVE[2027] / "USPTO-MIT-Synthesis/adapter_model.safetensors"),
        "lm_head": model.get_output_embeddings().weight[:native_vocab].detach().to(torch.bfloat16).cpu(),
        "panel": str(PILOT.resolve()), "panel_sha256": sha256_file(PILOT),
        "split_assignment": assignment, "split_hash": stable_hash(assignment),
        "seconds": time.perf_counter() - started, "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()),
    }
    destination.parent.mkdir(parents=True, exist_ok=True); torch.save(payload, destination)
    print(json.dumps({"stage": "view_cache_complete", "seconds": payload["seconds"], "bytes": destination.stat().st_size}), flush=True)


def canonical_atom_states(record: dict, canonical: str) -> torch.Tensor:
    correspondence = canonical_atom_correspondence(canonical, record["target"])
    states = record["states"].float(); mapping = record["product_atom_map"]
    result = [None] * Chem.MolFromSmiles(canonical).GetNumAtoms()
    for view_atom, canonical_atom in correspondence.items():
        tokens = [int(token) for token, atoms in mapping.items() if view_atom in atoms]
        if not tokens: raise ValueError("graph atom has no aligned product token")
        result[canonical_atom] = states[tokens].mean(0)
    if any(value is None for value in result): raise ValueError("incomplete canonical atom alignment")
    return torch.stack(result)


def graph_unit(molecule, atom: int) -> set[int]:
    unit = {atom} | {neighbor.GetIdx() for neighbor in molecule.GetAtomWithIdx(atom).GetNeighbors()}
    matches = [set(match) for query in MOTIF_QUERIES.values() for match in molecule.GetSubstructMatches(query, uniquify=True) if atom in match]
    if matches:
        # Prefer the smallest chemically named group containing the next atom;
        # radius-one context ensures singleton/edge matches are not raw spans.
        unit |= min(matches, key=lambda value: (len(value), sorted(value)))
    return unit


def target_rows(payload: dict):
    grouped = defaultdict(list)
    for record in payload["records"]: grouped[record["group_id"]].append(record)
    output = []
    lm_head = payload["lm_head"].float()
    for group_id, views in sorted(grouped.items()):
        views.sort(key=lambda row: row["view"]); reference = views[0]
        canonical = Chem.MolToSmiles(Chem.MolFromSmiles(reference["target"]), isomericSmiles=True)
        atom_views = [canonical_atom_states(record, canonical) for record in views]
        molecule = Chem.MolFromSmiles(canonical)
        inverse = defaultdict(list)
        correspondence = canonical_atom_correspondence(canonical, reference["target"])
        for token, atoms in reference["product_atom_map"].items():
            for atom in atoms: inverse[int(token)].append(correspondence[int(atom)])
        product = reference["product_indices"]
        states = reference["states"].float(); ids = reference["input_ids"].long()
        for offset in range(len(product) - 1):
            current, future = int(product[offset]), int(product[offset + 1])
            if future not in inverse: continue
            atom = min(inverse[future]); unit = sorted(graph_unit(molecule, atom))
            cross = torch.stack([value[unit].mean(0) for value in atom_views[1:]]).mean(0)
            logits = lm_head @ states[current]
            gold = int(ids[future]); gold_logit = float(logits[gold]); rank = int((logits > logits[gold]).sum()) + 1
            strongest_other = float(torch.cat((logits[:gold], logits[gold + 1:])).max())
            log_prob = float(torch.log_softmax(logits, 0)[gold])
            output.append({
                "group_id": group_id, "split": reference["split"], "current": current, "future": future,
                "x": states[current], "adjacent": states[future],
                "semantic": atom_views[0][unit].mean(0), "cross_serialization": cross,
                "gold_log_prob": log_prob, "ntp_loss": -log_prob, "gold_rank": rank,
                "gold_margin": gold_logit - strongest_other, "unit_atoms": len(unit),
            })
    return output


def per_reaction_coupling(rows, errors, field: str, favorable: bool) -> dict:
    by_reaction = defaultdict(list)
    for row, error in zip(rows, errors.tolist()): by_reaction[row["group_id"]].append((error, float(row[field])))
    x = np.asarray([np.mean([v[0] for v in values]) for values in by_reaction.values()])
    y = np.asarray([np.mean([v[1] for v in values]) for values in by_reaction.values()])
    rho, p = spearmanr(x, y)
    # Positive means better target prediction (lower error) accompanies better
    # decoder behavior, independent of the metric's native direction.
    adjusted = float(rho if not favorable else -rho)
    return {"spearman_error_vs_metric": float(rho), "direction_adjusted_behavioral_coupling": adjusted,
            "pvalue": float(p), "reactions": len(by_reaction)}


def paired_coupling_contrast(error_maps, behavior, candidate: str, field: str, favorable: bool) -> dict:
    identities = sorted(behavior)
    sign = -1.0 if favorable else 1.0
    def effect(indices):
        values = []
        metric = [behavior[identities[i]][field] for i in indices]
        for target in (candidate, "adjacent"):
            error = [error_maps[target][identities[i]] for i in indices]
            values.append(sign * float(spearmanr(error, metric).statistic))
        return values[0] - values[1]
    observed = effect(list(range(len(identities))))
    rng = np.random.default_rng(PROBE_SEED + len(candidate) * 17 + len(field))
    draws = [effect(rng.integers(0, len(identities), len(identities)).tolist()) for _ in range(4000)]
    return {"candidate_minus_adjacent": observed,
            "bootstrap_ci95": [float(np.quantile(draws, .025)), float(np.quantile(draws, .975))],
            "reactions": len(identities), "bootstrap_draws": len(draws)}


def run_certificate(overwrite: bool = False) -> None:
    destination = OUT / "certificate" / "results.json"
    if destination.exists() and not overwrite:
        print(json.dumps({"stage": "certificate_reused"}), flush=True); return
    cache_path = OUT / "certificate" / "native_2027_five_view.pt"
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    rows = target_rows(payload)
    split_rows = {split: [row for row in rows if row["split"] == split] for split in ("train", "validation", "test")}
    results, error_maps = {}, {}
    behavior = {}
    for row in split_rows["test"]:
        behavior.setdefault(row["group_id"], defaultdict(list))
        for field in ("ntp_loss", "gold_log_prob", "gold_rank", "gold_margin"):
            behavior[row["group_id"]][field].append(float(row[field]))
    behavior = {identity: {field: float(np.mean(values)) for field, values in fields.items()}
                for identity, fields in behavior.items()}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for target in ("adjacent", "semantic", "cross_serialization"):
        matrices = {}
        for split, values in split_rows.items():
            matrices[split] = (torch.stack([row["x"] for row in values]), torch.stack([row[target] for row in values]))
        tx, ty = matrices["train"]; vx, vy = matrices["validation"]; ex, ey = matrices["test"]
        basis = TargetBasis.fit(ty.to(device), 128, PROBE_SEED)
        xs = Standardizer.fit(tx); txs = xs(tx); vxs = xs(vx); exs = xs(ex)
        tys, vys = basis.encode(ty), basis.encode(vy)
        probe = RidgeProbe(txs.shape[1], tys.shape[1]).to(device)
        probe, trace = fit_probe(probe, txs, tys, vxs, vys, weight_decay=1e-3, epochs=20, batch_size=512, seed=PROBE_SEED)
        predictions = []
        with torch.inference_mode():
            for start in range(0, len(ex), 512): predictions.append(probe(exs[start:start + 512].to(device)).cpu())
        prediction = basis.decode(torch.cat(predictions)); metrics = latent_metrics(ey, prediction, basis.mean)
        errors = (prediction - ey).square().mean(-1)
        grouped_errors = defaultdict(list)
        for row, error in zip(split_rows["test"], errors.tolist()): grouped_errors[row["group_id"]].append(error)
        error_maps[target] = {identity: float(np.mean(values)) for identity, values in grouped_errors.items()}
        coupling = {
            "ntp_loss": per_reaction_coupling(split_rows["test"], errors, "ntp_loss", False),
            "gold_log_probability": per_reaction_coupling(split_rows["test"], errors, "gold_log_prob", True),
            "gold_rank": per_reaction_coupling(split_rows["test"], errors, "gold_rank", False),
            "gold_margin": per_reaction_coupling(split_rows["test"], errors, "gold_margin", True),
        }
        results[target] = {
            "train_positions": len(tx), "validation_positions": len(vx), "test_positions": len(ex),
            "train_reactions": 160, "validation_reactions": 48, "test_reactions": 48,
            "target_pca_rank": 128, "pca_variance_coverage": basis.variance_coverage,
            "probe_parameters": sum(p.numel() for p in probe.parameters()), "fit_trace": trace,
            "heldout": metrics, "behavioral_coupling": coupling,
        }
        del probe, txs, vxs, exs, tys, vys, prediction
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    contrasts = {}
    for candidate in ("semantic", "cross_serialization"):
        contrasts[candidate] = {
            field: paired_coupling_contrast(error_maps, behavior, candidate, field, favorable)
            for field, favorable in (("ntp_loss", False), ("gold_log_prob", True),
                                      ("gold_rank", False), ("gold_margin", True))
        }
    result = {
        "type": "frozen_candidate_target_certificate", "cache": str(cache_path.resolve()),
        "cache_sha256": sha256_file(cache_path), "native_checkpoint": payload["native_checkpoint"],
        "native_adapter_sha256": payload["native_adapter_sha256"], "panel": payload["panel"],
        "panel_sha256": payload["panel_sha256"], "split_hash": payload["split_hash"],
        "split_counts": {"train": 160, "validation": 48, "test": 48},
        "causal_input": "final post-norm h[t] only; no future token, state, view, graph, or target information",
        "position_filter": "view-0 product transitions whose next serialized token maps to a graph atom",
        "semantic_target": "view-0 mean atom state over radius-1 graph neighborhood, expanded by smallest named SMARTS motif containing next atom",
        "cross_serialization_target": "mean of the same canonical graph unit pooled in existing R-SMILES views 1-4",
        "behavioral_metrics": "live-head teacher-forced metrics for the gold next atom token at h[t]",
        "probe": "same 2048-to-128 linear ridge, train-only target PCA and input standardization, fixed 20-epoch fitter",
        "results": results, "paired_behavioral_coupling_contrasts": contrasts,
        "test_reaction_mean_target_mse": error_maps,
    }
    write_json(destination, result)
    print(json.dumps({"stage": "certificate_complete", "positions": {k: len(v) for k, v in split_rows.items()}}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("gradient", "extract", "certificate", "all"))
    parser.add_argument("--gradient-batches", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.stage in {"gradient", "all"}:
        for seed in (2027, 3163):
            for auxiliary_seed in (533, 917): run_gradient(seed, args.gradient_batches, auxiliary_seed)
    if args.stage in {"extract", "all"}: extract_views()
    if args.stage in {"certificate", "all"}: run_certificate(args.overwrite)


if __name__ == "__main__": main()
