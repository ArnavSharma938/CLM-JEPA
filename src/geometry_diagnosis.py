from __future__ import annotations

import argparse
import csv
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from chemfm import MODEL_DIR, TOKENIZER_DIR, ReactionCollator, load_lora_model, load_reaction_tokenizer
from jepa import add_predictor_tokens, extract_source_and_target
from metrics import identity_mappings
from train import load_adapter_checkpoint


ROOT = Path(__file__).resolve().parents[1]
ATOM_PATTERN = re.compile(r"Cl|Br|Si|Se|Na|Li|Mg|Al|Ca|[A-Z]|[cnopsb]")


@torch.inference_mode()
def encode_position(model, rows, pad_token_id: int, *, batch_size: int = 16) -> torch.Tensor:
    states = []
    for start in range(0, len(rows), batch_size):
        chunk = rows[start:start + batch_size]
        padded = pad_sequence(chunk, batch_first=True, padding_value=pad_token_id).to(model.device)
        attention = padded.ne(pad_token_id)
        hidden = model(
            input_ids=padded, attention_mask=attention, output_hidden_states=True
        ).hidden_states[-1]
        indices = attention.sum(dim=1) - 1
        states.append(hidden[torch.arange(len(chunk), device=model.device), indices].float().cpu())
    return torch.cat(states)


def sample_metadata(records, targets):
    identities = [".".join(sorted(record["tgt"].split("."))) for record in records]
    if len(set(identities)) != len(records):
        raise ValueError("the geometry assay requires one row per target identity")
    lengths_by_identity = defaultdict(list)
    for identity, target in zip(identities, targets):
        lengths_by_identity[identity].append(len(target))
    token_lengths = {
        identity: round(sum(lengths) / len(lengths))
        for identity, lengths in lengths_by_identity.items()
    }
    heavy_atoms = {
        identity: len(ATOM_PATTERN.findall(identity)) for identity in identities
    }
    return identities, token_lengths, heavy_atoms


def cache_condition(
    label: str,
    checkpoint: Path | None,
    k: int,
    records,
    cache_path: Path,
    batch_size: int,
) -> dict:
    torch.manual_seed(533)
    tokenizer = load_reaction_tokenizer(TOKENIZER_DIR)
    chemfm_vocab_size = len(tokenizer)
    predictor_ids = add_predictor_tokens(tokenizer)
    model = load_lora_model(
        MODEL_DIR, tokenizer, chemfm_vocab_size=chemfm_vocab_size
    ).cuda().eval()
    if checkpoint is not None:
        load_adapter_checkpoint(model, checkpoint.resolve())
    collator = ReactionCollator(tokenizer, task="forward")
    batch = collator(records)
    tensor_batch = {key: value for key, value in batch.items() if torch.is_tensor(value)}
    sources, targets = extract_source_and_target(tensor_batch)
    predictor_suffix = list(reversed(predictor_ids[:k]))
    source_rows = [
        torch.cat((source, source.new_tensor(predictor_suffix)))
        if predictor_suffix else source
        for source in sources
    ]
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    target_states = encode_position(model, targets, tokenizer.pad_token_id, batch_size=batch_size)
    source_states = encode_position(model, source_rows, tokenizer.pad_token_id, batch_size=batch_size)
    elapsed = time.perf_counter() - started
    identities, token_lengths, heavy_atoms = sample_metadata(records, targets)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "label": label,
            "checkpoint": None if checkpoint is None else str(checkpoint.resolve()),
            "source_states": source_states,
            "target_states": target_states,
            "identities": identities,
            "token_lengths": token_lengths,
            "heavy_atoms": heavy_atoms,
            "k": k,
        },
        cache_path,
    )
    result = {
        "cache": str(cache_path),
        "inference_seconds": elapsed,
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()),
    }
    del model, source_states, target_states, batch, tensor_batch
    torch.cuda.empty_cache()
    return result


def effective_rank(values: torch.Tensor) -> float:
    centered = values - values.mean(0, keepdim=True)
    gram = centered @ centered.T
    energy = torch.linalg.eigvalsh(gram).clamp_min(0)
    probabilities = energy / energy.sum().clamp_min(1e-30)
    entropy = -(probabilities * probabilities.clamp_min(1e-30).log()).sum()
    return float(entropy.exp())


def pair_metrics(
    sources: torch.Tensor,
    targets: torch.Tensor,
    matched_indices: torch.Tensor,
    candidate_indices: torch.Tensor,
) -> dict:
    sources = F.normalize(sources, dim=-1)
    targets = F.normalize(targets, dim=-1)
    correct = (sources * targets).sum(-1)
    matched = (sources * targets[matched_indices]).sum(-1)
    candidate_scores = (sources[:, None, :] * targets[candidate_indices]).sum(-1)
    ranks = (candidate_scores > candidate_scores[:, :1]).sum(-1) + 1
    return {
        "correct_cosine": float(correct.mean()),
        "matched_shuffle_cosine": float(matched.mean()),
        "correct_minus_matched": float((correct - matched).mean()),
        "retrieval_top1": float((ranks == 1).float().mean()),
        "retrieval_mrr": float((1.0 / ranks.float()).mean()),
        "retrieval_chance_top1": 1.0 / candidate_indices.shape[1],
    }


def analyze_cache(cache_path: Path) -> dict:
    cached = torch.load(cache_path, map_location="cpu", weights_only=True)
    sources = cached["source_states"].float().cuda()
    targets = cached["target_states"].float().cuda()
    identities = cached["identities"]
    token_lengths = cached["token_lengths"]
    heavy_atoms = cached["heavy_atoms"]
    _, matched_map, matched_cost = identity_mappings(
        identities, token_lengths, heavy_atoms, 533
    )
    identity_to_index = {identity: index for index, identity in enumerate(identities)}
    matched_indices = torch.tensor(
        [identity_to_index[matched_map[identity]] for identity in identities],
        device=sources.device,
    )
    candidates = []
    for identity in identities:
        negatives = sorted(
            (other for other in identities if other != identity),
            key=lambda other: (
                abs(token_lengths[identity] - token_lengths[other])
                + abs(heavy_atoms[identity] - heavy_atoms[other]),
                other,
            ),
        )[:3]
        candidates.append([identity_to_index[identity], *map(identity_to_index.get, negatives)])
    candidate_indices = torch.tensor(candidates, device=sources.device)

    raw = pair_metrics(sources, targets, matched_indices, candidate_indices)
    raw.update({
        "source_variance": float(sources.var(0, unbiased=False).mean()),
        "source_effective_rank": effective_rank(sources),
        "source_mean_direction_energy": float(
            sources.mean(0).square().sum()
            / sources.square().sum(1).mean().clamp_min(1e-30)
        ),
        "target_variance": float(targets.var(0, unbiased=False).mean()),
        "target_effective_rank": effective_rank(targets),
        "target_mean_direction_energy": float(
            targets.mean(0).square().sum()
            / targets.square().sum(1).mean().clamp_min(1e-30)
        ),
        "matched_assignment_cost": matched_cost,
    })

    joint = torch.cat((sources, targets))
    joint_mean = joint.mean(0, keepdim=True)
    centered_sources = sources - joint_mean
    centered_targets = targets - joint_mean
    centered_joint = torch.cat((centered_sources, centered_targets))
    torch.manual_seed(533)
    _, _, components = torch.pca_lowrank(
        centered_joint, q=8, center=False, niter=6
    )
    transformed = {
        "raw": raw,
        "mean_centered": pair_metrics(
            centered_sources, centered_targets, matched_indices, candidate_indices
        ),
    }
    for count in (1, 2, 4):
        basis = components[:, :count]
        projected_sources = centered_sources - (centered_sources @ basis) @ basis.T
        projected_targets = centered_targets - (centered_targets @ basis) @ basis.T
        transformed[f"mean_centered_remove_pc{count}"] = pair_metrics(
            projected_sources, projected_targets, matched_indices, candidate_indices
        )
    transformed["whitening"] = {
        "status": "not_run",
        "reason": (
            "full joint whitening is not stable without regularization: the 2N by D "
            "centered matrix is rank-deficient by construction, and choosing a shrinkage "
            "constant would add a new unfrozen hyperparameter"
        ),
    }
    del sources, targets, joint, centered_joint
    torch.cuda.empty_cache()
    return transformed


def main() -> None:
    parser = argparse.ArgumentParser(description="Inference-only common-direction diagnosis")
    parser.add_argument(
        "--manifest", type=Path,
        default=ROOT / "data" / "gate3" / "uspto_mit_synthesis.csv",
    )
    parser.add_argument("--native-checkpoint", type=Path)
    parser.add_argument("--clm-checkpoint", type=Path)
    parser.add_argument("--target-sg-checkpoint", type=Path)
    parser.add_argument("--sigreg-k0-checkpoint", type=Path)
    parser.add_argument("--sigreg-k1-checkpoint", type=Path)
    parser.add_argument("--sigreg-k0-b128-checkpoint", type=Path)
    parser.add_argument(
        "--reuse-geometry", type=Path,
        help="reuse already-computed conditions from an identical identity panel",
    )
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "runs" / "diagnostics" / "geometry_cache")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--examples", type=int, default=1024)
    parser.add_argument(
        "--identity-reference", type=Path,
        help="JSONL artifact whose reaction_identity values define the exact analysis subset",
    )
    args = parser.parse_args()
    with args.manifest.open(newline="", encoding="utf-8") as handle:
        raw_records = list(csv.DictReader(handle))
    if args.identity_reference is not None:
        with args.identity_reference.open(encoding="utf-8") as handle:
            reference_rows = [json.loads(line) for line in handle if line.strip()]
        reference_rows.sort(key=lambda row: row["panel_index"])
        wanted = [row["reaction_identity"] for row in reference_rows]
        raw_by_identity = {row.get("reaction_identity"): row for row in raw_records}
        missing = [identity for identity in wanted if identity not in raw_by_identity]
        if missing:
            raise ValueError(f"identity reference contains {len(missing)} unknown reactions")
        raw_records = [raw_by_identity[identity] for identity in wanted]

    records = []
    seen_groups = set()
    for row in raw_records:
        group = row.get("group_id", row.get("example_id", str(len(records))))
        if group in seen_groups:
            continue
        seen_groups.add(group)
        records.append({
            **row,
            "src": row.get("src") or row.get("source", ""),
            "tgt": row.get("tgt") or row.get("target", ""),
        })
        if len(records) == args.examples:
            break
    if len(records) != args.examples:
        raise ValueError(
            f"expected {args.examples} unique reaction identities, got {len(records)}"
        )

    conditions = []
    reused = None
    if args.reuse_geometry is not None:
        reused = json.loads(args.reuse_geometry.read_text(encoding="utf-8"))
        if Path(reused["manifest"]).resolve() != args.manifest.resolve():
            raise ValueError("reused geometry manifest does not match")
        if reused["examples"] != len(records) or reused["seed"] != 533:
            raise ValueError("reused geometry panel or seed does not match")
    else:
        if args.native_checkpoint is None or args.clm_checkpoint is None:
            raise ValueError("native and cLM-JEPA checkpoints are required without --reuse-geometry")
        conditions.extend([
            ("base", None, 1),
            ("native_reference", args.native_checkpoint, 1),
            ("clm_jepa_reference", args.clm_checkpoint, 1),
        ])
    if args.target_sg_checkpoint is not None:
        conditions.append(("clm_jepa_target_sg_reference", args.target_sg_checkpoint, 1))
    if args.sigreg_k0_checkpoint is not None:
        conditions.append(("clm_jepa_sigreg_k0_epoch2", args.sigreg_k0_checkpoint, 0))
    if args.sigreg_k1_checkpoint is not None:
        conditions.append(("clm_jepa_sigreg_k1_epoch2", args.sigreg_k1_checkpoint, 1))
    if args.sigreg_k0_b128_checkpoint is not None:
        conditions.append((
            "clm_jepa_sigreg_k0_b128_epoch2",
            args.sigreg_k0_b128_checkpoint,
            0,
        ))
    output = {
        "manifest": str(args.manifest.resolve()),
        "examples": len(records),
        "seed": 533,
        "k": 1,
        "pca_policy": (
            "subtract one mean fitted to concatenated source and target states; fit shared "
            "PCs to the concatenated centered states and project both views through the same basis"
        ),
        "conditions": {} if reused is None else reused["conditions"],
        "reused_geometry": (
            None if args.reuse_geometry is None else str(args.reuse_geometry.resolve())
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for label, checkpoint, k in conditions:
        cache_path = args.cache_dir / f"{label}.pt"
        output["conditions"][label] = cache_condition(
            label, checkpoint, k, records, cache_path, args.batch_size
        )
        output["conditions"][label]["metrics"] = analyze_cache(cache_path)
        args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"completed": label, **output["conditions"][label]}), flush=True)
    output["new_inference_seconds"] = sum(
        output["conditions"][label]["inference_seconds"]
        for label, _, _ in conditions
    )
    output["total_inference_seconds"] = sum(
        condition["inference_seconds"] for condition in output["conditions"].values()
    )
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
