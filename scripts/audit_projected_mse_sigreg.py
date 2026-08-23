"""Audit projected MSE/SIGReg geometry and held-out-NTP gradient alignment.

This script is read-only: it constructs no optimizer and updates no model or
projector parameter.  Each selected checkpoint is measured in raw ChemFM space
and disposable projection space.  Auxiliary gradients are evaluated with the
training-time logical-batch BatchNorm statistic, then mapped back through the
endpoint VJP and compared with NTP gradients from disjoint validation rows.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Iterable, Mapping

import torch
import torch.nn.functional as F
from transformers import set_seed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from audit_chemfm_mechanism import device_batch, disable_stochastic_behavior, geometry  # noqa: E402
from audit_sigreg_pair_specificity import (  # noqa: E402
    accumulate_autograd_result,
    empty_vector,
    endpoint_forward,
    endpoint_vjps,
    lora_parameters,
    ntp_gradient,
    relation,
    vector_norm,
)
from chemfm import MODEL_DIR, TOKENIZER_DIR, ReactionCollator, load_lora_model, load_reaction_tokenizer  # noqa: E402
from jepa import CLMJEPA, ProjectionHead, SIGReg  # noqa: E402
from train import load_adapter_checkpoint, read_rows, validate_serialization_endings  # noqa: E402


SEED = 533
LOGICAL_BATCH = 16
OUTER_COEFFICIENT = 2.0
SIGREG_COEFFICIENT = 4.0 * 0.01 / 0.99
DEFAULT_TRAIN = ROOT / "data" / "clm_jepa_uspto_mit_pilot_1280" / "uspto_mit_train.csv"
DEFAULT_HELDOUT = (
    ROOT / "data" / "clm_jepa_uspto_mit_validation_256"
    / "uspto_mit_validation_length_stratified_256.csv"
)


def fixed_chunks(rows, collator, *, batches: int, physical_batch: int, seed: int):
    required = batches * LOGICAL_BATCH
    if len(rows) < required:
        raise ValueError(f"audit needs at least {required} rows")
    order = torch.randperm(len(rows), generator=torch.Generator().manual_seed(seed)).tolist()
    selected = [rows[index] for index in order[:required]]
    return [
        [
            collator(logical[start:start + physical_batch])
            for start in range(0, LOGICAL_BATCH, physical_batch)
        ]
        for logical in (
            selected[start:start + LOGICAL_BATCH]
            for start in range(0, required, LOGICAL_BATCH)
        )
    ]


def collect_states(method, model, chunks):
    sources, targets = [], []
    with torch.no_grad():
        for raw in chunks:
            output = endpoint_forward(method, model, raw)
            sources.append(output.source_states.detach().float())
            targets.append(output.target_states.detach().float())
    return torch.cat(sources), torch.cat(targets)


def space_metrics(source: torch.Tensor, target: torch.Tensor) -> dict:
    centers = (source + target) * 0.5
    distances = torch.cdist(source.float(), target.float()).square() / source.size(-1)
    cosine = F.normalize(source.float(), dim=-1) @ F.normalize(target.float(), dim=-1).T
    indices = torch.arange(source.size(0), device=source.device)
    negative_distance = distances.masked_fill(torch.eye(len(source), device=source.device).bool(), math.inf).min(1).values
    negative_cosine = cosine.masked_fill(torch.eye(len(source), device=source.device).bool(), -math.inf).max(1).values
    return {
        "source": geometry(source),
        "target": geometry(target),
        "pair_centers": geometry(centers),
        "mse_alignment": float(F.mse_loss(source, target)),
        "euclidean_pair_margin": float((negative_distance - distances.diag()).mean()),
        "euclidean_retrieval_top1": float(distances.argmin(1).eq(indices).float().mean()),
        "cosine_pair_margin": float((cosine.diag() - negative_cosine).mean()),
        "cosine_retrieval_top1": float(cosine.argmax(1).eq(indices).float().mean()),
    }


def objective_endpoint_gradients(projector, source, target, *, seed: int):
    projector.train()
    source_leaf = source.detach().clone().requires_grad_(True)
    target_leaf = target.detach().clone().requires_grad_(True)
    projected = projector(torch.cat((source_leaf, target_leaf), dim=0))
    projected_source, projected_target = projected.split(source.size(0))
    mse = F.mse_loss(projected_source, projected_target)
    sigreg = SIGReg(seed=seed)(torch.stack((projected_source, projected_target)))
    projector_parameters = tuple(projector.parameters())
    mse_all = torch.autograd.grad(
        mse, (source_leaf, target_leaf, *projector_parameters), retain_graph=True,
    )
    sigreg_all = torch.autograd.grad(
        sigreg, (source_leaf, target_leaf, *projector_parameters),
    )
    endpoint = {
        "projected_mse": (mse_all[0], mse_all[1]),
        "projected_sigreg": (sigreg_all[0], sigreg_all[1]),
    }
    projector_norms = {}
    for name, gradients in (("projected_mse", mse_all[2:]), ("projected_sigreg", sigreg_all[2:])):
        projector_norms[name] = math.sqrt(sum(
            float(gradient.detach().float().square().sum())
            for gradient in gradients if gradient is not None
        ))
    return {
        "endpoint_gradients": endpoint,
        "projector_gradient_norms": projector_norms,
        "mse": float(mse.detach()),
        "sigreg": float(sigreg.detach()),
    }


def add_vectors(*terms: tuple[Mapping[str, torch.Tensor], float]):
    names = set().union(*(vector.keys() for vector, _ in terms))
    return {
        name: sum(
            (vector[name] * coefficient for vector, coefficient in terms if name in vector),
            torch.zeros_like(next(vector[name] for vector, _ in terms if name in vector)),
        )
        for name in names
    }


def summarize_alignment(vector, heldout_ntp):
    return {
        "norm": vector_norm(vector),
        "relative_norm_to_heldout_ntp": vector_norm(vector) / max(vector_norm(heldout_ntp), 1e-30),
        "relation_to_heldout_ntp": relation(vector, heldout_ntp),
    }


def load_projector(checkpoint: Path, device: torch.device) -> ProjectionHead:
    payload = torch.load(checkpoint / "projection_head.pt", map_location=device, weights_only=False)
    config = payload["configuration"]
    projector = ProjectionHead(
        int(config["input_dim"]), int(config["hidden_dim"]), int(config["output_dim"]),
    ).to(device)
    if projector.configuration() != config:
        raise ValueError("projection-head artifact does not match its declared architecture")
    projector.load_state_dict(payload["state_dict"])
    return projector


def audit_checkpoint(label, checkpoint, model, method, train_batches, heldout_batches, named_lora):
    load_adapter_checkpoint(model, checkpoint)
    projector = load_projector(checkpoint, model.device)
    heldout_chunks = [chunk for logical in heldout_batches for chunk in logical]
    heldout_loss, heldout_ntp, heldout_tokens = ntp_gradient(model, heldout_chunks, named_lora)
    batch_records = []
    for batch_index, chunks in enumerate(train_batches):
        source, target = collect_states(method, model, chunks)
        raw_geometry = space_metrics(source, target)
        projector.eval()
        with torch.no_grad():
            projected = projector(torch.cat((source, target), dim=0))
            projected_source, projected_target = projected.split(LOGICAL_BATCH)
            projected_geometry = space_metrics(projected_source, projected_target)
            projected_geometry["sigreg_geometry"] = float(
                SIGReg(seed=SEED)(torch.stack((projected_source, projected_target)))
            )
        objective = objective_endpoint_gradients(
            projector, source, target, seed=SEED + batch_index,
        )
        vectors = endpoint_vjps(
            model, method, chunks, objective["endpoint_gradients"], named_lora,
        )
        full = add_vectors(
            (vectors["projected_mse"], OUTER_COEFFICIENT),
            (vectors["projected_sigreg"], OUTER_COEFFICIENT * SIGREG_COEFFICIENT),
        )
        applied_sigreg = add_vectors(
            (vectors["projected_sigreg"], OUTER_COEFFICIENT * SIGREG_COEFFICIENT),
        )
        batch_records.append({
            "batch": batch_index,
            "raw_lm_space": raw_geometry,
            "projected_jepa_space": projected_geometry,
            "projected_losses": {
                "mse": objective["mse"],
                "sigreg": objective["sigreg"],
                "full_auxiliary": objective["mse"] + SIGREG_COEFFICIENT * objective["sigreg"],
            },
            "projector_gradient_norms": objective["projector_gradient_norms"],
            "lora_gradient_alignment": {
                "mse_raw": summarize_alignment(vectors["projected_mse"], heldout_ntp),
                "mse_applied": summarize_alignment(
                    add_vectors((vectors["projected_mse"], OUTER_COEFFICIENT)), heldout_ntp,
                ),
                "sigreg_raw": summarize_alignment(vectors["projected_sigreg"], heldout_ntp),
                "sigreg_applied": summarize_alignment(applied_sigreg, heldout_ntp),
                "full_auxiliary_applied": summarize_alignment(full, heldout_ntp),
            },
        })
    return {
        "checkpoint": label,
        "path": str(checkpoint.resolve()),
        "heldout_ntp_loss": heldout_loss,
        "heldout_ntp_tokens": heldout_tokens,
        "batches": batch_records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--epochs", type=int, nargs="+", default=(1, 2, 4))
    parser.add_argument("--train-manifest", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--heldout-manifest", type=Path, default=DEFAULT_HELDOUT)
    parser.add_argument("--physical-batch", type=int, default=2)
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise EnvironmentError("the projected-objective audit requires CUDA")
    if LOGICAL_BATCH % args.physical_batch:
        raise ValueError("physical batch must divide the logical batch of 16")
    set_seed(SEED)
    train_rows = read_rows("uspto_mit_synthesis", path=args.train_manifest)
    heldout_rows = read_rows("uspto_mit_synthesis", path=args.heldout_manifest)
    tokenizer = load_reaction_tokenizer(TOKENIZER_DIR)
    collator = ReactionCollator(tokenizer, task="forward")
    validate_serialization_endings(collator, train_rows, tokenizer.eos_token_id)
    validate_serialization_endings(collator, heldout_rows, tokenizer.eos_token_id)
    train_batches = fixed_chunks(
        train_rows, collator, batches=args.batches,
        physical_batch=args.physical_batch, seed=SEED,
    )
    heldout_batches = fixed_chunks(
        heldout_rows, collator, batches=args.batches,
        physical_batch=args.physical_batch, seed=SEED + 1,
    )
    model = load_lora_model(
        MODEL_DIR, tokenizer, attention_dropout=0.0,
        chemfm_vocab_size=len(tokenizer),
    ).cuda().eval()
    disable_stochastic_behavior(model)
    method = CLMJEPA([], tokenizer.eos_token_id, tokenizer.pad_token_id, sigreg_seed=SEED)
    named_lora = lora_parameters(model)
    checkpoints = []
    for epoch in args.epochs:
        checkpoint = args.checkpoint_root / f"epoch_{epoch}"
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        checkpoints.append(audit_checkpoint(
            f"epoch_{epoch}", checkpoint, model, method,
            train_batches, heldout_batches, named_lora,
        ))
    payload = {
        "schema_version": 1,
        "objective": {
            "projection": "D->2048->2048->64",
            "outer_coefficient": OUTER_COEFFICIENT,
            "sigreg_relative_coefficient": SIGREG_COEFFICIENT,
            "logical_batch": LOGICAL_BATCH,
        },
        "checkpoints": checkpoints,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "epochs": args.epochs}))


if __name__ == "__main__":
    main()
