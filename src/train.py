"""Shared ChemFM trainer with explicit method-family dispatch.

Native ChemFM, endpoint cLM-JEPA (``src/jepa.py``), and the dense causal
V-JEPA-2.1-style adaptation (``src/vjepa2_1.py``) share data loading, LoRA,
NTP, optimization, checkpointing, validation, and generation.  Only the
training-only auxiliary method differs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
import numpy as np
from peft.utils.save_and_load import load_peft_weights, set_peft_model_state_dict
from rdkit import Chem
from rdkit.Chem import rdFMCS
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader
from transformers import get_scheduler, set_seed

ROOT = Path(__file__).resolve().parents[1]

from chemfm import (  # noqa: E402
    IGNORE_INDEX, MODEL_DIR, TOKENIZER_DIR, ReactionCollator,
    generate_products_batch, canonicalize, load_lora_model, load_reaction_tokenizer,
)
from jepa import (  # noqa: E402
    CLMJEPA, add_predictor_tokens, extract_source_and_target,
    matched_derangement,
)
from gradient_interaction import (  # noqa: E402
    CAGRAD_C, apply_combination, combine_gradients,
)
from metrics import (  # noqa: E402
    canonical_set, effective_rank, rank_augmented_candidates,
    pca_structure, relationship_metrics, score_candidates,
)
from vjepa2_1 import (  # noqa: E402
    VJEPA21_PAPER, VJEPA21_UPSTREAM_COMMIT, DenseVJEPA21,
    DenseVJEPA21Config, component_gradient_norms, dense_trainable_parameters,
)
from stp import (  # noqa: E402
    STP_PAPER, STP_UPSTREAM_COMMIT, STP_UPSTREAM_REPOSITORY,
    PaperSemanticTubePrediction, SemanticTubePrediction,
)


ADAPTER_NAME = "USPTO-MIT-Synthesis"
ADAM_BETAS = (0.9, 0.999)
ADAM_EPSILON = 1e-8
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.05
MIN_LEARNING_RATE = 1e-5


TASKS = {
    "uspto_mit_synthesis": "forward",
    "orderly_forward": "forward",
    "non_uspto_forward": "forward",
    "metatrans_full": "metabolism",
    "uspto_50k_retro": "retro",
    "uspto_480k_template_heldout": "retro",
    "non_uspto_retro": "retro",
}
SIGREG_TRADEOFF = 0.01
PAIR_RESIDUAL_SHUFFLE_SEED = 1907

NATIVE_CONDITION = "native"
STP_CONDITION = "stp"
RELEASED_STP_CONDITION = "stp_released"
PAPER_STP_CONDITION = "stp_paper"
DENSE_VJEPA21_CONDITION = "clm_jepa_vjepa2_1"
ENDPOINT_JEPA_CONDITIONS = frozenset({
    "monitor",
    "clm_jepa",
    "clm_jepa_target_sg",
    "clm_jepa_mse",
    "clm_jepa_mse_sigreg",
    "clm_jepa_pair_residual",
    "shuffled",
    "jepa_only",
})
TRAINING_CONDITIONS = (
    NATIVE_CONDITION,
    STP_CONDITION,
    RELEASED_STP_CONDITION,
    PAPER_STP_CONDITION,
    *sorted(ENDPOINT_JEPA_CONDITIONS),
    DENSE_VJEPA21_CONDITION,
)


def condition_family(condition: str) -> str:
    """Return the maintained implementation family for a CLI condition."""
    if condition == NATIVE_CONDITION:
        return "native"
    if condition in {STP_CONDITION, RELEASED_STP_CONDITION}:
        return "semantic_tube_prediction_released"
    if condition == PAPER_STP_CONDITION:
        return "semantic_tube_prediction_paper"
    if condition in ENDPOINT_JEPA_CONDITIONS:
        return "endpoint_clm_jepa"
    if condition == DENSE_VJEPA21_CONDITION:
        return "dense_vjepa2_1"
    raise ValueError(f"unknown training condition: {condition}")


DATASET_SPLITS = {
    "uspto_mit_synthesis": {
        "train": "uspto_mit_synthesis/train_r_smiles.csv",
        "validation": "uspto_mit_synthesis/validation_r_smiles.csv",
        "test": "uspto_mit_synthesis/test_r_smiles.csv",
    },
    "orderly_forward": {
        "train": "orderly_forward/train.csv",
        "test": "orderly_forward/test.csv",
    },
    "non_uspto_forward": {"test": "non_uspto_forward/test.csv"},
    "metatrans_full": {
        "train": "metatrans/train.csv",
        "released_validation": "metatrans/released_validation.csv",
        "test": "metatrans/heldout_drug_test.csv",
    },
    "uspto_50k_retro": {
        "train": "uspto_50k/train_single.csv",
        "train_r_smiles": "uspto_50k/train_r_smiles.csv",
        "validation_r_smiles": "uspto_50k/validation_r_smiles.csv",
        "test_r_smiles": "uspto_50k/test_r_smiles.csv",
    },
    "uspto_480k_template_heldout": {
        "train": "uspto_480k_template_heldout/train.csv",
        "validation": "uspto_480k_template_heldout/validation.csv",
        "test": "uspto_480k_template_heldout/test.csv",
    },
    "non_uspto_retro": {"test": "non_uspto_retro/test.csv"},
}


@dataclass(frozen=True)
class TrackingContext:
    task: str
    dataset: str
    condition: str
    seed: int
    data_fraction: float
    resolved_hyperparameters: Mapping[str, Any]


def _safe_config(context: TrackingContext) -> dict[str, Any]:
    config = asdict(context)
    flattened = str(config).lower()
    if any(marker in flattened for marker in ("api_key", "password", "secret", "token=")):
        raise ValueError("tracking configuration must not contain credentials")
    return config


class WandbTracker:
    """Section 13 logging with credentials supplied only through the environment."""

    def __init__(self, context: TrackingContext, *, run_name: str, enabled: bool = True, wandb_module=None):
        self.enabled = enabled
        self.started_at = time.perf_counter()
        self.total_tokens = 0
        self.jepa_active_batches = 0
        self.run = None
        if not enabled:
            return
        offline = os.environ.get("WANDB_MODE", "").lower() in {"offline", "disabled"}
        if not offline and not os.environ.get("WANDB_API_KEY"):
            raise EnvironmentError("WANDB_API_KEY must be set in the environment")
        project = os.environ.get("WANDB_PROJECT", "clm-jepa")
        if project != "clm-jepa":
            raise ValueError("WANDB_PROJECT must be clm-jepa")
        if wandb_module is None:
            import wandb as wandb_module
        wandb_dir = os.environ.get("WANDB_DIR", "runs/wandb")
        Path(wandb_dir).mkdir(parents=True, exist_ok=True)
        kwargs = {
            "project": project,
            "name": run_name,
            "config": _safe_config(context),
            "job_type": "fine-tuning",
            "reinit": True,
            "dir": wandb_dir,
        }
        entity = os.environ.get("WANDB_ENTITY")
        if entity:
            kwargs["entity"] = entity
        self.run = wandb_module.init(**kwargs)

    def log_training_step(
        self, *, step: int, native_loss: float, jepa_loss: float | None,
        sigreg_loss: float | None = None,
        jepa_objective_loss: float | None = None,
        total_loss: float, gradient_norm: float, learning_rate: float,
        jepa_active: bool, batch_tokens: int, model_calls: int,
        effective_tokens: int, peak_vram_bytes: int,
        max_gradient_parameter: str, max_parameter_gradient_norm: float,
        estimated_flops: float | None = None,
        gradient_interaction: Mapping[str, Any] | None = None,
        extra_metrics: Mapping[str, float] | None = None,
    ) -> None:
        self.total_tokens += int(batch_tokens)
        self.jepa_active_batches += int(jepa_active)
        elapsed = max(time.perf_counter() - self.started_at, 1e-12)
        payload = {
            "train/native_loss": native_loss,
            "train/jepa_loss": jepa_loss,
            "train/sigreg_loss": sigreg_loss,
            "train/jepa_objective_loss": jepa_objective_loss,
            "train/total_loss": total_loss,
            "train/gradient_norm": gradient_norm,
            "train/max_gradient_parameter": max_gradient_parameter,
            "train/max_parameter_gradient_norm": max_parameter_gradient_norm,
            "train/learning_rate": learning_rate,
            "compute/jepa_active_batch": int(jepa_active),
            "compute/jepa_active_batches": self.jepa_active_batches,
            "compute/batch_tokens": int(batch_tokens),
            "compute/total_tokens": self.total_tokens,
            "compute/model_calls": int(model_calls),
            "compute/effective_tokens": int(effective_tokens),
            "compute/wall_time_seconds": elapsed,
            "compute/tokens_per_second": self.total_tokens / elapsed,
            "compute/peak_vram_bytes": int(peak_vram_bytes),
        }
        if estimated_flops is not None:
            payload["compute/estimated_flops"] = estimated_flops
        if gradient_interaction is not None:
            for name in (
                "cosine", "raw_auxiliary_to_main_norm_ratio",
                "auxiliary_to_main_norm_ratio", "modification_norm",
                "modification_relative_to_raw_sum", "main_coefficient",
                "auxiliary_coefficient", "auxiliary_gate",
                "cagrad_weight_main", "cagrad_weight_auxiliary",
                "cagrad_lambda",
            ):
                value = gradient_interaction.get(name)
                if value is not None:
                    payload[f"gradient/{name}"] = value
            payload["gradient/conflict"] = int(
                gradient_interaction.get("conflict", False)
            )
        if extra_metrics is not None:
            payload.update({f"dense_jepa/{name}": value for name, value in extra_metrics.items()})
        if self.run is not None:
            self.run.log(payload, step=step)

    def log_evaluation(
        self, *, step: int, split: str, task_metrics: Mapping[str, float],
        validity: float, native_loss: float,
    ) -> None:
        payload = {
            f"{split}/native_loss": native_loss,
            f"{split}/validity": validity,
        }
        payload.update({f"{split}/{name}": value for name, value in task_metrics.items()})
        if self.run is not None:
            self.run.log(payload, step=step)

    def finish(self, summary: Mapping[str, Any] | None = None) -> None:
        if self.run is None:
            return
        if summary:
            self.run.summary.update(dict(summary))
        self.run.finish()


def read_rows(
    dataset: str, split: str = "train", path: Path | None = None
) -> list[dict[str, str]]:
    available = DATASET_SPLITS[dataset]
    if path is None:
        if split not in available:
            raise ValueError(
                f"{dataset!r} has no {split!r} split; available splits: "
                f"{', '.join(sorted(available))}. External test sets must never be fine-tuned."
            )
        path = ROOT / "data" / available[split]
    if not path.exists():
        raise FileNotFoundError(f"dataset manifest is missing: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return [
            {**row, "src": row["source"], "tgt": row["target"]}
            for row in csv.DictReader(handle)
        ]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def trainable_parameter_sha256(model) -> str:
    """Fingerprint the exact initialized trainable state before optimization."""
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        value = parameter.detach().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


def reaction_row_fingerprint(row: Mapping[str, str]) -> str:
    payload = {
        key: row.get(key, "")
        for key in ("source", "target", "src", "tgt", "group_id", "example_id")
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _model_inputs(batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: batch[name] for name in ("input_ids", "attention_mask", "labels")}


def _target_identity(smiles: str, task: str) -> str:
    return canonical_set(smiles) if task == "retro" else canonicalize(smiles)


def _heavy_atoms(smiles: str) -> int:
    total = 0
    for component in smiles.split("."):
        molecule = Chem.MolFromSmiles(component)
        if molecule is None:
            return 0
        total += molecule.GetNumHeavyAtoms()
    return total


def attach_matched_targets(rows, tokenizer, task: str, seed: int) -> str:
    """Attach a global, identity-deranged target matched on tokens and heavy atoms."""
    count = len(rows)
    if count < 2:
        raise ValueError("matched-shuffled JEPA requires at least two training rows")
    identities = [_target_identity(row["tgt"], task) for row in rows]
    if any(not identity for identity in identities):
        raise ValueError("matched-shuffled JEPA requires valid target molecules")
    lengths = [
        len(ids) for ids in tokenizer(
            [row["tgt"] for row in rows], add_special_tokens=False
        )["input_ids"]
    ]
    atoms = [_heavy_atoms(row["tgt"]) for row in rows]
    rng = np.random.default_rng(seed)
    tie_scale = count * count + 1
    tie_breakers = rng.integers(0, tie_scale, size=(count, count), dtype=np.int64)
    costs = np.empty((count, count), dtype=np.int64)
    for left in range(count):
        for right in range(count):
            base = abs(lengths[left] - lengths[right]) + abs(atoms[left] - atoms[right])
            costs[left, right] = base * tie_scale + tie_breakers[left, right]
    forbidden = int(costs.max()) * count + 1
    for left in range(count):
        for right in range(count):
            if left == right or identities[left] == identities[right]:
                costs[left, right] = forbidden
    left_indices, right_indices = linear_sum_assignment(costs)
    assignment = [0] * count
    for left, right in zip(left_indices.tolist(), right_indices.tolist()):
        assignment[left] = right
    if any(
        left == right or identities[left] == identities[right]
        for left, right in enumerate(assignment)
    ):
        raise ValueError("no chemically unequal matched-target derangement exists")
    mapping = []
    for left, right in enumerate(assignment):
        rows[left]["jepa_tgt"] = rows[right]["tgt"]
        rows[left]["jepa_target_example_id"] = rows[right].get("example_id", str(right))
        mapping.append((rows[left].get("example_id", str(left)), rows[left]["jepa_target_example_id"]))
    return hashlib.sha256(json.dumps(mapping, separators=(",", ":")).encode()).hexdigest()


def validate_serialization_endings(collator, rows, eos_token_id: int) -> None:
    for start in range(0, len(rows), 64):
        batch = collator(rows[start:start + 64])
        sources, targets = extract_source_and_target(batch)
        if any(int(source[-1]) != eos_token_id for source in sources):
            raise ValueError("source truncation removed a required <eos>")
        if any(int(target[-1]) != eos_token_id for target in targets):
            raise ValueError("target truncation removed a required <eos>")
        if "jepa_target_ids" in batch:
            jepa_targets = [
                row[mask.bool()]
                for row, mask in zip(
                    batch["jepa_target_ids"], batch["jepa_target_attention_mask"]
                )
            ]
            if any(int(target[-1]) != eos_token_id for target in jepa_targets):
                raise ValueError("shuffled target truncation removed a required <eos>")


def native_loss(model, loader) -> float:
    model.eval()
    total = tokens = 0
    with torch.inference_mode():
        for batch in loader:
            inputs = {
                k: v.to(model.device, non_blocking=loader.pin_memory)
                for k, v in batch.items() if torch.is_tensor(v)
            }
            count = int(inputs["labels"].ne(IGNORE_INDEX).sum())
            total += float(model(**_model_inputs(inputs)).loss) * count
            tokens += count
    return total / max(1, tokens)


def beam_evaluate(
    model, tokenizer, collator, rows, task, windows=10,
    generation_batch_size: int = 1,
):
    if generation_batch_size < 1:
        raise ValueError("generation batch size must be positive")
    grouped: dict[str, list[dict[str, str]]] = {}
    for index, row in enumerate(rows):
        grouped.setdefault(row.get("group_id", f"row-{index}"), []).append(row)
    evaluation_rows = []
    prompts = []
    prompt_locations = []
    augmentations = []
    model.eval()
    for group_index, group in enumerate(grouped.values()):
        target_identity = _target_identity(group[0]["tgt"], task)
        if any(_target_identity(row["tgt"], task) != target_identity for row in group):
            raise ValueError("every R-SMILES validation group must share one target identity")
        evaluation_rows.append(group[0])
        augmentations.append([None] * len(group))
        group_prompts = collator(group)["generation_prompts"]
        for augmentation_index, prompt in enumerate(group_prompts):
            prompts.append(prompt)
            prompt_locations.append((group_index, augmentation_index))

    encoded_prompts = tokenizer(
        prompts, add_special_tokens=False, truncation=True
    )["input_ids"]
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    prompt_order = list(range(len(prompts)))
    if generation_batch_size > 1:
        prompt_order.sort(key=lambda index: (len(encoded_prompts[index]), index))
    for start in range(0, len(prompt_order), generation_batch_size):
        indices = prompt_order[start:start + generation_batch_size]
        predictions = generate_products_batch(
            model, tokenizer, [prompts[index] for index in indices],
            max_length=collator.source_max_len + collator.target_max_len,
            num_beams=windows, num_return_sequences=windows,
            pad_unequal_prompts=generation_batch_size > 1,
        )
        for prompt_index, candidates in zip(indices, predictions):
            group_index, augmentation_index = prompt_locations[prompt_index]
            augmentations[group_index][augmentation_index] = candidates

    if any(candidates is None for group in augmentations for candidates in group):
        raise RuntimeError("generation did not populate every R-SMILES view")
    workers = min(8, max(1, (os.cpu_count() or 2) - 1), len(augmentations))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        candidate_rows = list(executor.map(
            lambda values: rank_augmented_candidates(values, task, windows),
            augmentations,
        ))
    return score_candidates(evaluation_rows, candidate_rows, task)


def _representation_sample(rows, task: str, limit: int = 32):
    """Select one canonical target identity per diagnostic example."""
    sample = []
    seen = set()
    for row in rows:
        identity = _target_identity(row["tgt"], task)
        if not identity or identity in seen:
            continue
        seen.add(identity)
        sample.append(row)
        if len(sample) == limit:
            break
    if len(sample) < 2:
        raise ValueError("representation diagnostics require two canonical target identities")
    return sample


def _mcs_overlap_atoms(component: str, target: str) -> int:
    component_molecule = Chem.MolFromSmiles(component)
    if component_molecule is None:
        return 0
    best = 0
    for target_component in target.split("."):
        target_molecule = Chem.MolFromSmiles(target_component)
        if target_molecule is None:
            continue
        result = rdFMCS.FindMCS(
            [component_molecule, target_molecule],
            ringMatchesRingOnly=True,
            completeRingsOnly=True,
            timeout=1,
        )
        best = max(best, int(result.numAtoms))
    return best


def _largest_target_overlap_component(row) -> tuple[int, str]:
    """Return a transparent proxy for a target-contributing source component."""
    candidates = []
    for index, component in enumerate(row["src"].split(".")):
        molecule = Chem.MolFromSmiles(component)
        heavy_atoms = molecule.GetNumHeavyAtoms() if molecule else 0
        candidates.append(
            (_mcs_overlap_atoms(component, row["tgt"]), heavy_atoms, -index, index, component)
        )
    _, _, _, index, component = max(candidates)
    return index, component


def _space_geometry_metrics(
    sources: torch.Tensor, targets: torch.Tensor,
) -> dict[str, float]:
    sources = sources.float()
    targets = targets.float()
    centers = (sources + targets) * 0.5
    def mean_direction_energy(values: torch.Tensor) -> float:
        return float(
            values.mean(dim=0).square().sum()
            / values.square().sum(dim=1).mean().clamp_min(1e-30)
        )
    return {
        "source_variance": float(sources.var(dim=0, unbiased=False).mean()),
        "target_variance": float(targets.var(dim=0, unbiased=False).mean()),
        "pair_center_spread": float(
            centers.var(dim=0, unbiased=False).mean().sqrt()
        ),
        "source_effective_rank": effective_rank(sources),
        "target_effective_rank": effective_rank(targets),
        "source_mean_direction_energy": mean_direction_energy(sources),
        "target_mean_direction_energy": mean_direction_energy(targets),
    }


def representation_diagnostics(
    model, method, collator, rows, k, seed, task,
    limit: int = 32,
    physical_batch_size: int = 4,
):
    if physical_batch_size < 1:
        raise ValueError("diagnostic physical batch size must be positive")
    sample = _representation_sample(rows, task, limit=limit)
    model.eval()
    def states(view, *, include_targets: bool = False):
        source_chunks = []
        target_chunks = []
        for start in range(0, len(view), physical_batch_size):
            raw = collator(view[start : start + physical_batch_size])
            tensors = {
                name: value.to(model.device)
                for name, value in raw.items() if torch.is_tensor(value)
            }
            output = method(
                model, tensors, k=k, jepa_weight=0.0, monitor_only=True,
                jepa_loss_type="mse", force_jepa_active=True,
                endpoint_only=True, representation_only=True,
            )
            source_chunks.append(output.source_states.float())
            if include_targets:
                target_chunks.append(output.target_states.float())
        sources = torch.cat(source_chunks)
        if not include_targets:
            return sources
        return sources, torch.cat(target_chunks)

    with torch.inference_mode():
        sources, targets = states(sample, include_targets=True)
    identities = [_target_identity(row["tgt"], task) for row in sample]
    encoded_targets = collator.tokenizer(
        [row["tgt"] for row in sample], add_special_tokens=False
    )["input_ids"]
    token_lengths = {
        identity: len(encoded)
        for identity, encoded in zip(identities, encoded_targets)
    }
    heavy_atoms = {
        identity: _heavy_atoms(row["tgt"])
        for identity, row in zip(identities, sample)
    }
    metrics = relationship_metrics(sources, targets, identities, token_lengths, heavy_atoms, seed)
    metrics.update(_space_geometry_metrics(sources, targets))
    centers = (sources + targets) * 0.5
    metrics["pca_structure"] = {
        "source": pca_structure(sources),
        "target": pca_structure(targets),
        "pair_center": pca_structure(centers),
    }
    joint = torch.cat((sources, targets), dim=0).float()
    joint_mean = joint.mean(dim=0, keepdim=True)
    centered_joint = joint - joint_mean
    q = min(8, centered_joint.size(0) - 1, centered_joint.size(1))
    _, _, components = torch.pca_lowrank(
        centered_joint, q=q, center=False, niter=6,
    )
    basis = components[:, : min(2, q)]
    residual_sources = sources.float() - joint_mean
    residual_targets = targets.float() - joint_mean
    residual_sources -= (residual_sources @ basis) @ basis.T
    residual_targets -= (residual_targets @ basis) @ basis.T
    metrics["residual_pc2"] = relationship_metrics(
        residual_sources, residual_targets, identities,
        token_lengths, heavy_atoms, seed,
    )
    differences = sources - targets
    singular = torch.linalg.svdvals(differences - differences.mean(0, keepdim=True))
    energy = singular.square()
    metrics["source_target_difference_effective_rank"] = effective_rank(differences)
    metrics["source_target_difference_top_singular_energy"] = float(
        energy[0] / energy.sum().clamp_min(1e-30)
    )
    alternate, ablated, replaced = [], [], []
    contributors = [_largest_target_overlap_component(row) for row in sample]
    contributor_molecules = [Chem.MolFromSmiles(component) for _, component in contributors]
    contributor_sizes = [
        molecule.GetNumHeavyAtoms() if molecule else 0
        for molecule in contributor_molecules
    ]
    for row_index, row in enumerate(sample):
        components = row["src"].split(".")
        randomized = []
        for component in components:
            molecule = Chem.MolFromSmiles(component)
            randomized.append(Chem.MolToSmiles(molecule, doRandom=True) if molecule else component)
        alternate.append({"src": ".".join(randomized), "tgt": row["tgt"]})
        contributor_index, contributor = contributors[row_index]
        without = [
            component for index, component in enumerate(components)
            if index != contributor_index
        ]
        ablated.append({"src": ".".join(without), "tgt": row["tgt"]})
        alternatives = [
            index for index, other in enumerate(contributors)
            if index != row_index and other[1] != contributor
        ]
        replacement_index = min(
            alternatives,
            key=lambda index: (
                abs(contributor_sizes[row_index] - contributor_sizes[index]),
                contributors[index][1],
            ),
        )
        replacement_components = list(components)
        replacement_components[contributor_index] = contributors[replacement_index][1]
        replaced.append({"src": ".".join(replacement_components), "tgt": row["tgt"]})
    with torch.inference_mode():
        alternate_states = states(alternate)
        ablated_states = states(ablated)
        replaced_states = states(replaced)
    metrics["diagnostic_sample_rows"] = len(sample)
    metrics["diagnostic_unique_target_identities"] = len(set(identities))
    metrics["necessary_component_policy"] = (
        "source component with the largest ring-aware MCS atom overlap to any target component; "
        "heavy-atom count and original order break ties"
    )
    metrics["alternate_smiles_cosine"] = float(
        F.cosine_similarity(sources, alternate_states, dim=-1).mean()
    )
    metrics["necessary_component_ablation_cosine"] = float(
        F.cosine_similarity(sources, ablated_states, dim=-1).mean()
    )
    metrics["necessary_component_sensitivity"] = 1.0 - metrics["necessary_component_ablation_cosine"]
    metrics["necessary_component_replacement_cosine"] = float(
        F.cosine_similarity(sources, replaced_states, dim=-1).mean()
    )
    metrics["necessary_component_replacement_sensitivity"] = (
        1.0 - metrics["necessary_component_replacement_cosine"]
    )
    return metrics


def validation_selector(metrics: Mapping[str, float], task: str) -> tuple[float, ...]:
    if task == "metabolism":
        return (metrics["recall_at5"], metrics["lower_bound_precision_at5"])
    return (metrics["exact_top1"],)


def _adapter_weights_dir(checkpoint: Path) -> Path:
    nested = checkpoint / ADAPTER_NAME
    return nested if nested.exists() else checkpoint


def load_adapter_checkpoint(model, checkpoint: Path) -> None:
    weights = load_peft_weights(str(_adapter_weights_dir(checkpoint)), device=str(model.device))
    result = set_peft_model_state_dict(model, weights, adapter_name=ADAPTER_NAME)
    if getattr(result, "unexpected_keys", None):
        raise RuntimeError(f"unexpected adapter checkpoint keys: {result.unexpected_keys}")


def save_training_checkpoint(
    checkpoint: Path, model, tokenizer, optimizer, scheduler, generator, method,
    *, epoch: int, global_step: int, planned_epochs: int, curves, epoch_history,
    best_selector, best_checkpoint: str | None, elapsed_wall_time_seconds: float,
) -> None:
    checkpoint.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(
        checkpoint, safe_serialization=True, selected_adapters=[ADAPTER_NAME],
        # embed_tokens and lm_head are already preserved by ChemFM's
        # modules_to_save; PEFT's resized-embedding auto-export duplicates
        # wrapper/original tensors and cannot be reloaded by PEFT 0.13.2.
        save_embedding_layers=False,
    )
    tokenizer.save_pretrained(checkpoint)
    state = {
        "epoch": epoch,
        "global_step": global_step,
        "planned_epochs": planned_epochs,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "loader_generator_state": generator.get_state(),
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_states": torch.cuda.get_rng_state_all(),
        "curves": curves,
        "epoch_history": epoch_history,
        "best_selector": best_selector,
        "best_checkpoint": best_checkpoint,
        "elapsed_wall_time_seconds": elapsed_wall_time_seconds,
    }
    if isinstance(method, DenseVJEPA21):
        state["dense_vjepa2_1"] = method.checkpoint_state()
    elif isinstance(method, SemanticTubePrediction):
        state["stp_generator_state"] = method.generator_state()
    else:
        state["jepa_dropout_generator_state"] = method.jepa_dropout_generator.get_state()
        state["sigreg_global_step"] = method.sigreg.global_step
    torch.save(state, checkpoint / "training_state.pt")


def restore_training_checkpoint(
    checkpoint: Path, model, optimizer, scheduler, generator,
    planned_epochs: int, method=None,
):
    load_adapter_checkpoint(model, checkpoint)
    state = torch.load(
        checkpoint / "training_state.pt", map_location=model.device, weights_only=False
    )
    if state["planned_epochs"] != planned_epochs:
        raise ValueError("resume must use the checkpoint's original planned epoch budget")
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    # torch.load(map_location=model.device) also relocates the CPU DataLoader
    # generator state. CPU generators reject CUDA byte tensors, so restore the
    # state explicitly on its required device.
    generator.set_state(state["loader_generator_state"].cpu())
    random.setstate(state["python_rng_state"])
    np.random.set_state(state["numpy_rng_state"])
    torch.set_rng_state(state["torch_rng_state"].cpu())
    torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda_rng_states"]])
    if method is not None and "jepa_dropout_generator_state" in state:
        method.jepa_dropout_generator.set_state(
            state["jepa_dropout_generator_state"].cpu()
        )
    if method is not None and "sigreg_global_step" in state:
        method.sigreg.global_step = state["sigreg_global_step"]
    if isinstance(method, SemanticTubePrediction):
        if "stp_generator_state" not in state:
            raise ValueError("resume checkpoint is missing STP sampler state")
        method.set_generator_state(state["stp_generator_state"], model.device)
    if isinstance(method, DenseVJEPA21):
        if "dense_vjepa2_1" not in state:
            raise ValueError("resume checkpoint is missing dense V-JEPA 2.1 state")
        method.load_checkpoint_state(state["dense_vjepa2_1"], model)
    return state


def gradient_diagnostics(
    model, extra_modules: Sequence[tuple[str, torch.nn.Module]] = (),
) -> tuple[float, tuple[str, float]]:
    """Clip gradients and preserve the established per-parameter diagnostics."""
    named_parameters = [(f"model.{name}", parameter) for name, parameter in model.named_parameters()]
    for prefix, module in extra_modules:
        named_parameters.extend(
            (f"{prefix}.{name}", parameter)
            for name, parameter in module.named_parameters()
            if parameter.requires_grad
        )
    total = torch.nn.utils.clip_grad_norm_(
        [parameter for _, parameter in named_parameters], 1.0,
    )
    total_value = float(total)
    clip_coefficient = min(1.0, 1.0 / (total_value + 1e-6))
    names = []
    clipped_norms = []
    for name, parameter in named_parameters:
        if parameter.grad is None:
            continue
        names.append(name)
        clipped_norms.append(parameter.grad.detach().float().norm())
    if not clipped_norms:
        return total_value, ("", 0.0)
    # Preserve the exact diagnostic while replacing one host synchronization
    # per tensor with one batched reduction and one final transfer.
    norms = torch.stack(clipped_norms)
    largest_index = int(norms.argmax())
    original_norm = float(norms[largest_index]) / max(clip_coefficient, 1e-30)
    return total_value, (names[largest_index], original_norm)


def synchronized_loss_means(records: list[dict[str, Any]]) -> dict[str, float | None]:
    """Aggregate GPU-resident logging scalars with one device synchronization."""
    keys = ("native_loss", "jepa_loss", "sigreg_loss", "jepa_objective_loss", "total_loss")
    means: list[torch.Tensor] = []
    present: list[bool] = []
    device = next(
        value.device
        for row in records for value in row.values()
        if torch.is_tensor(value)
    )
    for key in keys:
        values = [row[key] for row in records if row[key] is not None]
        present.append(bool(values))
        means.append(
            torch.stack(values).mean() if values else torch.zeros((), device=device)
        )
    synchronized = torch.stack(means).float().cpu().tolist()
    return {
        key: (value if exists else None)
        for key, value, exists in zip(keys, synchronized, present)
    }


def _rng_snapshot() -> tuple[torch.Tensor, list[torch.Tensor]]:
    return torch.get_rng_state(), torch.cuda.get_rng_state_all()


def _restore_rng(snapshot: tuple[torch.Tensor, list[torch.Tensor]]) -> None:
    cpu, cuda = snapshot
    torch.set_rng_state(cpu)
    torch.cuda.set_rng_state_all(cuda)


def raw_auxiliary_vjp(
    sigreg,
    source_states: torch.Tensor,
    target_states: torch.Tensor,
    *,
    sigreg_coefficient: float,
) -> dict[str, torch.Tensor]:
    """Differentiate exact MSE+SIGReg on the raw logical endpoint batch."""
    if source_states.ndim != 2 or target_states.shape != source_states.shape:
        raise ValueError("JEPA expects matched 2D logical endpoint batches")
    if source_states.size(0) < 2:
        raise ValueError("SIGReg requires at least two pairs")
    sources = source_states.float().detach().requires_grad_(True)
    targets = target_states.float().detach().requires_grad_(True)
    mse = F.mse_loss(sources, targets)
    sigreg_loss = sigreg(torch.stack((sources, targets)))
    objective = mse + sigreg_coefficient * sigreg_loss
    source_gradients, target_gradients = torch.autograd.grad(
        objective, (sources, targets)
    )
    return {
        "mse": mse.detach(),
        "sigreg": sigreg_loss.detach(),
        "objective": objective.detach(),
        "source_gradients": source_gradients.detach(),
        "target_gradients": target_gradients.detach(),
    }


def raw_pair_residual_vjp(
    source_states: torch.Tensor,
    target_states: torch.Tensor,
    permutation: Sequence[int],
) -> dict[str, torch.Tensor | float]:
    """Differentiate the audited pair-specific MSE gradient residual.

    The intervention is a gradient contrast, not a replacement JEPA loss:
    ``grad(MSE(source, true_target)) - grad(MSE(source, shuffled_target))``.
    SIGReg is absent because its marginal statistic is invariant to target
    permutation and therefore cancels from the exact true-minus-shuffled
    auxiliary gradient.
    """
    if source_states.ndim != 2 or target_states.shape != source_states.shape:
        raise ValueError("pair-residual JEPA expects matched 2D endpoint batches")
    count = source_states.size(0)
    if count < 2:
        raise ValueError("pair-residual JEPA requires at least two pairs")
    permutation = list(permutation)
    if sorted(permutation) != list(range(count)):
        raise ValueError("pair-residual shuffle must be a permutation")
    if any(index == shuffled for index, shuffled in enumerate(permutation)):
        raise ValueError("pair-residual shuffle must be a derangement")

    sources = source_states.float().detach().requires_grad_(True)
    targets = target_states.float().detach().requires_grad_(True)
    true_mse = F.mse_loss(sources, targets)
    shuffled_mse = F.mse_loss(sources, targets[permutation])
    true_source, true_target = torch.autograd.grad(
        true_mse, (sources, targets), retain_graph=True,
    )
    shuffled_source, shuffled_target = torch.autograd.grad(
        shuffled_mse, (sources, targets),
    )
    residual_source = true_source - shuffled_source
    residual_target = true_target - shuffled_target

    true_flat = torch.cat((true_source.flatten(), true_target.flatten()))
    shuffled_flat = torch.cat((shuffled_source.flatten(), shuffled_target.flatten()))
    denominator = true_flat.norm() * shuffled_flat.norm()
    cosine = (
        float(torch.dot(true_flat, shuffled_flat) / denominator)
        if float(denominator) > 0.0 else float("nan")
    )
    true_norm = float(true_flat.norm())
    residual_norm = float(torch.cat(
        (residual_source.flatten(), residual_target.flatten())
    ).norm())
    return {
        "true_mse": true_mse.detach(),
        "shuffled_mse": shuffled_mse.detach(),
        "residual_objective": (true_mse - shuffled_mse).detach(),
        "source_gradients": residual_source.detach(),
        "target_gradients": residual_target.detach(),
        "true_source_gradients": true_source.detach(),
        "true_target_gradients": true_target.detach(),
        "shuffled_source_gradients": shuffled_source.detach(),
        "shuffled_target_gradients": shuffled_target.detach(),
        "endpoint_true_shuffle_gradient_cosine": cosine,
        "endpoint_residual_over_true_norm": (
            residual_norm / true_norm if true_norm else float("nan")
        ),
    }


def _accumulate_gradients(
    buffers: list[torch.Tensor | None],
    gradients: tuple[torch.Tensor | None, ...],
) -> None:
    for index, gradient in enumerate(gradients):
        if gradient is None:
            continue
        detached = gradient.detach()
        if buffers[index] is None:
            buffers[index] = detached
        else:
            buffers[index].add_(detached)


def adamw_residual_update_diagnostics(
    optimizer: torch.optim.Optimizer,
    parameters: Sequence[torch.nn.Parameter],
    main_gradients: Sequence[torch.Tensor],
    applied_residual_gradients: Sequence[torch.Tensor],
    *, native_gradient_scale: float = 1.0,
    combined_gradient_scale: float = 1.0,
) -> dict[str, float]:
    """Measure the residual's counterfactual effect after AdamW preconditioning.

    This is read-only: it evaluates the next adaptive gradient update from the
    current moments once with the native gradient and once with the combined
    gradient. Decoupled weight decay cancels from their difference and is
    intentionally excluded from both vectors.
    """
    if not parameters or not (
        len(parameters) == len(main_gradients) == len(applied_residual_gradients)
    ):
        raise ValueError("AdamW diagnostic inputs must be nonempty and equal length")
    groups = {
        id(parameter): group
        for group in optimizer.param_groups for parameter in group["params"]
    }
    device = parameters[0].device
    main_squared = torch.zeros((), device=device, dtype=torch.float64)
    effect_squared = torch.zeros_like(main_squared)
    dot = torch.zeros_like(main_squared)
    for parameter, main, residual in zip(
        parameters, main_gradients, applied_residual_gradients
    ):
        group = groups.get(id(parameter))
        if group is None:
            raise ValueError("diagnosed parameter is absent from optimizer")
        if group.get("amsgrad", False) or group.get("maximize", False):
            raise ValueError("diagnostic supports the experiment's standard AdamW only")
        beta1, beta2 = group["betas"]
        epsilon = group["eps"]
        learning_rate = group["lr"]
        state = optimizer.state.get(parameter, {})
        step_value = state.get("step", 0)
        step = int(step_value.item()) if torch.is_tensor(step_value) else int(step_value)
        next_step = step + 1
        exp_avg = state.get("exp_avg")
        exp_avg_sq = state.get("exp_avg_sq")
        if exp_avg is None:
            exp_avg_float = torch.zeros_like(parameter, dtype=torch.float32)
            exp_avg_sq_float = torch.zeros_like(parameter, dtype=torch.float32)
        else:
            exp_avg_float = exp_avg.float()
            exp_avg_sq_float = exp_avg_sq.float()

        def adaptive_update(gradient: torch.Tensor) -> torch.Tensor:
            gradient = gradient.float()
            moment = beta1 * exp_avg_float + (1.0 - beta1) * gradient
            variance = beta2 * exp_avg_sq_float + (1.0 - beta2) * gradient.square()
            corrected_moment = moment / (1.0 - beta1 ** next_step)
            corrected_variance = variance / (1.0 - beta2 ** next_step)
            return learning_rate * corrected_moment / (
                corrected_variance.sqrt() + epsilon
            )

        main_update = adaptive_update(main * native_gradient_scale)
        combined_update = adaptive_update(
            (main + residual) * combined_gradient_scale
        )
        effect = combined_update - main_update
        main_squared += main_update.double().square().sum()
        effect_squared += effect.double().square().sum()
        dot += (main_update.double() * effect.double()).sum()
    main_norm = main_squared.sqrt()
    effect_norm = effect_squared.sqrt()
    denominator = main_norm * effect_norm
    main_value, effect_value, dot_value, denominator_value = torch.stack(
        (main_norm, effect_norm, dot, denominator)
    ).cpu().tolist()
    return {
        "native_adaptive_update_norm": main_value,
        "residual_adaptive_update_effect_norm": effect_value,
        "residual_to_native_adaptive_update_norm_ratio": (
            effect_value / main_value if main_value else float("nan")
        ),
        "residual_effect_native_update_cosine": (
            dot_value / denominator_value if denominator_value else float("nan")
        ),
    }


def read_only_global_gradient_norm(model) -> float:
    """Return the model-wide gradient norm without clipping or mutation."""
    norms = [
        parameter.grad.detach().float().norm()
        for parameter in model.parameters() if parameter.grad is not None
    ]
    if not norms:
        return 0.0
    return float(torch.linalg.vector_norm(torch.stack(norms)))


def train_mse_sigreg_gradient_epoch(
    *, model, method, loader, optimizer, scheduler, epoch: int,
    logical_batch_size: int, physical_batch_size: int,
    actual_lambda: float, native_weight: float, sigreg_tradeoff: float,
    sigreg_relative_scale: float,
    gradient_interaction: str,
    profile_training_phases: bool,
    jepa_ratio: float, non_embedding_parameters: int, pin_memory: bool,
    tracker: WandbTracker, global_step: int,
) -> tuple[list[dict[str, Any]], int]:
    """Train exact raw-endpoint MSE+SIGReg and combine LoRA gradients."""
    if logical_batch_size % physical_batch_size:
        raise ValueError("SIGReg logical batch must be divisible by physical batch size")
    chunks_per_step = logical_batch_size // physical_batch_size
    if len(loader) % chunks_per_step:
        raise ValueError("training exposure must divide into complete SIGReg logical batches")
    relative_coefficient = (
        sigreg_relative_scale * sigreg_tradeoff / (1.0 - sigreg_tradeoff)
    )
    named_trainable = [
        (name, parameter) for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    trainable = [parameter for _, parameter in named_trainable]
    lora_indices = [
        index for index, (name, _) in enumerate(named_trainable)
        if "lora_A" in name or "lora_B" in name
    ]
    if not lora_indices:
        raise RuntimeError("gradient interaction requires trainable LoRA A/B parameters")
    lora_parameters = [trainable[index] for index in lora_indices]
    lora_index_set = set(lora_indices)
    iterator = iter(loader)
    records: list[dict[str, Any]] = []
    for logical_index in range(len(loader) // chunks_per_step):
        data_started = time.perf_counter()
        raw_chunks = list(itertools.islice(iterator, chunks_per_step))
        data_seconds = time.perf_counter() - data_started
        if len(raw_chunks) != chunks_per_step:
            raise RuntimeError("incomplete SIGReg logical batch")
        active = method.sample_jepa_activity(jepa_ratio)
        source_chunks: list[torch.Tensor] = []
        target_chunks: list[torch.Tensor] = []
        replay_states: list[tuple[torch.Tensor, list[torch.Tensor]]] = []
        first_pass_seconds = 0.0
        jepa_mse = None
        jepa_sigreg = None
        jepa_objective = None
        source_gradients = None
        target_gradients = None
        if active:
            if profile_training_phases:
                torch.cuda.synchronize()
            first_started = time.perf_counter()
            with torch.no_grad():
                for raw in raw_chunks:
                    replay_states.append(_rng_snapshot())
                    batch = {
                        name: value.to(model.device, non_blocking=pin_memory)
                        for name, value in raw.items() if torch.is_tensor(value)
                    }
                    output = method(
                        model, batch, k=0, jepa_weight=0.0,
                        native_weight=native_weight, monitor_only=True,
                        stop_gradient_target=False, jepa_loss_type="mse",
                        sigreg_tradeoff=0.0, representation_only=True,
                        jepa_ratio=jepa_ratio, force_jepa_active=True,
                        # The statistics pass consumes only endpoint states.  Skipping
                        # the LM head is mathematically exact and avoids projecting
                        # every token twice, independently of the optional compact
                        # native-logit path used by the gradient-bearing pass.
                        endpoint_only=True,
                    )
                    source_chunks.append(output.source_states.float())
                    target_chunks.append(output.target_states.float())
            final_rng = _rng_snapshot()
            raw_sources, raw_targets = torch.cat(source_chunks), torch.cat(target_chunks)
            if raw_sources.size(0) != logical_batch_size:
                raise RuntimeError("auxiliary did not receive the complete logical JEPA batch")
            auxiliary_result = raw_auxiliary_vjp(
                method.sigreg, raw_sources, raw_targets,
                sigreg_coefficient=relative_coefficient,
            )
            jepa_mse = auxiliary_result["mse"]
            jepa_sigreg = auxiliary_result["sigreg"]
            jepa_objective = auxiliary_result["objective"]
            source_gradients = auxiliary_result["source_gradients"].split(
                physical_batch_size
            )
            target_gradients = auxiliary_result["target_gradients"].split(
                physical_batch_size
            )
            if profile_training_phases:
                torch.cuda.synchronize()
            first_pass_seconds = time.perf_counter() - first_started

        loss_records: list[dict[str, Any]] = []
        main_gradients: list[torch.Tensor | None] = [None] * len(trainable)
        auxiliary_gradients: list[torch.Tensor | None] = [None] * len(trainable)
        batch_tokens = 0
        effective_tokens = 0
        if profile_training_phases:
            torch.cuda.synchronize()
        gradient_started = time.perf_counter()
        for chunk_index, raw in enumerate(raw_chunks):
            if active:
                _restore_rng(replay_states[chunk_index])
            raw_tokens = int(raw["attention_mask"].sum())
            batch = {
                name: value.to(model.device, non_blocking=pin_memory)
                for name, value in raw.items() if torch.is_tensor(value)
            }
            output = method(
                model, batch, k=0,
                jepa_weight=0.0,
                native_weight=native_weight,
                monitor_only=active,
                stop_gradient_target=False, jepa_loss_type="mse",
                sigreg_tradeoff=0.0,
                jepa_ratio=jepa_ratio, force_jepa_active=active,
                representation_only=active,
            )
            native_component = output.native_loss * native_weight / chunks_per_step
            if active:
                auxiliary_surrogate = (
                    output.source_states
                    * source_gradients[chunk_index].to(output.source_states.dtype)
                ).sum() + (
                    output.target_states
                    * target_gradients[chunk_index].to(output.target_states.dtype)
                ).sum()
                gradients_main = torch.autograd.grad(
                    native_component, trainable, retain_graph=True,
                    allow_unused=True,
                )
                gradients_auxiliary = torch.autograd.grad(
                    auxiliary_surrogate, trainable, allow_unused=True,
                )
                _accumulate_gradients(main_gradients, gradients_main)
                _accumulate_gradients(auxiliary_gradients, gradients_auxiliary)
            else:
                native_component.backward()
            if not torch.isfinite(output.native_loss):
                raise FloatingPointError(
                    f"non-finite loss in epoch {epoch}, SIGReg group {logical_index + 1}"
                )
            batch_tokens += raw_tokens
            effective_tokens += raw_tokens * (4 if active else 1)
            loss_records.append({
                "native_loss": output.native_loss.detach(),
                "jepa_loss": None,
                "sigreg_loss": None,
                "jepa_objective_loss": None,
                "total_loss": output.loss.detach(),
            })
        interaction_metrics = None
        if active:
            _restore_rng(final_rng)
            zero = lambda parameter: torch.zeros_like(parameter)
            complete_main = [
                gradient if gradient is not None else zero(parameter)
                for parameter, gradient in zip(trainable, main_gradients)
            ]
            complete_auxiliary = [
                gradient if gradient is not None else zero(parameter)
                for parameter, gradient in zip(trainable, auxiliary_gradients)
            ]
            for index, (parameter, main_gradient, auxiliary_gradient) in enumerate(zip(
                trainable, complete_main, complete_auxiliary
            )):
                if index in lora_index_set:
                    continue
                parameter.grad = (
                    main_gradient + actual_lambda * auxiliary_gradient
                ).to(dtype=parameter.dtype)
            lora_main = [complete_main[index] for index in lora_indices]
            lora_auxiliary_raw = [
                complete_auxiliary[index] for index in lora_indices
            ]
            lora_auxiliary_applied = [
                actual_lambda * gradient for gradient in lora_auxiliary_raw
            ]
            raw_statistics = combine_gradients(
                "weighted_sum", lora_main, lora_auxiliary_raw,
            )
            combination = combine_gradients(
                gradient_interaction, lora_main, lora_auxiliary_applied,
                cagrad_c=CAGRAD_C,
            )
            apply_combination(
                lora_parameters, lora_main, lora_auxiliary_applied, combination,
            )
            interaction_metrics = combination.as_dict()
            interaction_metrics.update({
                "scope": "LoRA A/B parameters only",
                "parameter_tensors": len(lora_parameters),
                "parameters": sum(parameter.numel() for parameter in lora_parameters),
                "raw_auxiliary_to_main_norm_ratio": (
                    raw_statistics.auxiliary_to_main_norm_ratio
                ),
                "active_auxiliary_coefficient": actual_lambda,
            })
        if profile_training_phases:
            torch.cuda.synchronize()
        gradient_seconds = time.perf_counter() - gradient_started

        optimizer_started = time.perf_counter()
        learning_rate = optimizer.param_groups[0]["lr"]
        gradient_norm, largest_gradient = gradient_diagnostics(model)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        if profile_training_phases:
            torch.cuda.synchronize()
        optimizer_seconds = time.perf_counter() - optimizer_started
        global_step += 1

        means = synchronized_loss_means(loss_records)
        if active:
            mse_value = float(jepa_mse)
            sigreg_value = float(jepa_sigreg)
            means["jepa_loss"] = mse_value
            means["sigreg_loss"] = sigreg_value
            means["jepa_objective_loss"] = float(jepa_objective)
            means["total_loss"] = (
                means["native_loss"]
                + actual_lambda * means["jepa_objective_loss"]
            )
        record = {
            "step": global_step,
            "epoch": epoch,
            **means,
            "jepa_active": active,
            "jepa_active_microbatches": chunks_per_step if active else 0,
            "sigreg_distribution_samples_per_view": logical_batch_size if active else 0,
            "gradient_interaction": interaction_metrics,
            "learning_rate": learning_rate,
            "gradient_norm": gradient_norm,
            "max_gradient_parameter": largest_gradient[0],
            "max_parameter_gradient_norm": largest_gradient[1],
            "batch_tokens": batch_tokens,
            "effective_tokens": effective_tokens,
            "model_calls": chunks_per_step * (2 if active else 1),
            "data_seconds": data_seconds,
            "auxiliary_statistics_vjp_seconds": first_pass_seconds,
            "gradient_forward_backward_seconds": gradient_seconds,
            "optimizer_seconds": optimizer_seconds,
        }
        record["estimated_flops"] = 6.0 * effective_tokens * non_embedding_parameters
        records.append(record)
        tracker.log_training_step(
            step=global_step, native_loss=record["native_loss"],
            jepa_loss=record["jepa_loss"], sigreg_loss=record["sigreg_loss"],
            jepa_objective_loss=record["jepa_objective_loss"],
            total_loss=record["total_loss"], gradient_norm=gradient_norm,
            max_gradient_parameter=largest_gradient[0],
            max_parameter_gradient_norm=largest_gradient[1],
            learning_rate=learning_rate, jepa_active=active,
            batch_tokens=batch_tokens, model_calls=record["model_calls"],
            effective_tokens=effective_tokens,
            peak_vram_bytes=torch.cuda.max_memory_allocated(),
            estimated_flops=record["estimated_flops"],
            gradient_interaction=interaction_metrics,
        )
    return records, global_step


def train_pair_residual_epoch(
    *, model, method, loader, optimizer, scheduler, epoch: int,
    logical_batch_size: int, physical_batch_size: int,
    actual_lambda: float, native_weight: float,
    jepa_ratio: float, non_embedding_parameters: int, pin_memory: bool,
    tracker: WandbTracker, global_step: int,
) -> tuple[list[dict[str, Any]], int]:
    """Train NTP while adding only the audited true-minus-shuffled LoRA gradient."""
    if logical_batch_size % physical_batch_size:
        raise ValueError("pair-residual logical batch must divide into physical batches")
    chunks_per_step = logical_batch_size // physical_batch_size
    if len(loader) % chunks_per_step:
        raise ValueError("training exposure must divide into complete residual batches")

    named_lora = [
        (name, parameter) for name, parameter in model.named_parameters()
        if parameter.requires_grad and ("lora_A" in name or "lora_B" in name)
    ]
    if not named_lora:
        raise RuntimeError("pair-residual JEPA requires trainable LoRA A/B parameters")
    lora_parameters = [parameter for _, parameter in named_lora]
    iterator = iter(loader)
    records: list[dict[str, Any]] = []

    for logical_index in range(len(loader) // chunks_per_step):
        data_started = time.perf_counter()
        raw_chunks = list(itertools.islice(iterator, chunks_per_step))
        data_seconds = time.perf_counter() - data_started
        if len(raw_chunks) != chunks_per_step:
            raise RuntimeError("incomplete pair-residual logical batch")
        active = method.sample_jepa_activity(jepa_ratio)

        # This pass is intentionally identical to ordinary native training.
        # Auxiliary computation later restores this post-NTP RNG snapshot so
        # it cannot perturb future NTP dropout streams in the matched control.
        native_started = time.perf_counter()
        loss_records: list[dict[str, Any]] = []
        batch_tokens = 0
        for raw in raw_chunks:
            raw_tokens = int(raw["attention_mask"].sum())
            batch = {
                name: value.to(model.device, non_blocking=pin_memory)
                for name, value in raw.items() if torch.is_tensor(value)
            }
            output = method(
                model, batch, k=0, jepa_weight=0.0,
                native_weight=native_weight, force_jepa_active=False,
            )
            native_component = output.native_loss * native_weight / chunks_per_step
            native_component.backward()
            if not torch.isfinite(output.native_loss):
                raise FloatingPointError(
                    f"non-finite native loss in epoch {epoch}, residual group "
                    f"{logical_index + 1}"
                )
            batch_tokens += raw_tokens
            loss_records.append({
                "native_loss": output.native_loss.detach(),
                "jepa_loss": None,
                "sigreg_loss": None,
                "jepa_objective_loss": None,
                "total_loss": output.native_loss.detach(),
            })
        native_seconds = time.perf_counter() - native_started
        post_native_rng = _rng_snapshot()

        interaction_metrics = None
        true_mse = None
        shuffled_mse = None
        residual_objective = None
        auxiliary_seconds = 0.0
        derangement = None
        target_length_cost = None
        if active:
            auxiliary_started = time.perf_counter()
            all_targets: list[torch.Tensor] = []
            for raw in raw_chunks:
                _, chunk_targets = extract_source_and_target(raw)
                all_targets.extend(chunk_targets)
            shuffle_seed = PAIR_RESIDUAL_SHUFFLE_SEED + global_step
            derangement = matched_derangement(all_targets, shuffle_seed)
            target_length_cost = sum(
                abs(len(all_targets[index]) - len(all_targets[shuffled]))
                for index, shuffled in enumerate(derangement)
            )

            replay_states: list[tuple[torch.Tensor, list[torch.Tensor]]] = []
            source_chunks: list[torch.Tensor] = []
            target_chunks: list[torch.Tensor] = []
            with torch.no_grad():
                for raw in raw_chunks:
                    replay_states.append(_rng_snapshot())
                    batch = {
                        name: value.to(model.device, non_blocking=pin_memory)
                        for name, value in raw.items() if torch.is_tensor(value)
                    }
                    output = method(
                        model, batch, k=0, jepa_weight=0.0,
                        native_weight=native_weight, monitor_only=True,
                        stop_gradient_target=False, jepa_loss_type="mse",
                        sigreg_tradeoff=0.0, jepa_ratio=jepa_ratio,
                        force_jepa_active=True, endpoint_only=True,
                    )
                    source_chunks.append(output.source_states.float())
                    target_chunks.append(output.target_states.float())
            raw_sources = torch.cat(source_chunks)
            raw_targets = torch.cat(target_chunks)
            if raw_sources.size(0) != logical_batch_size:
                raise RuntimeError("residual did not receive the complete logical batch")
            residual_result = raw_pair_residual_vjp(
                raw_sources, raw_targets, derangement,
            )
            true_mse = residual_result["true_mse"]
            shuffled_mse = residual_result["shuffled_mse"]
            residual_objective = residual_result["residual_objective"]
            source_gradients = residual_result["source_gradients"].split(
                physical_batch_size
            )
            target_gradients = residual_result["target_gradients"].split(
                physical_batch_size
            )

            residual_gradients: list[torch.Tensor | None] = [None] * len(lora_parameters)
            for chunk_index, raw in enumerate(raw_chunks):
                _restore_rng(replay_states[chunk_index])
                batch = {
                    name: value.to(model.device, non_blocking=pin_memory)
                    for name, value in raw.items() if torch.is_tensor(value)
                }
                output = method(
                    model, batch, k=0, jepa_weight=0.0,
                    native_weight=native_weight, monitor_only=True,
                    stop_gradient_target=False, jepa_loss_type="mse",
                    sigreg_tradeoff=0.0, jepa_ratio=jepa_ratio,
                    force_jepa_active=True, endpoint_only=True,
                )
                surrogate = (
                    output.source_states
                    * source_gradients[chunk_index].to(output.source_states.dtype)
                ).sum() + (
                    output.target_states
                    * target_gradients[chunk_index].to(output.target_states.dtype)
                ).sum()
                gradients = torch.autograd.grad(
                    surrogate, lora_parameters, allow_unused=True,
                )
                _accumulate_gradients(residual_gradients, gradients)

            # Auxiliary dropout and replay must not move the native-control RNG
            # trajectory. Only the added LoRA gradient is the independent variable.
            _restore_rng(post_native_rng)
            zero = lambda parameter: torch.zeros_like(parameter)
            lora_main = [
                parameter.grad.detach().clone()
                if parameter.grad is not None else zero(parameter)
                for parameter in lora_parameters
            ]
            lora_residual_raw = [
                gradient if gradient is not None else zero(parameter)
                for parameter, gradient in zip(lora_parameters, residual_gradients)
            ]
            lora_residual_applied = [
                actual_lambda * gradient for gradient in lora_residual_raw
            ]
            raw_statistics = combine_gradients(
                "weighted_sum", lora_main, lora_residual_raw,
            )
            combination = combine_gradients(
                "weighted_sum", lora_main, lora_residual_applied,
            )
            native_preclip_norm = read_only_global_gradient_norm(model)
            apply_combination(
                lora_parameters, lora_main, lora_residual_applied, combination,
            )
            combined_preclip_norm = read_only_global_gradient_norm(model)
            native_clip_coefficient = min(
                1.0, 1.0 / (native_preclip_norm + 1e-6)
            )
            combined_clip_coefficient = min(
                1.0, 1.0 / (combined_preclip_norm + 1e-6)
            )
            interaction_metrics = combination.as_dict()
            interaction_metrics.update({
                "scope": "LoRA A/B parameters only",
                "parameter_tensors": len(lora_parameters),
                "parameters": sum(parameter.numel() for parameter in lora_parameters),
                "raw_auxiliary_to_main_norm_ratio": (
                    raw_statistics.auxiliary_to_main_norm_ratio
                ),
                "active_auxiliary_coefficient": actual_lambda,
                "true_mse": float(true_mse),
                "shuffled_mse": float(shuffled_mse),
                "residual_objective": float(residual_objective),
                "endpoint_true_shuffle_gradient_cosine": residual_result[
                    "endpoint_true_shuffle_gradient_cosine"
                ],
                "endpoint_residual_over_true_norm": residual_result[
                    "endpoint_residual_over_true_norm"
                ],
                "shuffle_seed": shuffle_seed,
                "target_length_assignment_cost": target_length_cost,
                "native_preclip_global_gradient_norm": native_preclip_norm,
                "combined_preclip_global_gradient_norm": combined_preclip_norm,
                "native_global_clip_coefficient": native_clip_coefficient,
                "combined_global_clip_coefficient": combined_clip_coefficient,
            })
            interaction_metrics.update(adamw_residual_update_diagnostics(
                optimizer, lora_parameters, lora_main, lora_residual_applied,
                native_gradient_scale=native_clip_coefficient,
                combined_gradient_scale=combined_clip_coefficient,
            ))
            raw_gradient_ratio = interaction_metrics[
                "auxiliary_to_main_norm_ratio"
            ]
            interaction_metrics["adamw_preconditioning_amplification"] = (
                interaction_metrics[
                    "residual_to_native_adaptive_update_norm_ratio"
                ] / raw_gradient_ratio
                if raw_gradient_ratio else float("nan")
            )
            auxiliary_seconds = time.perf_counter() - auxiliary_started

        learning_rate = optimizer.param_groups[0]["lr"]
        gradient_norm, largest_gradient = gradient_diagnostics(model)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        global_step += 1

        means = synchronized_loss_means(loss_records)
        if active:
            means["jepa_loss"] = float(true_mse)
            means["jepa_objective_loss"] = float(residual_objective)
            means["total_loss"] = (
                means["native_loss"] + actual_lambda * means["jepa_objective_loss"]
            )
        record = {
            "step": global_step,
            "epoch": epoch,
            **means,
            "pair_residual_true_mse": None if true_mse is None else float(true_mse),
            "pair_residual_shuffled_mse": (
                None if shuffled_mse is None else float(shuffled_mse)
            ),
            "pair_residual_scalar": (
                None if residual_objective is None else float(residual_objective)
            ),
            "pair_residual_derangement": derangement,
            "pair_residual_target_length_cost": target_length_cost,
            "jepa_active": active,
            "jepa_active_microbatches": chunks_per_step if active else 0,
            "sigreg_distribution_samples_per_view": 0,
            "gradient_interaction": interaction_metrics,
            "learning_rate": learning_rate,
            "gradient_norm": gradient_norm,
            "max_gradient_parameter": largest_gradient[0],
            "max_parameter_gradient_norm": largest_gradient[1],
            "batch_tokens": batch_tokens,
            "effective_tokens": batch_tokens * (7 if active else 1),
            "model_calls": chunks_per_step * (3 if active else 1),
            "data_seconds": data_seconds,
            "native_forward_backward_seconds": native_seconds,
            "pair_residual_statistics_vjp_seconds": auxiliary_seconds,
        }
        record["estimated_flops"] = (
            6.0 * record["effective_tokens"] * non_embedding_parameters
        )
        records.append(record)
        tracker.log_training_step(
            step=global_step, native_loss=record["native_loss"],
            jepa_loss=record["jepa_loss"], sigreg_loss=None,
            jepa_objective_loss=record["jepa_objective_loss"],
            total_loss=record["total_loss"], gradient_norm=gradient_norm,
            max_gradient_parameter=largest_gradient[0],
            max_parameter_gradient_norm=largest_gradient[1],
            learning_rate=learning_rate, jepa_active=active,
            batch_tokens=batch_tokens, model_calls=record["model_calls"],
            effective_tokens=record["effective_tokens"],
            peak_vram_bytes=torch.cuda.max_memory_allocated(),
            estimated_flops=record["estimated_flops"],
            gradient_interaction=interaction_metrics,
        )
    return records, global_step


def train(args):
    if not torch.cuda.is_available():
        raise EnvironmentError("Gate 4/5 ChemFM-1B fine-tuning requires CUDA")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    set_seed(args.seed)
    task = TASKS[args.dataset]
    evaluation_beam_size = 20 if task == "metabolism" else 10
    train_path = args.train_manifest.resolve()
    validation_path = args.validation_manifest.resolve()
    train_rows = read_rows(args.dataset, path=train_path)
    val_rows = read_rows(args.dataset, split="validation", path=validation_path)
    if args.max_train_rows is not None:
        train_rows = train_rows[:args.max_train_rows]
    if args.max_validation_rows is not None:
        val_rows = val_rows[:args.max_validation_rows]
    if not train_rows or not val_rows:
        raise ValueError("training and validation manifests must both be nonempty")
    method_family = condition_family(args.condition)
    has_mse_sigreg = args.condition == "clm_jepa_mse_sigreg"
    has_pair_residual = args.condition == "clm_jepa_pair_residual"
    has_dense_vjepa = method_family == "dense_vjepa2_1"
    has_released_stp = method_family == "semantic_tube_prediction_released"
    has_paper_stp = method_family == "semantic_tube_prediction_paper"
    has_stp = has_released_stp or has_paper_stp
    tokenizer = load_reaction_tokenizer(TOKENIZER_DIR)
    chemfm_vocab_size = len(tokenizer)
    # Preserve the historical native/endpoint vocabulary for checkpoint and
    # control parity. The dense predictor is latent-only and keeps ChemFM's
    # unextended vocabulary.
    predictor_ids = [] if has_dense_vjepa else add_predictor_tokens(tokenizer)
    collator = ReactionCollator(tokenizer, task=task)
    shuffle_manifest_sha256 = None
    if args.condition == "shuffled":
        shuffle_manifest_sha256 = attach_matched_targets(
            train_rows, tokenizer, task, args.seed
        )
    validate_serialization_endings(collator, train_rows, tokenizer.eos_token_id)
    validate_serialization_endings(collator, val_rows, tokenizer.eos_token_id)
    model = load_lora_model(
        MODEL_DIR, tokenizer, chemfm_vocab_size=chemfm_vocab_size,
        attn_implementation=args.attention_implementation,
        lora_rank=args.lora_rank, lora_alpha=args.lora_alpha,
    ).cuda()
    initial_trainable_sha256 = trainable_parameter_sha256(model)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.enable_input_require_grads()
    non_embedding_parameters = model.num_parameters(exclude_embeddings=True)
    generator = torch.Generator().manual_seed(args.seed)
    worker_kwargs = {
        "num_workers": args.dataloader_workers,
        "persistent_workers": args.dataloader_workers > 0,
    }
    if args.dataloader_workers > 0:
        worker_kwargs["prefetch_factor"] = args.dataloader_prefetch_factor
    loader = DataLoader(
        train_rows, batch_size=args.batch_size, shuffle=True,
        generator=generator, collate_fn=collator, pin_memory=args.pin_memory,
        **worker_kwargs,
    )
    validation = DataLoader(
        val_rows, batch_size=args.batch_size, shuffle=False,
        collate_fn=collator, pin_memory=args.pin_memory,
        **worker_kwargs,
    )
    sigreg_tradeoff = (
        SIGREG_TRADEOFF
        if args.sigreg_tradeoff is None else args.sigreg_tradeoff
    )
    streaming_sigreg = has_mse_sigreg or has_pair_residual
    updates_per_epoch = (
        len(train_rows) // args.sigreg_batch_size
        if streaming_sigreg
        else max(1, math.ceil(len(loader) / args.gradient_accumulation_steps))
    )
    steps = max(1, args.epochs * updates_per_epoch)
    if has_dense_vjepa:
        method = DenseVJEPA21(
            DenseVJEPA21Config(
                encoder_dim=model.config.hidden_size,
                seed=args.seed,
            ),
            total_steps=steps,
            ignore_index=IGNORE_INDEX,
        ).to(device=model.device, dtype=model.dtype)
        method.initialize_ema(model)
    elif has_stp:
        stp_class = (
            PaperSemanticTubePrediction if has_paper_stp
            else SemanticTubePrediction
        )
        method = stp_class(
            seed=args.seed,
            reactant_start_token_id=tokenizer.convert_tokens_to_ids("<rstart>"),
            product_start_token_id=tokenizer.convert_tokens_to_ids("<prostart>"),
            eos_token_id=tokenizer.eos_token_id,
        )
    else:
        method = CLMJEPA(
            predictor_ids, tokenizer.eos_token_id, tokenizer.pad_token_id,
            sigreg_seed=args.seed,
            optimized_native_logits=args.optimized_jepa_forward,
        )
    optimizer_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if has_dense_vjepa:
        optimizer_parameters.extend(dense_trainable_parameters(method))
    optimizer = torch.optim.AdamW(
        optimizer_parameters,
        lr=args.learning_rate, betas=ADAM_BETAS, eps=ADAM_EPSILON,
        weight_decay=WEIGHT_DECAY, fused=args.fused_adamw,
    )
    scheduler = get_scheduler(
        "cosine_with_min_lr", optimizer,
        num_warmup_steps=int(steps * WARMUP_RATIO),
        num_training_steps=steps,
        scheduler_specific_kwargs={"min_lr": MIN_LEARNING_RATE},
    )
    has_jepa = method_family != "native"
    stop_gradient_target = args.condition == "clm_jepa_target_sg"
    has_sigreg = has_mse_sigreg
    jepa_loss_type = (
        "dense_vjepa2_1" if has_dense_vjepa else
        "mse" if args.condition in {
            "clm_jepa_mse", "clm_jepa_mse_sigreg",
            "clm_jepa_pair_residual",
        }
        else "cosine"
    )
    # With two global views, LeJEPA's view-center prediction loss is exactly
    # raw pairwise MSE / 4. This scale retains coefficient one on raw MSE.
    sigreg_relative_scale = 4.0 if jepa_loss_type == "mse" else 1.0
    ratio = 1.0 if has_dense_vjepa or has_stp else 1.0 - args.dropout
    resolved_ratio = ratio if has_jepa else -1.0
    actual_lambda = (
        args.stp_lambda if has_stp else
        args.lambda_eff / ratio if has_jepa else
        0.0
    )
    native_weight = 0.0 if args.condition == "jepa_only" else 1.0
    config = {
        "method_family": method_family,
        "learning_rate": args.learning_rate, "epochs": args.epochs,
        "resource_budget_epochs": args.stop_after_epoch,
        "physical_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_batch_size": (
            args.sigreg_batch_size
            if streaming_sigreg
            else args.batch_size * args.gradient_accumulation_steps
        ),
        "k": None if has_dense_vjepa or has_stp else args.k,
        "lambda_eff": args.lambda_eff, "actual_lambda": actual_lambda,
        "initial_trainable_sha256": initial_trainable_sha256,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_scaling_alpha_over_rank": args.lora_alpha / args.lora_rank,
        "lora_dropout": 0.1,
        "lora_target_modules": [
            "q_proj", "v_proj", "k_proj", "o_proj", "gate_proj",
            "up_proj", "down_proj",
        ],
        "lora_modules_to_save": ["embed_tokens", "lm_head"],
        "lora_use_rslora": False,
        "jepa_loss_dropout": (
            None if has_dense_vjepa or has_stp else args.dropout if has_jepa else None
        ),
        "jepa_ratio": resolved_ratio,
        "jepa_target_stop_gradient": has_dense_vjepa or stop_gradient_target,
        "jepa_target_encoder": "EMA ChemFM causal encoder" if has_dense_vjepa else None,
        "jepa_loss_type": jepa_loss_type if has_jepa else None,
        "semantic_tube_prediction": has_stp,
        "stp_formulation": (
            "paper_equation" if has_paper_stp
            else "released_patch_vs_complement" if has_released_stp
            else None
        ),
        "stp_paper": STP_PAPER if has_stp else None,
        "stp_upstream_repository": STP_UPSTREAM_REPOSITORY if has_stp else None,
        "stp_upstream_commit": STP_UPSTREAM_COMMIT if has_stp else None,
        "stp_executable_mode": (
            "--linear=random_span" if has_released_stp else None
        ),
        "stp_lambda": args.stp_lambda if has_stp else None,
        "stp_hidden_layer": "final" if has_stp else None,
        "stp_spans_per_example": 1 if has_stp else None,
        "stp_span_sampler": (
            "released default: start uniform, end uniform conditional on start, "
            "reject only the full content span"
            if has_released_stp else
            "released outer sampler; reject intervals without an interior; "
            "sample r uniformly from the valid interior"
            if has_paper_stp else None
        ),
        "stp_content_regions": (
            "reactant and product SMILES tokens; ChemFM framing excluded"
            if has_stp else None
        ),
        "stp_gradient_flow": "symmetric/no stop-gradient" if has_stp else None,
        "stp_loss_reduction_dtype": "float32" if has_stp else None,
        "sigreg": has_sigreg,
        "sigreg_formulation": (
            "LeJEPA Epps-Pulley, 17 knots on [0,3], 1024 random unit slices, "
            "independent source/target view statistics"
            if has_sigreg else None
        ),
        "sigreg_tradeoff": sigreg_tradeoff if has_sigreg else None,
        "sigreg_relative_coefficient": (
            sigreg_relative_scale * sigreg_tradeoff / (1.0 - sigreg_tradeoff)
            if has_sigreg else None
        ),
        "sigreg_relative_scale_from_view_center": (
            sigreg_relative_scale if has_sigreg else None
        ),
        "sigreg_distribution_batch_size_per_view": (
            args.sigreg_batch_size if has_sigreg else None
        ),
        "sigreg_exact_chunk_recomputation": has_mse_sigreg,
        "raw_endpoint_auxiliary": has_mse_sigreg or has_pair_residual,
        "pair_specific_residual": has_pair_residual,
        "pair_specific_residual_definition": (
            "2*(grad MSE(source,true_target) - "
            "grad MSE(source,matched_shuffled_target)) on active steps"
            if has_pair_residual else None
        ),
        "pair_specific_residual_expected_coefficient": (
            args.lambda_eff if has_pair_residual else None
        ),
        "pair_specific_residual_logical_batch_size": (
            args.sigreg_batch_size if has_pair_residual else None
        ),
        "pair_specific_residual_shuffle_seed_rule": (
            f"{PAIR_RESIDUAL_SHUFFLE_SEED} + zero_based_global_step"
            if has_pair_residual else None
        ),
        "pair_specific_residual_ntp_rng_isolated": has_pair_residual,
        "pair_specific_residual_sigreg_cancellation": (
            "SIGReg omitted because its target-permutation-invariant gradient "
            "cancels exactly in true-minus-shuffled contrast"
            if has_pair_residual else None
        ),
        "dense_vjepa2_1": has_dense_vjepa,
        "dense_vjepa2_1_paper": VJEPA21_PAPER if has_dense_vjepa else None,
        "dense_vjepa2_1_upstream_commit": (
            VJEPA21_UPSTREAM_COMMIT if has_dense_vjepa else None
        ),
        "dense_vjepa2_1_config": (
            asdict(method.config) if has_dense_vjepa else None
        ),
        "dense_vjepa2_1_context_warmup_steps": (
            [method.context_schedule.start, method.context_schedule.end]
            if has_dense_vjepa else None
        ),
        "dense_vjepa2_1_total_planned_steps": steps if has_dense_vjepa else None,
        "generation_vocabulary_unchanged": has_dense_vjepa,
        "gradient_interaction": (
            "pair_specific_residual" if has_pair_residual else
            args.gradient_interaction if has_mse_sigreg else None
        ),
        "gradient_interaction_scope": (
            "LoRA A/B parameters only"
            if has_mse_sigreg or has_pair_residual else None
        ),
        "cagrad_c": (
            CAGRAD_C if has_mse_sigreg and args.gradient_interaction == "cagrad"
            else None
        ),
        "optimizer_steps_per_epoch": updates_per_epoch,
        "train_size": len(train_rows), "validation_size": len(val_rows),
        "train_manifest": str(train_path),
        "train_manifest_sha256": file_sha256(train_path),
        "validation_manifest": str(validation_path),
        "validation_manifest_sha256": file_sha256(validation_path),
        "shuffle_manifest_sha256": shuffle_manifest_sha256,
        "optimizer": "adamw_torch",
        "fused_adamw": args.fused_adamw,
        "gradient_checkpointing": args.gradient_checkpointing,
        "pin_memory": args.pin_memory,
        "dataloader_workers": args.dataloader_workers,
        "dataloader_prefetch_factor": (
            args.dataloader_prefetch_factor if args.dataloader_workers else None
        ),
        "attention_implementation": getattr(
            model.config, "_attn_implementation", "unknown"
        ),
        "optimized_jepa_forward": args.optimized_jepa_forward,
        "profile_training_phases": args.profile_training_phases,
        "torch_profile_output": (
            str(args.torch_profile_output) if args.torch_profile_output else None
        ),
        "evaluation_generation_batch_size": args.eval_generation_batch_size,
        "evaluation_beam_size": evaluation_beam_size,
        "adam_beta1": ADAM_BETAS[0], "adam_beta2": ADAM_BETAS[1],
        "adam_epsilon": ADAM_EPSILON, "weight_decay": WEIGHT_DECAY,
        "scheduler": "cosine_with_min_lr", "warmup_ratio": WARMUP_RATIO,
        "min_learning_rate": MIN_LEARNING_RATE,
        "upstream_llm_jepa_commit": "ea0017c654ad917066ff32afc88276bea8ca5f7e",
        "upstream_lejepa_commit": (
            "c293d291ca87cd4fddee9d3fffe4e914c7272052" if has_sigreg else None
        ),
        "evaluation_epochs": list(args.evaluation_epochs),
    }
    tracker = WandbTracker(
        TrackingContext(task, args.dataset, args.condition, args.seed, args.data_fraction, config),
        run_name=(
            f"gate{args.gate}-{args.dataset}-{args.condition}-"
            f"{args.gradient_interaction}-lambda{args.lambda_eff}-s{args.seed}"
        ),
        enabled=not args.no_wandb,
    )
    curves = []
    epoch_history = []
    start = time.perf_counter()
    global_step = 0
    start_epoch = 0
    best_selector = None
    best_checkpoint = None
    previous_elapsed_seconds = 0.0
    if args.resume_from is not None:
        state = restore_training_checkpoint(
            args.resume_from.resolve(), model, optimizer, scheduler, generator,
            args.epochs, method,
        )
        start_epoch = state["epoch"]
        global_step = state["global_step"]
        curves = state["curves"]
        epoch_history = state["epoch_history"]
        best_selector = state["best_selector"]
        best_checkpoint = state["best_checkpoint"]
        previous_elapsed_seconds = state.get(
            "elapsed_wall_time_seconds", args.prior_wall_time_seconds
        )
        if start_epoch > args.stop_after_epoch:
            raise ValueError(
                "resume checkpoint is beyond the requested resource budget"
            )

    optimizer.zero_grad(set_to_none=True)
    try:
        for epoch_index in range(start_epoch, args.stop_after_epoch):
            model.train()
            if has_dense_vjepa:
                method.train()
            torch.cuda.synchronize()
            epoch_training_started = time.perf_counter()
            if has_pair_residual:
                epoch_records, global_step = train_pair_residual_epoch(
                    model=model, method=method, loader=loader,
                    optimizer=optimizer, scheduler=scheduler,
                    epoch=epoch_index + 1,
                    logical_batch_size=args.sigreg_batch_size,
                    physical_batch_size=args.batch_size,
                    actual_lambda=actual_lambda,
                    native_weight=native_weight,
                    jepa_ratio=resolved_ratio,
                    non_embedding_parameters=non_embedding_parameters,
                    pin_memory=args.pin_memory, tracker=tracker,
                    global_step=global_step,
                )
                curves.extend(epoch_records)
                training_loader = ()
            elif has_mse_sigreg:
                profile_this_epoch = (
                    args.torch_profile_output is not None
                    and epoch_index == start_epoch
                )
                profile_context = (
                    torch.profiler.profile(
                        activities=(
                            torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA,
                        ),
                        record_shapes=False,
                    )
                    if profile_this_epoch else nullcontext()
                )
                with profile_context as epoch_profiler:
                    epoch_records, global_step = train_mse_sigreg_gradient_epoch(
                        model=model, method=method, loader=loader,
                        optimizer=optimizer, scheduler=scheduler,
                        epoch=epoch_index + 1,
                        logical_batch_size=args.sigreg_batch_size,
                        physical_batch_size=args.batch_size,
                        actual_lambda=actual_lambda,
                        native_weight=native_weight,
                        sigreg_tradeoff=sigreg_tradeoff,
                        sigreg_relative_scale=sigreg_relative_scale,
                        gradient_interaction=args.gradient_interaction,
                        profile_training_phases=args.profile_training_phases,
                        jepa_ratio=resolved_ratio,
                        non_embedding_parameters=non_embedding_parameters,
                        pin_memory=args.pin_memory, tracker=tracker,
                        global_step=global_step,
                    )
                if profile_this_epoch:
                    profile_path = args.torch_profile_output.resolve()
                    profile_path.parent.mkdir(parents=True, exist_ok=True)
                    epoch_profiler.export_chrome_trace(str(profile_path))
                    profile_path.with_suffix(".txt").write_text(
                        epoch_profiler.key_averages().table(
                            sort_by="self_cuda_time_total", row_limit=100,
                        ),
                        encoding="utf-8",
                    )
                curves.extend(epoch_records)
                training_loader = ()
            else:
                training_loader = loader
            window_records = []
            for batch_index, raw in enumerate(training_loader):
                batch_tokens = int(raw["attention_mask"].sum())
                batch = {
                    name: value.to(model.device, non_blocking=args.pin_memory)
                    for name, value in raw.items() if torch.is_tensor(value)
                }
                jepa_targets = None
                if args.condition == "shuffled":
                    jepa_targets = [
                        row[mask.bool()]
                        for row, mask in zip(
                            batch["jepa_target_ids"],
                            batch["jepa_target_attention_mask"],
                        )
                    ]
                window_start = (
                    batch_index // args.gradient_accumulation_steps
                ) * args.gradient_accumulation_steps
                window_size = min(
                    args.gradient_accumulation_steps, len(loader) - window_start
                )
                microbatch_number = epoch_index * len(loader) + batch_index + 1
                if has_stp:
                    output = method(
                        model, batch, stp_weight=actual_lambda,
                    )
                    dense_metrics = {
                        "mean_sampled_span_fraction": sum(
                            ((span[2] - span[0]) / span[3])
                            if len(span) == 4 else
                            ((span[1] - span[0]) / span[2])
                            for span in output.sampled_spans
                        ) / len(output.sampled_spans),
                    }
                    jepa_active = True
                    sigreg_value = None
                    objective_value = float(output.jepa_loss.detach())
                elif has_dense_vjepa:
                    output = method(
                        model, batch, jepa_weight=actual_lambda,
                        global_step=global_step,
                    )
                    dense_metrics = {
                        "mask_loss": float(output.mask_loss.detach()),
                        "context_loss": float(output.context_loss.detach()),
                        "context_coefficient": output.context_coefficient,
                        "mean_horizon_tokens": float(output.mask.horizons.float().mean()),
                        "long_horizon_fraction": float(
                            (output.mask.mask_token_indices >= method.config.short_mask_tokens)
                            .float().mean()
                        ),
                    }
                    for depth in method.supervised_depths:
                        dense_metrics[f"mask_loss_depth_{depth}"] = float(
                            output.mask_loss_by_depth[depth].detach()
                        )
                        dense_metrics[f"context_loss_depth_{depth}"] = float(
                            output.context_loss_by_depth[depth].detach()
                        )
                        dense_metrics[f"student_scale_depth_{depth}"] = float(
                            output.student_scale_by_depth[depth].detach()
                        )
                        dense_metrics[f"target_scale_depth_{depth}"] = float(
                            output.target_scale_by_depth[depth].detach()
                        )
                    # V-JEPA 2.1 deep-supervision attribution is expensive
                    # (one VJP per component), so collect it once at the first
                    # step where the progressive context loss is fully active.
                    if (
                        global_step == method.context_schedule.end
                        and batch_index % args.gradient_accumulation_steps == 0
                    ):
                        level_scale = 1.0 / len(method.supervised_depths)
                        component_losses = {}
                        for depth in method.supervised_depths:
                            component_losses[f"mask_depth_{depth}"] = (
                                level_scale * output.mask_loss_by_depth[depth]
                            )
                            component_losses[f"context_depth_{depth}"] = (
                                level_scale
                                * output.context_coefficient
                                * output.context_loss_by_depth[depth]
                            )
                        dense_metrics.update(component_gradient_norms(
                            component_losses,
                            {
                                "chemfm": tuple(
                                    parameter for parameter in model.parameters()
                                    if parameter.requires_grad
                                ),
                                "predictor": tuple(
                                    dense_trainable_parameters(method)
                                ),
                            },
                        ))
                    jepa_active = True
                    sigreg_value = None
                    objective_value = float(output.jepa_loss.detach())
                else:
                    output = method(
                        model, batch, k=args.k,
                        jepa_weight=actual_lambda if has_jepa else 0.0,
                        native_weight=native_weight,
                        monitor_only=args.condition == "monitor",
                        stop_gradient_target=stop_gradient_target,
                        jepa_loss_type=jepa_loss_type,
                        sigreg_tradeoff=sigreg_tradeoff if has_sigreg else 0.0,
                        sigreg_relative_scale=sigreg_relative_scale,
                        jepa_ratio=resolved_ratio,
                        jepa_targets=jepa_targets,
                    )
                    dense_metrics = {}
                    jepa_active = output.jepa_active
                    sigreg_value = (
                        None if output.sigreg_loss is None
                        else float(output.sigreg_loss.detach())
                    )
                    objective_value = (
                        None if output.jepa_objective_loss is None
                        else float(output.jepa_objective_loss.detach())
                    )
                if not torch.isfinite(output.loss):
                    raise FloatingPointError(
                        f"non-finite loss at microbatch {microbatch_number}"
                    )
                (output.loss / window_size).backward()
                effective_tokens = batch_tokens
                if has_dense_vjepa:
                    # One causal student pass and one causal EMA-target pass.
                    effective_tokens = 2 * batch_tokens
                elif output.jepa_active:
                    effective_tokens += batch_tokens + args.k * len(batch["input_ids"])
                window_records.append({
                    "native_loss": float(output.native_loss.detach()),
                    "jepa_loss": (
                        None if output.jepa_loss is None
                        else float(output.jepa_loss.detach())
                    ),
                    "sigreg_loss": sigreg_value,
                    "jepa_objective_loss": objective_value,
                    "total_loss": float(output.loss.detach()),
                    "jepa_active": jepa_active,
                    "batch_tokens": batch_tokens,
                    "effective_tokens": effective_tokens,
                    "dense_metrics": dense_metrics,
                })
                boundary = (
                    (batch_index + 1) % args.gradient_accumulation_steps == 0
                    or batch_index + 1 == len(loader)
                )
                if not boundary:
                    continue
                learning_rate = optimizer.param_groups[0]["lr"]
                total_gradient_norm, largest_gradient = gradient_diagnostics(
                    model,
                    (("dense_jepa", method),) if has_dense_vjepa else (),
                )
                optimizer.step()
                ema_coefficient = method.update_ema(model) if has_dense_vjepa else None
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                active_losses = [
                    row["jepa_loss"] for row in window_records
                    if row["jepa_loss"] is not None
                ]
                sigreg_losses = [
                    row["sigreg_loss"] for row in window_records
                    if row["sigreg_loss"] is not None
                ]
                objective_losses = [
                    row["jepa_objective_loss"] for row in window_records
                    if row["jepa_objective_loss"] is not None
                ]
                record = {
                    "step": global_step,
                    "epoch": epoch_index + 1,
                    "native_loss": sum(row["native_loss"] for row in window_records) / len(window_records),
                    "jepa_loss": (
                        None if not active_losses
                        else sum(active_losses) / len(active_losses)
                    ),
                    "sigreg_loss": (
                        None if not sigreg_losses
                        else sum(sigreg_losses) / len(sigreg_losses)
                    ),
                    "jepa_objective_loss": (
                        None if not objective_losses
                        else sum(objective_losses) / len(objective_losses)
                    ),
                    "total_loss": sum(row["total_loss"] for row in window_records) / len(window_records),
                    "jepa_active": any(row["jepa_active"] for row in window_records),
                    "jepa_active_microbatches": sum(
                        row["jepa_active"] for row in window_records
                    ),
                    "learning_rate": learning_rate,
                    "gradient_norm": total_gradient_norm,
                    "max_gradient_parameter": largest_gradient[0],
                    "max_parameter_gradient_norm": largest_gradient[1],
                    "batch_tokens": sum(
                        row["batch_tokens"] for row in window_records
                    ),
                    "effective_tokens": sum(
                        row["effective_tokens"] for row in window_records
                    ),
                    "model_calls": len(window_records) * (2 if has_dense_vjepa else 1),
                }
                dense_keys = sorted(set().union(*(
                    row["dense_metrics"] for row in window_records
                )))
                dense_record = {
                    key: sum(
                        row["dense_metrics"][key] for row in window_records
                        if key in row["dense_metrics"]
                    ) / sum(key in row["dense_metrics"] for row in window_records)
                    for key in dense_keys
                }
                if ema_coefficient is not None:
                    dense_record["ema_coefficient"] = ema_coefficient
                record["dense_vjepa2_1"] = (
                    dense_record if has_dense_vjepa and dense_record else None
                )
                record["stp"] = dense_record if has_stp and dense_record else None
                record["estimated_flops"] = (
                    6.0 * record["effective_tokens"] * non_embedding_parameters
                )
                curves.append(record)
                tracker.log_training_step(
                    step=global_step, native_loss=record["native_loss"],
                    jepa_loss=record["jepa_loss"], total_loss=record["total_loss"],
                    sigreg_loss=record["sigreg_loss"],
                    jepa_objective_loss=record["jepa_objective_loss"],
                    gradient_norm=total_gradient_norm,
                    max_gradient_parameter=largest_gradient[0],
                    max_parameter_gradient_norm=largest_gradient[1],
                    learning_rate=learning_rate,
                    jepa_active=record["jepa_active"],
                    batch_tokens=record["batch_tokens"],
                    model_calls=record["model_calls"],
                    effective_tokens=record["effective_tokens"],
                    peak_vram_bytes=torch.cuda.max_memory_allocated(),
                    estimated_flops=record["estimated_flops"],
                    extra_metrics=dense_record if dense_record else None,
                )
                window_records = []

            torch.cuda.synchronize()
            epoch_training_seconds = time.perf_counter() - epoch_training_started
            checkpoint = args.checkpoint_dir.resolve() / f"epoch_{epoch_index + 1}"
            evaluate_epoch = epoch_index + 1 in args.evaluation_epochs
            if evaluate_epoch:
                val_loss = native_loss(model, validation)
                metrics, predictions = beam_evaluate(
                    model, tokenizer, collator, val_rows, task,
                    windows=evaluation_beam_size,
                    generation_batch_size=args.eval_generation_batch_size,
                )
                selector = validation_selector(metrics, task)
                if best_selector is None or selector > tuple(best_selector):
                    best_selector = selector
                    best_checkpoint = str(checkpoint)
            else:
                val_loss = None
                metrics = None
                predictions = None
                selector = None
            epoch_record = {
                "epoch": epoch_index + 1,
                "global_step": global_step,
                "validation_native_loss": val_loss,
                "validation_metrics": metrics,
                "predictions": predictions,
                "selector": selector,
                "checkpoint": str(checkpoint),
                "training_seconds": epoch_training_seconds,
            }
            epoch_history.append(epoch_record)
            if evaluate_epoch:
                tracker.log_evaluation(
                    step=global_step, split="validation", task_metrics=metrics,
                    validity=metrics["valid_rate"], native_loss=val_loss,
                )
            if not args.final_checkpoint_only or epoch_index + 1 == args.stop_after_epoch:
                save_training_checkpoint(
                    checkpoint, model, tokenizer, optimizer, scheduler, generator,
                    method,
                    epoch=epoch_index + 1, global_step=global_step,
                    planned_epochs=args.epochs, curves=curves,
                    epoch_history=epoch_history, best_selector=best_selector,
                    best_checkpoint=best_checkpoint,
                    elapsed_wall_time_seconds=(
                        previous_elapsed_seconds + time.perf_counter() - start
                    ),
                )
    except Exception:
        tracker.finish({"status": "failed"})
        raise

    if best_checkpoint is None:
        raise RuntimeError("no validation checkpoint was produced")
    load_adapter_checkpoint(model, Path(best_checkpoint))
    selected = next(
        row for row in epoch_history if row["checkpoint"] == best_checkpoint
    )
    val_loss = selected["validation_native_loss"]
    metrics = selected["validation_metrics"]
    predictions = selected["predictions"]
    if has_stp:
        stp_rows = [row for row in curves if row["jepa_loss"] is not None]
        diagnostics = {
            "type": "official_stp_training_summary",
            "upstream_commit": STP_UPSTREAM_COMMIT,
            "first_epoch_mean_stp_loss": float(np.mean([
                row["jepa_loss"] for row in stp_rows if row["epoch"] == 1
            ])),
            "final_epoch_mean_stp_loss": float(np.mean([
                row["jepa_loss"]
                for row in stp_rows if row["epoch"] == selected["epoch"]
            ])),
            "final_epoch_mean_sampled_span_fraction": float(np.mean([
                row["stp"]["mean_sampled_span_fraction"]
                for row in stp_rows if row["epoch"] == selected["epoch"]
            ])),
        }
    elif has_dense_vjepa:
        diagnostics = {
            "type": "dense_vjepa2_1_training_summary",
            "supervised_depths": list(method.supervised_depths),
            "last_training_step": curves[-1].get("dense_vjepa2_1"),
            "note": (
                "causal token-representation comparison is computed by the "
                "frozen feasibility evaluator, not endpoint-EOS diagnostics"
            ),
        }
    else:
        diagnostics = representation_diagnostics(
            model, method, collator, val_rows, args.k, args.seed, task,
        )
    result = {
        "gate": args.gate, "dataset": args.dataset, "task": task,
        "condition": args.condition, "seed": args.seed, "config": config,
        "validation_native_loss": val_loss, "validation_metrics": metrics,
        "diagnostics": diagnostics, "curves": curves,
        "epoch_history": [
            {key: value for key, value in row.items() if key != "predictions"}
            for row in epoch_history
        ],
        "selected_checkpoint": best_checkpoint,
        "selected_epoch": selected["epoch"],
        "predictions": predictions,
        "compute": {
            "optimizer_steps": global_step,
            "wall_time_seconds": (
                previous_elapsed_seconds + time.perf_counter() - start
            ),
            "peak_vram_bytes": torch.cuda.max_memory_allocated(),
            "jepa_active_microbatches": sum(
                row["jepa_active_microbatches"] for row in curves
            ),
            "model_calls": sum(row["model_calls"] for row in curves),
            "native_tokens": sum(row["batch_tokens"] for row in curves),
            "effective_tokens": sum(row["effective_tokens"] for row in curves),
            "estimated_flops": sum(row["estimated_flops"] for row in curves),
            "effective_tokens_per_second": (
                sum(row["effective_tokens"] for row in curves)
                / max(time.perf_counter() - start, 1e-12)
            ),
        },
    }
    tracker.finish({
        "validation_primary": metrics["exact_top1"],
        "validation_validity": metrics["valid_rate"],
        "validation_native_loss": val_loss,
    })
    return result


def main():
    parser = argparse.ArgumentParser(description="Quick, controlled Gate 4/5 ChemFM-cLM-JEPA trial")
    parser.add_argument("--gate", type=int, choices=(4, 5), required=True)
    parser.add_argument("--dataset", choices=sorted(TASKS), required=True)
    parser.add_argument(
        "--condition",
        choices=TRAINING_CONDITIONS,
        required=True,
    )
    parser.add_argument("--seed", type=int, default=533)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=8)
    parser.add_argument(
        "--stp-lambda", type=float, default=0.02,
        help="official released Llama-1B STP coefficient",
    )
    parser.add_argument("--k", type=int, choices=(0, 1, 2, 3), default=0)
    parser.add_argument(
        "--lambda-eff", type=float, choices=(0.25, 0.5, 1.0, 2.0),
        default=1.0,
    )
    parser.add_argument(
        "--gradient-interaction",
        choices=("weighted_sum", "pcgrad", "cagrad", "aux_similarity"),
        default="weighted_sum",
        help="published LoRA-gradient combination used by direct MSE+SIGReg",
    )
    parser.add_argument("--dropout", type=float, choices=(0.0, 0.5, 0.75), default=0.5)
    parser.add_argument("--epochs", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument(
        "--stop-after-epoch", type=int, choices=(1, 2, 4),
        help=(
            "successive-halving resource rung; the scheduler still uses --epochs "
            "as its fixed maximum budget"
        ),
    )
    parser.add_argument(
        "--prior-wall-time-seconds", type=float, default=0.0,
        help="fallback only for resuming a legacy checkpoint without elapsed time",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument(
        "--sigreg-batch-size", type=int, default=2,
        help=(
            "logical endpoint batch size for exact SIGReg or pair-residual "
            "training; values above the physical batch use exact replay"
        ),
    )
    parser.add_argument(
        "--sigreg-tradeoff", type=float,
        help=(
            "LeJEPA mixture trade-off; omitted retains the controlled MSE+SIGReg "
            "value of 0.01."
        ),
    )
    parser.add_argument(
        "--evaluation-epochs", type=int, nargs="+",
        help=(
            "epochs at which to run validation CE and generation; every epoch is "
            "still checkpointed. Omitted evaluates every resource-budget epoch."
        ),
    )
    parser.add_argument(
        "--gradient-checkpointing", action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--fused-adamw", action=argparse.BooleanOptionalAction, default=False,
    )
    parser.add_argument(
        "--final-checkpoint-only", action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "save only the final resource-budget checkpoint; requires evaluation "
            "only at that epoch and intentionally disables intermediate resume"
        ),
    )
    parser.add_argument(
        "--optimized-jepa-forward", action=argparse.BooleanOptionalAction,
        default=False,
        help="project logits only for native rows during active JEPA updates",
    )
    parser.add_argument(
        "--attention-implementation",
        choices=("eager", "sdpa", "flash_attention_2"),
        help="Transformers attention backend; omitted retains library auto-selection",
    )
    parser.add_argument(
        "--profile-training-phases", action=argparse.BooleanOptionalAction,
        default=False,
        help="synchronize benchmark phase timers; disable for maximum throughput",
    )
    parser.add_argument(
        "--torch-profile-output", type=Path,
        help="optional Chrome trace for the first JEPA training epoch",
    )
    parser.add_argument(
        "--pin-memory", action=argparse.BooleanOptionalAction, default=False,
    )
    parser.add_argument("--dataloader-workers", type=int, default=0)
    parser.add_argument("--dataloader-prefetch-factor", type=int, default=2)
    parser.add_argument(
        "--eval-generation-batch-size", type=int, choices=(1, 2, 4), default=1,
        help=(
            "number of length-sorted, left-padded prompts evaluated together; does not "
            "change beam width, stopping, generation length, or R-SMILES aggregation"
        ),
    )
    parser.add_argument("--data-fraction", type=float, default=1.0)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--max-train-rows", type=int, help="integration-test limit only")
    parser.add_argument("--max-validation-rows", type=int, help="integration-test limit only")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.stop_after_epoch is None:
        args.stop_after_epoch = args.epochs
    if args.evaluation_epochs is None:
        args.evaluation_epochs = list(range(1, args.stop_after_epoch + 1))
    args.evaluation_epochs = sorted(set(args.evaluation_epochs))
    if not args.evaluation_epochs or any(
        epoch < 1 or epoch > args.stop_after_epoch
        for epoch in args.evaluation_epochs
    ):
        raise ValueError(
            "evaluation epochs must be within the requested resource budget"
        )
    if args.stop_after_epoch > args.epochs:
        raise ValueError("stop-after epoch cannot exceed the planned epoch budget")
    if args.prior_wall_time_seconds < 0:
        raise ValueError("prior wall time cannot be negative")
    if args.batch_size < 1 or args.gradient_accumulation_steps < 1:
        raise ValueError("batch size and gradient accumulation must be positive")
    if args.lora_rank < 1 or args.lora_alpha < 1:
        raise ValueError("LoRA rank and alpha must be positive")
    if args.final_checkpoint_only and args.evaluation_epochs != [args.stop_after_epoch]:
        raise ValueError(
            "--final-checkpoint-only requires evaluation only at stop-after epoch"
        )
    if args.dataloader_workers < 0 or args.dataloader_prefetch_factor < 1:
        raise ValueError("DataLoader workers must be nonnegative and prefetch positive")
    if args.sigreg_batch_size < 2:
        raise ValueError("SIGReg batch size must be at least two")
    if args.sigreg_tradeoff is not None and not 0.0 <= args.sigreg_tradeoff < 1.0:
        raise ValueError("SIGReg trade-off must be in [0, 1)")
    if args.stp_lambda <= 0.0:
        raise ValueError("STP lambda must be positive")
    sigreg_conditions = {"clm_jepa_mse_sigreg"}
    logical_endpoint_conditions = sigreg_conditions | {"clm_jepa_pair_residual"}
    if args.condition not in sigreg_conditions and args.sigreg_tradeoff is not None:
        raise ValueError("--sigreg-tradeoff only applies to a SIGReg condition")
    if (
        args.condition not in logical_endpoint_conditions
        and args.sigreg_batch_size != 2
    ):
        raise ValueError(
            "--sigreg-batch-size only applies to a logical-batch endpoint condition"
        )
    if (
        args.condition == "clm_jepa_mse_sigreg"
        and args.sigreg_tradeoff is not None
        and args.sigreg_tradeoff != SIGREG_TRADEOFF
    ):
        raise ValueError("MSE+SIGReg is frozen to sigreg-tradeoff=0.01")
    if (
        args.condition in logical_endpoint_conditions
        and args.sigreg_batch_size
        != args.batch_size * args.gradient_accumulation_steps
    ):
        raise ValueError(
            "logical endpoint batch must equal physical batch times "
            "accumulation to preserve the controlled cadence"
        )
    if args.condition in logical_endpoint_conditions and (
        args.k != 0 or args.dropout != 0.5
    ):
        raise ValueError(
            "logical endpoint conditions are frozen to k=0 and dropout=0.5"
        )
    if args.condition == "clm_jepa_pair_residual" and (
        args.lambda_eff != 1.0
        or args.gradient_interaction != "weighted_sum"
        or args.optimized_jepa_forward
    ):
        raise ValueError(
            "pair-residual JEPA is frozen to lambda-eff=1, direct weighted "
            "addition, and its exact endpoint replay"
        )
    if (
        args.condition == "clm_jepa_mse_sigreg"
        and args.gradient_interaction != "weighted_sum"
        and args.lambda_eff != 1.0
    ):
        raise ValueError("gradient-interaction baselines are frozen to lambda-eff=1.0")
    if (
        args.condition != "clm_jepa_mse_sigreg"
        and args.gradient_interaction != "weighted_sum"
    ):
        raise ValueError("--gradient-interaction only applies to MSE+SIGReg")
    if args.condition == NATIVE_CONDITION and (args.lambda_eff != 1.0 or args.dropout != 0.5):
        raise ValueError("native trials must leave irrelevant JEPA defaults unchanged")
    if args.condition in {STP_CONDITION, RELEASED_STP_CONDITION, PAPER_STP_CONDITION} and (
        args.k != 0
        or args.lambda_eff != 1.0
        or args.dropout != 0.5
        or args.gradient_interaction != "weighted_sum"
        or args.optimized_jepa_forward
    ):
        raise ValueError(
            "STP uses one every-batch sample and no endpoint-JEPA options"
        )
    if args.condition == STP_CONDITION and args.stp_lambda != 0.02:
        raise ValueError("legacy --condition stp remains frozen to released lambda=0.02")
    if args.condition == DENSE_VJEPA21_CONDITION and (
        args.k != 0
        or args.lambda_eff != 1.0
        or args.dropout != 0.5
        or args.gradient_interaction != "weighted_sum"
        or args.optimized_jepa_forward
    ):
        raise ValueError(
            "dense V-JEPA 2.1 is frozen to lambda-eff=1, k=0, default dropout "
            "placeholder, weighted-sum gradients, and the ordinary native-logit path"
        )
    result = train(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output), "dataset": args.dataset,
        "condition": args.condition, "seed": args.seed,
        "primary": result["validation_metrics"]["exact_top1"],
        "validity": result["validation_metrics"]["valid_rate"],
        "native_loss": result["validation_native_loss"],
    }), flush=True)


if __name__ == "__main__":
    main()
