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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

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
from jepa import CLMJEPA, add_predictor_tokens, extract_source_and_target  # noqa: E402
from metrics import (  # noqa: E402
    canonical_set, effective_rank, rank_augmented_candidates,
    relationship_metrics, score_candidates,
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
SIGREG_TRADEOFF = 0.05

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
        raise FileNotFoundError(
            f"dataset manifest is missing: {path}; Gate 3 samples live under data/gate3"
        )
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


def split_rows(rows: list[dict[str, str]], seed: int, train_size: int, val_size: int):
    # Hash grouping keeps repeated sources/parents wholly in one side.
    groups: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault(row["src"], []).append(row)
    ordered = sorted(
        groups.items(),
        key=lambda item: hashlib.sha256(f"{seed}|{item[0]}".encode()).digest(),
    )
    train, val = [], []
    for _, group in ordered:
        destination = train if len(train) < train_size else val
        destination.extend(group)
        if len(val) >= val_size:
            break
    return train[:train_size], val[:val_size]


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


def representation_diagnostics(model, method, collator, rows, k, seed, task):
    sample = _representation_sample(rows, task)
    batch = collator(sample)
    tensors = {name: value.to(model.device) for name, value in batch.items() if torch.is_tensor(value)}
    model.eval()
    with torch.inference_mode():
        output = method(model, tensors, k=k, jepa_weight=1.0, monitor_only=True)
    sources = output.source_states.float()
    targets = output.target_states.float()
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
        def states(view):
            raw = collator(view)
            tensors = {name: value.to(model.device) for name, value in raw.items() if torch.is_tensor(value)}
            return method(model, tensors, k=k, jepa_weight=1.0, monitor_only=True).source_states.float()
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
        "jepa_dropout_generator_state": method.jepa_dropout_generator.get_state(),
        "sigreg_global_step": method.sigreg.global_step,
        "curves": curves,
        "epoch_history": epoch_history,
        "best_selector": best_selector,
        "best_checkpoint": best_checkpoint,
        "elapsed_wall_time_seconds": elapsed_wall_time_seconds,
    }
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
    return state


def gradient_diagnostics(model) -> tuple[float, tuple[str, float]]:
    """Clip gradients and preserve the established per-parameter diagnostics."""
    total = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    total_value = float(total)
    clip_coefficient = min(1.0, 1.0 / (total_value + 1e-6))
    largest = ("", 0.0)
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        clipped_norm = float(parameter.grad.detach().float().norm())
        original_norm = clipped_norm / max(clip_coefficient, 1e-30)
        if original_norm > largest[1]:
            largest = (name, original_norm)
    return total_value, largest


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


def train_streaming_sigreg_epoch(
    *, model, method, loader, optimizer, scheduler, epoch: int,
    logical_batch_size: int, physical_batch_size: int,
    actual_lambda: float, native_weight: float, sigreg_tradeoff: float,
    jepa_ratio: float, non_embedding_parameters: int, pin_memory: bool,
    tracker: WandbTracker, global_step: int,
) -> tuple[list[dict[str, Any]], int]:
    """Train one epoch with exact recomputed SIGReg statistics over logical batches.

    The first pass accumulates global ECF sufficient statistics without graphs.
    A second RNG-replayed pass computes native/cosine gradients and injects the
    exact SIGReg representation VJP. Parameters remain fixed across all chunks
    until the logical-batch gradient is complete.
    """
    if logical_batch_size % physical_batch_size:
        raise ValueError("SIGReg logical batch must be divisible by physical batch size")
    chunks_per_step = logical_batch_size // physical_batch_size
    if len(loader) % chunks_per_step:
        raise ValueError("training exposure must divide into complete SIGReg logical batches")
    relative_coefficient = sigreg_tradeoff / (1.0 - sigreg_tradeoff)
    iterator = iter(loader)
    records: list[dict[str, Any]] = []
    for logical_index in range(len(loader) // chunks_per_step):
        data_started = time.perf_counter()
        raw_chunks = list(itertools.islice(iterator, chunks_per_step))
        data_seconds = time.perf_counter() - data_started
        if len(raw_chunks) != chunks_per_step:
            raise RuntimeError("incomplete SIGReg logical batch")
        active = method.sample_jepa_activity(jepa_ratio)
        prepared = None
        replay_states: list[tuple[torch.Tensor, list[torch.Tensor]]] = []
        first_pass_seconds = 0.0
        if active:
            accumulator = None
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
                        stop_gradient_target=False, sigreg_tradeoff=0.0,
                        jepa_ratio=jepa_ratio, force_jepa_active=True,
                    )
                    states = torch.stack((output.source_states, output.target_states))
                    if accumulator is None:
                        accumulator = method.sigreg.start_streaming(
                            views=2, dimensions=states.size(-1),
                            expected_samples=logical_batch_size,
                            device=states.device,
                        )
                    accumulator.update(states)
            final_rng = _rng_snapshot()
            if accumulator is None:
                raise RuntimeError("SIGReg accumulator was not initialized")
            prepared = accumulator.finalize()
            torch.cuda.synchronize()
            first_pass_seconds = time.perf_counter() - first_started

        loss_records: list[dict[str, Any]] = []
        batch_tokens = 0
        effective_tokens = 0
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
                jepa_weight=actual_lambda if active else 0.0,
                native_weight=native_weight,
                stop_gradient_target=False, sigreg_tradeoff=0.0,
                jepa_ratio=jepa_ratio, force_jepa_active=active,
            )
            loss = output.loss / chunks_per_step
            if active:
                states = torch.stack((output.source_states, output.target_states))
                loss = loss + actual_lambda * relative_coefficient * prepared.surrogate(states)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite loss in epoch {epoch}, SIGReg group {logical_index + 1}"
                )
            loss.backward()
            batch_tokens += raw_tokens
            effective_tokens += raw_tokens * (4 if active else 1)
            loss_records.append({
                "native_loss": output.native_loss.detach(),
                "jepa_loss": None if output.jepa_loss is None else output.jepa_loss.detach(),
                "sigreg_loss": None,
                "jepa_objective_loss": None,
                "total_loss": output.loss.detach(),
            })
        if active:
            _restore_rng(final_rng)
        torch.cuda.synchronize()
        gradient_seconds = time.perf_counter() - gradient_started

        optimizer_started = time.perf_counter()
        learning_rate = optimizer.param_groups[0]["lr"]
        gradient_norm, largest_gradient = gradient_diagnostics(model)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        optimizer_seconds = time.perf_counter() - optimizer_started
        global_step += 1

        means = synchronized_loss_means(loss_records)
        if active:
            sigreg_value = float(prepared.loss)
            means["sigreg_loss"] = sigreg_value
            means["jepa_objective_loss"] = (
                means["jepa_loss"] + relative_coefficient * sigreg_value
            )
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
            "learning_rate": learning_rate,
            "gradient_norm": gradient_norm,
            "max_gradient_parameter": largest_gradient[0],
            "max_parameter_gradient_norm": largest_gradient[1],
            "batch_tokens": batch_tokens,
            "effective_tokens": effective_tokens,
            "model_calls": chunks_per_step * (2 if active else 1),
            "data_seconds": data_seconds,
            "sigreg_statistics_forward_seconds": first_pass_seconds,
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
    tokenizer = load_reaction_tokenizer(TOKENIZER_DIR)
    chemfm_vocab_size = len(tokenizer)
    predictor_ids = add_predictor_tokens(tokenizer)
    collator = ReactionCollator(tokenizer, task=task)
    shuffle_manifest_sha256 = None
    if args.condition == "shuffled":
        shuffle_manifest_sha256 = attach_matched_targets(
            train_rows, tokenizer, task, args.seed
        )
    validate_serialization_endings(collator, train_rows, tokenizer.eos_token_id)
    validate_serialization_endings(collator, val_rows, tokenizer.eos_token_id)
    model = load_lora_model(
        MODEL_DIR, tokenizer, chemfm_vocab_size=chemfm_vocab_size
    ).cuda()
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.enable_input_require_grads()
    method = CLMJEPA(
        predictor_ids, tokenizer.eos_token_id, tokenizer.pad_token_id,
        sigreg_seed=args.seed,
    )
    non_embedding_parameters = model.num_parameters(exclude_embeddings=True)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        train_rows, batch_size=args.batch_size, shuffle=True,
        generator=generator, collate_fn=collator, pin_memory=args.pin_memory,
    )
    validation = DataLoader(
        val_rows, batch_size=args.batch_size, shuffle=False,
        collate_fn=collator, pin_memory=args.pin_memory,
    )
    streaming_sigreg = (
        args.condition == "clm_jepa_sigreg"
        and args.sigreg_batch_size > args.batch_size
    )
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate, betas=ADAM_BETAS, eps=ADAM_EPSILON,
        weight_decay=WEIGHT_DECAY, fused=args.fused_adamw,
    )
    updates_per_epoch = (
        len(train_rows) // args.sigreg_batch_size
        if streaming_sigreg
        else max(1, math.ceil(len(loader) / args.gradient_accumulation_steps))
    )
    steps = max(1, args.epochs * updates_per_epoch)
    scheduler = get_scheduler(
        "cosine_with_min_lr", optimizer,
        num_warmup_steps=int(steps * WARMUP_RATIO),
        num_training_steps=steps,
        scheduler_specific_kwargs={"min_lr": MIN_LEARNING_RATE},
    )
    has_jepa = args.condition in {
        "monitor", "clm_jepa", "clm_jepa_target_sg", "clm_jepa_sigreg",
        "shuffled", "jepa_only",
    }
    stop_gradient_target = args.condition == "clm_jepa_target_sg"
    has_sigreg = args.condition == "clm_jepa_sigreg"
    ratio = 1.0 - args.dropout
    resolved_ratio = ratio if has_jepa else -1.0
    actual_lambda = args.lambda_eff / ratio if has_jepa else 0.0
    native_weight = 0.0 if args.condition == "jepa_only" else 1.0
    config = {
        "learning_rate": args.learning_rate, "epochs": args.epochs,
        "resource_budget_epochs": args.stop_after_epoch,
        "physical_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_batch_size": (
            args.sigreg_batch_size
            if streaming_sigreg
            else args.batch_size * args.gradient_accumulation_steps
        ),
        "k": args.k,
        "lambda_eff": args.lambda_eff, "actual_lambda": actual_lambda,
        "jepa_loss_dropout": args.dropout if has_jepa else None,
        "jepa_ratio": resolved_ratio,
        "jepa_target_stop_gradient": stop_gradient_target,
        "sigreg": has_sigreg,
        "sigreg_formulation": (
            "LeJEPA Epps-Pulley, 17 knots on [0,3], 1024 random unit slices, "
            "independent source/target view statistics"
            if has_sigreg else None
        ),
        "sigreg_tradeoff": SIGREG_TRADEOFF if has_sigreg else None,
        "sigreg_relative_coefficient": (
            SIGREG_TRADEOFF / (1.0 - SIGREG_TRADEOFF) if has_sigreg else None
        ),
        "sigreg_distribution_batch_size_per_view": (
            args.sigreg_batch_size if has_sigreg else None
        ),
        "sigreg_exact_chunk_recomputation": streaming_sigreg,
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
    }
    tracker = WandbTracker(
        TrackingContext(task, args.dataset, args.condition, args.seed, args.data_fraction, config),
        run_name=f"gate{args.gate}-{args.dataset}-{args.condition}-s{args.seed}",
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
            if streaming_sigreg:
                epoch_records, global_step = train_streaming_sigreg_epoch(
                    model=model, method=method, loader=loader,
                    optimizer=optimizer, scheduler=scheduler,
                    epoch=epoch_index + 1,
                    logical_batch_size=args.sigreg_batch_size,
                    physical_batch_size=args.batch_size,
                    actual_lambda=actual_lambda,
                    native_weight=native_weight,
                    sigreg_tradeoff=SIGREG_TRADEOFF,
                    jepa_ratio=resolved_ratio,
                    non_embedding_parameters=non_embedding_parameters,
                    pin_memory=args.pin_memory, tracker=tracker,
                    global_step=global_step,
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
                output = method(
                    model, batch, k=args.k,
                    jepa_weight=actual_lambda if has_jepa else 0.0,
                    native_weight=native_weight,
                    monitor_only=args.condition == "monitor",
                    stop_gradient_target=stop_gradient_target,
                    sigreg_tradeoff=SIGREG_TRADEOFF if has_sigreg else 0.0,
                    jepa_ratio=resolved_ratio,
                    jepa_targets=jepa_targets,
                )
                if not torch.isfinite(output.loss):
                    raise FloatingPointError(
                        f"non-finite loss at microbatch {microbatch_number}"
                    )
                (output.loss / window_size).backward()
                effective_tokens = batch_tokens
                if output.jepa_active:
                    effective_tokens += batch_tokens + args.k * len(batch["input_ids"])
                window_records.append({
                    "native_loss": float(output.native_loss.detach()),
                    "jepa_loss": (
                        None if output.jepa_loss is None
                        else float(output.jepa_loss.detach())
                    ),
                    "sigreg_loss": (
                        None if output.sigreg_loss is None
                        else float(output.sigreg_loss.detach())
                    ),
                    "jepa_objective_loss": (
                        None if output.jepa_objective_loss is None
                        else float(output.jepa_objective_loss.detach())
                    ),
                    "total_loss": float(output.loss.detach()),
                    "jepa_active": output.jepa_active,
                    "batch_tokens": batch_tokens,
                    "effective_tokens": effective_tokens,
                })
                boundary = (
                    (batch_index + 1) % args.gradient_accumulation_steps == 0
                    or batch_index + 1 == len(loader)
                )
                if not boundary:
                    continue
                learning_rate = optimizer.param_groups[0]["lr"]
                total_gradient_norm, largest_gradient = gradient_diagnostics(model)
                optimizer.step()
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
                    "model_calls": len(window_records),
                }
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
                )
                window_records = []

            val_loss = native_loss(model, validation)
            metrics, predictions = beam_evaluate(
                model, tokenizer, collator, val_rows, task,
                windows=evaluation_beam_size,
                generation_batch_size=args.eval_generation_batch_size,
            )
            selector = validation_selector(metrics, task)
            checkpoint = args.checkpoint_dir.resolve() / f"epoch_{epoch_index + 1}"
            if best_selector is None or selector > tuple(best_selector):
                best_selector = selector
                best_checkpoint = str(checkpoint)
            epoch_record = {
                "epoch": epoch_index + 1,
                "global_step": global_step,
                "validation_native_loss": val_loss,
                "validation_metrics": metrics,
                "predictions": predictions,
                "selector": selector,
                "checkpoint": str(checkpoint),
            }
            epoch_history.append(epoch_record)
            tracker.log_evaluation(
                step=global_step, split="validation", task_metrics=metrics,
                validity=metrics["valid_rate"], native_loss=val_loss,
            )
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
    diagnostics = representation_diagnostics(
        model, method, collator, val_rows, args.k, args.seed, task
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
        choices=(
            "native", "monitor", "clm_jepa", "clm_jepa_target_sg",
            "clm_jepa_sigreg",
            "shuffled", "jepa_only",
        ),
        required=True,
    )
    parser.add_argument("--seed", type=int, default=533)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--k", type=int, choices=(0, 1, 2, 3), default=0)
    parser.add_argument("--lambda-eff", type=float, choices=(0.5, 1.0, 2.0, 4.0), default=1.0)
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
            "representations per view in each SIGReg distribution estimate; values "
            "above the physical batch use exact sufficient-statistic recomputation"
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
        "--pin-memory", action=argparse.BooleanOptionalAction, default=False,
    )
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
    if args.stop_after_epoch > args.epochs:
        raise ValueError("stop-after epoch cannot exceed the planned epoch budget")
    if args.prior_wall_time_seconds < 0:
        raise ValueError("prior wall time cannot be negative")
    if args.batch_size < 1 or args.gradient_accumulation_steps < 1:
        raise ValueError("batch size and gradient accumulation must be positive")
    if args.sigreg_batch_size < 2:
        raise ValueError("SIGReg batch size must be at least two")
    if args.condition != "clm_jepa_sigreg" and args.sigreg_batch_size != 2:
        raise ValueError("--sigreg-batch-size only applies to clm_jepa_sigreg")
    if args.condition == "native" and (args.lambda_eff != 1.0 or args.dropout != 0.5):
        raise ValueError("native trials must leave irrelevant JEPA defaults unchanged")
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
