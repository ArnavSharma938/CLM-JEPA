"""Native ChemFM and released/paper STP training. Frozen checkpoint compatibility lives in chemfm.py."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import get_scheduler, set_seed
ROOT = Path(__file__).resolve().parents[1]
from chemfm import add_predictor_tokens, load_adapter_checkpoint, IGNORE_INDEX, MODEL_DIR, TOKENIZER_DIR, ReactionCollator, generate_products_batch, canonicalize, load_lora_model, load_reaction_tokenizer
from metrics import canonical_set, rank_augmented_candidates, score_candidates
from decoder_projected import DecoderProjectedObjective, FullStateObjective
from faithful_nextlat import FaithfulNextLatObjective
from stp import STP_PAPER, STP_UPSTREAM_COMMIT, STP_UPSTREAM_REPOSITORY, PaperSemanticTubePrediction, SemanticTubePrediction
ADAPTER_NAME = 'USPTO-MIT-Synthesis'
ADAM_BETAS = (0.9, 0.999)
ADAM_EPSILON = 1e-08
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.05
MIN_LEARNING_RATE = 1e-05
TASKS = {
    'uspto_mit_synthesis': 'forward',
    'orderly_forward': 'forward',
    'non_uspto_forward': 'forward',
    'metatrans_full': 'metabolism',
    'uspto_50k_retro': 'retro',
    'uspto_480k_template_heldout': 'retro',
    'non_uspto_retro': 'retro',
}
NATIVE_CONDITION = 'native'
STP_CONDITION = 'stp'
RELEASED_STP_CONDITION = 'stp_released'
PAPER_STP_CONDITION = 'stp_paper'
PROJECTED_CONDITION = 'decoder_projected'
FULL_STATE_CONDITION = 'full_state_nextlat'
FAITHFUL_NEXTLAT_CONDITION = 'faithful_nextlat'
TRAINING_CONDITIONS = (PROJECTED_CONDITION, FULL_STATE_CONDITION, FAITHFUL_NEXTLAT_CONDITION, NATIVE_CONDITION, STP_CONDITION, RELEASED_STP_CONDITION, PAPER_STP_CONDITION)

def condition_family(condition):
    if condition == PROJECTED_CONDITION:
        return 'decoder_projected'
    if condition == FULL_STATE_CONDITION:
        return 'full_state_nextlat'
    if condition == FAITHFUL_NEXTLAT_CONDITION:
        return 'faithful_nextlat'
    if condition == 'native':
        return 'native'
    if condition in {'stp', 'stp_released'}:
        return 'semantic_tube_prediction_released'
    if condition == 'stp_paper':
        return 'semantic_tube_prediction_paper'
    raise ValueError(f'Unsupported training condition: {condition}; legacy objectives are retired')
DATASET_SPLITS = {
    'uspto_mit_synthesis': {'train': 'uspto_mit_synthesis/train_r_smiles.csv', 'validation': 'uspto_mit_synthesis/validation_r_smiles.csv', 'test': 'uspto_mit_synthesis/test_r_smiles.csv'},
    'orderly_forward': {'train': 'orderly_forward/train.csv', 'test': 'orderly_forward/test.csv'},
    'non_uspto_forward': {'test': 'non_uspto_forward/test.csv'},
    'metatrans_full': {'train': 'metatrans/train.csv', 'released_validation': 'metatrans/released_validation.csv', 'test': 'metatrans/heldout_drug_test.csv'},
    'uspto_50k_retro': {
        'train': 'uspto_50k/train_single.csv',
        'train_r_smiles': 'uspto_50k/train_r_smiles.csv',
        'validation_r_smiles': 'uspto_50k/validation_r_smiles.csv',
        'test_r_smiles': 'uspto_50k/test_r_smiles.csv',
    },
    'uspto_480k_template_heldout': {'train': 'uspto_480k_template_heldout/train.csv', 'validation': 'uspto_480k_template_heldout/validation.csv', 'test': 'uspto_480k_template_heldout/test.csv'},
    'non_uspto_retro': {'test': 'non_uspto_retro/test.csv'},
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

def validate_serialization_endings(collator, rows, eos_token_id: int) -> None:
    for start in range(0, len(rows), 64):
        batch = collator(rows[start:start + 64])
        # Use the existing collator's suffix labels; the retired JEPA extraction
        # helper is no longer present in this repository.
        sources = [ids[attention.bool() & labels.eq(IGNORE_INDEX)]
                   for ids, attention, labels in zip(batch['input_ids'], batch['attention_mask'], batch['labels'])]
        targets = [ids[attention.bool() & labels.ne(IGNORE_INDEX)]
                   for ids, attention, labels in zip(batch['input_ids'], batch['attention_mask'], batch['labels'])]
        if any(source.numel() == 0 or target.numel() == 0 for source, target in zip(sources, targets)):
            raise ValueError('serialization requires nonempty source and target')
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

def validation_selector(metrics: Mapping[str, float], task: str) -> tuple[float, ...]:
    if task == "metabolism":
        return (metrics["recall_at5"], metrics["lower_bound_precision_at5"])
    return (metrics["exact_top1"],)

def save_training_checkpoint(checkpoint: Path, model, tokenizer, optimizer, scheduler, generator, method, *, epoch: int, global_step: int, planned_epochs: int, curves, epoch_history, best_selector, best_checkpoint: str | None, elapsed_wall_time_seconds: float) -> None:
    checkpoint.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint, safe_serialization=True, selected_adapters=[ADAPTER_NAME], save_embedding_layers=False)
    tokenizer.save_pretrained(checkpoint)
    state = {
        'epoch': epoch,
        'global_step': global_step,
        'planned_epochs': planned_epochs,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'loader_generator_state': generator.get_state(),
        'python_rng_state': random.getstate(),
        'numpy_rng_state': np.random.get_state(),
        'torch_rng_state': torch.get_rng_state(),
        'cuda_rng_states': torch.cuda.get_rng_state_all(),
        'curves': curves,
        'epoch_history': epoch_history,
        'best_selector': best_selector,
        'best_checkpoint': best_checkpoint,
        'elapsed_wall_time_seconds': elapsed_wall_time_seconds,
    }
    if isinstance(method, SemanticTubePrediction):
        state['stp_generator_state'] = method.generator_state()
    if hasattr(method, 'save_training_state'):
        method.save_training_state(checkpoint / 'auxiliary_training_state.pt')
    torch.save(state, checkpoint / 'training_state.pt')

def restore_training_checkpoint(checkpoint: Path, model, optimizer, scheduler, generator, planned_epochs: int, method=None):
    load_adapter_checkpoint(model, checkpoint)
    state = torch.load(checkpoint / 'training_state.pt', map_location=model.device, weights_only=False)
    if state['planned_epochs'] != planned_epochs:
        raise ValueError("resume must use the checkpoint's original planned epoch budget")
    if hasattr(method, 'restore_training_state'):
        method.restore_training_state(checkpoint / 'auxiliary_training_state.pt')
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])
    generator.set_state(state['loader_generator_state'].cpu())
    random.setstate(state['python_rng_state'])
    np.random.set_state(state['numpy_rng_state'])
    torch.set_rng_state(state['torch_rng_state'].cpu())
    torch.cuda.set_rng_state_all([value.cpu() for value in state['cuda_rng_states']])
    if isinstance(method, SemanticTubePrediction):
        if 'stp_generator_state' not in state:
            raise ValueError('resume checkpoint is missing STP sampler state')
        method.set_generator_state(state['stp_generator_state'], model.device)
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

class NativeObjective:
    """Ordinary label-shifted ChemFM loss, with the trainer's logging fields."""

    def __call__(self, model, batch):
        from types import SimpleNamespace
        result = model(**{key: batch[key] for key in ('input_ids', 'attention_mask', 'labels')})
        return SimpleNamespace(loss=result.loss, native_loss=result.loss, logits=result.logits, jepa_loss=None, sigreg_loss=None, jepa_objective_loss=None, jepa_active=False)

def train(args):
    if not torch.cuda.is_available():
        raise EnvironmentError('Gate 4/5 ChemFM-1B fine-tuning requires CUDA')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')
    set_seed(args.seed)
    task = TASKS[args.dataset]
    evaluation_beam_size = 20 if task == 'metabolism' else 10
    train_path = args.train_manifest.resolve()
    validation_path = args.validation_manifest.resolve()
    train_rows = read_rows(args.dataset, path=train_path)
    val_rows = read_rows(args.dataset, split='validation', path=validation_path)
    if args.max_train_rows is not None:
        train_rows = train_rows[:args.max_train_rows]
    if args.max_validation_rows is not None:
        val_rows = val_rows[:args.max_validation_rows]
    if not train_rows or not val_rows:
        raise ValueError('training and validation manifests must both be nonempty')
    method_family = condition_family(args.condition)
    has_released_stp = method_family == 'semantic_tube_prediction_released'
    has_paper_stp = method_family == 'semantic_tube_prediction_paper'
    has_stp = has_released_stp or has_paper_stp
    has_projected = method_family in {'decoder_projected', 'full_state_nextlat'}
    has_full_state = method_family == 'full_state_nextlat'
    has_faithful_nextlat = method_family == 'faithful_nextlat'
    has_auxiliary = has_projected or has_faithful_nextlat
    tokenizer = load_reaction_tokenizer(TOKENIZER_DIR)
    chemfm_vocab_size = len(tokenizer)
    add_predictor_tokens(tokenizer)
    collator = ReactionCollator(tokenizer, task=task)
    validate_serialization_endings(collator, train_rows, tokenizer.eos_token_id)
    validate_serialization_endings(collator, val_rows, tokenizer.eos_token_id)
    model = load_lora_model(MODEL_DIR, tokenizer, chemfm_vocab_size=chemfm_vocab_size, attn_implementation=args.attention_implementation, lora_rank=args.lora_rank, lora_alpha=args.lora_alpha).cuda()
    initial_trainable_sha256 = trainable_parameter_sha256(model)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        model.enable_input_require_grads()
    non_embedding_parameters = model.num_parameters(exclude_embeddings=True)
    generator = torch.Generator().manual_seed(args.seed)
    worker_kwargs = {'num_workers': args.dataloader_workers, 'persistent_workers': args.dataloader_workers > 0}
    if args.dataloader_workers > 0:
        worker_kwargs['prefetch_factor'] = args.dataloader_prefetch_factor
    loader = DataLoader(train_rows, batch_size=args.batch_size, shuffle=True, generator=generator, collate_fn=collator, pin_memory=args.pin_memory, **worker_kwargs)
    validation = DataLoader(val_rows, batch_size=args.batch_size, shuffle=False, collate_fn=collator, pin_memory=args.pin_memory, **worker_kwargs)
    updates_per_epoch = max(1, math.ceil(len(loader) / args.gradient_accumulation_steps))
    steps = max(1, args.epochs * updates_per_epoch)
    if has_stp:
        stp_class = PaperSemanticTubePrediction if has_paper_stp else SemanticTubePrediction
        method = stp_class(seed=args.seed, reactant_start_token_id=tokenizer.convert_tokens_to_ids('<rstart>'), product_start_token_id=tokenizer.convert_tokens_to_ids('<prostart>'), eos_token_id=tokenizer.eos_token_id)
    elif has_auxiliary:
        args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        # Isolate predictor initialization from Native's transformer dropout RNG.
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            objective_class = (FaithfulNextLatObjective if has_faithful_nextlat else
                               FullStateObjective if has_full_state else DecoderProjectedObjective)
            objective_kwargs = dict(
                product_start_token_id=tokenizer.convert_tokens_to_ids('<prostart>'),
                eos_token_id=tokenizer.eos_token_id,
                resume_state=(args.resume_from / 'auxiliary_training_state.pt') if args.resume_from else None)
            if has_faithful_nextlat:
                objective_kwargs['native_vocab_size'] = chemfm_vocab_size
            method = objective_class(
                model.get_output_embeddings().weight[:392], **objective_kwargs
            ).to(model.device)
        if args.resume_from is None and has_projected:
            calibration_generator = torch.Generator().manual_seed(args.seed)
            calibration_loader = DataLoader(train_rows, batch_size=args.batch_size, shuffle=True,
                generator=calibration_generator, collate_fn=collator)
            calibration_batch = {k: v.to(model.device) for k, v in next(iter(calibration_loader)).items() if torch.is_tensor(v)}
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                method.calibrate(model, calibration_batch)
            args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            (args.checkpoint_dir / 'auxiliary_calibration.json').write_text(json.dumps(
                {'calibration': method.calibration, 'svd': method.metadata}, indent=2))
    else:
        method = NativeObjective()
    optimizer_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if has_auxiliary:
        optimizer_parameters.extend(method.parameters())
    optimizer = torch.optim.AdamW(optimizer_parameters, lr=args.learning_rate, betas=ADAM_BETAS, eps=ADAM_EPSILON, weight_decay=WEIGHT_DECAY, fused=args.fused_adamw)
    scheduler = get_scheduler('cosine_with_min_lr', optimizer, num_warmup_steps=int(steps * WARMUP_RATIO), num_training_steps=steps, scheduler_specific_kwargs={'min_lr': MIN_LEARNING_RATE})
    has_jepa = method_family != 'native'
    jepa_loss_type = 'cosine'
    ratio = 1.0
    resolved_ratio = ratio if has_jepa else -1.0
    actual_lambda = args.stp_lambda if has_stp else 0.0
    config = {
        'method_family': method_family,
        'learning_rate': args.learning_rate,
        'epochs': args.epochs,
        'resource_budget_epochs': args.stop_after_epoch,
        'physical_batch_size': args.batch_size,
        'gradient_accumulation_steps': args.gradient_accumulation_steps,
        'effective_batch_size': args.batch_size * args.gradient_accumulation_steps,
        'k': None if has_stp else args.k,
        'lambda_eff': args.lambda_eff,
        'actual_lambda': actual_lambda,
        'initial_trainable_sha256': initial_trainable_sha256,
        'lora_rank': args.lora_rank,
        'lora_alpha': args.lora_alpha,
        'lora_scaling_alpha_over_rank': args.lora_alpha / args.lora_rank,
        'lora_dropout': 0.1,
        'lora_target_modules': ['q_proj', 'v_proj', 'k_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
        'lora_modules_to_save': ['embed_tokens', 'lm_head'],
        'lora_use_rslora': False,
        'jepa_loss_dropout': None if has_stp else args.dropout if has_jepa else None,
        'jepa_ratio': resolved_ratio,
        'jepa_target_stop_gradient': has_auxiliary,
        'jepa_target_encoder': None,
        'jepa_loss_type': jepa_loss_type if has_jepa else None,
        'semantic_tube_prediction': has_stp,
        'stp_formulation': 'paper_equation' if has_paper_stp else 'released_patch_vs_complement' if has_released_stp else None,
        'stp_paper': STP_PAPER if has_stp else None,
        'stp_upstream_repository': STP_UPSTREAM_REPOSITORY if has_stp else None,
        'stp_upstream_commit': STP_UPSTREAM_COMMIT if has_stp else None,
        'stp_executable_mode': '--linear=random_span' if has_released_stp else None,
        'stp_lambda': args.stp_lambda if has_stp else None,
        'stp_hidden_layer': 'final' if has_stp else None,
        'stp_spans_per_example': 1 if has_stp else None,
        'stp_span_sampler': 'released default: start uniform, end uniform conditional on start, reject only the full content span' if has_released_stp else 'released outer sampler; reject intervals without an interior; sample r uniformly from the valid interior' if has_paper_stp else None,
        'stp_content_regions': 'reactant and product SMILES tokens; ChemFM framing excluded' if has_stp else None,
        'stp_gradient_flow': 'symmetric/no stop-gradient' if has_stp else None,
        'stp_loss_reduction_dtype': 'float32' if has_stp else None,
        'generation_vocabulary_unchanged': False,
        'optimizer_steps_per_epoch': updates_per_epoch,
        'train_size': len(train_rows),
        'validation_size': len(val_rows),
        'train_manifest': str(train_path),
        'train_manifest_sha256': file_sha256(train_path),
        'validation_manifest': str(validation_path),
        'validation_manifest_sha256': file_sha256(validation_path),
        'optimizer': 'adamw_torch',
        'fused_adamw': args.fused_adamw,
        'gradient_checkpointing': args.gradient_checkpointing,
        'pin_memory': args.pin_memory,
        'dataloader_workers': args.dataloader_workers,
        'dataloader_prefetch_factor': args.dataloader_prefetch_factor if args.dataloader_workers else None,
        'attention_implementation': getattr(model.config, '_attn_implementation', 'unknown'),
        'evaluation_generation_batch_size': args.eval_generation_batch_size,
        'evaluation_beam_size': evaluation_beam_size,
        'adam_beta1': ADAM_BETAS[0],
        'adam_beta2': ADAM_BETAS[1],
        'adam_epsilon': ADAM_EPSILON,
        'weight_decay': WEIGHT_DECAY,
        'scheduler': 'cosine_with_min_lr',
        'warmup_ratio': WARMUP_RATIO,
        'min_learning_rate': MIN_LEARNING_RATE,
        'upstream_llm_jepa_commit': 'ea0017c654ad917066ff32afc88276bea8ca5f7e',
        'evaluation_epochs': list(args.evaluation_epochs),
    }
    tracker = WandbTracker(TrackingContext(task, args.dataset, args.condition, args.seed, args.data_fraction, config), run_name=f'gate{args.gate}-{args.dataset}-{args.condition}-s{args.seed}', enabled=not args.no_wandb)
    curves = []
    epoch_history = []
    start = time.perf_counter()
    global_step = 0
    start_epoch = 0
    best_selector = None
    best_checkpoint = None
    previous_elapsed_seconds = 0.0
    if has_auxiliary:
        config.update({'decoder_projected': method.metadata if method_family == 'decoder_projected' else None,
            'full_state_nextlat': method.metadata if has_full_state else None,
            'faithful_nextlat': method.metadata if has_faithful_nextlat else None,
            'auxiliary_calibration': method.calibration,
            'jepa_loss_type': 'smooth_l1_plus_teacher_to_prediction_kl', 'jepa_loss_dropout': None,
            'actual_lambda': 1.0, 'generation_vocabulary_unchanged': True,
            'auxiliary_inference_components': False})
    if args.resume_from is not None:
        state = restore_training_checkpoint(args.resume_from.resolve(), model, optimizer, scheduler, generator, args.epochs, method)
        start_epoch = state['epoch']
        global_step = state['global_step']
        curves = state['curves']
        epoch_history = state['epoch_history']
        best_selector = state['best_selector']
        best_checkpoint = state['best_checkpoint']
        previous_elapsed_seconds = state.get('elapsed_wall_time_seconds', args.prior_wall_time_seconds)
        if has_auxiliary:
            config['auxiliary_calibration'] = method.calibration
        if start_epoch > args.stop_after_epoch:
            raise ValueError('resume checkpoint is beyond the requested resource budget')
        if has_auxiliary:
            (args.checkpoint_dir / 'auxiliary_steps.jsonl').write_text(
                ''.join(json.dumps(record) + '\n' for record in curves))
    optimizer.zero_grad(set_to_none=True)
    try:
        for epoch_index in range(start_epoch, args.stop_after_epoch):
            model.train()
            torch.cuda.synchronize()
            epoch_training_started = time.perf_counter()
            training_loader = loader
            window_records = []
            for (batch_index, raw) in enumerate(training_loader):
                batch_tokens = int(raw['attention_mask'].sum())
                batch = {name: value.to(model.device, non_blocking=args.pin_memory) for (name, value) in raw.items() if torch.is_tensor(value)}
                window_start = batch_index // args.gradient_accumulation_steps * args.gradient_accumulation_steps
                window_size = min(args.gradient_accumulation_steps, len(loader) - window_start)
                microbatch_number = epoch_index * len(loader) + batch_index + 1
                if has_stp:
                    output = method(model, batch, stp_weight=actual_lambda)
                    stp_metrics = {'mean_sampled_span_fraction': sum(((span[2] - span[0]) / span[3] if len(span) == 4 else (span[1] - span[0]) / span[2] for span in output.sampled_spans)) / len(output.sampled_spans)}
                    jepa_active = True
                    sigreg_value = None
                    objective_value = float(output.jepa_loss.detach())
                else:
                    output = method(model, batch)
                    stp_metrics = ({'L_z': float(output.lz.detach()), 'L_KL': float(output.kl.detach())}
                        if has_auxiliary else {})
                    if has_projected:
                        stp_metrics.update(alpha=method.alpha,
                            calibration_auxiliary_ntp_ratio=method.calibration['expected_auxiliary_ntp_ratio'])
                    jepa_active = output.jepa_active
                    sigreg_value = None if output.sigreg_loss is None else float(output.sigreg_loss.detach())
                    objective_value = None if output.jepa_objective_loss is None else float(output.jepa_objective_loss.detach())
                if not torch.isfinite(output.loss):
                    raise FloatingPointError(f'non-finite loss at microbatch {microbatch_number}')
                (output.loss / window_size).backward()
                effective_tokens = batch_tokens
                window_records.append({
                    'native_loss': float(output.native_loss.detach()),
                    'jepa_loss': None if output.jepa_loss is None else float(output.jepa_loss.detach()),
                    'sigreg_loss': sigreg_value,
                    'jepa_objective_loss': objective_value,
                    'total_loss': float(output.loss.detach()),
                    'jepa_active': jepa_active,
                    'batch_tokens': batch_tokens,
                    'effective_tokens': effective_tokens,
                    'stp_metrics': stp_metrics,
                })
                boundary = (batch_index + 1) % args.gradient_accumulation_steps == 0 or batch_index + 1 == len(loader)
                if not boundary:
                    continue
                learning_rate = optimizer.param_groups[0]['lr']
                predictor_gradient_norm = (float(torch.stack([p.grad.detach().float().square().sum()
                    for p in method.parameters() if p.grad is not None]).sum().sqrt()) if has_auxiliary else None)
                # Upstream registers the dynamics model inside the trainable
                # model, so its parameters participate in the same global
                # max-norm clipping. Preserve legacy projected-arm behavior.
                extra_clip_modules = (("faithful_nextlat", method),) if has_faithful_nextlat else ()
                (total_gradient_norm, largest_gradient) = gradient_diagnostics(
                    model, extra_clip_modules)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                active_losses = [row['jepa_loss'] for row in window_records if row['jepa_loss'] is not None]
                sigreg_losses = [row['sigreg_loss'] for row in window_records if row['sigreg_loss'] is not None]
                objective_losses = [row['jepa_objective_loss'] for row in window_records if row['jepa_objective_loss'] is not None]
                record = {
                    'step': global_step,
                    'epoch': epoch_index + 1,
                    'native_loss': sum((row['native_loss'] for row in window_records)) / len(window_records),
                    'jepa_loss': None if not active_losses else sum(active_losses) / len(active_losses),
                    'sigreg_loss': None if not sigreg_losses else sum(sigreg_losses) / len(sigreg_losses),
                    'jepa_objective_loss': None if not objective_losses else sum(objective_losses) / len(objective_losses),
                    'total_loss': sum((row['total_loss'] for row in window_records)) / len(window_records),
                    'jepa_active': any((row['jepa_active'] for row in window_records)),
                    'jepa_active_microbatches': sum((row['jepa_active'] for row in window_records)),
                    'learning_rate': learning_rate,
                    'gradient_norm': total_gradient_norm,
                    'max_gradient_parameter': largest_gradient[0],
                    'max_parameter_gradient_norm': largest_gradient[1],
                    'batch_tokens': sum((row['batch_tokens'] for row in window_records)),
                    'effective_tokens': sum((row['effective_tokens'] for row in window_records)),
                    'model_calls': len(window_records),
                }
                stp_keys = sorted(set().union(*(row['stp_metrics'] for row in window_records)))
                stp_record = {key: sum((row['stp_metrics'][key] for row in window_records if key in row['stp_metrics'])) / sum((key in row['stp_metrics'] for row in window_records)) for key in stp_keys}
                record['stp'] = stp_record if has_stp and stp_record else None
                if has_auxiliary:
                    record['auxiliary'] = {**stp_record, 'predictor_gradient_norm': predictor_gradient_norm,
                        'chemfm_total_gradient_norm': total_gradient_norm,
                        'peak_vram_bytes': torch.cuda.max_memory_allocated(),
                        'tokens_per_second': sum(row['effective_tokens'] for row in curves + [record]) / max(time.perf_counter() - start, 1e-12)}
                    with (args.checkpoint_dir / 'auxiliary_steps.jsonl').open('a') as log:
                        log.write(json.dumps(record) + '\n')
                record['estimated_flops'] = 6.0 * record['effective_tokens'] * non_embedding_parameters
                curves.append(record)
                tracker.log_training_step(step=global_step, native_loss=record['native_loss'], jepa_loss=record['jepa_loss'], total_loss=record['total_loss'], sigreg_loss=record['sigreg_loss'], jepa_objective_loss=record['jepa_objective_loss'], gradient_norm=total_gradient_norm, max_gradient_parameter=largest_gradient[0], max_parameter_gradient_norm=largest_gradient[1], learning_rate=learning_rate, jepa_active=record['jepa_active'], batch_tokens=record['batch_tokens'], model_calls=record['model_calls'], effective_tokens=record['effective_tokens'], peak_vram_bytes=torch.cuda.max_memory_allocated(), estimated_flops=record['estimated_flops'], extra_metrics=stp_record if stp_record else None)
                window_records = []
            torch.cuda.synchronize()
            epoch_training_seconds = time.perf_counter() - epoch_training_started
            checkpoint = args.checkpoint_dir.resolve() / f'epoch_{epoch_index + 1}'
            evaluate_epoch = epoch_index + 1 in args.evaluation_epochs
            if evaluate_epoch:
                val_loss = native_loss(model, validation)
                (metrics, predictions) = beam_evaluate(model, tokenizer, collator, val_rows, task, windows=evaluation_beam_size, generation_batch_size=args.eval_generation_batch_size)
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
                'epoch': epoch_index + 1,
                'global_step': global_step,
                'validation_native_loss': val_loss,
                'validation_metrics': metrics,
                'predictions': predictions,
                'selector': selector,
                'checkpoint': str(checkpoint),
                'training_seconds': epoch_training_seconds,
                'checkpoint_saved': False,
                'checkpoint_seconds': 0.0,
            }
            epoch_history.append(epoch_record)
            if evaluate_epoch:
                tracker.log_evaluation(step=global_step, split='validation', task_metrics=metrics, validity=metrics['valid_rate'], native_loss=val_loss)
            if not args.final_checkpoint_only or epoch_index + 1 == args.stop_after_epoch:
                checkpoint_started = time.perf_counter()
                save_training_checkpoint(checkpoint, model, tokenizer, optimizer, scheduler, generator, method, epoch=epoch_index + 1, global_step=global_step, planned_epochs=args.epochs, curves=curves, epoch_history=epoch_history, best_selector=best_selector, best_checkpoint=best_checkpoint, elapsed_wall_time_seconds=previous_elapsed_seconds + time.perf_counter() - start)
                epoch_record['checkpoint_saved'] = True
                epoch_record['checkpoint_seconds'] = time.perf_counter() - checkpoint_started
    except Exception:
        tracker.finish({'status': 'failed'})
        raise
    if best_checkpoint is None:
        raise RuntimeError('no validation checkpoint was produced')
    load_adapter_checkpoint(model, Path(best_checkpoint))
    selected = next((row for row in epoch_history if row['checkpoint'] == best_checkpoint))
    val_loss = selected['validation_native_loss']
    metrics = selected['validation_metrics']
    predictions = selected['predictions']
    if has_stp:
        stp_rows = [row for row in curves if row['jepa_loss'] is not None]
        diagnostics = {
            'type': 'official_stp_training_summary',
            'upstream_commit': STP_UPSTREAM_COMMIT,
            'first_epoch_mean_stp_loss': float(np.mean([row['jepa_loss'] for row in stp_rows if row['epoch'] == 1])),
            'final_epoch_mean_stp_loss': float(np.mean([row['jepa_loss'] for row in stp_rows if row['epoch'] == selected['epoch']])),
            'final_epoch_mean_sampled_span_fraction': float(np.mean([row['stp']['mean_sampled_span_fraction'] for row in stp_rows if row['epoch'] == selected['epoch']])),
        }
    elif has_auxiliary:
        diagnostics = {'type': method_family, 'method': method.metadata, 'calibration': method.calibration}
    else:
        diagnostics = {'type': 'native_training_summary'}
    result = {
        'gate': args.gate,
        'dataset': args.dataset,
        'task': task,
        'condition': args.condition,
        'seed': args.seed,
        'config': config,
        'validation_native_loss': val_loss,
        'validation_metrics': metrics,
        'diagnostics': diagnostics,
        'curves': curves,
        'epoch_history': [{key: value for (key, value) in row.items() if key != 'predictions'} for row in epoch_history],
        'selected_checkpoint': best_checkpoint,
        'selected_epoch': selected['epoch'],
        'predictions': predictions,
        'compute': {
            'optimizer_steps': global_step,
            'wall_time_seconds': previous_elapsed_seconds + time.perf_counter() - start,
            'peak_vram_bytes': torch.cuda.max_memory_allocated(),
            'jepa_active_microbatches': sum((row['jepa_active_microbatches'] for row in curves)),
            'model_calls': sum((row['model_calls'] for row in curves)),
            'native_tokens': sum((row['batch_tokens'] for row in curves)),
            'effective_tokens': sum((row['effective_tokens'] for row in curves)),
            'estimated_flops': sum((row['estimated_flops'] for row in curves)),
            'effective_tokens_per_second': sum((row['effective_tokens'] for row in curves)) / max(time.perf_counter() - start, 1e-12),
        },
    }
    tracker.finish({'validation_primary': metrics['exact_top1'], 'validation_validity': metrics['valid_rate'], 'validation_native_loss': val_loss})
    return result

def main():
    parser = argparse.ArgumentParser(description='Controlled Native/STP ChemFM training')
    parser.add_argument('--gate', type=int, choices=(4, 5), required=True)
    parser.add_argument('--dataset', choices=sorted(TASKS), required=True)
    parser.add_argument('--condition', choices=TRAINING_CONDITIONS, required=True)
    parser.add_argument('--seed', type=int, default=533)
    parser.add_argument('--learning-rate', type=float, default=0.0001)
    parser.add_argument('--lora-rank', type=int, default=8)
    parser.add_argument('--lora-alpha', type=int, default=8)
    parser.add_argument('--stp-lambda', type=float, default=0.02, help='official released Llama-1B STP coefficient')
    parser.add_argument('--k', type=int, choices=(0,), default=0, help='legacy wrapper compatibility; no predictor-token objective')
    parser.add_argument('--lambda-eff', type=float, choices=(1.0,), default=1.0, help='legacy wrapper compatibility; use --stp-lambda')
    parser.add_argument('--dropout', type=float, choices=(0.5,), default=0.5, help='legacy wrapper compatibility; no auxiliary dropout')
    parser.add_argument('--epochs', type=int, choices=(1, 2, 4), default=1)
    parser.add_argument('--stop-after-epoch', type=int, choices=(1, 2, 4), help='successive-halving resource rung; the scheduler still uses --epochs as its fixed maximum budget')
    parser.add_argument('--prior-wall-time-seconds', type=float, default=0.0, help='fallback only for resuming a legacy checkpoint without elapsed time')
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--gradient-accumulation-steps', type=int, default=4)
    parser.add_argument('--evaluation-epochs', type=int, nargs='+', help='epochs at which to run validation CE and generation; every epoch is still checkpointed. Omitted evaluates every resource-budget epoch.')
    parser.add_argument('--gradient-checkpointing', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--fused-adamw', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--final-checkpoint-only', action=argparse.BooleanOptionalAction, default=False, help='save only the final resource-budget checkpoint; requires evaluation only at that epoch and intentionally disables intermediate resume')
    parser.add_argument('--attention-implementation', choices=('eager', 'sdpa', 'flash_attention_2'), help='Transformers attention backend; omitted retains library auto-selection')
    parser.add_argument('--pin-memory', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--dataloader-workers', type=int, default=0)
    parser.add_argument('--dataloader-prefetch-factor', type=int, default=2)
    parser.add_argument('--eval-generation-batch-size', type=int, choices=(1, 2, 4), default=1, help='number of length-sorted, left-padded prompts evaluated together; does not change beam width, stopping, generation length, or R-SMILES aggregation')
    parser.add_argument('--data-fraction', type=float, default=1.0)
    parser.add_argument('--train-manifest', type=Path, required=True)
    parser.add_argument('--validation-manifest', type=Path, required=True)
    parser.add_argument('--checkpoint-dir', type=Path, required=True)
    parser.add_argument('--resume-from', type=Path)
    parser.add_argument('--max-train-rows', type=int, help='integration-test limit only')
    parser.add_argument('--max-validation-rows', type=int, help='integration-test limit only')
    parser.add_argument('--no-wandb', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.stop_after_epoch is None:
        args.stop_after_epoch = args.epochs
    if args.evaluation_epochs is None:
        args.evaluation_epochs = list(range(1, args.stop_after_epoch + 1))
    args.evaluation_epochs = sorted(set(args.evaluation_epochs))
    if not args.evaluation_epochs or any((epoch < 1 or epoch > args.stop_after_epoch for epoch in args.evaluation_epochs)):
        raise ValueError('evaluation epochs must be within the requested resource budget')
    if args.stop_after_epoch > args.epochs:
        raise ValueError('stop-after epoch cannot exceed the planned epoch budget')
    if args.prior_wall_time_seconds < 0:
        raise ValueError('prior wall time cannot be negative')
    if args.batch_size < 1 or args.gradient_accumulation_steps < 1:
        raise ValueError('batch size and gradient accumulation must be positive')
    if args.lora_rank < 1 or args.lora_alpha < 1:
        raise ValueError('LoRA rank and alpha must be positive')
    if args.final_checkpoint_only and args.evaluation_epochs != [args.stop_after_epoch]:
        raise ValueError('--final-checkpoint-only requires evaluation only at stop-after epoch')
    if args.dataloader_workers < 0 or args.dataloader_prefetch_factor < 1:
        raise ValueError('DataLoader workers must be nonnegative and prefetch positive')
    if args.stp_lambda <= 0.0:
        raise ValueError('STP lambda must be positive')
    if args.condition == NATIVE_CONDITION and (args.lambda_eff != 1.0 or args.dropout != 0.5):
        raise ValueError('native trials must leave irrelevant JEPA defaults unchanged')
    if args.condition == STP_CONDITION and args.stp_lambda != 0.02:
        raise ValueError('legacy --condition stp remains frozen to released lambda=0.02')
    result = train(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({
        'output': str(args.output),
        'dataset': args.dataset,
        'condition': args.condition,
        'seed': args.seed,
        'primary': result['validation_metrics']['exact_top1'],
        'validity': result['validation_metrics']['valid_rate'],
        'native_loss': result['validation_native_loss'],
    }), flush=True)
if __name__ == '__main__':
    main()
