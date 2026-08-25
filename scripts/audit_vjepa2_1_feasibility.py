"""Frozen dense-V-JEPA feasibility and local target-token comparison.

This evaluator does not optimize a model.  It summarizes the selected training
curve (including the one-shot component VJPs) and compares native, direct
endpoint MSE+SIGReg, and dense V-JEPA checkpoints at the exact teacher-forced
states whose following logits predict product tokens.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from peft.utils.save_and_load import load_peft_weights, set_peft_model_state_dict


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chemfm import (  # noqa: E402
    IGNORE_INDEX, MODEL_DIR, TOKENIZER_DIR, ReactionCollator, load_lora_model,
    load_reaction_tokenizer,
)
from jepa import add_predictor_tokens  # noqa: E402
from train import ADAPTER_NAME, read_rows  # noqa: E402
from vjepa2_1 import LayerCapture, VJEPA21_PAPER, VJEPA21_UPSTREAM_COMMIT  # noqa: E402


DEFAULT_NATIVE = (
    ROOT / "runs" / "sigreg_batch16_pilot" / "matched_b4"
    / "native_checkpoints" / "epoch_4"
)
DEFAULT_ENDPOINT = (
    ROOT / "runs" / "mse_ablation" / "stage1"
    / "mse_sigreg_checkpoints" / "epoch_4"
)
DEFAULT_PANEL = (
    ROOT / "data" / "clm_jepa_uspto_mit_validation_256"
    / "uspto_mit_validation_length_stratified_256.csv"
)
DEPTHS = (6, 11, 17, 22)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def adapter_dir(checkpoint: Path) -> Path:
    nested = checkpoint / ADAPTER_NAME
    return nested if nested.exists() else checkpoint


def load_condition(checkpoint: Path, *, legacy_predictor_vocabulary: bool):
    tokenizer = load_reaction_tokenizer(TOKENIZER_DIR)
    chemfm_vocab_size = len(tokenizer)
    if legacy_predictor_vocabulary:
        add_predictor_tokens(tokenizer)
    model = load_lora_model(
        MODEL_DIR, tokenizer, attention_dropout=0.0,
        chemfm_vocab_size=chemfm_vocab_size,
        attn_implementation="sdpa",
    ).cuda().eval()
    state = load_peft_weights(str(adapter_dir(checkpoint)), device="cpu")
    loaded = set_peft_model_state_dict(model, state, adapter_name=ADAPTER_NAME)
    if getattr(loaded, "unexpected_keys", None):
        raise RuntimeError(f"unexpected adapter keys in {checkpoint}: {loaded.unexpected_keys}")
    return tokenizer, ReactionCollator(tokenizer, task="forward"), model


def linear_cka(first: torch.Tensor, second: torch.Tensor) -> float:
    """Feature-space linear CKA without allocating an N-by-N token Gram."""
    x = first.double() - first.double().mean(dim=0, keepdim=True)
    y = second.double() - second.double().mean(dim=0, keepdim=True)
    cross = x.T @ y
    denominator = (x.T @ x).square().sum().sqrt() * (y.T @ y).square().sum().sqrt()
    return float(cross.square().sum() / denominator.clamp_min(torch.finfo(x.dtype).eps))


@torch.inference_mode()
def collect_target_prediction_states(
    checkpoint: Path,
    rows: Sequence[Mapping[str, str]],
    *,
    batch_size: int,
    max_tokens: int,
    legacy_predictor_vocabulary: bool,
) -> dict[str, Any]:
    tokenizer, collator, model = load_condition(
        checkpoint, legacy_predictor_vocabulary=legacy_predictor_vocabulary,
    )
    features = {depth: [] for depth in DEPTHS}
    final_norm_features = []
    token_losses = []
    correct = 0
    token_count = 0
    backbone = model.get_base_model().model
    final_norm_output: list[torch.Tensor] = []

    def final_norm_hook(_module, _inputs, output):
        final_norm_output.append(output)

    final_handle = backbone.norm.register_forward_hook(final_norm_hook)
    try:
        for start in range(0, len(rows), batch_size):
            if token_count >= max_tokens:
                break
            raw = collator(rows[start : start + batch_size])
            batch = {
                key: value.to(model.device)
                for key, value in raw.items() if torch.is_tensor(value)
            }
            final_norm_output.clear()
            with LayerCapture(backbone, DEPTHS) as capture:
                output = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    use_cache=False,
                    return_dict=True,
                )
            prediction_mask = batch["labels"][:, 1:].ne(IGNORE_INDEX)
            labels = batch["labels"][:, 1:][prediction_mask]
            logits = output.logits[:, :-1][prediction_mask].float()
            available = min(max_tokens - token_count, labels.numel())
            if available <= 0:
                break
            labels = labels[:available]
            logits = logits[:available]
            token_losses.append(F.cross_entropy(logits, labels, reduction="none").cpu())
            correct += int(logits.argmax(dim=-1).eq(labels).sum())
            for depth, values in zip(DEPTHS, capture.values()):
                features[depth].append(values[:, :-1][prediction_mask][:available].float().cpu())
            if len(final_norm_output) != 1:
                raise RuntimeError("final ChemFM norm was not captured exactly once")
            final_norm_features.append(
                final_norm_output[0][:, :-1][prediction_mask][:available].float().cpu()
            )
            token_count += available
    finally:
        final_handle.remove()
        del model
        torch.cuda.empty_cache()
    if not token_count:
        raise RuntimeError("evaluation panel produced no target-token predictions")
    return {
        "target_token_count": token_count,
        "normalized_target_token_ce": float(torch.cat(token_losses).mean()),
        "teacher_forced_top1": correct / token_count,
        "features": {depth: torch.cat(parts) for depth, parts in features.items()},
        "final_norm_features": torch.cat(final_norm_features),
        "tokenizer_size": len(tokenizer),
    }


def representation_change(candidate: torch.Tensor, native: torch.Tensor) -> dict[str, float]:
    difference = candidate - native
    native_norm = native.norm(dim=-1)
    return {
        "linear_cka_with_native": linear_cka(candidate, native),
        "mean_cosine_with_native": float(F.cosine_similarity(candidate, native, dim=-1).mean()),
        "relative_rms_displacement": float(
            difference.square().mean().sqrt()
            / native.square().mean().sqrt().clamp_min(torch.finfo(native.dtype).eps)
        ),
        "mean_candidate_norm": float(candidate.norm(dim=-1).mean()),
        "mean_native_norm": float(native_norm.mean()),
    }


def training_summary(result_path: Path) -> dict[str, Any]:
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    curves = payload["curves"]
    if not curves:
        raise ValueError("training result contains no optimization curve")
    selected_vjp = [
        row for row in curves
        if row.get("dense_vjepa2_1")
        and any("gradient_norm" in key for key in row["dense_vjepa2_1"])
    ]
    return {
        "path": str(result_path.resolve()),
        "sha256": file_sha256(result_path),
        "steps": len(curves),
        "initial": curves[0],
        "final": curves[-1],
        "component_gradient_diagnostic": selected_vjp[0] if selected_vjp else None,
        "selected_epoch": payload.get("selected_epoch"),
        "validation_native_loss": payload.get("validation_native_loss"),
        "validation_metrics": payload.get("validation_metrics"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dense-checkpoint", type=Path, required=True)
    parser.add_argument("--dense-result", type=Path, required=True)
    parser.add_argument("--native-checkpoint", type=Path, default=DEFAULT_NATIVE)
    parser.add_argument("--endpoint-checkpoint", type=Path, default=DEFAULT_ENDPOINT)
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise EnvironmentError("frozen feasibility evaluation requires CUDA")
    if args.limit < 1 or args.batch_size < 1 or args.max_tokens < 2:
        raise ValueError("limit, batch size, and max tokens must be positive")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    rows = read_rows("uspto_mit_synthesis", split="validation", path=args.panel)[: args.limit]
    checkpoints = {
        "native": (args.native_checkpoint, True),
        "endpoint_mse_sigreg": (args.endpoint_checkpoint, True),
        "dense_vjepa2_1": (args.dense_checkpoint, False),
    }
    collected = {}
    for label, (checkpoint, legacy) in checkpoints.items():
        print(json.dumps({"collecting": label, "checkpoint": str(checkpoint)}), flush=True)
        collected[label] = collect_target_prediction_states(
            checkpoint, rows, batch_size=args.batch_size,
            max_tokens=args.max_tokens,
            legacy_predictor_vocabulary=legacy,
        )
    native = collected["native"]
    conditions = {}
    for label, values in collected.items():
        condition = {
            "checkpoint": str(checkpoints[label][0].resolve()),
            "target_token_count": values["target_token_count"],
            "normalized_target_token_ce": values["normalized_target_token_ce"],
            "teacher_forced_top1": values["teacher_forced_top1"],
            "tokenizer_size": values["tokenizer_size"],
            "by_depth": {},
        }
        if label != "native":
            for depth in DEPTHS:
                condition["by_depth"][str(depth)] = representation_change(
                    values["features"][depth], native["features"][depth]
                )
            condition["final_chemfm_norm"] = representation_change(
                values["final_norm_features"], native["final_norm_features"]
            )
            condition["target_ce_change_vs_native"] = (
                values["normalized_target_token_ce"]
                - native["normalized_target_token_ce"]
            )
            condition["teacher_forced_top1_change_vs_native"] = (
                values["teacher_forced_top1"] - native["teacher_forced_top1"]
            )
        conditions[label] = condition
    payload = {
        "protocol": {
            "paper": VJEPA21_PAPER,
            "upstream_commit": VJEPA21_UPSTREAM_COMMIT,
            "panel": str(args.panel.resolve()),
            "panel_sha256": file_sha256(args.panel),
            "reaction_limit": args.limit,
            "max_aligned_target_tokens": args.max_tokens,
            "depths": list(DEPTHS),
            "position_semantics": (
                "h[:, :-1] where labels[:, 1:] is a product target; these are "
                "the exact teacher-forced states feeding next-token prediction"
            ),
        },
        "training": training_summary(args.dense_result),
        "local_target_token_behavior": conditions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"completed": str(args.output), "tokens": native["target_token_count"]}))


if __name__ == "__main__":
    main()
