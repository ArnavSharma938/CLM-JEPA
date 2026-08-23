"""Held-out LoRA-gradient audit for gradient-interaction checkpoints.

This is evaluation-only: it constructs no optimizer and changes no parameters.
Every checkpoint is measured on the same frozen logical validation batch in
model evaluation mode.  MSE and SIGReg endpoint VJPs are separated before the
full controlled auxiliary and published combination are reconstructed.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chemfm import (  # noqa: E402
    MODEL_DIR,
    TOKENIZER_DIR,
    ReactionCollator,
    load_lora_model,
    load_reaction_tokenizer,
)
from gradient_interaction import CAGRAD_C, combine_gradients  # noqa: E402
from jepa import CLMJEPA, add_predictor_tokens  # noqa: E402
from train import (  # noqa: E402
    TASKS,
    file_sha256,
    load_adapter_checkpoint,
    read_rows,
)


SIGREG_RELATIVE_COEFFICIENT = 4.0 * 0.01 / 0.99


def accumulate(
    buffers: list[torch.Tensor], gradients: Sequence[torch.Tensor | None],
) -> None:
    for buffer, gradient in zip(buffers, gradients):
        if gradient is not None:
            buffer.add_(gradient.detach())


def scale(
    gradients: Sequence[torch.Tensor], coefficient: float,
) -> list[torch.Tensor]:
    return [coefficient * gradient for gradient in gradients]


def add(
    left: Sequence[torch.Tensor], right: Sequence[torch.Tensor],
    right_coefficient: float = 1.0,
) -> list[torch.Tensor]:
    return [
        left_gradient + right_coefficient * right_gradient
        for left_gradient, right_gradient in zip(left, right)
    ]


def relation(
    main: Sequence[torch.Tensor], auxiliary: Sequence[torch.Tensor],
) -> dict[str, float | bool]:
    result = combine_gradients("weighted_sum", main, auxiliary)
    return {
        "cosine_with_heldout_ntp": result.cosine,
        "gradient_norm_ratio_to_heldout_ntp": (
            result.auxiliary_to_main_norm_ratio
        ),
        "conflict": result.conflict,
    }


def endpoint_forward(method, model, raw, *, endpoint_only: bool):
    batch = {
        name: value.to(model.device, non_blocking=True)
        for name, value in raw.items() if torch.is_tensor(value)
    }
    return method(
        model,
        batch,
        k=0,
        jepa_weight=0.0,
        native_weight=1.0,
        monitor_only=True,
        stop_gradient_target=False,
        jepa_loss_type="mse",
        sigreg_tradeoff=0.0,
        jepa_ratio=1.0,
        force_jepa_active=True,
        representation_only=True,
        endpoint_only=endpoint_only,
    )


def audit_checkpoint(
    model,
    method,
    chunks,
    checkpoint: Path,
    *,
    interaction: str,
    actual_lambda: float,
) -> dict:
    load_adapter_checkpoint(model, checkpoint)
    state = torch.load(
        checkpoint / "training_state.pt", map_location="cpu", weights_only=False,
    )
    method.sigreg.global_step = int(state["sigreg_global_step"])
    sigreg_slice = method.sigreg.global_step
    model.eval()

    sources, targets = [], []
    with torch.no_grad():
        for raw in chunks:
            output = endpoint_forward(method, model, raw, endpoint_only=True)
            sources.append(output.source_states.float())
            targets.append(output.target_states.float())
    sources = torch.cat(sources).detach().requires_grad_(True)
    targets = torch.cat(targets).detach().requires_grad_(True)
    mse = torch.nn.functional.mse_loss(sources, targets)
    sigreg = method.sigreg(torch.stack((sources, targets)))
    mse_endpoint = torch.autograd.grad(mse, (sources, targets), retain_graph=True)
    sigreg_endpoint = torch.autograd.grad(sigreg, (sources, targets))

    named_lora = [
        (name, parameter) for name, parameter in model.named_parameters()
        if parameter.requires_grad and ("lora_A" in name or "lora_B" in name)
    ]
    if not named_lora:
        raise RuntimeError("checkpoint model has no trainable LoRA parameters")
    parameters = [parameter for _, parameter in named_lora]
    zeros = lambda: [torch.zeros_like(parameter) for parameter in parameters]
    main_gradients = zeros()
    mse_gradients = zeros()
    sigreg_gradients = zeros()
    offset = 0
    for raw in chunks:
        size = int(raw["input_ids"].size(0))
        output = endpoint_forward(method, model, raw, endpoint_only=False)
        main = output.native_loss / len(chunks)
        mse_surrogate = (
            output.source_states
            * mse_endpoint[0][offset:offset + size].to(output.source_states.dtype)
        ).sum() + (
            output.target_states
            * mse_endpoint[1][offset:offset + size].to(output.target_states.dtype)
        ).sum()
        sigreg_surrogate = (
            output.source_states
            * sigreg_endpoint[0][offset:offset + size].to(output.source_states.dtype)
        ).sum() + (
            output.target_states
            * sigreg_endpoint[1][offset:offset + size].to(output.target_states.dtype)
        ).sum()
        accumulate(
            main_gradients,
            torch.autograd.grad(main, parameters, retain_graph=True, allow_unused=True),
        )
        accumulate(
            mse_gradients,
            torch.autograd.grad(
                mse_surrogate, parameters, retain_graph=True, allow_unused=True,
            ),
        )
        accumulate(
            sigreg_gradients,
            torch.autograd.grad(sigreg_surrogate, parameters, allow_unused=True),
        )
        offset += size

    full_raw = add(
        mse_gradients, sigreg_gradients, SIGREG_RELATIVE_COEFFICIENT,
    )
    full_active = scale(full_raw, actual_lambda)
    combination = combine_gradients(
        interaction,
        main_gradients,
        full_active,
        cagrad_c=CAGRAD_C,
    )
    combined = add(
        scale(main_gradients, combination.main_coefficient),
        full_active,
        combination.auxiliary_coefficient,
    )
    raw_sum = add(main_gradients, full_active)
    modification = add(combined, raw_sum, -1.0)
    modification_relation = combine_gradients(
        "weighted_sum", raw_sum, modification,
    )
    metrics = {
        "epoch": int(state["epoch"]),
        "global_step": int(state["global_step"]),
        "sigreg_slice": sigreg_slice,
        "checkpoint": str(checkpoint.resolve()),
        "heldout_native_loss": float(sum(
            endpoint_forward(method, model, raw, endpoint_only=False).native_loss.detach()
            for raw in chunks
        ) / len(chunks)),
        "mse_loss": float(mse.detach()),
        "sigreg_loss": float(sigreg.detach()),
        "full_auxiliary_loss": float(
            mse.detach() + SIGREG_RELATIVE_COEFFICIENT * sigreg.detach()
        ),
        "gradient_scope": {
            "description": "LoRA A/B parameters only",
            "parameter_tensors": len(parameters),
            "parameters": sum(parameter.numel() for parameter in parameters),
        },
        "gradient_alignment": {
            "mse_raw": relation(main_gradients, mse_gradients),
            "sigreg_raw": relation(main_gradients, sigreg_gradients),
            "full_auxiliary_raw": relation(main_gradients, full_raw),
            "full_auxiliary_active_weighted": relation(
                main_gradients, full_active,
            ),
        },
        "combination": combination.as_dict(),
        "combination_modification": {
            "norm": modification_relation.auxiliary_to_main_norm_ratio
            * math.sqrt(sum(float(value.float().square().sum()) for value in raw_sum)),
            "relative_to_raw_sum": (
                modification_relation.auxiliary_to_main_norm_ratio
            ),
        },
    }
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit MSE/SIGReg/full auxiliary alignment with held-out NTP",
    )
    parser.add_argument("--run-json", type=Path, nargs="+", required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--dataset", choices=tuple(TASKS), default="uspto_mit_synthesis")
    parser.add_argument("--epochs", type=int, nargs="+", default=(1, 2, 4))
    parser.add_argument("--logical-batch-size", type=int, default=16)
    parser.add_argument("--physical-batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=533)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.logical_batch_size % args.physical_batch_size:
        raise ValueError("logical batch must divide into physical chunks")
    torch.manual_seed(args.seed)
    tokenizer = load_reaction_tokenizer(TOKENIZER_DIR)
    chemfm_vocab_size = len(tokenizer)
    predictor_ids = add_predictor_tokens(tokenizer)
    collator = ReactionCollator(tokenizer, task=TASKS[args.dataset])
    rows = read_rows(
        args.dataset, "validation", path=args.validation_manifest.resolve(),
    )[:args.logical_batch_size]
    chunks = [
        collator(rows[start:start + args.physical_batch_size])
        for start in range(0, len(rows), args.physical_batch_size)
    ]
    model = load_lora_model(
        MODEL_DIR,
        tokenizer,
        chemfm_vocab_size=chemfm_vocab_size,
        attn_implementation="sdpa",
    ).cuda().eval()
    method = CLMJEPA(
        predictor_ids,
        tokenizer.eos_token_id,
        tokenizer.pad_token_id,
        sigreg_seed=args.seed,
        optimized_native_logits=True,
    )
    output = {
        "schema_version": 1,
        "evaluation_only": True,
        "model_mode": "eval",
        "seed": args.seed,
        "dataset": args.dataset,
        "validation_manifest": str(args.validation_manifest.resolve()),
        "validation_manifest_sha256": file_sha256(args.validation_manifest.resolve()),
        "logical_batch_size": args.logical_batch_size,
        "physical_batch_size": args.physical_batch_size,
        "sigreg_relative_coefficient": SIGREG_RELATIVE_COEFFICIENT,
        "conditions": {},
    }
    for run_path in args.run_json:
        run = json.loads(run_path.read_text(encoding="utf-8"))
        label = run_path.parent.name
        root = Path(run["selected_checkpoint"]).parent
        interaction = run["config"]["gradient_interaction"]
        actual_lambda = float(run["config"]["actual_lambda"])
        output["conditions"][label] = {
            "gradient_interaction": interaction,
            "lambda_eff": run["config"]["lambda_eff"],
            "actual_active_lambda": actual_lambda,
            "checkpoints": [],
        }
        for epoch in args.epochs:
            checkpoint = root / f"epoch_{epoch}"
            output["conditions"][label]["checkpoints"].append(
                audit_checkpoint(
                    model,
                    method,
                    chunks,
                    checkpoint,
                    interaction=interaction,
                    actual_lambda=actual_lambda,
                )
            )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(output, indent=2) + "\n", encoding="utf-8",
            )
    print(json.dumps({"output": str(args.output), "conditions": len(output["conditions"])}))


if __name__ == "__main__":
    main()
