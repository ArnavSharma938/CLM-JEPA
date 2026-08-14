from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import statistics
import subprocess
import sys
import threading
import time
import types
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from peft import set_peft_model_state_dict
from peft.utils.save_and_load import load_peft_weights
from rdkit import Chem, RDLogger
from scipy.stats import beta, binom, binomtest

from chemfm import (
    END,
    MODEL_DIR,
    PRODUCT_START,
    REACTANT_START,
    TOKENIZER_DIR,
    generate_products_batch,
    load_lora_model,
    load_reaction_tokenizer,
)
from jepa import add_predictor_tokens


RDLogger.DisableLog("rdApp.*")
ADAPTER_NAME = "USPTO-MIT-Synthesis"
BEAM_SIZE = 10
VIEWS = 5
POWER_ALPHA = 0.05
POWER_TARGET = 0.80
MINIMUM_EFFECT = 0.01
PILOT_REACTIONS = 256
PILOT_DISCORDANT = 4


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonicalize_official(smiles: str, *, clear_map: bool = False) -> str:
    """Match ChemFM evaluate.py/score.py canonicalization for one molecule."""
    molecule = Chem.MolFromSmiles(smiles, sanitize=True)
    if molecule is None:
        return ""
    if clear_map:
        for atom in molecule.GetAtoms():
            if atom.HasProp("molAtomMapNumber"):
                atom.ClearProp("molAtomMapNumber")
    try:
        return Chem.MolToSmiles(molecule, isomericSmiles=True)
    except Exception:
        return ""


def canonical_source(smiles: str) -> str:
    components = [canonicalize_official(value, clear_map=True) for value in smiles.split(".")]
    return ".".join(sorted(components)) if components and all(components) else ""


def reaction_identity(source: str, target: str) -> str:
    payload = f"{source}>>{target}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _canonicalize_group(item: tuple[int, list[dict[str, str]]]) -> dict:
    group_index, rows = item
    sources = [canonical_source(row["source"]) for row in rows]
    targets = [canonicalize_official(row["target"], clear_map=True) for row in rows]
    if not all(targets) or len(set(targets)) != 1:
        raise ValueError(f"official group {group_index} does not share one valid canonical target")
    source_consistent = all(sources) and len(set(sources)) == 1
    canonical_reaction_source = sources[0] if source_consistent else ""
    # The official evaluator tokenizes every source even when RDKit cannot
    # sanitize a charged source serialization. Preserve those benchmark rows;
    # the consecutive official group index is the authoritative identity.
    identity_payload = f"official-uspto-mit-test:{group_index}:{targets[0]}"
    return {
        "official_group_index": group_index,
        "reaction_identity": hashlib.sha256(identity_payload.encode("utf-8")).hexdigest(),
        "canonical_source": canonical_reaction_source,
        "source_canonicalization_consistent": source_consistent,
        "canonical_target": targets[0],
        "sources": [row["source"] for row in rows],
        "targets": [row["target"] for row in rows],
        "example_ids": [row.get("example_id", "") for row in rows],
        "source_character_lengths": [len(row["source"]) for row in rows],
    }


def read_official_groups(path: Path, processes: int) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"source", "target"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"official test file must contain {sorted(required)}")
    if len(rows) % VIEWS:
        raise ValueError("official R-SMILES test rows must divide into groups of five")
    items = [(index, rows[index * VIEWS:(index + 1) * VIEWS]) for index in range(len(rows) // VIEWS)]
    if processes <= 1:
        groups = [_canonicalize_group(item) for item in items]
    else:
        import multiprocessing as mp

        with mp.Pool(processes=processes) as pool:
            groups = pool.map(_canonicalize_group, items, chunksize=64)
    identities = [group["reaction_identity"] for group in groups]
    if len(identities) != len(set(identities)):
        raise ValueError("official test group identities are unexpectedly duplicated")
    return groups


def clopper_pearson_interval(successes: int, trials: int, confidence: float = 0.95) -> list[float]:
    tail = (1.0 - confidence) / 2.0
    lower = 0.0 if successes == 0 else float(beta.ppf(tail, successes, trials - successes + 1))
    upper = 1.0 if successes == trials else float(
        beta.ppf(1.0 - tail, successes + 1, trials - successes)
    )
    return [lower, upper]


def exact_mcnemar_power(sample_size: int, discordance: float, effect: float) -> float:
    """Unconditional power for the two-sided exact conditional McNemar test."""
    if not 0.0 < effect <= discordance <= 1.0:
        raise ValueError("effect must be positive and no larger than discordance")
    win_probability = (discordance + effect) / (2.0 * discordance)
    low = max(0, int(binom.ppf(1e-12, sample_size, discordance)))
    high = min(sample_size, int(binom.ppf(1.0 - 1e-12, sample_size, discordance)))
    discordant_counts = np.arange(low, high + 1)
    discordant_mass = binom.pmf(discordant_counts, sample_size, discordance)
    # For D discordant pairs, reject in either exact Binomial(.5) tail.
    upper_threshold = binom.isf(POWER_ALPHA / 2.0, discordant_counts, 0.5).astype(int) + 1
    conditional_power = (
        binom.sf(upper_threshold - 1, discordant_counts, win_probability)
        + binom.cdf(discordant_counts - upper_threshold, discordant_counts, win_probability)
    )
    conditional_power[discordant_counts == 0] = 0.0
    return float(discordant_mass @ conditional_power)


def required_sample_size(discordance: float, effect: float, power: float) -> tuple[int, float]:
    lower, upper = 1, 1
    while exact_mcnemar_power(upper, discordance, effect) < power:
        upper *= 2
    while lower < upper:
        middle = (lower + upper) // 2
        if exact_mcnemar_power(middle, discordance, effect) >= power:
            upper = middle
        else:
            lower = middle + 1
    return lower, exact_mcnemar_power(lower, discordance, effect)


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(path)


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def prepare_manifest(args) -> None:
    groups = read_official_groups(args.official_test, args.processes)
    interval = clopper_pearson_interval(PILOT_DISCORDANT, PILOT_REACTIONS)
    required, achieved = required_sample_size(interval[1], MINIMUM_EFFECT, POWER_TARGET)
    chosen = len(groups) if args.full_test else max(required, args.round_up_to)
    if chosen > len(groups):
        chosen = len(groups)
    generator = random.Random(args.seed)
    selected_indices = sorted(generator.sample(range(len(groups)), chosen))
    selected = [groups[index] for index in selected_indices]
    for panel_index, group in enumerate(selected):
        group["panel_index"] = panel_index
    write_jsonl(args.output, selected)
    selected_set = set(selected_indices)
    if args.equivalence_output is not None:
        candidates = [
            group for index, group in enumerate(groups) if index not in selected_set
        ]
        candidates.sort(key=lambda group: (max(group["source_character_lengths"]), group["official_group_index"]))
        if args.equivalence_groups > len(candidates):
            raise ValueError("equivalence panel exceeds the reactions outside the primary panel")
        positions = np.linspace(0, len(candidates) - 1, args.equivalence_groups, dtype=int)
        equivalence = [dict(candidates[position], panel_index=index) for index, position in enumerate(positions)]
        write_jsonl(args.equivalence_output, equivalence)
    if args.full_manifest_output is not None:
        write_jsonl(
            args.full_manifest_output,
            (dict(group, panel_index=index) for index, group in enumerate(groups)),
        )
    metadata = {
        "created_before_model_inference": True,
        "public_repository_commit": args.repository_commit,
        "official_test": str(args.official_test.resolve()),
        "official_test_sha256": file_sha256(args.official_test),
        "official_augmented_rows": len(groups) * VIEWS,
        "available_unique_reactions": len(groups),
        "views_per_reaction": VIEWS,
        "groups_with_consistent_rdkit_canonical_source": sum(
            group["source_canonicalization_consistent"] for group in groups
        ),
        "groups_with_unsanitizable_or_inconsistent_source_views": sum(
            not group["source_canonicalization_consistent"] for group in groups
        ),
        "selection": "seeded simple random sample without replacement over unique canonical reactions",
        "seed": args.seed,
        "selected_reactions": len(selected),
        "manifest": str(args.output.resolve()),
        "manifest_sha256": file_sha256(args.output),
        "equivalence_manifest": (
            None if args.equivalence_output is None else str(args.equivalence_output.resolve())
        ),
        "equivalence_manifest_sha256": (
            None if args.equivalence_output is None else file_sha256(args.equivalence_output)
        ),
        "equivalence_reactions": (
            0 if args.equivalence_output is None else args.equivalence_groups
        ),
        "full_manifest": (
            None if args.full_manifest_output is None else str(args.full_manifest_output.resolve())
        ),
        "full_manifest_sha256": (
            None if args.full_manifest_output is None else file_sha256(args.full_manifest_output)
        ),
        "power": {
            "primary_endpoint": "paired exact top-1",
            "test": "two-sided exact McNemar",
            "alpha": POWER_ALPHA,
            "target_power": POWER_TARGET,
            "minimum_effect_absolute": MINIMUM_EFFECT,
            "pilot_reactions": PILOT_REACTIONS,
            "pilot_discordant_pairs": PILOT_DISCORDANT,
            "pilot_discordance_point": PILOT_DISCORDANT / PILOT_REACTIONS,
            "pilot_discordance_exact_95_ci": interval,
            "planning_discordance": interval[1],
            "exact_required_reactions": required,
            "power_at_exact_requirement": achieved,
            "power_at_selected_size": exact_mcnemar_power(
                len(selected), interval[1], MINIMUM_EFFECT
            ),
        },
    }
    metadata_path = args.output.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, sort_keys=True))


def official_rank(view_candidates: list[list[str]], n_best: int = BEAM_SIZE) -> tuple[list[str], dict]:
    """Exact score.py reciprocal-rank aggregation, including stable tie order."""
    if len(view_candidates) != VIEWS or any(len(values) != BEAM_SIZE for values in view_candidates):
        raise ValueError("official ranking requires five views with ten beams each")
    scores: dict[str, float] = {}
    invalid_by_beam = [0] * BEAM_SIZE
    unique_by_view = []
    for predictions in view_candidates:
        for beam_index, prediction in enumerate(predictions):
            invalid_by_beam[beam_index] += int(not prediction)
        # score.py uses set followed by the original encounter order.
        unique = list(dict.fromkeys(value for value in predictions if value))
        unique_by_view.append(unique)
        for rank, value in enumerate(unique):
            scores[value] = scores.get(value, 0.0) + 1.0 / (rank + 1.0)
    # Python's stable sort preserves dict insertion order for score ties.
    ranked_items = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:n_best]
    ranked = [item[0] for item in ranked_items]
    ranked += [""] * (n_best - len(ranked))
    return ranked, {
        "scores": {value: score for value, score in ranked_items},
        "invalid_by_beam": invalid_by_beam,
        "unique_valid_per_view": [len(values) for values in unique_by_view],
    }


def load_endpoint(checkpoint: Path):
    import torch

    if os.environ.get("CHEMFM_DISABLE_PYTHON_NATIVE_TRITON") == "1":
        # PyTorch 2.13 can route eager ops through experimental Python/Triton
        # implementations. Keep the newer CUDA/SDPA stack while allowing this
        # independently switchable (and parity-tested) dispatch layer off.
        torch.backends.python_native.triton.enabled = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if os.environ.get("CHEMFM_FORCE_FLASH_SDPA") == "1":
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(False)
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(533)
    tokenizer = load_reaction_tokenizer(TOKENIZER_DIR)
    chemfm_vocab_size = len(tokenizer)
    add_predictor_tokens(tokenizer)
    model = load_lora_model(
        MODEL_DIR, tokenizer, chemfm_vocab_size=chemfm_vocab_size
    ).cuda().eval()
    attention_implementation = os.environ.get("CHEMFM_ATTENTION_IMPLEMENTATION")
    if attention_implementation:
        if hasattr(model, "set_attn_implementation"):
            model.set_attn_implementation(attention_implementation)
        else:
            model.config._attn_implementation = attention_implementation
    nested = checkpoint / ADAPTER_NAME
    weights_path = nested if nested.exists() else checkpoint
    weights = load_peft_weights(str(weights_path), device=str(model.device))
    result = set_peft_model_state_dict(model, weights, adapter_name=ADAPTER_NAME)
    if getattr(result, "unexpected_keys", None):
        raise RuntimeError(f"unexpected checkpoint keys: {result.unexpected_keys}")
    if os.environ.get("CHEMFM_LORA_BF16") == "1":
        # Remove PEFT's decode-time BF16->FP32 casts and FP32 rank-8 matmuls
        # while retaining the unfused adapter execution order. This candidate
        # is accepted only if the ordered beam lists remain exactly identical.
        for name, parameter in model.named_parameters():
            if ".lora_A." in name or ".lora_B." in name:
                parameter.data = parameter.data.to(dtype=torch.bfloat16)
    if os.environ.get("CHEMFM_COMPILE_LORA_KERNEL") == "1":
        import torch.nn.functional as functional
        from peft.tuners.lora.layer import Linear as LoraLinear

        def lora_linear_eval(x, base_weight, base_bias, lora_a, lora_b, scaling):
            # Keep PEFT's exact eager ordering and FP32 adapter arithmetic:
            # base(x) first, then A/B, FP32 add, finally cast to base dtype.
            base = functional.linear(x, base_weight, base_bias)
            adapter_input = x.to(lora_a.dtype)
            adapter = functional.linear(
                functional.linear(adapter_input, lora_a, None), lora_b, None
            ) * scaling
            return (base + adapter).to(base.dtype)

        compiled_lora_linear_eval = torch.compile(
            lora_linear_eval, mode="default", fullgraph=True, dynamic=True
        )

        def compiled_forward(layer, x, *args, **kwargs):
            if args or kwargs or layer.disable_adapters or layer.merged:
                return LoraLinear.forward(layer, x, *args, **kwargs)
            active = [name for name in layer.active_adapters if name in layer.lora_A]
            if len(active) != 1 or active[0] in layer.lora_variant:
                return LoraLinear.forward(layer, x, *args, **kwargs)
            adapter_name = active[0]
            dropout = layer.lora_dropout[adapter_name]
            if layer.training or getattr(dropout, "training", False):
                return LoraLinear.forward(layer, x, *args, **kwargs)
            base_layer = layer.get_base_layer()
            return compiled_lora_linear_eval(
                x,
                base_layer.weight,
                base_layer.bias,
                layer.lora_A[adapter_name].weight,
                layer.lora_B[adapter_name].weight,
                layer.scaling[adapter_name],
            )

        for module in model.modules():
            if isinstance(module, LoraLinear):
                module.forward = types.MethodType(compiled_forward, module)
    if os.environ.get("CHEMFM_LORA_WORKSPACE") == "1":
        from peft.tuners.lora.layer import Linear as LoraLinear

        def workspace_forward(layer, x, *args, **kwargs):
            if args or kwargs or layer.disable_adapters or layer.merged or layer.training:
                return LoraLinear.forward(layer, x, *args, **kwargs)
            active = [name for name in layer.active_adapters if name in layer.lora_A]
            if len(active) != 1 or active[0] in layer.lora_variant:
                return LoraLinear.forward(layer, x, *args, **kwargs)
            adapter_name = active[0]
            dropout = layer.lora_dropout[adapter_name]
            if getattr(dropout, "training", False):
                return LoraLinear.forward(layer, x, *args, **kwargs)
            base_result = layer.base_layer(x)
            # Prefill shapes occur once and gain nothing from workspace setup.
            # Reuse fixed decode buffers only for [beams, 1, hidden] calls.
            if x.ndim != 3 or x.shape[1] != 1:
                torch_dtype = base_result.dtype
                adapter_input = layer._cast_input_dtype(
                    x, layer.lora_A[adapter_name].weight.dtype
                )
                return (
                    base_result
                    + layer.lora_B[adapter_name](
                        layer.lora_A[adapter_name](dropout(adapter_input))
                    ) * layer.scaling[adapter_name]
                ).to(torch_dtype)
            rows = x.shape[0]
            input_width = x.shape[-1]
            output_width = base_result.shape[-1]
            rank = layer.lora_A[adapter_name].weight.shape[0]
            key = (rows, input_width, rank, output_width, x.device)
            workspaces = getattr(layer, "_inference_workspaces", None)
            if workspaces is None:
                workspaces = {}
                layer._inference_workspaces = workspaces
            workspace = workspaces.get(key)
            if workspace is None:
                workspace = (
                    torch.empty((rows, input_width), dtype=torch.float32, device=x.device),
                    torch.empty((rows, rank), dtype=torch.float32, device=x.device),
                    torch.empty((rows, output_width), dtype=torch.float32, device=x.device),
                )
                workspaces[key] = workspace
            cast_input, rank_output, adapter_output = workspace
            cast_input.copy_(x.reshape(rows, input_width))
            torch.mm(
                cast_input,
                layer.lora_A[adapter_name].weight.t(),
                out=rank_output,
            )
            torch.mm(
                rank_output,
                layer.lora_B[adapter_name].weight.t(),
                out=adapter_output,
            )
            adapter_output.mul_(layer.scaling[adapter_name])
            adapter_output.add_(base_result.reshape(rows, output_width))
            base_result.copy_(adapter_output.view_as(base_result))
            return base_result

        for module in model.modules():
            if isinstance(module, LoraLinear):
                module.forward = types.MethodType(workspace_forward, module)
    if os.environ.get("CHEMFM_EXACT_LORA_FASTPATH") == "1":
        from peft.tuners.lora.layer import Linear as LoraLinear

        # Beam-10 decoding repeatedly evaluates [10, 1, hidden] tensors. PEFT's
        # generic inference path allocates fresh FP32 cast, rank, adapter, add,
        # and BF16-cast tensors in every LoRA module at every token. Specialize
        # only that stable decode shape while retaining the identical operation
        # order and dtypes: BF16 base linear; BF16->FP32 cast; A then B FP32
        # matmuls; FP32 scale; FP32 add of the BF16 base result; BF16 cast.
        # Prefill and any unexpected call shape use PEFT unchanged.
        use_cuda_graph = os.environ.get("CHEMFM_EXACT_LORA_CUDAGRAPH") == "1"
        mlp_graph_lora_ids: set[int] = set()
        if os.environ.get("CHEMFM_EXACT_MLP_CUDAGRAPH") == "1":
            from transformers.models.llama.modeling_llama import LlamaMLP

            mlp_graph_lora_ids = {
                id(child)
                for parent in model.modules() if isinstance(parent, LlamaMLP)
                for child in parent.modules() if isinstance(child, LoraLinear)
            }
        qkv_graph_lora_ids: set[int] = set()
        if os.environ.get("CHEMFM_EXACT_QKV_CUDAGRAPH") == "1":
            from transformers.models.llama.modeling_llama import LlamaAttention

            qkv_graph_lora_ids = {
                id(projection)
                for attention in model.modules() if isinstance(attention, LlamaAttention)
                for projection in (attention.q_proj, attention.k_proj, attention.v_proj)
            }
        for module in tuple(model.modules()):
            if not isinstance(module, LoraLinear):
                continue
            active = [name for name in module.active_adapters if name in module.lora_A]
            if len(active) != 1 or active[0] in module.lora_variant:
                continue
            adapter_name = active[0]
            original_forward = module.forward
            base_layer = module.get_base_layer()
            a_weight = module.lora_A[adapter_name].weight
            b_weight = module.lora_B[adapter_name].weight
            a_weight_t = a_weight.t()
            b_weight_t = b_weight.t()
            scaling = module.scaling[adapter_name]
            workspace = [None]
            graph_state = [None]

            def exact_decode_impl(
                x,
                _base=base_layer,
                _a=a_weight,
                _a_t=a_weight_t,
                _b_t=b_weight_t,
                _scale=scaling,
                _workspace=workspace,
            ):
                base_result = _base(x)
                rows, input_width = x.shape[0], x.shape[-1]
                rank = _a.shape[0]
                output_width = base_result.shape[-1]
                if _workspace[0] is None:
                    _workspace[0] = (
                        torch.empty((rows, input_width), dtype=_a.dtype, device=x.device),
                        torch.empty((rows, rank), dtype=_a.dtype, device=x.device),
                        torch.empty((rows, output_width), dtype=_a.dtype, device=x.device),
                    )
                cast_input, rank_output, adapter_output = _workspace[0]
                cast_input.copy_(x.reshape(rows, input_width))
                torch.mm(cast_input, _a_t, out=rank_output)
                torch.mm(rank_output, _b_t, out=adapter_output)
                adapter_output.mul_(_scale)
                adapter_output.add_(base_result.reshape(rows, output_width))
                base_result.copy_(adapter_output.view_as(base_result))
                return base_result

            def exact_decode_forward(
                layer, x, *args,
                _original=original_forward,
                _base=base_layer,
                _a=a_weight,
                _b=b_weight,
                _a_t=a_weight_t,
                _b_t=b_weight_t,
                _scale=scaling,
                _workspace=workspace,
                _impl=exact_decode_impl,
                _graph_state=graph_state,
                _use_cuda_graph=(
                    use_cuda_graph
                    and id(module) not in mlp_graph_lora_ids
                    and id(module) not in qkv_graph_lora_ids
                ),
                **kwargs,
            ):
                if (
                    args or kwargs or layer.training or layer.disable_adapters or layer.merged
                    or x.ndim != 3 or x.shape[0] != BEAM_SIZE or x.shape[1] != 1
                ):
                    return _original(x, *args, **kwargs)
                if not _use_cuda_graph:
                    return _impl(x)
                if _graph_state[0] is None:
                    static_input = torch.empty_like(x)
                    static_input.copy_(x)
                    current_stream = torch.cuda.current_stream(x.device)
                    warmup_stream = torch.cuda.Stream(device=x.device)
                    warmup_stream.wait_stream(current_stream)
                    with torch.cuda.stream(warmup_stream):
                        for _ in range(3):
                            _impl(static_input)
                    current_stream.wait_stream(warmup_stream)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=warmup_stream):
                        static_output = _impl(static_input)
                    _graph_state[0] = (static_input, static_output, graph)
                static_input, static_output, graph = _graph_state[0]
                static_input.copy_(x)
                graph.replay()
                return static_output

            module.forward = types.MethodType(exact_decode_forward, module)
    if os.environ.get("CHEMFM_EXACT_QKV_CUDAGRAPH") == "1":
        from transformers.models.llama.modeling_llama import LlamaAttention

        for attention in tuple(model.modules()):
            if not isinstance(attention, LlamaAttention):
                continue
            q_projection = attention.q_proj
            k_projection = attention.k_proj
            v_projection = attention.v_proj
            original_q = q_projection.forward
            original_k = k_projection.forward
            original_v = v_projection.forward
            graph_state = [None]
            pending = [None]

            def graphed_q_forward(
                layer, x, *args,
                _q=original_q,
                _k=original_k,
                _v=original_v,
                _graph_state=graph_state,
                _pending=pending,
                **kwargs,
            ):
                if args or kwargs or x.ndim != 3 or x.shape[0] != BEAM_SIZE or x.shape[1] != 1:
                    _pending[0] = None
                    return _q(x, *args, **kwargs)
                if _graph_state[0] is None:
                    static_input = torch.empty_like(x)
                    static_input.copy_(x)
                    current_stream = torch.cuda.current_stream(x.device)
                    warmup_stream = torch.cuda.Stream(device=x.device)
                    warmup_stream.wait_stream(current_stream)
                    with torch.cuda.stream(warmup_stream):
                        for _ in range(3):
                            _q(static_input), _k(static_input), _v(static_input)
                    current_stream.wait_stream(warmup_stream)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=warmup_stream):
                        static_q = _q(static_input)
                        static_k = _k(static_input)
                        static_v = _v(static_input)
                    _graph_state[0] = (static_input, static_q, static_k, static_v, graph)
                static_input, static_q, static_k, static_v, graph = _graph_state[0]
                static_input.copy_(x)
                graph.replay()
                _pending[0] = (static_k, static_v)
                return static_q

            def graphed_k_forward(layer, x, *args, _original=original_k, _pending=pending, **kwargs):
                if not args and not kwargs and _pending[0] is not None and x.shape[0] == BEAM_SIZE and x.shape[1] == 1:
                    return _pending[0][0]
                return _original(x, *args, **kwargs)

            def graphed_v_forward(layer, x, *args, _original=original_v, _pending=pending, **kwargs):
                if not args and not kwargs and _pending[0] is not None and x.shape[0] == BEAM_SIZE and x.shape[1] == 1:
                    value = _pending[0][1]
                    _pending[0] = None
                    return value
                return _original(x, *args, **kwargs)

            q_projection.forward = types.MethodType(graphed_q_forward, q_projection)
            k_projection.forward = types.MethodType(graphed_k_forward, k_projection)
            v_projection.forward = types.MethodType(graphed_v_forward, v_projection)
    if os.environ.get("CHEMFM_EXACT_RMSNORM_FASTPATH") == "1":
        from transformers.models.llama.modeling_llama import LlamaRMSNorm

        for module in tuple(model.modules()):
            if not isinstance(module, LlamaRMSNorm):
                continue
            original_forward = module.forward
            workspace = [None]

            def exact_rmsnorm_decode_forward(
                layer, hidden_states,
                _original=original_forward,
                _workspace=workspace,
            ):
                if (
                    layer.training or hidden_states.ndim != 3
                    or hidden_states.shape[0] != BEAM_SIZE or hidden_states.shape[1] != 1
                ):
                    return _original(hidden_states)
                if _workspace[0] is None:
                    shape = hidden_states.shape
                    reduced_shape = (*shape[:-1], 1)
                    _workspace[0] = (
                        torch.empty(shape, dtype=torch.float32, device=hidden_states.device),
                        torch.empty(shape, dtype=torch.float32, device=hidden_states.device),
                        torch.empty(reduced_shape, dtype=torch.float32, device=hidden_states.device),
                        torch.empty(reduced_shape, dtype=torch.float32, device=hidden_states.device),
                        torch.empty(shape, dtype=hidden_states.dtype, device=hidden_states.device),
                        torch.empty(shape, dtype=hidden_states.dtype, device=hidden_states.device),
                    )
                fp32_values, square_or_normalized, variance, inverse, bf16_values, output = _workspace[0]
                fp32_values.copy_(hidden_states)
                torch.pow(fp32_values, 2, out=square_or_normalized)
                torch.mean(square_or_normalized, dim=-1, keepdim=True, out=variance)
                torch.add(variance, layer.variance_epsilon, out=inverse)
                torch.rsqrt(inverse, out=inverse)
                torch.mul(fp32_values, inverse, out=square_or_normalized)
                bf16_values.copy_(square_or_normalized)
                torch.mul(layer.weight, bf16_values, out=output)
                return output

            module.forward = types.MethodType(exact_rmsnorm_decode_forward, module)
    if os.environ.get("CHEMFM_EXACT_RMSNORM_CUDAGRAPH") == "1":
        from transformers.models.llama.modeling_llama import LlamaRMSNorm

        for module in tuple(model.modules()):
            if not isinstance(module, LlamaRMSNorm):
                continue
            original_forward = module.forward
            graph_state = [None]

            def graphed_rmsnorm_decode_forward(
                layer, hidden_states,
                _original=original_forward,
                _graph_state=graph_state,
            ):
                if (
                    layer.training or hidden_states.ndim != 3
                    or hidden_states.shape[0] != BEAM_SIZE or hidden_states.shape[1] != 1
                ):
                    return _original(hidden_states)
                if _graph_state[0] is None:
                    static_input = torch.empty_like(hidden_states)
                    static_input.copy_(hidden_states)
                    current_stream = torch.cuda.current_stream(hidden_states.device)
                    warmup_stream = torch.cuda.Stream(device=hidden_states.device)
                    warmup_stream.wait_stream(current_stream)
                    with torch.cuda.stream(warmup_stream):
                        for _ in range(3):
                            _original(static_input)
                    current_stream.wait_stream(warmup_stream)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=warmup_stream):
                        static_output = _original(static_input)
                    _graph_state[0] = (static_input, static_output, graph)
                static_input, static_output, graph = _graph_state[0]
                static_input.copy_(hidden_states)
                graph.replay()
                return static_output

            module.forward = types.MethodType(graphed_rmsnorm_decode_forward, module)
    if os.environ.get("CHEMFM_EXACT_ROPE_CUDAGRAPH") == "1":
        from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

        for module in tuple(model.modules()):
            if not isinstance(module, LlamaRotaryEmbedding):
                continue
            original_forward = module.forward
            graph_state = [None]

            def graphed_rope_decode_forward(
                layer, x, position_ids,
                _original=original_forward,
                _graph_state=graph_state,
            ):
                if (
                    layer.training or x.ndim != 3 or x.shape[0] != BEAM_SIZE or x.shape[1] != 1
                    or position_ids.shape[-1] != 1
                ):
                    return _original(x, position_ids)
                if _graph_state[0] is None:
                    # RoPE uses x only for dtype/device, not its values.
                    static_x = torch.empty_like(x)
                    static_positions = torch.empty_like(position_ids)
                    static_positions.copy_(position_ids)
                    current_stream = torch.cuda.current_stream(x.device)
                    warmup_stream = torch.cuda.Stream(device=x.device)
                    warmup_stream.wait_stream(current_stream)
                    with torch.cuda.stream(warmup_stream):
                        for _ in range(3):
                            _original(static_x, static_positions)
                    current_stream.wait_stream(warmup_stream)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=warmup_stream):
                        static_cos, static_sin = _original(static_x, static_positions)
                    _graph_state[0] = (static_positions, static_cos, static_sin, graph)
                static_positions, static_cos, static_sin, graph = _graph_state[0]
                static_positions.copy_(position_ids)
                graph.replay()
                return static_cos, static_sin

            module.forward = types.MethodType(graphed_rope_decode_forward, module)
    if os.environ.get("CHEMFM_EXACT_APPLY_ROPE_CUDAGRAPH") == "1":
        import transformers.models.llama.modeling_llama as llama_modeling

        original_apply_rope = getattr(
            llama_modeling,
            "_chemfm_original_apply_rotary_pos_emb",
            llama_modeling.apply_rotary_pos_emb,
        )
        llama_modeling._chemfm_original_apply_rotary_pos_emb = original_apply_rope
        graph_state = [None]

        def graphed_apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
            if (
                position_ids is not None or unsqueeze_dim != 1
                or q.shape[0] != BEAM_SIZE or q.shape[-2] != 1
                or k.shape[0] != BEAM_SIZE or k.shape[-2] != 1
            ):
                return original_apply_rope(q, k, cos, sin, position_ids, unsqueeze_dim)
            if graph_state[0] is None:
                static_q = torch.empty_like(q)
                static_k = torch.empty_like(k)
                static_cos = torch.empty_like(cos)
                static_sin = torch.empty_like(sin)
                static_q.copy_(q)
                static_k.copy_(k)
                static_cos.copy_(cos)
                static_sin.copy_(sin)
                current_stream = torch.cuda.current_stream(q.device)
                warmup_stream = torch.cuda.Stream(device=q.device)
                warmup_stream.wait_stream(current_stream)
                with torch.cuda.stream(warmup_stream):
                    for _ in range(3):
                        original_apply_rope(static_q, static_k, static_cos, static_sin)
                current_stream.wait_stream(warmup_stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=warmup_stream):
                    static_q_embed, static_k_embed = original_apply_rope(
                        static_q, static_k, static_cos, static_sin
                    )
                graph_state[0] = (
                    static_q, static_k, static_cos, static_sin,
                    static_q_embed, static_k_embed, graph,
                )
            (
                static_q, static_k, static_cos, static_sin,
                static_q_embed, static_k_embed, graph,
            ) = graph_state[0]
            static_q.copy_(q)
            static_k.copy_(k)
            static_cos.copy_(cos)
            static_sin.copy_(sin)
            graph.replay()
            return static_q_embed, static_k_embed

        llama_modeling.apply_rotary_pos_emb = graphed_apply_rotary_pos_emb
    if os.environ.get("CHEMFM_EXACT_MLP_FASTPATH") == "1":
        import torch.nn.functional as functional
        from transformers.models.llama.modeling_llama import LlamaMLP

        for module in tuple(model.modules()):
            if not isinstance(module, LlamaMLP):
                continue
            original_forward = module.forward

            def exact_mlp_decode_forward(layer, x, _original=original_forward):
                if layer.training or x.ndim != 3 or x.shape[0] != BEAM_SIZE or x.shape[1] != 1:
                    return _original(x)
                gate = layer.gate_proj(x)
                functional.silu(gate, inplace=True)
                gate.mul_(layer.up_proj(x))
                return layer.down_proj(gate)

            module.forward = types.MethodType(exact_mlp_decode_forward, module)
    if os.environ.get("CHEMFM_EXACT_MLP_CUDAGRAPH") == "1":
        from transformers.models.llama.modeling_llama import LlamaMLP

        for module in tuple(model.modules()):
            if not isinstance(module, LlamaMLP):
                continue
            original_forward = module.forward
            graph_state = [None]

            def graphed_mlp_decode_forward(
                layer, x,
                _original=original_forward,
                _graph_state=graph_state,
            ):
                if layer.training or x.ndim != 3 or x.shape[0] != BEAM_SIZE or x.shape[1] != 1:
                    return _original(x)
                if _graph_state[0] is None:
                    static_input = torch.empty_like(x)
                    static_input.copy_(x)
                    current_stream = torch.cuda.current_stream(x.device)
                    warmup_stream = torch.cuda.Stream(device=x.device)
                    warmup_stream.wait_stream(current_stream)
                    with torch.cuda.stream(warmup_stream):
                        for _ in range(3):
                            _original(static_input)
                    current_stream.wait_stream(warmup_stream)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=warmup_stream):
                        static_output = _original(static_input)
                    _graph_state[0] = (static_input, static_output, graph)
                static_input, static_output, graph = _graph_state[0]
                static_input.copy_(x)
                graph.replay()
                return static_output

            module.forward = types.MethodType(graphed_mlp_decode_forward, module)
    if os.environ.get("CHEMFM_MERGE_LORA") == "1":
        # PEFT's supported inference path: fold the frozen low-rank deltas and
        # modules_to_save into the base model to remove adapter dispatch,
        # per-layer dtype casts, and the extra LoRA matmuls during decoding.
        model = model.merge_and_unload(safe_merge=True).eval()
    compile_mode = os.environ.get("CHEMFM_TORCH_COMPILE")
    if compile_mode:
        # PeftModel.generate delegates to the underlying CausalLM's generate,
        # so compiling PeftModel.forward does not touch decoding. Compile the
        # actual CausalLM forward used by GenerationMixin.
        compile_target = model.get_base_model()
        compile_target.forward = torch.compile(
            compile_target.forward,
            mode=compile_mode,
            fullgraph=False,
            dynamic=True,
        )
    return model, tokenizer


def prompts_for_group(group: dict) -> list[str]:
    if len(group["sources"]) != VIEWS:
        raise ValueError("each manifest reaction must contain exactly five sources")
    return [f"{REACTANT_START}{source}{END}{PRODUCT_START}" for source in group["sources"]]


def _evaluate_assigned_groups(
    model, tokenizer, groups: list[dict], prompt_batch_size: int, batch_mode: str
) -> list[dict]:
    if not groups:
        return []
    prompts = []
    locations = []
    per_group: list[list[list[str] | None]] = [[None] * VIEWS for _ in groups]
    for group_index, group in enumerate(groups):
        for view_index, prompt in enumerate(prompts_for_group(group)):
            prompts.append(prompt)
            locations.append((group_index, view_index))
    if prompt_batch_size == 1:
        batches = [[index] for index in range(len(prompts))]
    else:
        lengths = [
            len(values)
            for values in tokenizer(prompts, add_special_tokens=False, truncation=True)["input_ids"]
        ]
        if batch_mode == "equal-length":
            buckets: dict[int, list[int]] = {}
            for index, length in enumerate(lengths):
                buckets.setdefault(length, []).append(index)
            batches = [
                indices[start:start + prompt_batch_size]
                for _, indices in sorted(buckets.items())
                for start in range(0, len(indices), prompt_batch_size)
            ]
        elif batch_mode == "left-pad":
            order = sorted(range(len(prompts)), key=lambda index: (lengths[index], index))
            batches = [order[start:start + prompt_batch_size] for start in range(0, len(order), prompt_batch_size)]
        else:
            raise ValueError("batch mode must be equal-length or left-pad")
    for indices in batches:
        outputs = generate_products_batch(
            model,
            tokenizer,
            [prompts[index] for index in indices],
            max_length=1024,
            num_beams=BEAM_SIZE,
            num_return_sequences=BEAM_SIZE,
            pad_unequal_prompts=(len(indices) > 1 and batch_mode == "left-pad"),
        )
        for prompt_index, candidates in zip(indices, outputs):
            group_index, view_index = locations[prompt_index]
            per_group[group_index][view_index] = candidates
    records = []
    for group, raw_views in zip(groups, per_group):
        if any(values is None for values in raw_views):
            raise RuntimeError("generation did not fill every R-SMILES view")
        raw_views = [list(values) for values in raw_views]
        canonical_views = [
            [canonicalize_official(value) for value in values]
            for values in raw_views
        ]
        ranked, rank_details = official_rank(canonical_views)
        target = group["canonical_target"]
        records.append({
            "panel_index": group["panel_index"],
            "official_group_index": group["official_group_index"],
            "reaction_identity": group["reaction_identity"],
            "sources": group["sources"],
            "target": target,
            "raw_candidates_by_view": raw_views,
            "canonical_candidates_by_view": canonical_views,
            "ranked_candidates": ranked,
            "rank_scores": rank_details["scores"],
            "invalid_by_beam": rank_details["invalid_by_beam"],
            "unique_valid_per_view": rank_details["unique_valid_per_view"],
            "exact": [bool(value) and value == target for value in ranked],
        })
    return records


def worker(args) -> None:
    import torch

    manifest = read_jsonl(args.manifest)
    assigned = [
        group for position, group in enumerate(manifest)
        if position % args.workers == args.worker_index
    ]
    existing = read_jsonl(args.output)
    completed = {row["reaction_identity"] for row in existing}
    assigned = [group for group in assigned if group["reaction_identity"] not in completed]
    torch.cuda.reset_peak_memory_stats()
    load_started = time.perf_counter()
    model, tokenizer = load_endpoint(args.checkpoint)
    load_seconds = time.perf_counter() - load_started
    torch.cuda.synchronize()
    evaluation_started = time.perf_counter()
    if args.threads_per_worker == 1 or len(assigned) < 2:
        new_records = []
        # Persist short, atomic chunks so multi-hour official beam evaluation
        # is resumable without changing model calls, ordering, or ranking.
        chunk_size = 4
        for start in range(0, len(assigned), chunk_size):
            new_records.extend(_evaluate_assigned_groups(
                model,
                tokenizer,
                assigned[start:start + chunk_size],
                args.prompt_batch_size,
                args.batch_mode,
            ))
            progress = existing + new_records
            progress.sort(key=lambda row: row["panel_index"])
            write_jsonl(args.output, progress)
            args.output.with_suffix(".progress.json").write_text(
                json.dumps({
                    "completed_reactions": len(progress),
                    "assigned_reactions": len(existing) + len(assigned),
                    "updated_at_unix": time.time(),
                }, indent=2) + "\n",
                encoding="utf-8",
            )
    else:
        thread_count = min(args.threads_per_worker, len(assigned))
        thread_state = threading.local()

        def evaluate_partition(partition):
            if not hasattr(thread_state, "stream"):
                thread_state.stream = torch.cuda.Stream()
            with torch.cuda.stream(thread_state.stream):
                records = _evaluate_assigned_groups(
                    model, tokenizer, partition, args.prompt_batch_size, args.batch_mode
                )
            thread_state.stream.synchronize()
            return records

        new_records = []
        with ThreadPoolExecutor(max_workers=thread_count) as executor:
            for start in range(0, len(assigned), thread_count):
                wave = [[group] for group in assigned[start:start + thread_count]]
                for records in executor.map(evaluate_partition, wave):
                    new_records.extend(records)
                if len(new_records) % 24 < thread_count:
                    progress = existing + new_records
                    progress.sort(key=lambda row: row["panel_index"])
                    write_jsonl(args.output, progress)
                    args.output.with_suffix(".progress.json").write_text(
                        json.dumps({
                            "completed_reactions": len(progress),
                            "assigned_reactions": len(existing) + len(assigned),
                            "updated_at_unix": time.time(),
                        }, indent=2) + "\n",
                        encoding="utf-8",
                    )
    torch.cuda.synchronize()
    evaluation_seconds = time.perf_counter() - evaluation_started
    merged = existing + new_records
    merged.sort(key=lambda row: row["panel_index"])
    write_jsonl(args.output, merged)
    statistics_path = args.output.with_suffix(".stats.json")
    statistics_path.write_text(json.dumps({
        "worker_index": args.worker_index,
        "workers": args.workers,
        "prompt_batch_size": args.prompt_batch_size,
        "threads_per_worker": args.threads_per_worker,
        "batch_mode": args.batch_mode,
        "assigned_reactions": len(merged),
        "new_reactions": len(new_records),
        "load_seconds": load_seconds,
        "evaluation_seconds": evaluation_seconds,
        "reactions_per_second": len(new_records) / max(evaluation_seconds, 1e-12),
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()),
        "attention_implementation": getattr(model.config, "_attn_implementation", None),
        "sdpa_flash_enabled": bool(torch.backends.cuda.flash_sdp_enabled()),
        "sdpa_mem_efficient_enabled": bool(torch.backends.cuda.mem_efficient_sdp_enabled()),
        "cache_implementation": os.environ.get("CHEMFM_CACHE_IMPLEMENTATION"),
        "torch_compile_mode": os.environ.get("CHEMFM_TORCH_COMPILE"),
        "lora_merged": os.environ.get("CHEMFM_MERGE_LORA") == "1",
        "python_native_triton_disabled": (
            os.environ.get("CHEMFM_DISABLE_PYTHON_NATIVE_TRITON") == "1"
        ),
        "lora_bf16": os.environ.get("CHEMFM_LORA_BF16") == "1",
        "force_flash_sdpa": os.environ.get("CHEMFM_FORCE_FLASH_SDPA") == "1",
        "compiled_lora_kernel": os.environ.get("CHEMFM_COMPILE_LORA_KERNEL") == "1",
        "lora_workspace": os.environ.get("CHEMFM_LORA_WORKSPACE") == "1",
        "exact_lora_fastpath": os.environ.get("CHEMFM_EXACT_LORA_FASTPATH") == "1",
        "exact_lora_cudagraph": os.environ.get("CHEMFM_EXACT_LORA_CUDAGRAPH") == "1",
        "exact_rmsnorm_cudagraph": os.environ.get("CHEMFM_EXACT_RMSNORM_CUDAGRAPH") == "1",
        "exact_rope_cudagraph": os.environ.get("CHEMFM_EXACT_ROPE_CUDAGRAPH") == "1",
        "exact_apply_rope_cudagraph": (
            os.environ.get("CHEMFM_EXACT_APPLY_ROPE_CUDAGRAPH") == "1"
        ),
        "gpu_sample_interval_seconds": 10.0,
    }, indent=2) + "\n", encoding="utf-8")


@dataclass
class GPUSample:
    timestamp: float
    utilization_percent: float
    memory_mib: float
    power_watts: float


def _gpu_sampler(stop: threading.Event, samples: list[GPUSample]) -> None:
    # A one-second loop spent measurable CPU time spawning `nvidia-smi` while
    # four Python generation workers were launch-bound on a six-vCPU host.
    # Ten-second telemetry is still dense for multi-hour endpoint runs and
    # removes 90% of those monitoring subprocesses from the hot evaluation.
    while not stop.wait(10.0):
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used,power.draw",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            values = [float(value.strip()) for value in result.stdout.splitlines()[0].split(",")]
            samples.append(GPUSample(time.time(), *values))
        except Exception:
            pass


def launch_configuration(
    *, checkpoint: Path, manifest: Path, output_dir: Path,
    workers: int, prompt_batch_size: int, batch_mode: str,
    threads_per_worker: int = 1,
) -> dict:
    groups = read_jsonl(manifest)
    workers = min(workers, len(groups))
    output_dir.mkdir(parents=True, exist_ok=True)
    commands = []
    for worker_index in range(workers):
        output = output_dir / f"worker_{worker_index:02d}.jsonl"
        commands.append([
            sys.executable,
            str(Path(__file__).resolve()),
            "worker",
            "--checkpoint", str(checkpoint),
            "--manifest", str(manifest),
            "--workers", str(workers),
            "--worker-index", str(worker_index),
            "--prompt-batch-size", str(prompt_batch_size),
            "--batch-mode", batch_mode,
            "--threads-per-worker", str(threads_per_worker),
            "--output", str(output),
        ])
    samples: list[GPUSample] = []
    stop = threading.Event()
    monitor = threading.Thread(target=_gpu_sampler, args=(stop, samples), daemon=True)
    monitor.start()
    started = time.perf_counter()
    processes = []
    logs = []
    for worker_index, command in enumerate(commands):
        log = (output_dir / f"worker_{worker_index:02d}.log").open("w", encoding="utf-8")
        logs.append(log)
        processes.append(subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT))
    returncodes = [process.wait() for process in processes]
    wall_seconds = time.perf_counter() - started
    stop.set()
    monitor.join(timeout=2)
    for log in logs:
        log.close()
    if any(returncodes):
        raise RuntimeError(f"evaluation workers failed: {returncodes}")
    records = [
        row
        for worker_index in range(workers)
        for row in read_jsonl(output_dir / f"worker_{worker_index:02d}.jsonl")
    ]
    by_identity = {row["reaction_identity"]: row for row in records}
    expected = [group["reaction_identity"] for group in groups]
    if len(by_identity) != len(records) or set(by_identity) != set(expected):
        raise RuntimeError("worker outputs do not exactly cover the manifest")
    ordered = [by_identity[identity] for identity in expected]
    merged_path = output_dir / "predictions.jsonl"
    write_jsonl(merged_path, ordered)
    worker_stats = [
        json.loads((output_dir / f"worker_{worker_index:02d}.stats.json").read_text(encoding="utf-8"))
        for worker_index in range(workers)
    ]
    active_seconds = max((value["evaluation_seconds"] for value in worker_stats), default=0.0)
    summary = {
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_adapter_sha256": file_sha256(checkpoint / ADAPTER_NAME / "adapter_model.safetensors"),
        "manifest": str(manifest.resolve()),
        "manifest_sha256": file_sha256(manifest),
        "reactions": len(groups),
        "views_per_reaction": VIEWS,
        "beam_size": BEAM_SIZE,
        "returned_candidates_per_view": BEAM_SIZE,
        "workers": workers,
        "prompt_batch_size": prompt_batch_size,
        "batch_mode": batch_mode,
        "threads_per_worker": threads_per_worker,
        "wall_seconds_including_model_load": wall_seconds,
        "end_to_end_reactions_per_second": len(groups) / max(wall_seconds, 1e-12),
        "active_evaluation_seconds": active_seconds,
        "steady_state_reactions_per_second": len(groups) / max(active_seconds, 1e-12),
        "worker_statistics": worker_stats,
        "gpu": {
            "samples": len(samples),
            "mean_utilization_percent": statistics.fmean(value.utilization_percent for value in samples) if samples else None,
            "max_utilization_percent": max((value.utilization_percent for value in samples), default=None),
            "mean_memory_mib": statistics.fmean(value.memory_mib for value in samples) if samples else None,
            "peak_memory_mib": max((value.memory_mib for value in samples), default=None),
            "mean_power_watts": statistics.fmean(value.power_watts for value in samples) if samples else None,
        },
        "predictions": str(merged_path.resolve()),
        "predictions_sha256": file_sha256(merged_path),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def run_configuration(args) -> None:
    summary = launch_configuration(
        checkpoint=args.checkpoint,
        manifest=args.manifest,
        output_dir=args.output_dir,
        workers=args.workers,
        prompt_batch_size=args.prompt_batch_size,
        batch_mode=args.batch_mode,
        threads_per_worker=args.threads_per_worker,
    )
    print(json.dumps(summary, sort_keys=True))


def parity(reference: list[dict], candidate: list[dict]) -> dict:
    if [row["reaction_identity"] for row in reference] != [row["reaction_identity"] for row in candidate]:
        return {"identities": False, "raw_candidates": False, "canonical_candidates": False, "ranked_candidates": False, "exact_flags": False}
    fields = {
        "raw_candidates": "raw_candidates_by_view",
        "canonical_candidates": "canonical_candidates_by_view",
        "ranked_candidates": "ranked_candidates",
        "exact_flags": "exact",
    }
    return {
        "identities": True,
        **{
            label: all(left[field] == right[field] for left, right in zip(reference, candidate))
            for label, field in fields.items()
        },
    }


def benchmark(args) -> None:
    configurations = []
    for value in args.configurations.split(","):
        pieces = value.split("x", 3)
        workers, batch_size, mode = pieces[:3]
        threads = int(pieces[3]) if len(pieces) == 4 else 1
        configurations.append((int(workers), int(batch_size), mode, threads))
    if (1, 1, "left-pad", 1) not in configurations:
        configurations.insert(0, (1, 1, "left-pad", 1))
    results = []
    reference_records = None
    reference_summary = None
    for workers, batch_size, mode, threads in configurations:
        label = f"w{workers}_b{batch_size}_{mode}_t{threads}"
        try:
            summary = launch_configuration(
                checkpoint=args.checkpoint,
                manifest=args.manifest,
                output_dir=args.output_dir / label,
                workers=workers,
                prompt_batch_size=batch_size,
                batch_mode=mode,
                threads_per_worker=threads,
            )
        except Exception as error:
            results.append({
                "label": label,
                "summary": None,
                "parity": None,
                "exact_parity": False,
                "failure": f"{type(error).__name__}: {error}",
            })
            continue
        records = read_jsonl(Path(summary["predictions"]))
        if workers == 1 and batch_size == 1 and threads == 1:
            reference_records = records
            reference_summary = summary
            evidence = {"identities": True, "raw_candidates": True, "canonical_candidates": True, "ranked_candidates": True, "exact_flags": True}
        else:
            if reference_records is None:
                raise RuntimeError("sequential batch-1 reference must run first")
            evidence = parity(reference_records, records)
        exact = all(evidence.values())
        results.append({"label": label, "summary": summary, "parity": evidence, "exact_parity": exact})
    exact_results = [result for result in results if result["exact_parity"]]
    winner = max(exact_results, key=lambda result: result["summary"]["steady_state_reactions_per_second"])
    output = {
        "reference": reference_summary,
        "results": results,
        "winning_configuration": winner["label"],
        "winning_summary": winner["summary"],
        "steady_state_speedup": (
            winner["summary"]["steady_state_reactions_per_second"]
            / reference_summary["steady_state_reactions_per_second"]
        ),
        "selection_rule": "highest complete-five-view reactions/s among configurations with exact equality of every raw, canonical, ranked candidate list and exact flag",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "benchmark.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "winner": output["winning_configuration"],
        "speedup": output["steady_state_speedup"],
        "exact_configurations": [result["label"] for result in exact_results],
    }, sort_keys=True))


def topk_correct(row: dict, cutoff: int) -> bool:
    return row["target"] in row["ranked_candidates"][:cutoff]


def paired_endpoint(left: list[dict], right: list[dict], cutoff: int) -> dict:
    left_correct = np.asarray([topk_correct(row, cutoff) for row in left], dtype=bool)
    right_correct = np.asarray([topk_correct(row, cutoff) for row in right], dtype=bool)
    both = int(np.sum(left_correct & right_correct))
    left_only = int(np.sum(left_correct & ~right_correct))
    right_only = int(np.sum(~left_correct & right_correct))
    neither = int(np.sum(~left_correct & ~right_correct))
    discordant = left_only + right_only
    p_value = 1.0 if discordant == 0 else float(
        binomtest(right_only, discordant, 0.5, alternative="two-sided").pvalue
    )
    return {
        "native_accuracy": float(left_correct.mean()),
        "clm_jepa_accuracy": float(right_correct.mean()),
        "absolute_difference": float(right_correct.mean() - left_correct.mean()),
        "both_correct": both,
        "native_only_correct": left_only,
        "clm_jepa_only_correct": right_only,
        "neither_correct": neither,
        "discordant": discordant,
        "exact_mcnemar_two_sided_p": p_value,
    }


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values, key=p_values.get)
    adjusted = {}
    running = 0.0
    count = len(ordered)
    for rank, name in enumerate(ordered):
        value = min(1.0, (count - rank) * p_values[name])
        running = max(running, value)
        adjusted[name] = running
    return adjusted


def summarize(args) -> None:
    manifest = read_jsonl(args.manifest)
    native = read_jsonl(args.native_predictions)
    clm = read_jsonl(args.clm_predictions)
    expected = [row["reaction_identity"] for row in manifest]
    if [row["reaction_identity"] for row in native] != expected or [row["reaction_identity"] for row in clm] != expected:
        raise ValueError("both prediction files must exactly follow the frozen manifest")
    primary = paired_endpoint(native, clm, 1)
    differences = np.asarray([
        int(topk_correct(right, 1)) - int(topk_correct(left, 1))
        for left, right in zip(native, clm)
    ], dtype=np.int8)
    rng = np.random.default_rng(args.seed)
    bootstrap = np.empty(args.bootstrap_repetitions, dtype=np.float64)
    for start in range(0, args.bootstrap_repetitions, 1000):
        size = min(1000, args.bootstrap_repetitions - start)
        indices = rng.integers(0, len(differences), size=(size, len(differences)))
        bootstrap[start:start + size] = differences[indices].mean(axis=1)
    primary["paired_bootstrap_95_ci"] = [
        float(np.quantile(bootstrap, 0.025)),
        float(np.quantile(bootstrap, 0.975)),
    ]
    secondary = {f"top{cutoff}": paired_endpoint(native, clm, cutoff) for cutoff in (3, 5, 10)}
    adjusted = holm_adjust({name: value["exact_mcnemar_two_sided_p"] for name, value in secondary.items()})
    for name in secondary:
        secondary[name]["holm_adjusted_p_across_secondary_cutoffs"] = adjusted[name]
    def validity(rows: list[dict]) -> dict:
        view_slots = len(rows) * VIEWS * BEAM_SIZE
        ranked_slots = len(rows) * BEAM_SIZE
        return {
            "official_view_candidate_valid_rate": sum(
                bool(value)
                for row in rows
                for view in row["canonical_candidates_by_view"]
                for value in view
            ) / view_slots,
            "aggregated_ranked_valid_rate": sum(
                bool(value) for row in rows for value in row["ranked_candidates"]
            ) / ranked_slots,
            "mean_unique_valid_per_view": float(np.mean([
                value for row in rows for value in row["unique_valid_per_view"]
            ])),
        }
    output = {
        "protocol": {
            "manifest": str(args.manifest.resolve()),
            "manifest_sha256": file_sha256(args.manifest),
            "reactions": len(manifest),
            "views_per_reaction": VIEWS,
            "beam_size": BEAM_SIZE,
            "candidates_per_view": BEAM_SIZE,
            "ranking": "official ChemFM score.py reciprocal rank across five R-SMILES views, alpha=1",
            "primary": "paired exact top-1",
            "teacher_forced_ce": "not recomputed; kept separate from this behavioral endpoint assay",
        },
        "primary_top1": primary,
        "secondary": secondary,
        "validity": {"native": validity(native), "clm_jepa": validity(clm)},
        "artifacts": {
            "native_predictions": str(args.native_predictions.resolve()),
            "native_predictions_sha256": file_sha256(args.native_predictions),
            "clm_predictions": str(args.clm_predictions.resolve()),
            "clm_predictions_sha256": file_sha256(args.clm_predictions),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, sort_keys=True))


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark-faithful ChemFM five-view endpoint evaluation")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--official-test", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--processes", type=int, default=6)
    prepare.add_argument("--seed", type=int, default=533)
    prepare.add_argument("--round-up-to", type=int, default=3300)
    prepare.add_argument("--full-test", action="store_true")
    prepare.add_argument("--equivalence-output", type=Path)
    prepare.add_argument("--equivalence-groups", type=int, default=24)
    prepare.add_argument("--full-manifest-output", type=Path)
    prepare.add_argument("--repository-commit", required=True)
    prepare.set_defaults(function=prepare_manifest)

    worker_parser = subparsers.add_parser("worker")
    worker_parser.add_argument("--checkpoint", type=Path, required=True)
    worker_parser.add_argument("--manifest", type=Path, required=True)
    worker_parser.add_argument("--workers", type=int, required=True)
    worker_parser.add_argument("--worker-index", type=int, required=True)
    worker_parser.add_argument("--prompt-batch-size", type=int, default=1)
    worker_parser.add_argument("--batch-mode", choices=("left-pad", "equal-length"), default="left-pad")
    worker_parser.add_argument("--threads-per-worker", type=int, default=1)
    worker_parser.add_argument("--output", type=Path, required=True)
    worker_parser.set_defaults(function=worker)

    run = subparsers.add_parser("run")
    run.add_argument("--checkpoint", type=Path, required=True)
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--workers", type=int, required=True)
    run.add_argument("--prompt-batch-size", type=int, default=1)
    run.add_argument("--batch-mode", choices=("left-pad", "equal-length"), default="left-pad")
    run.add_argument("--threads-per-worker", type=int, default=1)
    run.add_argument("--output-dir", type=Path, required=True)
    run.set_defaults(function=run_configuration)

    benchmark_parser = subparsers.add_parser("benchmark")
    benchmark_parser.add_argument("--checkpoint", type=Path, required=True)
    benchmark_parser.add_argument("--manifest", type=Path, required=True)
    benchmark_parser.add_argument("--configurations", default="1x1xleft-pad,2x1xleft-pad,4x1xleft-pad,8x1xleft-pad,12x1xleft-pad,4x2xleft-pad,4x2xequal-length")
    benchmark_parser.add_argument("--output-dir", type=Path, required=True)
    benchmark_parser.set_defaults(function=benchmark)

    summary = subparsers.add_parser("summarize")
    summary.add_argument("--manifest", type=Path, required=True)
    summary.add_argument("--native-predictions", type=Path, required=True)
    summary.add_argument("--clm-predictions", type=Path, required=True)
    summary.add_argument("--output", type=Path, required=True)
    summary.add_argument("--seed", type=int, default=533)
    summary.add_argument("--bootstrap-repetitions", type=int, default=20000)
    summary.set_defaults(function=summarize)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    arguments.function(arguments)
