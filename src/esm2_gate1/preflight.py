from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .config import MODEL_ID, MODEL_REVISION
from .modeling import configure_adaptation, dense_targets, load_model, parameter_report, sha256_file


FIXED_SEQUENCES = (
    "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQANL",
    "GAVILMFYWSTNQCPHKRDEGAVILMFYWSTNQCPHKRDE",
    "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY",
)
MASK_POSITIONS = ((3, 11, 25), (2, 17, 35), (5, 20, 38))


def source_metadata() -> dict:
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    info = api.model_info(MODEL_ID, revision=MODEL_REVISION, files_metadata=True)
    if info.sha != MODEL_REVISION:
        raise RuntimeError("resolved model revision differs from the pinned SHA")
    weights = hf_hub_download(MODEL_ID, "model.safetensors", revision=MODEL_REVISION)
    return {
        "model_id": MODEL_ID,
        "resolved_model_revision": info.sha,
        "model_weight_file": str(weights),
        "model_weight_sha256": sha256_file(Path(weights)),
    }


def fair_esm_parity(output: Path) -> dict:
    import esm

    device = torch.device("cuda")
    fair_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    fair_model = fair_model.float().eval().to(device)
    converter = alphabet.get_batch_converter()
    _, _, fair_tokens = converter([(str(i), sequence) for i, sequence in enumerate(FIXED_SEQUENCES)])
    labels = fair_tokens.clone()
    selected = torch.zeros_like(fair_tokens, dtype=torch.bool)
    for row, positions in enumerate(MASK_POSITIONS):
        for position in positions:
            selected[row, position] = True
            fair_tokens[row, position] = alphabet.mask_idx
    with torch.no_grad():
        fair_logits = fair_model(fair_tokens.to(device))["logits"].cpu()
    fair_nll = float(F.cross_entropy(fair_logits[selected], labels[selected]).item())
    del fair_model
    gc.collect(); torch.cuda.empty_cache()

    hf_model, tokenizer = load_model(attention_backend="eager")
    hf_model = hf_model.float().eval().to(device)
    encoded = tokenizer(list(FIXED_SEQUENCES), return_tensors="pt", padding=True)
    hf_labels = encoded["input_ids"].clone()
    hf_selected = torch.zeros_like(hf_labels, dtype=torch.bool)
    for row, positions in enumerate(MASK_POSITIONS):
        for position in positions:
            hf_selected[row, position] = True
            encoded["input_ids"][row, position] = tokenizer.mask_token_id
    with torch.no_grad():
        hf_logits = hf_model(**{key: value.to(device) for key, value in encoded.items()}).logits.cpu()
    if list(alphabet.all_toks) != tokenizer.convert_ids_to_tokens(range(len(tokenizer))):
        raise RuntimeError("Fair-ESM and Hugging Face vocabularies differ in order")
    hf_nll = float(F.cross_entropy(hf_logits[hf_selected], hf_labels[hf_selected]).item())
    difference = (fair_logits[selected] - hf_logits[hf_selected]).abs()
    result = {
        "masked_logits_max_abs_difference": float(difference.max()),
        "masked_logits_mean_abs_difference": float(difference.mean()),
        "fair_esm_nll": fair_nll,
        "huggingface_nll": hf_nll,
        "nll_absolute_difference": abs(fair_nll - hf_nll),
        "tolerance": {"logits_max_abs": 5e-4, "nll_abs": 1e-5},
    }
    result["passed"] = (
        result["masked_logits_max_abs_difference"] <= result["tolerance"]["logits_max_abs"]
        and result["nll_absolute_difference"] <= result["tolerance"]["nll_abs"]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    if not result["passed"]:
        raise RuntimeError(f"unexplained Fair-ESM parity failure: {result}")
    return result


def _benchmark_one(backend: str, reference: dict | None, *, compile_model: bool = False) -> tuple[dict, dict]:
    device = torch.device("cuda")
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    model, tokenizer = load_model(attention_backend=backend)
    configure_adaptation(model, "dtft")
    model.to(device=device, dtype=torch.bfloat16).train()
    target = dense_targets(model)[0].module.weight
    if compile_model:
        model = torch.compile(model, mode="reduce-overhead")
    batch = tokenizer([FIXED_SEQUENCES[2] * 3, FIXED_SEQUENCES[1] * 3], return_tensors="pt", padding=True)
    labels = batch["input_ids"].clone()
    selected = torch.zeros_like(labels, dtype=torch.bool)
    for row in range(2):
        for position in (5, 20, 50, 90):
            selected[row, position] = True
            batch["input_ids"][row, position] = tokenizer.mask_token_id
    batch = {key: value.to(device) for key, value in batch.items()}
    labels = labels.to(device)
    for _ in range(2):
        model.zero_grad(set_to_none=True)
        logits = model(**batch).logits
        loss = F.cross_entropy(logits[selected.to(device)].float(), labels[selected.to(device)])
        loss.backward()
    torch.cuda.synchronize()
    start = time.perf_counter()
    steps = 5
    for _ in range(steps):
        model.zero_grad(set_to_none=True)
        logits = model(**batch).logits
        loss = F.cross_entropy(logits[selected.to(device)].float(), labels[selected.to(device)])
        loss.backward()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    observation = {
        "masked_logits": logits[selected.to(device)].detach().float().cpu(),
        "nll": float(loss.detach()),
        "gradient": target.grad.detach().float().cpu(),
    }
    tokens = int(batch["attention_mask"].sum()) * steps
    metrics = {
        "backend": backend,
        "torch_compile": compile_model,
        "tokens_per_second": tokens / elapsed,
        "peak_vram_bytes": torch.cuda.max_memory_allocated(),
        "nll": observation["nll"],
    }
    if reference is not None:
        candidate_logits = observation["masked_logits"]
        reference_logits = reference["masked_logits"]
        candidate_probabilities = candidate_logits.softmax(dim=-1)
        reference_probabilities = reference_logits.softmax(dim=-1)
        centered_candidate = candidate_logits - candidate_logits.mean(dim=-1, keepdim=True)
        centered_reference = reference_logits - reference_logits.mean(dim=-1, keepdim=True)
        grad_cosine = F.cosine_similarity(
            observation["gradient"].flatten(), reference["gradient"].flatten(), dim=0
        )
        gradient_relative_l2 = (
            (observation["gradient"] - reference["gradient"]).norm()
            / reference["gradient"].norm().clamp_min(1e-12)
        )
        metrics["parity"] = {
            "logits_max_abs": float((candidate_logits - reference_logits).abs().max()),
            "centered_logits_max_abs": float(
                (centered_candidate - centered_reference).abs().max()
            ),
            "probability_max_abs": float(
                (candidate_probabilities - reference_probabilities).abs().max()
            ),
            "reference_to_candidate_kl": float(
                (
                    reference_probabilities
                    * (
                        reference_probabilities.clamp_min(1e-12).log()
                        - candidate_probabilities.clamp_min(1e-12).log()
                    )
                ).sum(dim=-1).mean()
            ),
            "top1_agreement": float(
                (candidate_logits.argmax(dim=-1) == reference_logits.argmax(dim=-1))
                .float().mean()
            ),
            "nll_abs": abs(observation["nll"] - reference["nll"]),
            "gradient_cosine": float(grad_cosine),
            "gradient_relative_l2": float(gradient_relative_l2),
        }
    del model
    gc.collect(); torch.cuda.empty_cache()
    return metrics, observation


def _parity_passed(parity: dict) -> bool:
    """BF16 backend equivalence gate based on predictions, loss, and gradients."""
    return (
        parity["centered_logits_max_abs"] <= 0.15
        and parity["probability_max_abs"] <= 2e-3
        and parity["reference_to_candidate_kl"] <= 5e-5
        and parity["top1_agreement"] == 1.0
        and parity["nll_abs"] <= 2e-4
        and parity["gradient_cosine"] >= 0.9985
        and parity["gradient_relative_l2"] <= 0.08
    )


def benchmark_backends(output: Path) -> dict:
    results = []
    reference = None
    for backend in ("eager", "sdpa", "flash_attention_2"):
        try:
            metrics, observation = _benchmark_one(backend, reference)
            if reference is None:
                reference = observation
                metrics["passed"] = True
            else:
                parity = metrics["parity"]
                metrics["passed"] = _parity_passed(parity)
            results.append(metrics)
        except Exception as error:
            results.append({"backend": backend, "passed": False, "error": repr(error)})
    passing = [item for item in results if item.get("passed")]
    if not passing:
        raise RuntimeError("no attention backend passed")
    selected = max(passing, key=lambda item: item["tokens_per_second"])["backend"]
    selected_metrics = next(item for item in passing if item["backend"] == selected)
    compile_result = None
    try:
        compile_result, _ = _benchmark_one(selected, reference, compile_model=True)
        parity = compile_result.get("parity", {})
        compile_result["passed"] = _parity_passed(parity)
    except Exception as error:
        compile_result = {"backend": selected, "torch_compile": True, "passed": False, "error": repr(error)}
    use_compile = bool(
        compile_result.get("passed")
        and compile_result["tokens_per_second"] > selected_metrics["tokens_per_second"] * 1.05
    )
    payload = {
        "benchmarks": results,
        "selected_backend": selected,
        "torch_compile_benchmark": compile_result,
        "use_torch_compile": use_compile,
        "compile_minimum_improvement_fraction": 0.05,
        "bf16_parity_tolerances": {
            "centered_logits_max_abs": 0.15,
            "probability_max_abs": 2e-3,
            "reference_to_candidate_kl": 5e-5,
            "top1_agreement": 1.0,
            "nll_abs": 2e-4,
            "gradient_cosine_min": 0.9985,
            "gradient_relative_l2": 0.08,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def preflight(output: Path) -> dict:
    output.parent.mkdir(parents=True, exist_ok=True)
    model, _ = load_model(attention_backend="eager")
    report = parameter_report(model)
    config = model.config
    source_digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        source_digest.update(path.name.encode())
        source_digest.update(path.read_bytes())
    try:
        repository_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, text=True, capture_output=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        repository_commit = None
    report.update({
        "layers": config.num_hidden_layers,
        "hidden_dimension": config.hidden_size,
        "attention_heads": config.num_attention_heads,
        "intermediate_dimension": config.intermediate_size,
        "source": source_metadata(),
        "repository_commit": repository_commit,
        "esm2_gate1_source_sha256": source_digest.hexdigest(),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": importlib.metadata.version("transformers"),
            "datasets": importlib.metadata.version("datasets"),
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "gpu_vram_bytes": torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None,
            "free_disk_bytes": shutil.disk_usage(output.parent).free,
        },
    })
    del model
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("metadata", "parity", "benchmark"):
        item = sub.add_parser(command)
        item.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "metadata":
        preflight(args.output)
    elif args.command == "parity":
        fair_esm_parity(args.output)
    else:
        benchmark_backends(args.output)


if __name__ == "__main__":
    main()
