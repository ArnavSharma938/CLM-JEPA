"""A6000 frozen LLM-JEPA geometry diagnostics using upstream serialization."""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

UPSTREAM_DIR = Path(__file__).resolve().parents[1] / "references" / "llm-jepa"
sys.path.insert(0, str(UPSTREAM_DIR))

from finetune import (  # noqa: E402
    get_assistant_messages,
    get_messages,
    get_user_messages,
    setup_model_and_tokenizer,
)


BASE_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"


def load_checkpoint(spec: str, seed: int):
    if spec == "base":
        return setup_model_and_tokenizer(
            BASE_MODEL, use_lora=False, lora_rank=16, seed=seed
        )
    path = Path(spec)
    if (path / "adapter_config.json").exists():
        model, tokenizer = setup_model_and_tokenizer(
            BASE_MODEL, use_lora=False, lora_rank=16, seed=seed
        )
        model = PeftModel.from_pretrained(model, path)
        model = model.merge_and_unload()
        return model, tokenizer
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        low_cpu_mem_usage=True,
        use_cache=False,
    )
    return model, tokenizer


def serialize_pair(tokenizer, messages: list[dict]) -> tuple[str, str]:
    source_messages = get_user_messages(BASE_MODEL, copy.deepcopy(messages))
    source_messages[0]["content"] += "<|predictor_1|>"
    target_messages = get_assistant_messages(
        BASE_MODEL, "gsm8k_train.jsonl", copy.deepcopy(messages)
    )
    source = tokenizer.apply_chat_template(
        source_messages, tokenize=False, add_generation_prompt=False
    )
    target = tokenizer.apply_chat_template(
        target_messages, tokenize=False, add_generation_prompt=False
    )
    return source, target


def embed_prompts(model, tokenizer, prompts: list[str], batch_size: int) -> torch.Tensor:
    chunks = []
    model.eval()
    for start in range(0, len(prompts), batch_size):
        encoded = tokenizer(
            prompts[start : start + batch_size],
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt",
            add_special_tokens=True,
        )
        encoded = {key: value.to(model.device) for key, value in encoded.items()}
        with torch.inference_mode():
            output = model(
                **encoded,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
        chunks.append(output.hidden_states[-1][:, -1].detach().cpu().to(torch.bfloat16))
    return torch.cat(chunks)


def native_ntp_loss(model, tokenizer, prompts: list[str], batch_size: int) -> float:
    weighted_loss = 0.0
    token_count = 0
    model.eval()
    for start in range(0, len(prompts), batch_size):
        encoded = tokenizer(
            prompts[start : start + batch_size],
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt",
            add_special_tokens=True,
        )
        labels = encoded["input_ids"].clone()
        labels[labels == tokenizer.pad_token_id] = -100
        valid = int(labels[:, 1:].ne(-100).sum())
        encoded = {key: value.to(model.device) for key, value in encoded.items()}
        with torch.inference_mode():
            loss = model(**encoded, labels=labels.to(model.device), use_cache=False).loss
        weighted_loss += float(loss) * valid
        token_count += valid
    return weighted_loss / token_count


def effective_rank(values: torch.Tensor) -> float:
    centered = values - values.mean(0, keepdim=True)
    eigenvalues = torch.linalg.eigvalsh(centered @ centered.T).clamp_min(0)
    probabilities = eigenvalues / eigenvalues.sum().clamp_min(1e-30)
    entropy = -(probabilities * probabilities.clamp_min(1e-30).log()).sum()
    return float(entropy.exp())


def candidate_indices(count: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(82)
    order = torch.randperm(count, generator=generator).tolist()
    candidates = []
    for index in range(count):
        negatives = [value for value in order if value != index][:3]
        candidates.append([index, *negatives])
    return torch.tensor(candidates)


def pair_metrics(source: torch.Tensor, target: torch.Tensor) -> dict:
    source = F.normalize(source, dim=-1)
    target = F.normalize(target, dim=-1)
    count = source.shape[0]
    shuffled = torch.roll(torch.arange(count, device=source.device), shifts=1)
    correct = (source * target).sum(-1)
    shuffled_score = (source * target[shuffled]).sum(-1)
    similarities = source @ target.T
    full_top1 = similarities.argmax(dim=1).eq(
        torch.arange(count, device=source.device)
    ).float().mean()
    candidates = candidate_indices(count).to(source.device)
    candidate_scores = (source[:, None, :] * target[candidates]).sum(-1)
    ranks = (candidate_scores > candidate_scores[:, :1]).sum(-1) + 1
    return {
        "correct_cosine": float(correct.mean()),
        "shuffled_cosine": float(shuffled_score.mean()),
        "correct_minus_shuffled": float((correct - shuffled_score).mean()),
        "retrieval_full_top1": float(full_top1),
        "retrieval_full_chance": 1.0 / count,
        "retrieval_four_way_top1": float((ranks == 1).float().mean()),
        "retrieval_four_way_mrr": float((1.0 / ranks.float()).mean()),
        "retrieval_four_way_chance": 0.25,
    }


def stream_geometry(values: torch.Tensor) -> dict:
    mean = values.mean(0)
    mean_square_norm = values.square().sum(1).mean()
    return {
        "variance": float(values.var(0, unbiased=False).mean()),
        "effective_rank": effective_rank(values),
        "mean_direction_energy": float(
            mean.square().sum() / mean_square_norm.clamp_min(1e-30)
        ),
        "mean_embedding_norm": float(values.norm(dim=1).mean()),
        "rms_embedding_norm": float(mean_square_norm.sqrt()),
        "mean_vector_norm": float(mean.norm()),
    }


def analyze(source: torch.Tensor, target: torch.Tensor) -> dict:
    source = source.float().cuda()
    target = target.float().cuda()
    result = {
        "source": stream_geometry(source),
        "target": stream_geometry(target),
        "raw": pair_metrics(source, target),
    }
    joint = torch.cat((source, target))
    joint_mean = joint.mean(0, keepdim=True)
    centered_source = source - joint_mean
    centered_target = target - joint_mean
    result["mean_centered"] = pair_metrics(centered_source, centered_target)
    torch.manual_seed(82)
    _, _, components = torch.pca_lowrank(
        torch.cat((centered_source, centered_target)),
        q=8,
        center=False,
        niter=6,
    )
    for count in (1, 2, 4):
        basis = components[:, :count]
        residual_source = centered_source - (centered_source @ basis) @ basis.T
        residual_target = centered_target - (centered_target @ basis) @ basis.T
        result[f"mean_centered_remove_pc{count}"] = pair_metrics(
            residual_source, residual_target
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", action="append", required=True, help="LABEL=PATH or LABEL=base")
    parser.add_argument("--input-file", default="gsm8k_test.jsonl")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--seed", type=int, default=82)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset("json", data_files=args.input_file)["train"]
    if args.max_examples is not None:
        dataset = dataset.select(range(min(args.max_examples, len(dataset))))
    panel = [
        {
            "index": index,
            "question": row["messages"][1]["content"],
            "answer": row["messages"][2]["content"],
        }
        for index, row in enumerate(dataset)
    ]
    (output_dir / "panel.json").write_text(json.dumps(panel, indent=2), encoding="utf-8")

    all_results = {}
    for checkpoint_arg in args.checkpoint:
        label, spec = checkpoint_arg.split("=", 1)
        model, tokenizer = load_checkpoint(spec, args.seed)
        pairs = [serialize_pair(tokenizer, row["messages"]) for row in dataset]
        source_prompts = [pair[0] for pair in pairs]
        target_prompts = [pair[1] for pair in pairs]
        main_prompts = [
            tokenizer.apply_chat_template(
                get_messages(BASE_MODEL, copy.deepcopy(row["messages"])),
                tokenize=False,
                add_generation_prompt=False,
            )
            for row in dataset
        ]
        source = embed_prompts(model, tokenizer, source_prompts, args.batch_size)
        target = embed_prompts(model, tokenizer, target_prompts, args.batch_size)
        source_last_token_id = tokenizer(
            source_prompts[0], truncation=True, max_length=512
        )["input_ids"][-1]
        target_last_token_id = tokenizer(
            target_prompts[0], truncation=True, max_length=512
        )["input_ids"][-1]
        raw_path = output_dir / f"{label}_embeddings.pt"
        torch.save(
            {
                "source": source,
                "target": target,
                "source_last_token_id": source_last_token_id,
                "target_last_token_id": target_last_token_id,
                "checkpoint": spec,
            },
            raw_path,
        )
        metrics = analyze(source, target)
        metrics["native_ntp_loss"] = native_ntp_loss(
            model, tokenizer, main_prompts, args.batch_size
        )
        metrics["cosine_jepa_loss"] = 1.0 - metrics["raw"]["correct_cosine"]
        metrics["combined_lambda_0_5_loss"] = (
            metrics["native_ntp_loss"] + 0.5 * metrics["cosine_jepa_loss"]
        )
        metrics["checkpoint"] = spec
        metrics["examples"] = len(dataset)
        metrics["source_last_token"] = tokenizer.convert_ids_to_tokens(
            source_last_token_id
        )
        metrics["target_last_token"] = tokenizer.convert_ids_to_tokens(
            target_last_token_id
        )
        all_results[label] = metrics
        (output_dir / f"{label}_metrics.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8"
        )
        del model, tokenizer, source, target
        torch.cuda.empty_cache()

    (output_dir / "geometry_summary.json").write_text(
        json.dumps(all_results, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
