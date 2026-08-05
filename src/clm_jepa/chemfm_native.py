from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from peft import LoraConfig, get_peft_model
from rdkit import Chem
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoConfig, AutoTokenizer, LlamaForCausalLM


IGNORE_INDEX = -100
PAD_TOKEN = "[PAD]"
REACTANT_START = "<rstart>"
PRODUCT_START = "<prostart>"
END = "<eos>"


@dataclass
class ReactionCollator:
    """Official ChemFM synthesis collation, kept independent of their debug loop."""

    tokenizer: object
    source_max_len: int = 512
    target_max_len: int = 512

    def __call__(self, instances: Sequence[dict[str, str]]) -> dict[str, object]:
        sources = [f"{REACTANT_START}{x['src']}{END}" for x in instances]
        targets = [f"{PRODUCT_START}{x['tgt']}{END}" for x in instances]
        prompts = [source + PRODUCT_START for source in sources]
        source_ids = self.tokenizer(
            sources,
            max_length=self.source_max_len,
            truncation=True,
            add_special_tokens=False,
        )["input_ids"]
        target_ids = self.tokenizer(
            targets,
            max_length=self.target_max_len,
            truncation=True,
            add_special_tokens=False,
        )["input_ids"]

        input_ids = []
        labels = []
        for src, tgt in zip(source_ids, target_ids):
            input_ids.append(torch.tensor(src + tgt, dtype=torch.long))
            labels.append(
                torch.tensor([IGNORE_INDEX] * len(src) + copy.deepcopy(tgt), dtype=torch.long)
            )
        input_ids = pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return {
            "input_ids": input_ids,
            "attention_mask": input_ids.ne(self.tokenizer.pad_token_id),
            "labels": labels,
            "tgt_smiles": [x["tgt"] for x in instances],
            "generation_prompts": prompts,
        }


def load_reaction_tokenizer(tokenizer_path: Path):
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, padding_side="right", use_fast=True, trust_remote_code=False
    )
    tokenizer.add_special_tokens({"pad_token": PAD_TOKEN})
    return tokenizer


def load_lora_model(model_path: Path, tokenizer, attention_dropout: float = 0.1):
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=False)
    config.attention_dropout = attention_dropout
    model = LlamaForCausalLM.from_pretrained(
        model_path,
        config=config,
        # BF16 preserves the exponent range needed by this direct, single-GPU
        # loop while retaining the 2-byte footprint required by the pilot GPU.
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
    old_size = model.get_input_embeddings().weight.shape[0]
    model.resize_token_embeddings(len(tokenizer))
    if len(tokenizer) > old_size:
        with torch.no_grad():
            embedding = model.get_input_embeddings().weight
            embedding[old_size:] = embedding[:old_size].mean(dim=0, keepdim=True)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    lora = LoraConfig(
        task_type="CAUSAL_LM",
        r=8,
        target_modules=[
            "q_proj",
            "v_proj",
            "k_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        modules_to_save=["embed_tokens", "lm_head"],
        inference_mode=False,
        lora_alpha=8,
        lora_dropout=0.1,
        use_rslora=False,
    )
    return get_peft_model(model, lora, adapter_name="USPTO-MIT-Synthesis")


def canonicalize(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol) if mol is not None else ""


@torch.inference_mode()
def generate_products(model, tokenizer, prompts: list[str], max_new_tokens: int = 128):
    encoded = tokenizer(
        prompts, return_tensors="pt", padding=True, truncation=True, add_special_tokens=False
    )
    encoded.pop("token_type_ids", None)
    encoded = {key: value.to(model.device) for key, value in encoded.items()}
    model.config.use_cache = True
    outputs = model.generate(
        **encoded,
        max_new_tokens=max_new_tokens,
        num_return_sequences=1,
        do_sample=False,
        num_beams=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    model.config.use_cache = False
    prompt_width = encoded["input_ids"].shape[1]
    decoded = tokenizer.batch_decode(outputs[:, prompt_width:], skip_special_tokens=True)
    return [text.replace(" ", "") for text in decoded]
