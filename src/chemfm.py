from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from peft import LoraConfig, get_peft_model
from rdkit import Chem, RDLogger
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoConfig, AutoTokenizer, LlamaForCausalLM
from transformers.cache_utils import DynamicCache

ROOT = Path(__file__).resolve().parents[1]
TOKENIZER_DIR = ROOT / "references" / "chemfm" / "finetuning" / "reaction_prediction" / "tokenizer"
MODEL_DIR = Path(os.environ.get("CHEMFM_MODEL_PATH", ROOT / "models" / "ChemFM-1B"))

RDLogger.DisableLog("rdApp.*")


IGNORE_INDEX = -100
PAD_TOKEN = "[PAD]"
REACTANT_START = "<rstart>"
PRODUCT_START = "<prostart>"
END = "<eos>"


class ExactPreallocatedDynamicCache(DynamicCache):
    """Dynamic-shape KV cache without repeated prefix concatenation.

    Attention still receives a view ending at the current sequence length, so
    its tensor shapes and arithmetic match ``DynamicCache``.  Two fixed backing
    buffers let beam reordering ping-pong between allocations; appending a new
    token then copies only that token instead of the entire cached prefix.
    """

    def __init__(self, max_cache_len: int) -> None:
        super().__init__()
        self.max_cache_len = int(max_cache_len)
        self._key_buffers: list[list[torch.Tensor | int] | None] = []
        self._value_buffers: list[list[torch.Tensor | int] | None] = []

    def _ensure_layer(self, layer_idx: int, key_states, value_states) -> None:
        while len(self.key_cache) <= layer_idx:
            self.key_cache.append([])
            self.value_cache.append([])
            self._key_buffers.append(None)
            self._value_buffers.append(None)
        if self._key_buffers[layer_idx] is not None:
            key_buffer = self._key_buffers[layer_idx][0]
            value_buffer = self._value_buffers[layer_idx][0]
            if (
                key_buffer.shape[0] == key_states.shape[0]
                and key_buffer.shape[1] == key_states.shape[1]
                and key_buffer.shape[-1] == key_states.shape[-1]
                and value_buffer.shape[0] == value_states.shape[0]
                and value_buffer.shape[1] == value_states.shape[1]
                and value_buffer.shape[-1] == value_states.shape[-1]
            ):
                return
        key_shape = list(key_states.shape)
        value_shape = list(value_states.shape)
        key_shape[-2] = self.max_cache_len
        value_shape[-2] = self.max_cache_len
        self._key_buffers[layer_idx] = [
            torch.empty(key_shape, dtype=key_states.dtype, device=key_states.device),
            torch.empty(key_shape, dtype=key_states.dtype, device=key_states.device),
            0,
        ]
        self._value_buffers[layer_idx] = [
            torch.empty(value_shape, dtype=value_states.dtype, device=value_states.device),
            torch.empty(value_shape, dtype=value_states.dtype, device=value_states.device),
            0,
        ]

    def reset(self) -> None:
        """Clear logical lengths while retaining allocated backing buffers."""
        self._seen_tokens = 0
        for layer_idx in range(len(self.key_cache)):
            self.key_cache[layer_idx] = []
            self.value_cache[layer_idx] = []
            if self._key_buffers[layer_idx] is not None:
                self._key_buffers[layer_idx][2] = 0
                self._value_buffers[layer_idx][2] = 0

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        self._ensure_layer(layer_idx, key_states, value_states)
        old_length = (
            self.key_cache[layer_idx].shape[-2]
            if isinstance(self.key_cache[layer_idx], torch.Tensor)
            else 0
        )
        added = key_states.shape[-2]
        new_length = old_length + added
        if new_length > self.max_cache_len:
            raise ValueError(
                f"KV cache length {new_length} exceeds capacity {self.max_cache_len}"
            )
        if layer_idx == 0:
            self._seen_tokens += added
        key_buffers = self._key_buffers[layer_idx]
        value_buffers = self._value_buffers[layer_idx]
        key_buffer = key_buffers[int(key_buffers[2])]
        value_buffer = value_buffers[int(value_buffers[2])]
        key_buffer[..., old_length:new_length, :].copy_(key_states)
        value_buffer[..., old_length:new_length, :].copy_(value_states)
        self.key_cache[layer_idx] = key_buffer[..., :new_length, :]
        self.value_cache[layer_idx] = value_buffer[..., :new_length, :]
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        for layer_idx in range(len(self.key_cache)):
            if not isinstance(self.key_cache[layer_idx], torch.Tensor):
                continue
            length = self.key_cache[layer_idx].shape[-2]
            key_buffers = self._key_buffers[layer_idx]
            value_buffers = self._value_buffers[layer_idx]
            current = int(key_buffers[2])
            alternate = 1 - current
            device_indices = beam_idx.to(self.key_cache[layer_idx].device)
            torch.index_select(
                key_buffers[current][..., :length, :],
                0,
                device_indices,
                out=key_buffers[alternate][..., :length, :],
            )
            torch.index_select(
                value_buffers[current][..., :length, :],
                0,
                device_indices,
                out=value_buffers[alternate][..., :length, :],
            )
            key_buffers[2] = alternate
            value_buffers[2] = alternate
            self.key_cache[layer_idx] = key_buffers[alternate][..., :length, :]
            self.value_cache[layer_idx] = value_buffers[alternate][..., :length, :]


@dataclass
class ReactionCollator:
    """Official ChemFM synthesis collation, kept independent of their debug loop."""

    tokenizer: object
    task: str = "forward"
    source_max_len: int = 512
    target_max_len: int = 512

    def _target_text(self, smiles: str) -> str:
        marker = REACTANT_START if self.task == "retro" else PRODUCT_START
        return f"{marker}{smiles}{END}"

    def __call__(self, instances: Sequence[dict[str, str]]) -> dict[str, object]:
        if self.task == "retro":
            sources = [f"{PRODUCT_START}{x['src']}{END}" for x in instances]
            targets = [self._target_text(x["tgt"]) for x in instances]
            prompts = [source + REACTANT_START for source in sources]
        elif self.task in {"forward", "metabolism"}:
            sources = [f"{REACTANT_START}{x['src']}{END}" for x in instances]
            targets = [self._target_text(x["tgt"]) for x in instances]
            prompts = [source + PRODUCT_START for source in sources]
        else:
            raise ValueError(f"unsupported task: {self.task}")
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
        result = {
            "input_ids": input_ids,
            "attention_mask": input_ids.ne(self.tokenizer.pad_token_id),
            "labels": labels,
            "tgt_smiles": [x["tgt"] for x in instances],
            "generation_prompts": prompts,
        }
        if any("jepa_tgt" in instance for instance in instances):
            if not all("jepa_tgt" in instance for instance in instances):
                raise ValueError("every row in a shuffled batch must provide jepa_tgt")
            jepa_ids = self.tokenizer(
                [self._target_text(x["jepa_tgt"]) for x in instances],
                max_length=self.target_max_len,
                truncation=True,
                add_special_tokens=False,
            )["input_ids"]
            jepa_rows = [torch.tensor(row, dtype=torch.long) for row in jepa_ids]
            result["jepa_target_ids"] = pad_sequence(
                jepa_rows, batch_first=True, padding_value=self.tokenizer.pad_token_id
            )
            result["jepa_target_attention_mask"] = result["jepa_target_ids"].ne(
                self.tokenizer.pad_token_id
            )
            result["jepa_tgt_smiles"] = [x["jepa_tgt"] for x in instances]
        return result


def load_reaction_tokenizer(tokenizer_path: Path):
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, padding_side="right", use_fast=True, trust_remote_code=False
    )
    tokenizer.add_special_tokens({"pad_token": PAD_TOKEN})
    return tokenizer


def resize_chemfm_then_predictors(
    model, *, tokenizer_size: int, chemfm_vocab_size: int,
) -> None:
    original_size = model.get_input_embeddings().weight.shape[0]
    if not original_size <= chemfm_vocab_size <= tokenizer_size:
        raise ValueError("vocabulary sizes must satisfy model <= ChemFM <= JEPA")
    model.resize_token_embeddings(chemfm_vocab_size)
    if chemfm_vocab_size > original_size:
        # Exact ChemFM smart_tokenizer_and_embedding_resize behavior: only
        # added input rows are set to the old vocabulary mean. Output rows keep
        # the resize initializer.
        with torch.no_grad():
            embeddings = model.get_input_embeddings().weight
            embeddings[original_size:chemfm_vocab_size] = embeddings[
                :original_size
            ].mean(dim=0, keepdim=True)
    if tokenizer_size > chemfm_vocab_size:
        # Pinned LLM-JEPA adds predictor tokens afterward and retains the
        # resize initializer for distinct predictor input/output embeddings.
        model.resize_token_embeddings(tokenizer_size)


def load_lora_model(
    model_path: Path, tokenizer, attention_dropout: float = 0.1,
    *, chemfm_vocab_size: int | None = None,
    attn_implementation: str | None = None,
):
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=False)
    config.attention_dropout = attention_dropout
    load_kwargs = {
        "config": config,
        # BF16 preserves the exponent range needed by this direct, single-GPU
        # loop while retaining the 2-byte footprint required by the pilot GPU.
        "torch_dtype": torch.bfloat16,
        "low_cpu_mem_usage": True,
        "trust_remote_code": False,
    }
    if attn_implementation is not None:
        load_kwargs["attn_implementation"] = attn_implementation
    model = LlamaForCausalLM.from_pretrained(model_path, **load_kwargs)
    resize_chemfm_then_predictors(
        model,
        tokenizer_size=len(tokenizer),
        chemfm_vocab_size=(
            len(tokenizer) if chemfm_vocab_size is None else chemfm_vocab_size
        ),
    )
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
    mol = Chem.MolFromSmiles(smiles, sanitize=True)
    if mol is None:
        return ""
    for atom in mol.GetAtoms():
        if atom.HasProp("molAtomMapNumber"):
            atom.ClearProp("molAtomMapNumber")
    try:
        return Chem.MolToSmiles(mol, isomericSmiles=True)
    except Exception:
        return ""


@torch.inference_mode()
def generate_products_batch(
    model,
    tokenizer,
    prompts: list[str],
    max_length: int = 1024,
    num_beams: int = 1,
    num_return_sequences: int = 1,
    pad_unequal_prompts: bool = False,
):
    if not prompts:
        raise ValueError("at least one generation prompt is required")
    original_padding_side = tokenizer.padding_side
    if pad_unequal_prompts:
        tokenizer.padding_side = "left"
    try:
        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=False,
        )
    finally:
        tokenizer.padding_side = original_padding_side
    encoded.pop("token_type_ids", None)
    prompt_lengths = encoded["attention_mask"].sum(dim=1)
    if (
        not pad_unequal_prompts
        and not torch.equal(prompt_lengths, prompt_lengths[:1].expand_as(prompt_lengths))
    ):
        raise ValueError("batched ChemFM prompts must have equal tokenized lengths")
    encoded = {key: value.to(model.device) for key, value in encoded.items()}
    model.config.use_cache = True
    try:
        generation_kwargs = {}
        cache_implementation = os.environ.get("CHEMFM_CACHE_IMPLEMENTATION")
        if cache_implementation:
            generation_kwargs["cache_implementation"] = cache_implementation
        if os.environ.get("CHEMFM_EXACT_PREALLOCATED_CACHE") == "1":
            if cache_implementation:
                raise ValueError(
                    "preallocated dynamic cache and cache_implementation are exclusive"
                )
            exact_cache = getattr(model, "_chemfm_exact_preallocated_cache", None)
            if exact_cache is None or exact_cache.max_cache_len != max_length:
                exact_cache = ExactPreallocatedDynamicCache(max_length)
                model._chemfm_exact_preallocated_cache = exact_cache
            else:
                exact_cache.reset()
            generation_kwargs["past_key_values"] = exact_cache
        outputs = model.generate(
            **encoded,
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            do_sample=False,
            num_beams=num_beams,
            eos_token_id=tokenizer.eos_token_id,
            early_stopping="never",
            pad_token_id=tokenizer.pad_token_id,
            length_penalty=0.0,
            **generation_kwargs,
        )
    finally:
        model.config.use_cache = False
    prompt_width = encoded["input_ids"].shape[1]
    decoded = tokenizer.batch_decode(outputs[:, prompt_width:], skip_special_tokens=True)
    decoded = [text.replace(" ", "") for text in decoded]
    expected = len(prompts) * num_return_sequences
    if len(decoded) != expected:
        raise RuntimeError(f"expected {expected} generated sequences, got {len(decoded)}")
    return [
        decoded[start:start + num_return_sequences]
        for start in range(0, expected, num_return_sequences)
    ]


def generate_products(
    model,
    tokenizer,
    prompts: list[str],
    max_length: int = 1024,
    num_beams: int = 1,
    num_return_sequences: int = 1,
):
    if len(prompts) != 1:
        raise ValueError("ChemFM standalone evaluation generates one R-SMILES view at a time")
    return generate_products_batch(
        model,
        tokenizer,
        prompts,
        max_length=max_length,
        num_beams=num_beams,
        num_return_sequences=num_return_sequences,
        pad_unequal_prompts=False,
    )[0]
