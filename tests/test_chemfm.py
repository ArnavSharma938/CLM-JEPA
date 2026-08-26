import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import pytest
from transformers import LlamaConfig, LlamaForCausalLM

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chemfm import (
    IGNORE_INDEX, TOKENIZER_DIR, ExactPreallocatedDynamicCache,
    ReactionCollator, canonicalize,
    generate_products_batch, load_reaction_tokenizer,
    resize_chemfm_then_predictors,
)
from transformers.cache_utils import DynamicCache


def test_preallocated_cache_matches_dynamic_update_reorder_and_reset():
    torch.manual_seed(71)
    reference = DynamicCache()
    candidate = ExactPreallocatedDynamicCache(max_cache_len=8)
    for layer in range(2):
        key = torch.randn(3, 2, 2, 4)
        value = torch.randn(3, 2, 2, 4)
        expected = reference.update(key, value, layer)
        actual = candidate.update(key, value, layer)
        torch.testing.assert_close(actual[0], expected[0])
        torch.testing.assert_close(actual[1], expected[1])
    beam_indices = torch.tensor([2, 0, 0])
    reference.reorder_cache(beam_indices)
    candidate.reorder_cache(beam_indices)
    for layer in range(2):
        key = torch.randn(3, 2, 1, 4)
        value = torch.randn(3, 2, 1, 4)
        expected = reference.update(key, value, layer)
        actual = candidate.update(key, value, layer)
        torch.testing.assert_close(actual[0], expected[0])
        torch.testing.assert_close(actual[1], expected[1])
    candidate.reset()
    fresh = DynamicCache()
    key = torch.randn(3, 2, 3, 4)
    value = torch.randn(3, 2, 3, 4)
    expected = fresh.update(key, value, 0)
    actual = candidate.update(key, value, 0)
    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1])


def test_chemfm_and_predictor_tokens_use_respective_upstream_initializers():
    torch.manual_seed(19)
    model = LlamaForCausalLM(LlamaConfig(
        vocab_size=20, hidden_size=32, intermediate_size=64,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=64,
    ))
    original_mean = model.get_input_embeddings().weight.detach().mean(
        dim=0, keepdim=True
    )
    resize_chemfm_then_predictors(
        model, tokenizer_size=24, chemfm_vocab_size=22
    )
    embeddings = model.get_input_embeddings().weight.detach()
    torch.testing.assert_close(
        embeddings[20:22], original_mean.expand(2, -1)
    )
    assert not torch.equal(embeddings[22], embeddings[23])
    assert not torch.equal(embeddings[22], original_mean[0])


def test_source_labels_are_fully_masked():
    tokenizer = load_reaction_tokenizer(TOKENIZER_DIR)
    batch = ReactionCollator(tokenizer)([{"src": "CCO.O=O", "tgt": "CC(=O)O"}])
    target_start = batch["labels"][0].ne(IGNORE_INDEX).nonzero()[0].item()
    assert batch["labels"][0, :target_start].eq(IGNORE_INDEX).all()
    assert batch["labels"][0, target_start:].eq(batch["input_ids"][0, target_start:]).all()
    decoded = tokenizer.decode(batch["input_ids"][0, :target_start]).replace(" ", "")
    assert decoded == "<rstart>CCO.O=O<eos>"


def test_retro_collator_uses_product_then_reactant_markers():
    tokenizer = load_reaction_tokenizer(TOKENIZER_DIR)
    batch = ReactionCollator(tokenizer, task="retro")([{"src": "CCO", "tgt": "Br.CC"}])
    assert batch["generation_prompts"] == ["<prostart>CCO<eos><rstart>"]
    start = batch["labels"][0].ne(IGNORE_INDEX).nonzero()[0].item()
    assert tokenizer.decode(batch["input_ids"][0, :start]).replace(" ", "") == "<prostart>CCO<eos>"


def test_shuffled_target_view_uses_task_marker_and_target_eos():
    tokenizer = load_reaction_tokenizer(TOKENIZER_DIR)
    batch = ReactionCollator(tokenizer)([
        {"src": "CCO", "tgt": "CC=O", "jepa_tgt": "CCC"},
        {"src": "CC", "tgt": "CO", "jepa_tgt": "CCN"},
    ])
    rows = [
        row[mask] for row, mask in zip(
            batch["jepa_target_ids"], batch["jepa_target_attention_mask"]
        )
    ]
    assert all(int(row[-1]) == tokenizer.eos_token_id for row in rows)
    assert tokenizer.decode(rows[0]).replace(" ", "") == "<prostart>CCC<eos>"


def test_canonicalization_clears_atom_maps_and_retains_stereochemistry():
    assert canonicalize("[CH3:1][OH:2]") == "CO"
    assert canonicalize("F[C@H](Cl)Br") != canonicalize("F[C@@H](Cl)Br")


class GenerationTokenizer:
    eos_token_id = 2
    pad_token_id = 0
    padding_side = "right"

    def __call__(self, prompts, **kwargs):
        return {
            "input_ids": torch.tensor([[4, 5]] * len(prompts)),
            "attention_mask": torch.tensor([[1, 1]] * len(prompts)),
            "token_type_ids": torch.tensor([[0, 0]] * len(prompts)),
        }

    def batch_decode(self, values, skip_special_tokens):
        return [" C C "] * len(values)


class GenerationModel:
    device = torch.device("cpu")

    def __init__(self):
        self.config = SimpleNamespace(use_cache=False)
        self.kwargs = None

    def generate(self, **kwargs):
        self.kwargs = kwargs
        rows = []
        for _ in range(len(kwargs["input_ids"])):
            rows.extend([[4, 5, 6], [4, 5, 7]])
        return torch.tensor(rows)


def test_batched_generation_preserves_official_chemfm_beam_arguments():
    model = GenerationModel()
    values = generate_products_batch(
        model, GenerationTokenizer(), ["prompt"],
        max_length=1024, num_beams=2, num_return_sequences=2,
    )
    assert values == [["CC", "CC"]]
    assert model.kwargs["max_length"] == 1024
    assert model.kwargs["early_stopping"] == "never"
    assert model.kwargs["length_penalty"] == 0.0
    assert "max_new_tokens" not in model.kwargs
    assert model.config.use_cache is False


def test_equal_length_batched_generation_preserves_prompt_and_beam_order():
    model = GenerationModel()
    values = generate_products_batch(
        model, GenerationTokenizer(), ["view-1", "view-2"],
        max_length=1024, num_beams=2, num_return_sequences=2,
    )
    assert values == [["CC", "CC"], ["CC", "CC"]]
    assert model.kwargs["input_ids"].shape == (2, 2)
    assert model.kwargs["num_beams"] == 2
    assert model.kwargs["early_stopping"] == "never"
    assert model.config.use_cache is False


def test_batched_generation_rejects_unequal_token_lengths():
    class UnequalTokenizer(GenerationTokenizer):
        def __call__(self, prompts, **kwargs):
            return {
                "input_ids": torch.tensor([[4, 5, 0], [4, 5, 6]]),
                "attention_mask": torch.tensor([[1, 1, 0], [1, 1, 1]]),
                "token_type_ids": torch.zeros((2, 3), dtype=torch.long),
            }

    with pytest.raises(ValueError, match="equal tokenized lengths"):
        generate_products_batch(
            GenerationModel(), UnequalTokenizer(), ["short", "long"],
            num_beams=2, num_return_sequences=2,
        )


def test_left_padded_batched_generation_restores_tokenizer_and_output_order():
    tokenizer = GenerationTokenizer()
    model = GenerationModel()
    values = generate_products_batch(
        model,
        tokenizer,
        ["short", "long"],
        num_beams=2,
        num_return_sequences=2,
        pad_unequal_prompts=True,
    )
    assert values == [["CC", "CC"], ["CC", "CC"]]
    assert tokenizer.padding_side == "right"
