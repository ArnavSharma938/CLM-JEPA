import pytest
import torch

from src.constants import CANONICAL_AA, RITA_MODEL_ID, RITA_TOKENIZER_REVISION
from src.tokenization import (
    assert_no_special_positions,
    collate_encoded,
    encode_canonical,
    next_residue_transition_mask,
)


@pytest.fixture(scope="module")
def tokenizer():
    transformers = pytest.importorskip("transformers")
    return transformers.AutoTokenizer.from_pretrained(
        RITA_MODEL_ID, revision=RITA_TOKENIZER_REVISION
    )


def test_canonical_one_residue_per_position_and_encode_decode(tokenizer):
    encoded = encode_canonical(tokenizer, CANONICAL_AA)
    assert len(encoded.input_ids) == len(CANONICAL_AA) + 1
    assert encoded.input_ids[-1] == 2
    assert tokenizer.convert_ids_to_tokens(encoded.input_ids[:-1]) == list(CANONICAL_AA)
    # The published decoder inserts spaces; token identity, not string formatting,
    # is the lossless round-trip contract.
    decoded = tokenizer.decode(encoded.input_ids, skip_special_tokens=False)
    assert decoded.replace(" ", "") == CANONICAL_AA + "<EOS>"


def test_published_special_token_nuance_and_no_bos(tokenizer):
    assert tokenizer.bos_token_id is None
    assert tokenizer.pad_token_id is None
    assert tokenizer.eos_token_id is None
    assert tokenizer.encode("ACD")[-1] == 2
    assert len(tokenizer.encode("ACD")) == 4


def test_exact_causal_alignment_excludes_eos_and_pad(tokenizer):
    a = encode_canonical(tokenizer, "ACDE")
    b = encode_canonical(tokenizer, "FGH")
    batch = collate_encoded([a, b])
    mask = next_residue_transition_mask(batch["input_ids"], batch["attention_mask"])
    assert mask.tolist() == [
        [True, True, True, False],
        [True, True, False, False],
    ]
    assert_no_special_positions(batch["input_ids"][:, 1:], mask)


@pytest.mark.parametrize("bad", ["", "ACDX", "ABCD", "acd"])
def test_noncanonical_sequences_fail_closed(tokenizer, bad):
    with pytest.raises(ValueError):
        encode_canonical(tokenizer, bad)

