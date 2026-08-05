import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clm_jepa.chemfm_native import IGNORE_INDEX, ReactionCollator, load_reaction_tokenizer
from clm_jepa.paths import TOKENIZER_DIR


def test_source_labels_are_fully_masked():
    tokenizer = load_reaction_tokenizer(TOKENIZER_DIR)
    batch = ReactionCollator(tokenizer)([{"src": "CCO.O=O", "tgt": "CC(=O)O"}])
    target_start = batch["labels"][0].ne(IGNORE_INDEX).nonzero()[0].item()
    assert batch["labels"][0, :target_start].eq(IGNORE_INDEX).all()
    assert batch["labels"][0, target_start:].eq(batch["input_ids"][0, target_start:]).all()
    decoded = tokenizer.decode(batch["input_ids"][0, :target_start]).replace(" ", "")
    assert decoded == "<rstart>CCO.O=O<eos>"
