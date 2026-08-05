from __future__ import annotations

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TOKENIZER_DIR = ROOT / "assets" / "chemfm_reaction_tokenizer"
MODEL_DIR = Path(os.environ.get("CHEMFM_MODEL_PATH", ROOT / "models" / "ChemFM-1B"))
