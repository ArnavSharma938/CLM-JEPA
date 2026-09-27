from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import MODEL_ID, MODEL_REVISION
from .token_cache import build_token_cache


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache Propedia ESM-2 token IDs once")
    parser.add_argument("--pairs", type=Path, default=Path("data/protein/propedia26/unique_pairs.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/protein/propedia26/token_cache.pt"))
    args = parser.parse_args()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    print(json.dumps(build_token_cache(args.pairs, args.output, tokenizer), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
