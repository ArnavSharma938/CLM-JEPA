from __future__ import annotations

import argparse
from pathlib import Path

import torch

from official_five_view_evaluation import load_endpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model, _ = load_endpoint(args.checkpoint)
    input_ids = torch.arange(10, device="cuda", dtype=torch.long).remainder(model.config.vocab_size).view(10, 1)
    attention_mask = torch.ones_like(input_ids)
    with torch.inference_mode():
        logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
    torch.cuda.synchronize()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(logits.cpu(), args.output)


if __name__ == "__main__":
    main()
