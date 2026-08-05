from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clm_jepa.chemfm_native import ReactionCollator, load_lora_model, load_reaction_tokenizer  # noqa: E402
from clm_jepa.modeling import CLMJEPA, add_predictor_tokens  # noqa: E402
from clm_jepa.paths import MODEL_DIR, TOKENIZER_DIR  # noqa: E402


def main() -> None:
    tokenizer = load_reaction_tokenizer(TOKENIZER_DIR)
    predictor_ids = add_predictor_tokens(tokenizer)
    model = load_lora_model(MODEL_DIR, tokenizer).cuda().eval()
    with (ROOT / "data" / "manifests" / "gate1" / "train_32.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))[:2]
    batch = ReactionCollator(tokenizer)(rows)
    batch = {key: value.cuda() for key, value in batch.items() if torch.is_tensor(value)}
    method = CLMJEPA(predictor_ids, tokenizer.eos_token_id, tokenizer.pad_token_id)
    with torch.inference_mode():
        native = model(**batch)
        zero = method(model, batch, k=2, jepa_weight=0.0)
        active = method(model, batch, k=2, jepa_weight=1.0)
    result = {
        "checkpoint": "ChemFM/ChemFM-1B f99dc2e89726539bb9cf31b2e2b4360650bac6a8",
        "tokenizer_size": len(tokenizer),
        "predictor_token_ids": predictor_ids,
        "lambda_zero_loss_exact": bool(torch.equal(native.loss, zero.loss)),
        "lambda_zero_logits_exact": bool(torch.equal(native.logits, zero.logits)),
        "jepa_loss": float(active.jepa_loss),
        "source_state_shape": list(active.source_states.shape),
        "target_state_shape": list(active.target_states.shape),
        "finite": bool(torch.isfinite(active.loss)),
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
    }
    if not all((result["lambda_zero_loss_exact"], result["lambda_zero_logits_exact"], result["finite"])):
        raise AssertionError(result)
    output = ROOT / "artifacts" / "gate2" / "chemfm_1b_smoke.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
