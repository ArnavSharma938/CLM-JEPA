"""Explicit real-checkpoint integration check requested by the amendment."""
import torch
from torch.nn import functional as F

from src.modeling import load_rita
from src.tokenization import collate_encoded, encode_canonical


def test_real_pinned_final_hidden_state_reproduces_live_logits():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loaded = load_rita(device=device)
    encoded = [encode_canonical(loaded.tokenizer, "ACDEFGHIKLMNPQRSTVWY")]
    batch = {key: value.to(device) for key, value in collate_encoded(encoded).items() if key != "residue_mask"}
    with torch.inference_mode():
        outputs = loaded.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        reconstructed = F.linear(outputs.hidden_states, loaded.model.get_output_embeddings().weight)
    torch.testing.assert_close(reconstructed, outputs.logits, atol=2e-3, rtol=2e-3)
