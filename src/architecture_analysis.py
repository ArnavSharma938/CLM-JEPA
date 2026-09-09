"""Correct, shared primitives for architecture-level transition analysis."""
from __future__ import annotations

import hashlib
import json

import torch
from torch.nn import functional as F


def product_transition_rows(records):
    """Build product-local k=1 rows with explicit absolute-index checks."""
    rows = []
    for record in sorted(records, key=lambda value: value["reaction_identity"]):
        positions = list(map(int, record["product_indices"]))
        states = record["final_states"]
        ids = record["input_ids"].long()
        if len(states) != len(positions):
            raise ValueError("cache is not explicitly sliced to H[product_indices]")
        if positions != list(range(positions[0], positions[-1] + 1)):
            raise ValueError("product positions are not contiguous")
        # Retain another actual product token for gold-rank/margin metrics,
        # excluding EOS as the decoder target.
        for j in range(len(positions) - 2):
            absolute_t = positions[j]
            absolute_future = positions[j + 1]
            assert absolute_future == absolute_t + 1
            assert torch.equal(states[j], states[positions.index(absolute_t)])
            assert torch.equal(states[j + 1], states[positions.index(absolute_future)])
            oracle = int(ids[absolute_t + 1])
            gold = int(ids[absolute_t + 2])
            assert oracle == int(ids[absolute_future])
            assert gold == int(ids[positions[j + 2]])
            rows.append({
                "reaction_identity": record["reaction_identity"],
                "split": record["split"],
                "product_offset": j,
                "absolute_t": absolute_t,
                "absolute_future": absolute_future,
                "h_t": states[j].float(),
                "h_next": states[j + 1].float(),
                "oracle_token": oracle,
                "gold_token": gold,
            })
    return rows


def orthonormal_decoder_basis(projection, singular_values):
    """Recover V^T from stored P=Sigma V^T, without centering states."""
    singular_values = torch.as_tensor(singular_values, dtype=torch.float32)
    projection = projection.float()
    if projection.shape[0] != len(singular_values) or torch.any(singular_values <= 0):
        raise ValueError("invalid stored decoder projection/SVD")
    vt = projection / singular_values[:, None]
    torch.testing.assert_close(vt @ vt.T, torch.eye(len(vt)), atol=3e-4, rtol=3e-4)
    return vt


def decoder_coordinates(hidden, projection, vt):
    """Return weighted z, orthonormal q, visible h, and decoder-null h."""
    hidden = hidden.float()
    q = F.linear(hidden, vt.float())
    z = F.linear(hidden, projection.float())
    parallel = F.linear(q, vt.float().T)
    return {"orthonormal": q, "functional": z, "parallel": parallel, "null": hidden - parallel}


def functional_distribution_metrics(true_logits, predicted_logits, gold):
    """Correct JS and prediction-derived rank/margin metrics."""
    true_logits, predicted_logits = true_logits.float(), predicted_logits.float()
    true_logp = F.log_softmax(true_logits, -1)
    pred_logp = F.log_softmax(predicted_logits, -1)
    p, q = true_logp.exp(), pred_logp.exp()
    midpoint = 0.5 * (p + q)
    log_midpoint = midpoint.clamp_min(1e-30).log()
    js = 0.5 * ((p * (true_logp - log_midpoint)).sum(-1) +
                (q * (pred_logp - log_midpoint)).sum(-1))
    row = torch.arange(len(gold), device=predicted_logits.device)
    gold_logits = predicted_logits[row, gold]
    competitors = predicted_logits.clone()
    competitors[row, gold] = -torch.inf
    result = {
        "js": js,
        "top1_agreement": predicted_logits.argmax(-1).eq(true_logits.argmax(-1)),
        "gold_rank": predicted_logits.gt(gold_logits[:, None]).sum(-1) + 1,
        "gold_margin": gold_logits - competitors.max(-1).values,
        "gold_log_probability": pred_logp[row, gold],
    }
    for k in (5, 10):
        width = min(k, true_logits.shape[-1])
        left = true_logits.topk(width, -1).indices
        right = predicted_logits.topk(width, -1).indices
        result[f"top{k}_overlap"] = (
            left[:, :, None] == right[:, None, :]).any(-1).sum(-1).float() / width
    return result


def selected_key_hash(rows):
    keys = [(row["reaction_identity"], int(row["absolute_t"])) for row in rows]
    encoded = json.dumps(keys, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()
