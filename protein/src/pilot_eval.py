"""Batched frozen-model evaluation utilities for the causal pilot."""
from __future__ import annotations

import hashlib
import math
from collections import Counter
from typing import Iterable, Sequence

import numpy as np
import torch
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, average_precision_score
from torch.nn import functional as F

from .tokenization import collate_encoded, encode_canonical


def stable_int(value: str, salt: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{salt}\0{value}".encode()).digest()[:8], "big")


def _rotate_half(value: torch.Tensor) -> torch.Tensor:
    first, second = value.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class RITAKVCache:
    """Exact eval-only KV cache for the pinned RITA architecture.

    The checkpoint's remote-code forward ignores ``past_key_values``.  This
    implementation evaluates one new position at a time while retaining each
    layer's already-rotated keys and values.  It calls the live projection
    modules, so it also works for PEFT-wrapped RITA attention projections.
    """

    def __init__(self, model):
        self.model = model
        # PeftModel -> RITAModelForCausalLM -> RITAModel.  The loop also keeps
        # this robust to evaluation of the unwrapped checkpoint.
        base = model
        while not hasattr(base, "transformer"):
            if hasattr(base, "base_model"):
                base = base.base_model
            elif hasattr(base, "model"):
                base = base.model
            else:
                raise TypeError("could not locate RITA transformer")
        self.causal_lm = base
        self.transformer = base.transformer
        self.cache: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.position = 0

    def reset(self):
        self.cache = []
        self.position = 0

    def step(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Consume ``[batch, 1]`` token ids and return next-token logits."""
        if token_ids.ndim != 2 or token_ids.shape[1] != 1:
            raise ValueError("RITAKVCache.step requires [batch, 1] token ids")
        x = self.transformer.embedding(token_ids)
        new_cache = []
        for layer_index, layer in enumerate(self.transformer.layers):
            residual = x
            normalized = layer.attn_norm(x)
            attention = layer.self_attention
            batch, _, width = normalized.shape
            heads, head_dim = attention.num_heads, attention.head_dim
            q = attention.query(normalized).view(batch, 1, heads, head_dim).transpose(1, 2)
            k = attention.key(normalized).view(batch, 1, heads, head_dim).transpose(1, 2)
            v = attention.value(normalized).view(batch, 1, heads, head_dim).transpose(1, 2)
            inv_freq = attention.rotary_embedding.inv_freq
            position = torch.tensor([self.position], device=x.device, dtype=inv_freq.dtype)
            frequency = torch.einsum("i,j->ij", position, inv_freq)
            embedding = torch.cat((frequency, frequency), dim=-1).to(dtype=q.dtype)
            cosine = embedding.cos()[None, None]
            sine = embedding.sin()[None, None]
            q = q * cosine + _rotate_half(q) * sine
            k = k * cosine + _rotate_half(k) * sine
            if layer_index < len(self.cache):
                prior_k, prior_v = self.cache[layer_index]
                k_all = torch.cat((prior_k, k), dim=2)
                v_all = torch.cat((prior_v, v), dim=2)
            else:
                k_all, v_all = k, v
            scores = (q @ k_all.transpose(-2, -1)) * (head_dim ** -0.5)
            weights = F.softmax(scores, dim=-1)
            attended = (weights @ v_all).transpose(1, 2).contiguous().view(batch, 1, width)
            attended = attention.resid_drop(attention.proj(attended))
            x = residual + layer.attn_dropout(attended)
            x = x + layer.mlp_dropout(layer.mlp(layer.mlp_norm(x)))
            new_cache.append((k_all, v_all))
        self.cache = new_cache
        self.position += 1
        hidden = self.transformer.final_norm(x)
        return self.causal_lm.get_output_embeddings()(hidden)[:, -1]

    def prime(self, input_ids: torch.Tensor) -> torch.Tensor:
        self.reset()
        logits = None
        for position in range(input_ids.shape[1]):
            logits = self.step(input_ids[:, position:position + 1])
        if logits is None:
            raise ValueError("cannot prime an empty prompt")
        return logits


def sample_top_p(logits: torch.Tensor, temperature: float, top_p: float,
                 generator: torch.Generator | None = None) -> torch.Tensor:
    scaled = logits.float() / temperature
    sorted_logits, sorted_indices = torch.sort(scaled, descending=True, dim=-1)
    cumulative = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    remove = cumulative > top_p
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False
    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
    selected = torch.multinomial(F.softmax(sorted_logits, dim=-1), 1, generator=generator)
    return sorted_indices.gather(-1, selected)


def sample_top_p_uniform(logits: torch.Tensor, uniforms: torch.Tensor,
                         temperature: float, top_p: float) -> torch.Tensor:
    """Top-p inverse-CDF sampling with one externally fixed uniform per row."""
    sorted_logits, sorted_indices = torch.sort(logits.float() / temperature,
                                                descending=True, dim=-1)
    original_cumulative = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    remove = original_cumulative > top_p
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False
    probability = F.softmax(sorted_logits.masked_fill(remove, float("-inf")), dim=-1)
    cumulative = torch.cumsum(probability, dim=-1)
    chosen = (cumulative < uniforms[:, None]).sum(-1).clamp_max(logits.shape[-1] - 1)
    return sorted_indices.gather(-1, chosen[:, None])


@torch.inference_mode()
def cached_generate_rita(model, input_ids: torch.Tensor, max_new_tokens: int,
                         eos_token_id: int, temperature: float = 1.0,
                         top_p: float = 1.0,
                         generator: torch.Generator | None = None,
                         uniforms: torch.Tensor | None = None,
                         do_sample: bool = True) -> torch.Tensor:
    """Generate with the exact RITA cache; prompts in a batch must be unpadded."""
    cache = RITAKVCache(model)
    logits = cache.prime(input_ids)
    result = input_ids
    finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
    for _ in range(max_new_tokens):
        if do_sample and uniforms is not None:
            next_ids = sample_top_p_uniform(
                logits, uniforms[:, result.shape[1] - input_ids.shape[1]],
                temperature, top_p,
            )
        else:
            next_ids = (sample_top_p(logits, temperature, top_p, generator)
                        if do_sample else logits.argmax(-1, keepdim=True))
        result = torch.cat((result, next_ids), dim=1)
        finished |= next_ids.squeeze(1).eq(eos_token_id)
        if bool(finished.all()):
            break
        logits = cache.step(next_ids)
    return result


@torch.inference_mode()
def directional_lm_rows(model, tokenizer, rows: Sequence[dict], device: str,
                        batch_size: int = 64) -> list[dict]:
    """Per-protein/per-direction ordinary RITA NTP, including terminal EOS."""
    directional = []
    for row in rows:
        for orientation, sequence in (("forward", row["sequence"]),
                                      ("reverse", row["sequence"][::-1])):
            directional.append((row, orientation, sequence))
    # Evaluation order is irrelevant and length bucketing avoids padding every
    # heterogeneous UniRef batch to its longest member.
    directional.sort(key=lambda item: (len(item[2]), item[0]["id"], item[1]))
    output = []
    for start in range(0, len(directional), batch_size):
        selected = directional[start:start + batch_size]
        encoded = [encode_canonical(tokenizer, sequence) for _, _, sequence in selected]
        batch = collate_encoded(encoded)
        ids = batch["input_ids"].to(device)
        attention = batch["attention_mask"].to(device)
        with torch.autocast(
            device_type="cuda", dtype=torch.bfloat16,
            enabled=torch.device(device).type == "cuda",
        ):
            logits = model(input_ids=ids, attention_mask=attention, use_cache=False).logits
        token_loss = F.cross_entropy(
            logits[:, :-1].float().transpose(1, 2), ids[:, 1:], reduction="none"
        )
        valid = attention[:, 1:].bool()
        probability = F.softmax(logits[:, :-1].float(), -1).gather(
            -1, ids[:, 1:].unsqueeze(-1)
        ).squeeze(-1)
        top1 = logits[:, :-1].argmax(-1).eq(ids[:, 1:])
        for index, (row, orientation, _) in enumerate(selected):
            mask = valid[index]
            output.append({
                "id": row["id"], "cluster_id": row.get("cluster_id", row["id"]),
                "orientation": orientation, "length": len(row["sequence"]),
                "ntp_loss": float(token_loss[index][mask].mean()),
                "perplexity": float(torch.exp(token_loss[index][mask].mean())),
                "correct_probability": float(probability[index][mask].mean()),
                "top1": float(top1[index][mask].float().mean()),
                "scored_tokens": int(mask.sum()),
            })
    return output


def bidirectional_summary(rows: Sequence[dict]) -> list[dict]:
    grouped = {}
    for row in rows:
        grouped.setdefault(row["id"], {})[row["orientation"]] = row
    result = []
    for identifier, pair in grouped.items():
        if set(pair) != {"forward", "reverse"}:
            raise AssertionError(f"missing orientation for {identifier}")
        result.append({
            "id": identifier,
            "cluster_id": pair["forward"]["cluster_id"],
            "length": pair["forward"]["length"],
            "forward_ntp_loss": pair["forward"]["ntp_loss"],
            "reverse_ntp_loss": pair["reverse"]["ntp_loss"],
            "bidirectional_ntp_loss": .5 * (
                pair["forward"]["ntp_loss"] + pair["reverse"]["ntp_loss"]
            ),
            "forward_perplexity": pair["forward"]["perplexity"],
            "reverse_perplexity": pair["reverse"]["perplexity"],
            "correct_probability": .5 * (
                pair["forward"]["correct_probability"] + pair["reverse"]["correct_probability"]
            ),
            "top1": .5 * (pair["forward"]["top1"] + pair["reverse"]["top1"]),
        })
    return result


@torch.inference_mode()
def sequence_representations(model, tokenizer, rows: Sequence[dict], device: str,
                             batch_size: int = 64, reverse: bool = False):
    result = {}
    ordered = sorted(rows, key=lambda row: (len(row["sequence"]), row["id"]))
    for start in range(0, len(ordered), batch_size):
        selected = ordered[start:start + batch_size]
        sequences = [row["sequence"][::-1] if reverse else row["sequence"] for row in selected]
        batch = collate_encoded([encode_canonical(tokenizer, sequence) for sequence in sequences])
        ids = batch["input_ids"].to(device)
        attention = batch["attention_mask"].to(device)
        with torch.autocast(
            device_type="cuda", dtype=torch.bfloat16,
            enabled=torch.device(device).type == "cuda",
        ):
            states = model(
                input_ids=ids, attention_mask=attention, use_cache=False
            ).hidden_states.float()
        for index, row in enumerate(selected):
            length = len(row["sequence"])
            state = states[index, :length]
            if reverse:
                state = state.flip(0)
            result[row["id"]] = {
                "mean": state.mean(0).cpu().numpy().astype(np.float32),
                "tokens": state.cpu().numpy().astype(np.float32),
            }
    return result


def topk_accuracy(scores: np.ndarray, truth: np.ndarray, k: int) -> float:
    indices = np.argpartition(scores, -k, axis=1)[:, -k:]
    return float(np.mean(np.any(indices == truth[:, None], axis=1)))


def ec_probe(rows: Sequence[dict], forward: dict, reverse: dict) -> dict:
    labels = sorted({row["label"] for row in rows})
    encode = {label: index for index, label in enumerate(labels)}
    by_split = {split: [row for row in rows if row["split"] == split]
                for split in ("train", "dev", "test")}
    train = by_split["train"]
    y_train = np.array([encode[row["label"]] for row in train])
    result = {"classes": len(labels), "train_examples": len(train)}
    for orientation, features in (("forward", forward), ("reverse", reverse)):
        # Fit the same frozen protocol independently in each orientation.  A
        # forward-trained classifier evaluated on reverse states would confound
        # orientation quality with an avoidable coordinate-distribution shift.
        x_train = np.stack([features[row["id"]]["mean"] for row in train])
        probe = LogisticRegression(
            C=1.0, penalty="l2", solver="lbfgs", max_iter=1000,
            multi_class="multinomial", random_state=20260915,
        ).fit(x_train, y_train)
        for split in ("dev", "test"):
            selected = by_split[split]
            x = np.stack([features[row["id"]]["mean"] for row in selected])
            y = np.array([encode[row["label"]] for row in selected])
            scores = probe.predict_proba(x)
            result[f"{orientation}_{split}_accuracy"] = accuracy_score(y, scores.argmax(1))
            result[f"{orientation}_{split}_top5"] = topk_accuracy(scores, y, min(5, len(labels)))
    result["test_orientation_accuracy_delta_reverse_minus_forward"] = (
        result["reverse_test_accuracy"] - result["forward_test_accuracy"]
    )
    return result


def secondary_probe(rows: Sequence[dict], representations: dict) -> dict:
    train_rows = [row for row in rows if row["split"] == "train"]
    test_rows = [row for row in rows if row["split"] in {"casp12", "cb513", "ts115"}]
    def matrix(selected):
        xs, ys = [], []
        for row in selected:
            valid = np.asarray(row["valid"], dtype=bool)
            labels = np.asarray(row["labels"], dtype=np.int64)
            xs.append(representations[row["id"]]["tokens"][valid])
            ys.append(labels[valid])
        return np.concatenate(xs), np.concatenate(ys)
    x_train, y_train = matrix(train_rows)
    probe = RidgeClassifier(alpha=1.0).fit(x_train, y_train)
    result = {"train_proteins": len(train_rows), "train_residues": len(y_train)}
    for split in ("casp12", "cb513", "ts115"):
        selected = [row for row in test_rows if row["split"] == split]
        x, y = matrix(selected)
        scores = probe.decision_function(x)
        result[f"{split}_accuracy"] = accuracy_score(y, scores.argmax(1))
        result[f"{split}_top2"] = topk_accuracy(scores, y, 2)
        result[f"{split}_proteins"] = len(selected)
    result["macro_accuracy"] = float(np.mean([
        result[f"{split}_accuracy"] for split in ("casp12", "cb513", "ts115")
    ]))
    return result


def _contact_pairs(row: dict, salt: str, negative_ratio: int = 5):
    coords = np.asarray(row["coords"], dtype=np.float32)
    valid = np.asarray(row["valid"], dtype=bool)
    length = len(coords)
    i, j = np.triu_indices(length, k=24)
    keep = valid[i] & valid[j]
    i, j = i[keep], j[keep]
    distance = np.linalg.norm(coords[i] - coords[j], axis=1)
    positive = np.flatnonzero(distance < 8.0)
    negative = np.flatnonzero(distance >= 8.0)
    rng = np.random.default_rng(stable_int(row["id"], salt))
    if len(positive) > 512:
        positive = rng.choice(positive, 512, replace=False)
    n_negative = min(len(negative), max(1, negative_ratio * len(positive)))
    negative = rng.choice(negative, n_negative, replace=False)
    chosen = np.concatenate([positive, negative])
    return i[chosen], j[chosen], np.concatenate([
        np.ones(len(positive), dtype=np.int64),
        np.zeros(len(negative), dtype=np.int64),
    ])


def contact_probe(rows: Sequence[dict], representations: dict) -> dict:
    train = [row for row in rows if row["split"] == "valid"]
    test = [row for row in rows if row["split"] == "test"]
    pca_train = np.concatenate([
        representations[row["id"]]["tokens"] for row in train
    ], axis=0)
    reducer = PCA(n_components=128, svd_solver="randomized", random_state=20260915).fit(
        pca_train
    )
    del pca_train
    reduced = {
        row["id"]: reducer.transform(representations[row["id"]]["tokens"]).astype(np.float32)
        for row in rows
    }
    def matrix(selected, salt):
        xs, ys, proteins = [], [], []
        for row in selected:
            i, j, labels = _contact_pairs(row, salt)
            hidden = reduced[row["id"]]
            xs.append(np.concatenate([np.abs(hidden[i] - hidden[j]), hidden[i] * hidden[j]], axis=1))
            ys.append(labels)
            proteins.extend([row["id"]] * len(labels))
        return np.concatenate(xs), np.concatenate(ys), proteins
    x_train, y_train, _ = matrix(train, "contact-train-v1")
    x_test, y_test, protein = matrix(test, "contact-test-v1")
    probe = LogisticRegression(
        C=1.0, penalty="l2", solver="liblinear", max_iter=1000,
        random_state=20260915,
    ).fit(x_train, y_train)
    score = probe.predict_proba(x_test)[:, 1]
    per_protein = []
    for identifier in sorted(set(protein)):
        mask = np.asarray([value == identifier for value in protein])
        if y_test[mask].min() != y_test[mask].max():
            per_protein.append(average_precision_score(y_test[mask], score[mask]))
    return {
        "train_proteins": len(train), "test_proteins": len(test),
        "train_pairs": len(y_train), "test_pairs": len(y_test),
        "input_pca_components": 128,
        "input_pca_train_variance_retained": float(reducer.explained_variance_ratio_.sum()),
        "average_precision": average_precision_score(y_test, score),
        "protein_macro_average_precision": float(np.mean(per_protein)),
    }


def generated_quality(sequence: str, reference_composition: dict[str, float]) -> dict:
    canonical = [token for token in sequence if token in reference_composition]
    counts = Counter(canonical)
    length = len(sequence)
    probabilities = np.asarray([count / max(len(canonical), 1) for count in counts.values()])
    entropy = float(-(probabilities * np.log(probabilities + 1e-30)).sum())
    max_run = 0
    current_run = 0
    prior = None
    for token in sequence:
        current_run = current_run + 1 if token == prior else 1
        max_run = max(max_run, current_run)
        prior = token
    bigrams = [sequence[index:index + 2] for index in range(max(0, length - 1))]
    composition_l1 = sum(abs(counts.get(aa, 0) / max(len(canonical), 1) - expected)
                         for aa, expected in reference_composition.items())
    return {
        "length": length,
        "canonical_fraction": len(canonical) / max(length, 1),
        "entropy": entropy,
        "max_homopolymer": max_run,
        "unique_bigram_fraction": len(set(bigrams)) / max(len(bigrams), 1),
        "composition_l1": composition_l1,
    }


def reference_composition(rows: Sequence[dict]) -> dict[str, float]:
    counts = Counter("".join(row["sequence"] for row in rows))
    total = sum(counts.values())
    return {aa: counts[aa] / total for aa in sorted(counts)}
