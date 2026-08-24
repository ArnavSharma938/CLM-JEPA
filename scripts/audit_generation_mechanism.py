"""Frozen diagnostics connecting cLM-JEPA representations to ChemFM decoding.

This script never trains a model.  It uses the selected matched-native and
direct MSE+SIGReg epoch-4 checkpoints for four small diagnostics:

* layerwise JEPA-view versus autoregressive-view representation comparison;
* frozen cross-checkpoint activation patching;
* exact one-step AdamW counterfactuals from a saved optimizer state; and
* chemistry-aware rescoring of existing generated candidates.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F
from peft.utils.save_and_load import load_peft_weights, set_peft_model_state_dict
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from torch.nn.utils.rnn import pad_sequence
from transformers import set_seed


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from audit_chemfm_mechanism import (  # noqa: E402
    disable_stochastic_behavior,
    parameter_fingerprint,
)
from audit_contraction_ntp_direction import build_validation_batches  # noqa: E402
from audit_sigreg_pair_specificity import fixed_batches  # noqa: E402
from chemfm import (  # noqa: E402
    IGNORE_INDEX,
    MODEL_DIR,
    TOKENIZER_DIR,
    ReactionCollator,
    canonicalize,
    load_lora_model,
    load_reaction_tokenizer,
)
from jepa import CLMJEPA, add_predictor_tokens, extract_source_and_target  # noqa: E402
from train import (  # noqa: E402
    ADAM_BETAS,
    ADAM_EPSILON,
    WEIGHT_DECAY,
    raw_auxiliary_vjp,
    read_rows,
    validate_serialization_endings,
)


SEED = 533
ADAPTER_NAME = "USPTO-MIT-Synthesis"
NATIVE_CHECKPOINT = (
    ROOT / "runs" / "sigreg_batch16_pilot" / "matched_b4"
    / "native_checkpoints" / "epoch_4"
)
CLM_CHECKPOINT = (
    ROOT / "runs" / "mse_ablation" / "stage1"
    / "mse_sigreg_checkpoints" / "epoch_4"
)
TRAIN_MANIFEST = (
    ROOT / "data" / "clm_jepa_uspto_mit_pilot_1280" / "uspto_mit_train.csv"
)
REPRESENTATION_PANEL = (
    ROOT / "data" / "clm_jepa_uspto_mit_validation_256"
    / "uspto_mit_validation_length_stratified_256.csv"
)
NATIVE_PREDICTIONS = (
    ROOT / "runs" / "official_five_view_endpoint" / "stage1_native"
    / "predictions.jsonl"
)
CLM_PREDICTIONS = (
    ROOT / "runs" / "official_five_view_endpoint" / "stage1_clm"
    / "predictions.jsonl"
)
DEFAULT_OUTPUT_DIR = ROOT / "runs" / "diagnostics" / "generation_mechanism"

SIGREG_RELATIVE_COEFFICIENT = 4.0 * 0.01 / 0.99
ACTIVE_AUXILIARY_COEFFICIENT = 2.0


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_representation_panel(limit: int | None = None) -> list[dict[str, str]]:
    with REPRESENTATION_PANEL.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda row: int(row["pilot_panel_index"]))
    if len(rows) != 256 or len({row["reaction_identity"] for row in rows}) != 256:
        raise ValueError("expected the frozen 256-reaction representation panel")
    return rows if limit is None else rows[:limit]


def canonicalized_rows(rows: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    result = []
    for row in rows:
        source_parts = [canonicalize(part) for part in row["source"].split(".")]
        target = canonicalize(row["target"])
        if not target or not source_parts or not all(source_parts):
            raise ValueError(f"canonicalization failed for {row['reaction_identity']}")
        updated = dict(row)
        # Canonicalize each molecule independently and sort components.  This
        # removes the paired-root R-SMILES ordering rather than merely choosing
        # a different jointly aligned root.
        updated["source"] = ".".join(sorted(source_parts))
        updated["target"] = target
        updated["serialization"] = "independent_canonical_smiles"
        result.append(updated)
    return result


def centered_linear_gram(values: torch.Tensor) -> torch.Tensor:
    values = values.double() - values.double().mean(0, keepdim=True)
    return values @ values.T


def cka_from_grams(gram_x: torch.Tensor, gram_y: torch.Tensor) -> float:
    numerator = (gram_x * gram_y).sum()
    denominator = gram_x.square().sum().sqrt() * gram_y.square().sum().sqrt()
    return float(numerator / denominator.clamp_min(torch.finfo(numerator.dtype).eps))


def linear_cka(first: torch.Tensor, second: torch.Tensor) -> float:
    """Centered linear CKA using the equivalent sample-Gram formulation."""
    if first.ndim != 2 or second.ndim != 2 or first.size(0) != second.size(0):
        raise ValueError("linear CKA expects two 2D matrices with equal rows")
    return cka_from_grams(centered_linear_gram(first), centered_linear_gram(second))


def ridge_explained_variance(
    sources: torch.Tensor,
    targets: torch.Tensor,
    identities: Sequence[str],
    *,
    alpha: float = 1.0,
) -> dict[str, float | int]:
    heldout_count = max(2, math.ceil(0.2 * len(identities)))
    heldout = set(sorted(identities)[-heldout_count:])
    train = torch.tensor([identity not in heldout for identity in identities])
    test = ~train
    x_train = sources[train].double()
    y_train = targets[train].double()
    x_test = sources[test].double()
    y_test = targets[test].double()
    x_mean = x_train.mean(0, keepdim=True)
    y_mean = y_train.mean(0, keepdim=True)
    centered_x = x_train - x_mean
    coefficients = torch.linalg.solve(
        centered_x @ centered_x.T
        + alpha * torch.eye(len(centered_x), dtype=torch.float64),
        y_train - y_mean,
    )
    prediction = (x_test - x_mean) @ centered_x.T @ coefficients + y_mean
    residual = (y_test - prediction).square().sum()
    baseline = (y_test - y_mean).square().sum().clamp_min(1e-30)
    return {
        "explained_variance": float(1.0 - residual / baseline),
        "train_rows": int(train.sum()),
        "heldout_rows": int(test.sum()),
        "alpha": alpha,
    }


def retrieval_metrics(
    sources: torch.Tensor,
    targets: torch.Tensor,
    candidate_indices: Sequence[Sequence[int]] | None = None,
) -> dict[str, float | int]:
    source = F.normalize(sources.float(), dim=-1)
    target = F.normalize(targets.float(), dim=-1)
    scores = source @ target.T
    count = len(source)
    if candidate_indices is None:
        correct_scores = scores.diag()
        ranks = (scores > correct_scores[:, None]).sum(1) + 1
        wrong_scores = scores.clone()
        wrong_scores.fill_diagonal_(-torch.inf)
        best_wrong = wrong_scores.max(1).values
        candidates_per_query = count
    else:
        candidates = torch.tensor(candidate_indices, dtype=torch.long)
        if candidates.size(0) != count or not candidates[:, 0].equal(torch.arange(count)):
            raise ValueError("each retrieval row must place its true target first")
        candidate_scores = scores.gather(1, candidates)
        correct_scores = candidate_scores[:, 0]
        best_wrong = candidate_scores[:, 1:].max(1).values
        ranks = (candidate_scores > candidate_scores[:, :1]).sum(1) + 1
        candidates_per_query = int(candidates.size(1))
    return {
        "candidates_per_query": candidates_per_query,
        "top1": float(ranks.eq(1).float().mean()),
        "mrr": float((1.0 / ranks.float()).mean()),
        "mean_correct_cosine": float(correct_scores.mean()),
        "mean_best_wrong_cosine": float(best_wrong.mean()),
        "mean_margin_over_best_wrong": float((correct_scores - best_wrong).mean()),
    }


def bootstrap_mean_ci(
    values: Sequence[float], *, seed: int = SEED, draws: int = 10000,
) -> list[float]:
    tensor = torch.tensor(values, dtype=torch.float64)
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randint(len(tensor), (draws, len(tensor)), generator=generator)
    means = tensor[indices].mean(1)
    return [float(torch.quantile(means, 0.025)), float(torch.quantile(means, 0.975))]


def paired_summary(
    candidate: Sequence[float], reference: Sequence[float], *, seed: int = SEED,
) -> dict[str, Any]:
    if len(candidate) != len(reference) or not candidate:
        raise ValueError("paired summary requires equal nonempty vectors")
    differences = [float(left - right) for left, right in zip(candidate, reference)]
    return {
        "count": len(differences),
        "candidate_mean": statistics.fmean(candidate),
        "reference_mean": statistics.fmean(reference),
        "mean_difference": statistics.fmean(differences),
        "mean_difference_bootstrap_95_ci": bootstrap_mean_ci(differences, seed=seed),
        "candidate_better": sum(value > 0.0 for value in differences),
        "reference_better": sum(value < 0.0 for value in differences),
        "tied": sum(value == 0.0 for value in differences),
    }


def adapter_weights_dir(checkpoint: Path) -> Path:
    nested = checkpoint / ADAPTER_NAME
    return nested if nested.exists() else checkpoint


def load_adapter_state(checkpoint: Path) -> dict[str, torch.Tensor]:
    return load_peft_weights(str(adapter_weights_dir(checkpoint)), device="cpu")


def apply_adapter_state(model, state: Mapping[str, torch.Tensor], label: str) -> None:
    result = set_peft_model_state_dict(model, state, adapter_name=ADAPTER_NAME)
    if getattr(result, "unexpected_keys", None):
        raise RuntimeError(f"unexpected adapter keys for {label}: {result.unexpected_keys}")


def prepare_model():
    if not torch.cuda.is_available():
        raise EnvironmentError("the frozen mechanism audit requires local CUDA")
    set_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    tokenizer = load_reaction_tokenizer(TOKENIZER_DIR)
    chemfm_vocab_size = len(tokenizer)
    predictor_ids = add_predictor_tokens(tokenizer)
    collator = ReactionCollator(tokenizer, task="forward")
    model = load_lora_model(
        MODEL_DIR,
        tokenizer,
        attention_dropout=0.0,
        chemfm_vocab_size=chemfm_vocab_size,
        attn_implementation="sdpa",
    ).cuda().eval()
    controls = disable_stochastic_behavior(model)
    return tokenizer, predictor_ids, collator, model, controls


def causal_lm(model):
    return model.get_base_model() if hasattr(model, "get_base_model") else model


def llama_backbone(model):
    return causal_lm(model).model


def device_tensors(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _unpadded(values: torch.Tensor, mask: torch.Tensor) -> list[torch.Tensor]:
    return [row[row_mask.bool()] for row, row_mask in zip(values, mask)]


def build_three_view_batch(collator, rows: Sequence[Mapping[str, str]]):
    raw = collator([{"src": row["source"], "tgt": row["target"]} for row in rows])
    sources, targets = extract_source_and_target(raw)
    native_rows = _unpadded(raw["input_ids"], raw["attention_mask"])
    combined = native_rows + sources + targets
    padded = pad_sequence(
        combined, batch_first=True, padding_value=collator.tokenizer.pad_token_id
    )
    attention = padded.ne(collator.tokenizer.pad_token_id)
    return raw, sources, targets, native_rows, padded, attention


@torch.inference_mode()
def collect_layerwise_views(model, collator, rows, batch_size: int, *, final_only=False):
    names = (
        "source_only_eos",
        "target_only_eos",
        "native_source_eos",
        "autoregressive_prompt",
        "autoregressive_product_mean",
        "teacher_forced_target_eos",
    )
    collected: dict[str, list[list[torch.Tensor]] | None] = {name: None for name in names}
    layer_labels = None
    model.eval()
    for start in range(0, len(rows), batch_size):
        subset = rows[start:start + batch_size]
        raw, sources, targets, native_rows, padded, attention = build_three_view_batch(
            collator, subset
        )
        output = llama_backbone(model)(
            input_ids=padded.to(model.device),
            attention_mask=attention.to(model.device),
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = output.hidden_states[-1:] if final_only else output.hidden_states
        if layer_labels is None:
            layer_labels = (
                [model.config.num_hidden_layers]
                if final_only
                else list(range(len(hidden_states)))
            )
            for name in names:
                collected[name] = [[] for _ in hidden_states]
        batch_count = len(subset)
        for relative_layer, hidden in enumerate(hidden_states):
            for row_index in range(batch_count):
                source_length = len(sources[row_index])
                target_length = len(targets[row_index])
                native_length = len(native_rows[row_index])
                product_positions = hidden[
                    row_index, source_length:native_length - 1
                ]
                if product_positions.numel() == 0:
                    raise RuntimeError("teacher-forced product span is empty")
                values = {
                    "source_only_eos": hidden[
                        batch_count + row_index, source_length - 1
                    ],
                    "target_only_eos": hidden[
                        2 * batch_count + row_index, target_length - 1
                    ],
                    "native_source_eos": hidden[row_index, source_length - 1],
                    # The generation prompt ends at <prostart>; this state
                    # predicts the first molecular product token.
                    "autoregressive_prompt": hidden[row_index, source_length],
                    # These states predict the raw product tokens and final EOS.
                    "autoregressive_product_mean": product_positions.mean(0),
                    "teacher_forced_target_eos": hidden[row_index, native_length - 1],
                }
                for name, value in values.items():
                    assert collected[name] is not None
                    collected[name][relative_layer].append(value.float().cpu())
        del output, hidden_states
    result = {
        name: torch.stack([torch.stack(layer) for layer in layers])
        for name, layers in collected.items()
        if layers is not None
    }
    return layer_labels, result


def layerwise_comparison(states: Mapping[str, torch.Tensor], identities: Sequence[str]):
    rows = []
    for layer_index in range(states["source_only_eos"].size(0)):
        source = states["source_only_eos"][layer_index]
        target = states["target_only_eos"][layer_index]
        prompt = states["autoregressive_prompt"][layer_index]
        product = states["autoregressive_product_mean"][layer_index]
        native_source = states["native_source_eos"][layer_index]
        grams = {
            "source": centered_linear_gram(source),
            "target": centered_linear_gram(target),
            "prompt": centered_linear_gram(prompt),
            "product": centered_linear_gram(product),
            "native_source": centered_linear_gram(native_source),
        }
        rows.append({
            "source_vs_native_prefix_cka": cka_from_grams(grams["source"], grams["native_source"]),
            "source_vs_native_prefix_mean_l2": float((source - native_source).norm(dim=1).mean()),
            "source_vs_target_only_cka": cka_from_grams(grams["source"], grams["target"]),
            "source_vs_autoregressive_prompt_cka": cka_from_grams(grams["source"], grams["prompt"]),
            "source_vs_autoregressive_product_cka": cka_from_grams(grams["source"], grams["product"]),
            "target_only_vs_autoregressive_product_cka": cka_from_grams(grams["target"], grams["product"]),
            "source_to_target_only_linear": ridge_explained_variance(
                source, target, identities
            ),
            "source_to_autoregressive_product_linear": ridge_explained_variance(
                source, product, identities
            ),
            "source_to_target_only_retrieval": retrieval_metrics(source, target),
            "source_to_autoregressive_product_retrieval": retrieval_metrics(
                source, product
            ),
        })
    return rows


def molecule_scaffold(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return ""
    scaffold = MurckoScaffold.GetScaffoldForMol(molecule)
    return Chem.MolToSmiles(scaffold, isomericSmiles=True) if scaffold.GetNumAtoms() else ""


def hard_negative_candidates(rows, tokenizer, negatives: int = 3):
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    target_smiles = [canonicalize(row["target"]) for row in rows]
    molecules = [Chem.MolFromSmiles(value) for value in target_smiles]
    if any(molecule is None for molecule in molecules):
        raise ValueError("invalid product in hard-negative panel")
    fingerprints = [generator.GetFingerprint(molecule) for molecule in molecules]
    atoms = [molecule.GetNumHeavyAtoms() for molecule in molecules]
    token_lengths = [
        len(value)
        for value in tokenizer(target_smiles, add_special_tokens=False)["input_ids"]
    ]
    character_lengths = [len(value) for value in target_smiles]
    scaffolds = [molecule_scaffold(value) for value in target_smiles]
    candidates = []
    diagnostics = []
    for left in range(len(rows)):
        pool = []
        for right in range(len(rows)):
            if left == right:
                continue
            atom_difference = abs(atoms[left] - atoms[right])
            token_difference = abs(token_lengths[left] - token_lengths[right])
            character_difference = abs(character_lengths[left] - character_lengths[right])
            tanimoto = DataStructs.TanimotoSimilarity(
                fingerprints[left], fingerprints[right]
            )
            size_matched = (
                atom_difference <= max(2, math.ceil(0.12 * atoms[left]))
                and token_difference <= max(4, math.ceil(0.12 * token_lengths[left]))
            )
            pool.append((
                not size_matched,
                -tanimoto,
                0 if scaffolds[left] and scaffolds[left] == scaffolds[right] else 1,
                atom_difference,
                token_difference,
                character_difference,
                rows[right]["reaction_identity"],
                right,
                tanimoto,
            ))
        selected = sorted(pool)[:negatives]
        candidates.append([left] + [entry[7] for entry in selected])
        diagnostics.extend({
            "query_index": left,
            "negative_index": entry[7],
            "size_matched": not entry[0],
            "morgan_tanimoto": entry[8],
            "same_scaffold": entry[2] == 0,
            "heavy_atom_difference": entry[3],
            "token_length_difference": entry[4],
            "character_length_difference": entry[5],
        } for entry in selected)
    return candidates, {
        "negatives_per_query": negatives,
        "pairs": len(diagnostics),
        "size_matched_fraction": statistics.fmean(float(row["size_matched"]) for row in diagnostics),
        "same_scaffold_fraction": statistics.fmean(float(row["same_scaffold"]) for row in diagnostics),
        "mean_morgan_tanimoto": statistics.fmean(row["morgan_tanimoto"] for row in diagnostics),
        "median_morgan_tanimoto": statistics.median(row["morgan_tanimoto"] for row in diagnostics),
        "mean_heavy_atom_difference": statistics.fmean(row["heavy_atom_difference"] for row in diagnostics),
        "mean_token_length_difference": statistics.fmean(row["token_length_difference"] for row in diagnostics),
        "mean_character_length_difference": statistics.fmean(row["character_length_difference"] for row in diagnostics),
    }


def run_representations(output_dir: Path, batch_size: int) -> Path:
    rows = read_representation_panel()
    identities = [row["reaction_identity"] for row in rows]
    tokenizer, _, collator, model, controls = prepare_model()
    native_state = load_adapter_state(NATIVE_CHECKPOINT)
    clm_state = load_adapter_state(CLM_CHECKPOINT)
    hard_candidates, hard_diagnostics = hard_negative_candidates(rows, tokenizer)
    alternate_rows = canonicalized_rows(rows)
    alternate_hard_candidates, alternate_hard_diagnostics = hard_negative_candidates(
        alternate_rows, tokenizer
    )
    results = {}
    started = time.perf_counter()
    for label, state in (("native", native_state), ("mse_sigreg", clm_state)):
        apply_adapter_state(model, state, label)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        layer_labels, aligned = collect_layerwise_views(
            model, collator, rows, batch_size
        )
        _, alternate = collect_layerwise_views(
            model, collator, alternate_rows, batch_size, final_only=True
        )
        aligned_final_source = aligned["source_only_eos"][-1]
        aligned_final_target = aligned["target_only_eos"][-1]
        alternate_final_source = alternate["source_only_eos"][-1]
        alternate_final_target = alternate["target_only_eos"][-1]
        results[label] = {
            "layerwise": layerwise_comparison(aligned, identities),
            "shortcut_controls": {
                "aligned_r_smiles": {
                    "full_panel": retrieval_metrics(
                        aligned_final_source, aligned_final_target
                    ),
                    "hard_four_way": retrieval_metrics(
                        aligned_final_source, aligned_final_target, hard_candidates
                    ),
                },
                "independent_canonical_smiles": {
                    "full_panel": retrieval_metrics(
                        alternate_final_source, alternate_final_target
                    ),
                    "hard_four_way": retrieval_metrics(
                        alternate_final_source,
                        alternate_final_target,
                        alternate_hard_candidates,
                    ),
                },
            },
            "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()),
        }
        print(json.dumps({"completed": label, "seconds": time.perf_counter() - started}), flush=True)
    output = {
        "scope": "frozen layerwise representation and shortcut-control audit; no optimizer",
        "seed": SEED,
        "checkpoints": {
            "native": str(NATIVE_CHECKPOINT.resolve()),
            "mse_sigreg": str(CLM_CHECKPOINT.resolve()),
        },
        "panel": str(REPRESENTATION_PANEL.resolve()),
        "panel_sha256": file_sha256(REPRESENTATION_PANEL),
        "reactions": len(rows),
        "layer_index_semantics": "0 is embedding output; 1..22 are transformer-block outputs",
        "autoregressive_product_representation": "mean hidden state at positions predicting product tokens and EOS in the teacher-forced native row",
        "linear_cka": "centered linear CKA via sample Gram matrices",
        "linear_prediction_split": "last 20% of sorted reaction identities held out; dual ridge alpha=1",
        "hard_negative_construction": {
            "aligned_r_smiles": hard_diagnostics,
            "independent_canonical_smiles": alternate_hard_diagnostics,
        },
        "stochastic_controls": controls,
        "layers": layer_labels,
        "conditions": results,
        "runtime_seconds": time.perf_counter() - started,
        "gpu": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
    }
    return write_json(output_dir / "representation_pathway.json", output)


def canonical_fingerprint(smiles: str, generator):
    molecule = Chem.MolFromSmiles(smiles)
    return None if molecule is None else generator.GetFingerprint(molecule)


def chemistry_rows(records: Sequence[Mapping[str, Any]]):
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    output = []
    for record in records:
        target = canonicalize(record["target"])
        target_fp = canonical_fingerprint(target, generator)
        if target_fp is None:
            raise ValueError(f"invalid target for {record['reaction_identity']}")
        target_scaffold = molecule_scaffold(target)
        similarities = []
        scaffolds = []
        for candidate in record["ranked_candidates"][:10]:
            canonical = canonicalize(candidate) if candidate else ""
            fingerprint = canonical_fingerprint(canonical, generator) if canonical else None
            similarities.append(
                DataStructs.TanimotoSimilarity(target_fp, fingerprint)
                if fingerprint is not None else 0.0
            )
            scaffold = molecule_scaffold(canonical) if canonical else ""
            scaffolds.append(bool(target_scaffold) and scaffold == target_scaffold)
        similarities.extend([0.0] * (10 - len(similarities)))
        scaffolds.extend([False] * (10 - len(scaffolds)))
        output.append({
            "reaction_identity": record["reaction_identity"],
            "top1_exact": bool(record["ranked_candidates"] and canonicalize(record["ranked_candidates"][0]) == target),
            "top1_tanimoto": similarities[0],
            "best_top3_tanimoto": max(similarities[:3]),
            "best_top5_tanimoto": max(similarities[:5]),
            "best_top10_tanimoto": max(similarities[:10]),
            "top1_scaffold_match": scaffolds[0],
            "any_top3_scaffold_match": any(scaffolds[:3]),
            "any_top5_scaffold_match": any(scaffolds[:5]),
            "any_top10_scaffold_match": any(scaffolds[:10]),
            "target_has_scaffold": bool(target_scaffold),
        })
    return output


def run_chemistry(output_dir: Path) -> Path:
    native_records = sorted(read_jsonl(NATIVE_PREDICTIONS), key=lambda row: row["panel_index"])
    clm_records = sorted(read_jsonl(CLM_PREDICTIONS), key=lambda row: row["panel_index"])
    native_ids = [row["reaction_identity"] for row in native_records]
    if native_ids != [row["reaction_identity"] for row in clm_records]:
        raise ValueError("native and cLM official predictions are not identity aligned")
    native = chemistry_rows(native_records)
    clm = chemistry_rows(clm_records)
    metrics = (
        "top1_tanimoto",
        "best_top3_tanimoto",
        "best_top5_tanimoto",
        "best_top10_tanimoto",
        "top1_scaffold_match",
        "any_top3_scaffold_match",
        "any_top5_scaffold_match",
        "any_top10_scaffold_match",
    )
    paired = {
        metric: paired_summary(
            [float(row[metric]) for row in clm],
            [float(row[metric]) for row in native],
            seed=SEED + index,
        )
        for index, metric in enumerate(metrics)
    }
    neither_exact = [
        index for index, (left, right) in enumerate(zip(native, clm))
        if not left["top1_exact"] and not right["top1_exact"]
    ]
    paired_nonexact = {
        metric: paired_summary(
            [float(clm[index][metric]) for index in neither_exact],
            [float(native[index][metric]) for index in neither_exact],
            seed=SEED + 100 + position,
        )
        for position, metric in enumerate(metrics)
    }
    output = {
        "scope": "chemistry-aware rescoring of existing predictions only",
        "native_predictions": str(NATIVE_PREDICTIONS.resolve()),
        "native_predictions_sha256": file_sha256(NATIVE_PREDICTIONS),
        "mse_sigreg_predictions": str(CLM_PREDICTIONS.resolve()),
        "mse_sigreg_predictions_sha256": file_sha256(CLM_PREDICTIONS),
        "reactions": len(native),
        "fingerprint": "RDKit Morgan radius 2, 2048 bits",
        "scaffold": "exact isomeric Bemis-Murcko scaffold; empty scaffolds never match",
        "paired_all_reactions": paired,
        "jointly_nonexact_top1": {
            "reactions": len(neither_exact),
            "paired": paired_nonexact,
        },
        "exact_outcomes": {
            "native_top1": sum(row["top1_exact"] for row in native),
            "mse_sigreg_top1": sum(row["top1_exact"] for row in clm),
            "both": sum(a["top1_exact"] and b["top1_exact"] for a, b in zip(native, clm)),
            "native_only": sum(a["top1_exact"] and not b["top1_exact"] for a, b in zip(native, clm)),
            "mse_sigreg_only": sum(b["top1_exact"] and not a["top1_exact"] for a, b in zip(native, clm)),
        },
        "target_scaffold_available": sum(row["target_has_scaffold"] for row in native),
        "per_reaction": {
            "native": native,
            "mse_sigreg": clm,
        },
    }
    return write_json(output_dir / "chemical_similarity.json", output)


def target_prediction_mask(labels: torch.Tensor) -> torch.Tensor:
    """Hidden positions whose logits predict supervised target labels."""
    mask = torch.zeros_like(labels, dtype=torch.bool)
    mask[:, :-1] = labels[:, 1:].ne(IGNORE_INDEX)
    return mask


def per_reaction_ce(logits: torch.Tensor, labels: torch.Tensor, identities: Sequence[str]):
    shifted_logits = logits[:, :-1].float()
    shifted_labels = labels[:, 1:]
    rows = []
    for identity, row_logits, row_labels in zip(identities, shifted_logits, shifted_labels):
        active = row_labels.ne(IGNORE_INDEX)
        nll = F.cross_entropy(row_logits[active], row_labels[active], reduction="sum")
        tokens = int(active.sum())
        rows.append({
            "reaction_identity": identity,
            "target_nll": float(nll),
            "target_tokens": tokens,
            "target_ce": float(nll) / tokens,
        })
    return rows


def aggregate_ce(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    total_nll = sum(float(row["target_nll"]) for row in rows)
    total_tokens = sum(int(row["target_tokens"]) for row in rows)
    return {
        "target_ce": total_nll / total_tokens,
        "mean_reaction_ce": statistics.fmean(float(row["target_ce"]) for row in rows),
        "target_tokens": total_tokens,
        "reactions": list(rows),
    }


@torch.inference_mode()
def capture_activations(model, collator, rows, batch_size: int, boundaries: Sequence[int]):
    layers = llama_backbone(model).layers
    captures = {boundary: [] for boundary in boundaries}
    batches = []
    for start in range(0, len(rows), batch_size):
        subset = rows[start:start + batch_size]
        raw = collator([{"src": row["source"], "tgt": row["target"]} for row in subset])
        batch = device_tensors(raw, model.device)
        local: dict[int, torch.Tensor] = {}
        handles = []
        for boundary in boundaries:
            def hook(_module, _inputs, output, *, key=boundary):
                hidden = output[0] if isinstance(output, tuple) else output
                local[key] = hidden.detach().cpu()
            handles.append(layers[boundary].register_forward_hook(hook))
        try:
            llama_backbone(model)(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=False,
                return_dict=True,
            )
        finally:
            for handle in handles:
                handle.remove()
        for boundary in boundaries:
            captures[boundary].append(local[boundary])
        batches.append({
            "raw": raw,
            "identities": [row["reaction_identity"] for row in subset],
        })
    return batches, captures


@torch.inference_mode()
def evaluate_activation_condition(
    model,
    batches,
    *,
    donor_activations: Sequence[torch.Tensor] | None = None,
    boundary: int | None = None,
    position_scope: str = "all",
):
    if (donor_activations is None) != (boundary is None):
        raise ValueError("donor activations and boundary must be supplied together")
    if position_scope not in {"all", "context_only", "target_prediction_only"}:
        raise ValueError(f"unknown position scope {position_scope}")
    layers = llama_backbone(model).layers
    result = []
    for batch_index, entry in enumerate(batches):
        raw = entry["raw"]
        batch = device_tensors(raw, model.device)
        handle = None
        if boundary is not None:
            donor = donor_activations[batch_index].to(model.device)
            prediction_mask = target_prediction_mask(batch["labels"])

            def patch_hook(_module, _inputs, output):
                recipient = output[0] if isinstance(output, tuple) else output
                if donor.shape != recipient.shape:
                    raise RuntimeError(
                        f"donor/recipient activation mismatch {donor.shape} != {recipient.shape}"
                    )
                if position_scope == "all":
                    patched = donor.to(recipient.dtype)
                else:
                    selected = (
                        prediction_mask
                        if position_scope == "target_prediction_only"
                        else batch["attention_mask"].bool() & ~prediction_mask
                    )
                    patched = recipient.clone()
                    patched[selected] = donor.to(recipient.dtype)[selected]
                if isinstance(output, tuple):
                    return (patched,) + output[1:]
                return patched

            handle = layers[boundary].register_forward_hook(patch_hook)
        try:
            output = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=False,
                return_dict=True,
            )
            result.extend(per_reaction_ce(output.logits, batch["labels"], entry["identities"]))
        finally:
            if handle is not None:
                handle.remove()
    return aggregate_ce(result)


def paired_ce_change(candidate, reference, *, seed: int):
    candidate_by_id = {
        row["reaction_identity"]: float(row["target_ce"])
        for row in candidate["reactions"]
    }
    reference_by_id = {
        row["reaction_identity"]: float(row["target_ce"])
        for row in reference["reactions"]
    }
    identities = list(reference_by_id)
    differences = [candidate_by_id[key] - reference_by_id[key] for key in identities]
    return {
        "aggregate_target_ce_change": candidate["target_ce"] - reference["target_ce"],
        "mean_reaction_ce_change": statistics.fmean(differences),
        "mean_reaction_ce_change_bootstrap_95_ci": bootstrap_mean_ci(
            differences, seed=seed
        ),
        "improved_reactions": sum(value < 0.0 for value in differences),
        "worsened_reactions": sum(value > 0.0 for value in differences),
        "tied_reactions": sum(value == 0.0 for value in differences),
    }


def run_activation_patching(
    output_dir: Path,
    batch_size: int,
    limit: int,
    position_refine: bool,
) -> Path:
    rows = read_representation_panel(limit)
    tokenizer, _, collator, model, controls = prepare_model()
    del tokenizer
    native_state = load_adapter_state(NATIVE_CHECKPOINT)
    clm_state = load_adapter_state(CLM_CHECKPOINT)
    # Boundary labels are the outputs of the listed zero-based blocks.
    # 11 -> before the implicated 12-16 segment; 16 -> before 17-21;
    # 21 -> after every transformer block, before final norm/LM head.
    boundaries = (11, 16, 21)
    started = time.perf_counter()

    apply_adapter_state(model, native_state, "native")
    native_batches, native_activations = capture_activations(
        model, collator, rows, batch_size, boundaries
    )
    native_baseline = evaluate_activation_condition(model, native_batches)

    apply_adapter_state(model, clm_state, "mse_sigreg")
    clm_batches, clm_activations = capture_activations(
        model, collator, rows, batch_size, boundaries
    )
    clm_baseline = evaluate_activation_condition(model, clm_batches)

    base_path = output_dir / "activation_patching.json"
    if position_refine and base_path.exists():
        base = json.loads(base_path.read_text(encoding="utf-8"))
        results = dict(base["results"])
        comparisons = dict(base["paired_changes_vs_recipient"])
    else:
        results = {
            "full_native": native_baseline,
            "full_mse_sigreg": clm_baseline,
        }
        comparisons = {}
        apply_adapter_state(model, native_state, "native")
        for boundary in boundaries:
            label = f"native_with_mse_sigreg_activation_after_layer_{boundary}"
            results[label] = evaluate_activation_condition(
                model,
                native_batches,
                donor_activations=clm_activations[boundary],
                boundary=boundary,
            )
            comparisons[label] = paired_ce_change(
                results[label], native_baseline, seed=SEED + boundary
            )

        apply_adapter_state(model, clm_state, "mse_sigreg")
        for boundary in boundaries:
            label = f"mse_sigreg_with_native_activation_after_layer_{boundary}"
            results[label] = evaluate_activation_condition(
                model,
                clm_batches,
                donor_activations=native_activations[boundary],
                boundary=boundary,
            )
            comparisons[label] = paired_ce_change(
                results[label], clm_baseline, seed=SEED + 100 + boundary
            )

    refinements = {}
    if position_refine:
        # Refine only the boundary immediately before the harmful 17-21 region.
        boundary = 16
        for recipient, state, batches, donor, reference in (
            ("native", native_state, native_batches, clm_activations, native_baseline),
            ("mse_sigreg", clm_state, clm_batches, native_activations, clm_baseline),
        ):
            apply_adapter_state(model, state, recipient)
            donor_label = "mse_sigreg" if recipient == "native" else "native"
            for scope_index, scope in enumerate(("context_only", "target_prediction_only")):
                label = f"{recipient}_with_{donor_label}_activation_after_layer_{boundary}_{scope}"
                results[label] = evaluate_activation_condition(
                    model,
                    batches,
                    donor_activations=donor[boundary],
                    boundary=boundary,
                    position_scope=scope,
                )
                refinements[label] = paired_ce_change(
                    results[label], reference,
                    seed=SEED + 200 + 10 * scope_index + (0 if recipient == "native" else 1),
                )

    output = {
        "scope": "frozen cross-checkpoint activation patching; no parameter changes",
        "seed": SEED,
        "checkpoints": {
            "native": str(NATIVE_CHECKPOINT.resolve()),
            "mse_sigreg": str(CLM_CHECKPOINT.resolve()),
        },
        "panel": str(REPRESENTATION_PANEL.resolve()),
        "panel_sha256": file_sha256(REPRESENTATION_PANEL),
        "reactions": len(rows),
        "batch_size": batch_size,
        "boundaries": list(boundaries),
        "boundary_semantics": {
            "11": "after layer 11, before implicated beneficial layers 12-16",
            "16": "after layer 16, before implicated harmful layers 17-21",
            "21": "after final transformer layer, before final norm and LM head",
        },
        "position_refinement": bool(position_refine),
        "stochastic_controls": controls,
        "parameter_sha256_after": parameter_fingerprint(model),
        "results": results,
        "paired_changes_vs_recipient": comparisons,
        "position_refinements_vs_recipient": refinements,
        "runtime_seconds": time.perf_counter() - started,
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()),
        "gpu": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
    }
    suffix = "_position_refined" if position_refine else ""
    return write_json(output_dir / f"activation_patching{suffix}.json", output)


def named_trainable(model):
    return [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]


def empty_gradient_buffers(parameters: Sequence[torch.nn.Parameter]):
    return [None for _ in parameters]


def accumulate_gradients(buffers, gradients):
    for index, gradient in enumerate(gradients):
        if gradient is None:
            continue
        detached = gradient.detach()
        if buffers[index] is None:
            buffers[index] = detached.clone()
        else:
            buffers[index].add_(detached)


def complete_cpu_vector(names, parameters, buffers):
    return {
        name: (
            torch.zeros(parameter.shape, dtype=torch.float32)
            if gradient is None else gradient.detach().cpu().float().clone()
        )
        for name, parameter, gradient in zip(names, parameters, buffers)
    }


def vector_relation(first, second, names: Iterable[str] | None = None):
    chosen = sorted(set(first) & set(second) if names is None else set(names) & set(first) & set(second))
    dot = sum(float(torch.dot(first[name].flatten(), second[name].flatten())) for name in chosen)
    first_sq = sum(float(first[name].square().sum()) for name in chosen)
    second_sq = sum(float(second[name].square().sum()) for name in chosen)
    first_norm = math.sqrt(max(0.0, first_sq))
    second_norm = math.sqrt(max(0.0, second_sq))
    return {
        "parameter_tensors": len(chosen),
        "dot": dot,
        "first_norm": first_norm,
        "second_norm": second_norm,
        "first_over_second_norm": first_norm / second_norm if second_norm else None,
        "cosine": dot / (first_norm * second_norm) if first_norm and second_norm else None,
    }


def vector_add(*terms):
    names = set().union(*(vector for vector, _ in terms))
    return {
        name: sum(
            (coefficient * vector[name] for vector, coefficient in terms),
            torch.tensor(0.0),
        )
        for name in names
    }


def vector_dot(first, second):
    return sum(float(torch.dot(first[name].flatten(), second[name].flatten())) for name in set(first) & set(second))


def evaluate_chunked_ntp(model, chunks, *, with_grad: bool):
    named = named_trainable(model)
    parameters = tuple(parameter for _, parameter in named)
    buffers = empty_gradient_buffers(parameters)
    losses = []
    context = torch.enable_grad() if with_grad else torch.inference_mode()
    with context:
        for raw in chunks:
            batch = device_tensors(raw, model.device)
            output = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                use_cache=False,
                return_dict=True,
            )
            component = output.loss / len(chunks)
            losses.append(float(output.loss.detach()))
            if with_grad:
                gradients = torch.autograd.grad(component, parameters, allow_unused=True)
                accumulate_gradients(buffers, gradients)
    return statistics.fmean(losses), (complete_cpu_vector(
        [name for name, _ in named], parameters, buffers
    ) if with_grad else None)


def exact_training_gradients(model, method, chunks):
    named = named_trainable(model)
    names = [name for name, _ in named]
    parameters = tuple(parameter for _, parameter in named)
    source_chunks = []
    target_chunks = []
    with torch.no_grad():
        for raw in chunks:
            output = method(
                model,
                device_tensors(raw, model.device),
                k=0,
                jepa_weight=0.0,
                native_weight=1.0,
                monitor_only=True,
                stop_gradient_target=False,
                jepa_loss_type="mse",
                sigreg_tradeoff=0.0,
                jepa_ratio=1.0,
                force_jepa_active=True,
                endpoint_only=True,
            )
            source_chunks.append(output.source_states.float())
            target_chunks.append(output.target_states.float())
    auxiliary = raw_auxiliary_vjp(
        method.sigreg,
        torch.cat(source_chunks),
        torch.cat(target_chunks),
        sigreg_coefficient=SIGREG_RELATIVE_COEFFICIENT,
    )
    source_vjps = auxiliary["source_gradients"].split(len(chunks[0]["input_ids"]))
    target_vjps = auxiliary["target_gradients"].split(len(chunks[0]["input_ids"]))
    main_buffers = empty_gradient_buffers(parameters)
    auxiliary_buffers = empty_gradient_buffers(parameters)
    native_losses = []
    for chunk_index, raw in enumerate(chunks):
        output = method(
            model,
            device_tensors(raw, model.device),
            k=0,
            jepa_weight=0.0,
            native_weight=1.0,
            monitor_only=True,
            stop_gradient_target=False,
            jepa_loss_type="mse",
            sigreg_tradeoff=0.0,
            jepa_ratio=1.0,
            force_jepa_active=True,
            representation_only=True,
        )
        native_losses.append(float(output.native_loss.detach()))
        main = torch.autograd.grad(
            output.native_loss / len(chunks), parameters,
            retain_graph=True, allow_unused=True,
        )
        surrogate = (
            output.source_states
            * source_vjps[chunk_index].to(output.source_states.dtype)
        ).sum() + (
            output.target_states
            * target_vjps[chunk_index].to(output.target_states.dtype)
        ).sum()
        aux = torch.autograd.grad(surrogate, parameters, allow_unused=True)
        accumulate_gradients(main_buffers, main)
        accumulate_gradients(auxiliary_buffers, aux)
    main_vector = complete_cpu_vector(names, parameters, main_buffers)
    auxiliary_raw = complete_cpu_vector(names, parameters, auxiliary_buffers)
    auxiliary_active = {
        name: ACTIVE_AUXILIARY_COEFFICIENT * value
        for name, value in auxiliary_raw.items()
    }
    return {
        "native_loss": statistics.fmean(native_losses),
        "mse_loss": float(auxiliary["mse"]),
        "sigreg_loss": float(auxiliary["sigreg"]),
        "auxiliary_loss": float(auxiliary["objective"]),
        "main": main_vector,
        "auxiliary_active": auxiliary_active,
        "main_buffers": main_buffers,
        "auxiliary_buffers": [
            None if gradient is None else ACTIVE_AUXILIARY_COEFFICIENT * gradient
            for gradient in auxiliary_buffers
        ],
    }


def restore_trainable(parameters, initial):
    with torch.no_grad():
        for parameter, value in zip(parameters, initial):
            parameter.copy_(value)


def adamw_virtual_step(
    model,
    optimizer_state,
    initial_parameters,
    gradients,
    heldout_chunks,
):
    named = named_trainable(model)
    names = [name for name, _ in named]
    parameters = [parameter for _, parameter in named]
    restore_trainable(parameters, initial_parameters)
    optimizer = torch.optim.AdamW(
        parameters,
        lr=1e-4,
        betas=ADAM_BETAS,
        eps=ADAM_EPSILON,
        weight_decay=WEIGHT_DECAY,
        fused=True,
    )
    optimizer.load_state_dict(copy.deepcopy(optimizer_state))
    for parameter, gradient in zip(parameters, gradients):
        parameter.grad = None if gradient is None else gradient.detach().clone().to(parameter.dtype)
    raw_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
    optimizer.step()
    update = {
        name: (parameter.detach() - before).cpu().float()
        for name, parameter, before in zip(names, parameters, initial_parameters)
    }
    heldout_loss, _ = evaluate_chunked_ntp(model, heldout_chunks, with_grad=False)
    optimizer.zero_grad(set_to_none=True)
    del optimizer
    return {
        "gradient_norm_before_clip": float(raw_norm),
        "heldout_loss": heldout_loss,
        "update": update,
    }


def run_optimizer_counterfactual(output_dir: Path, physical_batch: int) -> Path:
    if 16 % physical_batch:
        raise ValueError("physical batch must divide logical batch 16")
    tokenizer, predictor_ids, collator, model, controls = prepare_model()
    training_rows = read_rows("uspto_mit_synthesis", path=TRAIN_MANIFEST)
    validation_rows = read_rows("uspto_mit_synthesis", path=REPRESENTATION_PANEL)
    train_indices, train_batches = fixed_batches(
        training_rows, collator, count=1, physical_batch=physical_batch
    )
    validation_indices, validation_batches = build_validation_batches(
        validation_rows, collator, count=1, physical_batch=physical_batch
    )
    train_chunks = train_batches[0]["chunks"]
    heldout_chunks = validation_batches[0]["chunks"]
    validate_serialization_endings(
        collator, train_batches[0]["rows"], tokenizer.eos_token_id
    )
    validate_serialization_endings(
        collator, validation_batches[0]["rows"], tokenizer.eos_token_id
    )

    apply_adapter_state(model, load_adapter_state(CLM_CHECKPOINT), "mse_sigreg")
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.enable_input_require_grads()
    model.train()
    disable_stochastic_behavior(model)
    method = CLMJEPA(
        predictor_ids,
        tokenizer.eos_token_id,
        tokenizer.pad_token_id,
        sigreg_seed=SEED,
    )
    state = torch.load(
        CLM_CHECKPOINT / "training_state.pt", map_location="cpu", weights_only=False
    )
    method.sigreg.global_step = int(state["sigreg_global_step"])
    named = named_trainable(model)
    names = [name for name, _ in named]
    parameters = [parameter for _, parameter in named]
    initial_parameters = [parameter.detach().clone() for parameter in parameters]
    initial_fingerprint = parameter_fingerprint(model)
    started = time.perf_counter()

    heldout_loss, heldout_gradient = evaluate_chunked_ntp(
        model, heldout_chunks, with_grad=True
    )
    gradients = exact_training_gradients(model, method, train_chunks)
    lora_names = [name for name in names if "lora_A" in name or "lora_B" in name]
    raw_relationships = {
        "all_trainable_main_vs_auxiliary": vector_relation(
            gradients["main"], gradients["auxiliary_active"]
        ),
        "lora_main_vs_auxiliary": vector_relation(
            gradients["main"], gradients["auxiliary_active"], lora_names
        ),
        "main_vs_heldout_ntp": vector_relation(
            gradients["main"], heldout_gradient
        ),
        "auxiliary_vs_heldout_ntp": vector_relation(
            gradients["auxiliary_active"], heldout_gradient
        ),
    }

    conditions = {
        "ntp_only": gradients["main_buffers"],
        "jepa_only": gradients["auxiliary_buffers"],
        "ntp_plus_jepa": [
            (None if main is None and auxiliary is None else
             (torch.zeros_like(parameter) if main is None else main)
             + (torch.zeros_like(parameter) if auxiliary is None else auxiliary))
            for parameter, main, auxiliary in zip(
                parameters, gradients["main_buffers"], gradients["auxiliary_buffers"]
            )
        ],
    }
    results = {}
    updates = {}
    for label, condition_gradients in conditions.items():
        result = adamw_virtual_step(
            model,
            state["optimizer"],
            initial_parameters,
            condition_gradients,
            heldout_chunks,
        )
        update = result.pop("update")
        updates[label] = update
        predicted_change = vector_dot(heldout_gradient, update)
        observed_change = result["heldout_loss"] - heldout_loss
        results[label] = {
            **result,
            "first_order_predicted_heldout_ntp_change": predicted_change,
            "observed_heldout_ntp_change": observed_change,
            "observed_minus_first_order": observed_change - predicted_change,
            "update_vs_negative_heldout_gradient": vector_relation(
                update,
                {name: -value for name, value in heldout_gradient.items()},
            ),
        }
        print(json.dumps({"completed": label, "observed_change": observed_change}), flush=True)

    update_relationships = {
        "ntp_vs_jepa": vector_relation(updates["ntp_only"], updates["jepa_only"]),
        "combined_vs_sum_of_separate": vector_relation(
            updates["ntp_plus_jepa"],
            vector_add((updates["ntp_only"], 1.0), (updates["jepa_only"], 1.0)),
        ),
        "combined_minus_ntp": vector_relation(
            vector_add(
                (updates["ntp_plus_jepa"], 1.0),
                (updates["ntp_only"], -1.0),
            ),
            updates["jepa_only"],
        ),
    }
    restore_trainable(parameters, initial_parameters)
    final_fingerprint = parameter_fingerprint(model)
    if final_fingerprint != initial_fingerprint:
        raise RuntimeError("virtual optimizer audit did not restore its checkpoint")
    output = {
        "scope": "exact one-step counterfactuals from frozen epoch-4 AdamW state; no persistent update",
        "checkpoint": str(CLM_CHECKPOINT.resolve()),
        "checkpoint_adapter_sha256": file_sha256(
            adapter_weights_dir(CLM_CHECKPOINT) / "adapter_model.safetensors"
        ),
        "training_state_sha256": file_sha256(CLM_CHECKPOINT / "training_state.pt"),
        "optimizer_global_step": int(state["global_step"]),
        "optimizer_lr": float(state["optimizer"]["param_groups"][0]["lr"]),
        "optimizer_betas": list(state["optimizer"]["param_groups"][0]["betas"]),
        "optimizer_eps": float(state["optimizer"]["param_groups"][0]["eps"]),
        "optimizer_weight_decay": float(state["optimizer"]["param_groups"][0]["weight_decay"]),
        "gradient_clipping_max_norm": 1.0,
        "active_auxiliary_formula": "2 * (MSE + 4*0.01/0.99 * SIGReg)",
        "sigreg_global_step": int(state["sigreg_global_step"]),
        "train_manifest": str(TRAIN_MANIFEST.resolve()),
        "train_indices_zero_based": train_indices,
        "heldout_manifest": str(REPRESENTATION_PANEL.resolve()),
        "heldout_indices_zero_based": validation_indices,
        "physical_batch": physical_batch,
        "logical_batch": 16,
        "training_objectives": {
            "native_loss": gradients["native_loss"],
            "mse_loss": gradients["mse_loss"],
            "sigreg_loss": gradients["sigreg_loss"],
            "unweighted_auxiliary_loss": gradients["auxiliary_loss"],
        },
        "heldout_ntp_before": heldout_loss,
        "raw_gradient_relationships": raw_relationships,
        "adamw_update_relationships": update_relationships,
        "counterfactuals": results,
        "stochastic_controls": controls,
        "checkpoint_restored_exactly": True,
        "runtime_seconds": time.perf_counter() - started,
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()),
        "gpu": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
    }
    return write_json(output_dir / "optimizer_counterfactual.json", output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("representations", "activation", "optimizer", "chemistry"),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--position-refine", action="store_true")
    args = parser.parse_args()
    if args.command == "representations":
        path = run_representations(args.output_dir, args.batch_size)
    elif args.command == "activation":
        path = run_activation_patching(
            args.output_dir, args.batch_size, args.limit, args.position_refine
        )
    elif args.command == "optimizer":
        path = run_optimizer_counterfactual(args.output_dir, args.batch_size)
    else:
        path = run_chemistry(args.output_dir)
    print(path)


if __name__ == "__main__":
    main()
