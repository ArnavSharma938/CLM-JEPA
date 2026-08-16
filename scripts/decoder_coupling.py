from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import binomtest, mannwhitneyu, spearmanr, wilcoxon
from torch.nn.utils.rnn import pad_sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chemfm import (
    IGNORE_INDEX,
    MODEL_DIR,
    TOKENIZER_DIR,
    ReactionCollator,
    canonicalize,
    generate_products,
    load_lora_model,
    load_reaction_tokenizer,
)
from jepa import add_predictor_tokens, extract_source_and_target
from metrics import identity_mappings, rank_augmented_candidates, score_candidates
from train import (
    _heavy_atoms,
    _largest_target_overlap_component,
    file_sha256,
    load_adapter_checkpoint,
)

SOURCE_VALIDATION = ROOT / "data" / "uspto_mit_synthesis" / "validation_r_smiles.csv"
PANEL_DIR = ROOT / "data" / "clm_jepa_uspto_mit_validation_1024"
PANEL_PATH = PANEL_DIR / "uspto_mit_validation_1024.csv"
PANEL_METADATA = PANEL_DIR / "manifest.json"
GROUP_SIZE = 5
PANEL_SIZE = 1024
SEED = 533


def canonical_source(smiles: str) -> str:
    components = [canonicalize(component) for component in smiles.split(".")]
    return ".".join(sorted(components)) if components and all(components) else ""


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prepare_panel() -> dict:
    if PANEL_PATH.exists() and PANEL_METADATA.exists():
        metadata = json.loads(PANEL_METADATA.read_text(encoding="utf-8"))
        if file_sha256(PANEL_PATH) != metadata["manifest_sha256"]:
            raise RuntimeError("existing decoder-coupling panel hash does not match metadata")
        return metadata
    if PANEL_PATH.exists() or PANEL_METADATA.exists():
        raise RuntimeError("partial panel artifacts exist; refusing to overwrite")

    candidates = []
    with SOURCE_VALIDATION.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        group = []
        group_index = 0
        for row in reader:
            group.append(row)
            if len(group) != GROUP_SIZE:
                continue
            score = hashlib.sha256(f"{SEED}|validation|{group_index}".encode()).digest()
            candidates.append((score, group_index, [dict(item) for item in group]))
            group = []
            group_index += 1
        if group:
            raise ValueError("validation row count is not divisible by five")

    selected = []
    seen_targets: set[str] = set()
    seen_reactions: set[str] = set()
    audit = {
        "source_groups_scanned": len(candidates),
        "invalid_groups_skipped": 0,
        "augmentation_identity_mismatch_groups_skipped": 0,
        "duplicate_target_identity_groups_skipped": 0,
        "duplicate_reaction_identity_groups_skipped": 0,
    }
    for _, group_index, group in sorted(candidates):
        target_ids = [canonicalize(item["target"]) for item in group]
        source_ids = [canonical_source(item["source"]) for item in group]
        if any(not value for value in target_ids + source_ids):
            audit["invalid_groups_skipped"] += 1
            continue
        if len(set(target_ids)) != 1 or len(set(source_ids)) != 1:
            audit["augmentation_identity_mismatch_groups_skipped"] += 1
            continue
        target_identity = target_ids[0]
        source_identity = source_ids[0]
        reaction_identity = f"{source_identity}>>{target_identity}"
        if target_identity in seen_targets:
            audit["duplicate_target_identity_groups_skipped"] += 1
            continue
        if reaction_identity in seen_reactions:
            audit["duplicate_reaction_identity_groups_skipped"] += 1
            continue
        seen_targets.add(target_identity)
        seen_reactions.add(reaction_identity)
        row = group[0]
        selected.append({
            **row,
            "group_id": f"validation-group-{group_index:09d}",
            "source_group": str(group_index),
            "augmentation_index": "0",
            "source_identity": source_identity,
            "target_identity": target_identity,
            "reaction_identity": reaction_identity,
            "selection_rank": str(len(selected)),
        })
        if len(selected) == PANEL_SIZE:
            break
    if len(selected) != PANEL_SIZE:
        raise RuntimeError(f"only {len(selected)} unique valid identities were found")

    write_csv(PANEL_PATH, selected)
    metadata = {
        "schema_version": 1,
        "dataset": "uspto_mit_synthesis",
        "split": "validation",
        "selection_seed": SEED,
        "selection": (
            "ascending sha256(533|validation|official_five-row-group-index); first official "
            "enumeration; require one unique canonical source/product reaction and product"
        ),
        "source_validation": str(SOURCE_VALIDATION.relative_to(ROOT)),
        "source_validation_sha256": file_sha256(SOURCE_VALIDATION),
        "identities": PANEL_SIZE,
        "official_augmentations_per_identity": GROUP_SIZE,
        "evaluated_augmentations_per_identity": 1,
        "audit": audit,
        "manifest": str(PANEL_PATH.relative_to(ROOT)),
        "manifest_sha256": file_sha256(PANEL_PATH),
    }
    PANEL_METADATA.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def read_panel() -> list[dict[str, str]]:
    prepare_panel()
    with PANEL_PATH.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != PANEL_SIZE:
        raise ValueError(f"expected {PANEL_SIZE} panel rows, got {len(rows)}")
    return rows


def read_analysis_panel(
    reference: Path | None = None, limit: int | None = None,
) -> list[dict[str, str]]:
    """Return the frozen parent panel or the exact identity subset in a JSONL artifact."""
    rows = read_panel()
    if reference is None:
        selected = rows
    else:
        records = sorted(read_jsonl(reference), key=lambda row: row["panel_index"])
        identities = [record["reaction_identity"] for record in records]
        if not identities or len(identities) != len(set(identities)):
            raise ValueError("panel reference must contain unique reaction identities")
        by_identity = {row["reaction_identity"]: row for row in rows}
        missing = [identity for identity in identities if identity not in by_identity]
        if missing:
            raise ValueError(f"panel reference contains {len(missing)} unknown identities")
        selected = [by_identity[identity] for identity in identities]
    if limit is not None:
        if limit < 1 or limit > len(selected):
            raise ValueError(f"panel limit must be in [1, {len(selected)}]")
        selected = selected[:limit]
    return selected


def load_model(checkpoint: Path):
    torch.manual_seed(SEED)
    tokenizer = load_reaction_tokenizer(TOKENIZER_DIR)
    chemfm_vocab_size = len(tokenizer)
    predictor_ids = add_predictor_tokens(tokenizer)
    model = load_lora_model(
        MODEL_DIR, tokenizer, chemfm_vocab_size=chemfm_vocab_size
    ).cuda().eval()
    load_adapter_checkpoint(model, checkpoint.resolve())
    return tokenizer, predictor_ids, model


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()


@torch.inference_mode()
def generate_equal_length_batch(model, tokenizer, prompts: list[str]) -> list[list[str]]:
    encoded = tokenizer(
        prompts, return_tensors="pt", padding=True, truncation=True,
        add_special_tokens=False,
    )
    encoded.pop("token_type_ids", None)
    if not encoded["attention_mask"].all():
        raise ValueError("batched diagnostic generation requires equal-length prompts")
    encoded = {key: value.to(model.device) for key, value in encoded.items()}
    model.config.use_cache = True
    try:
        outputs = model.generate(
            **encoded,
            max_length=1024,
            num_return_sequences=10,
            do_sample=False,
            num_beams=10,
            eos_token_id=tokenizer.eos_token_id,
            early_stopping="never",
            pad_token_id=tokenizer.pad_token_id,
            length_penalty=0.0,
        )
    finally:
        model.config.use_cache = False
    prompt_width = encoded["input_ids"].shape[1]
    decoded = tokenizer.batch_decode(outputs[:, prompt_width:], skip_special_tokens=True)
    decoded = [text.replace(" ", "") for text in decoded]
    return [decoded[start:start + 10] for start in range(0, len(decoded), 10)]


def generate_panel(args) -> None:
    rows = read_analysis_panel(args.panel_reference, args.panel_limit)
    if args.shard_count < 1:
        raise ValueError("shard count must be positive")
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard index must be in [0, shard_count)")
    rows = [
        row for position, row in enumerate(rows)
        if position % args.shard_count == args.shard_index
    ]
    completed = {record["reaction_identity"] for record in read_jsonl(args.output)}
    missing = [row for row in rows if row["reaction_identity"] not in completed]
    if not missing:
        print(json.dumps({"status": "complete", "rows": len(rows)}))
        return
    tokenizer, _, model = load_model(args.checkpoint)
    collator = ReactionCollator(tokenizer, task="forward")
    prompts = {
        row["reaction_identity"]: collator(
            [{"src": row["source"], "tgt": row["target"]}]
        )["generation_prompts"][0]
        for row in missing
    }
    prompt_ids = tokenizer(list(prompts.values()), add_special_tokens=False)["input_ids"]
    buckets: dict[int, list[dict[str, str]]] = {}
    for row, ids in zip(missing, prompt_ids):
        buckets.setdefault(len(ids), []).append(row)
    batches = []
    for bucket in buckets.values():
        for start in range(0, len(bucket), args.generation_batch_size):
            batches.append(bucket[start:start + args.generation_batch_size])

    run_started = time.perf_counter()
    generated = 0
    for generation_rows in batches:
        started = time.perf_counter()
        raw_batches = generate_equal_length_batch(
            model, tokenizer,
            [prompts[row["reaction_identity"]] for row in generation_rows],
        )
        batch_elapsed = time.perf_counter() - started
        for row, raw in zip(generation_rows, raw_batches):
            candidates = rank_augmented_candidates([raw], "forward", 10)
            _, scored = score_candidates(
                [{"src": row["source"], "tgt": row["target"]}], [candidates], "forward"
            )
            append_jsonl(args.output, {
                "condition": args.condition,
                "panel_index": int(row["selection_rank"]),
                "reaction_identity": row["reaction_identity"],
                "group_id": row["group_id"],
                **scored[0],
                "raw_candidates": raw,
                "generation_seconds": batch_elapsed / len(generation_rows),
                "generation_batch_size": len(generation_rows),
            })
            generated += 1
        total_complete = len(completed) + generated
        if generated <= len(generation_rows) or generated % 10 == 0:
            elapsed = time.perf_counter() - run_started
            print(json.dumps({
                "condition": args.condition,
                "complete": total_complete,
                "total": len(rows),
                "new": generated,
                "elapsed_seconds": elapsed,
                "seconds_per_new_reaction": elapsed / generated,
            }), flush=True)
        if args.max_new and generated >= args.max_new:
            break

@torch.inference_mode()
def encode_position(model, rows, pad_token_id: int, batch_size: int) -> torch.Tensor:
    states = []
    for start in range(0, len(rows), batch_size):
        chunk = rows[start:start + batch_size]
        padded = pad_sequence(chunk, batch_first=True, padding_value=pad_token_id).to(model.device)
        attention = padded.ne(pad_token_id)
        hidden = model(
            input_ids=padded, attention_mask=attention, output_hidden_states=True
        ).hidden_states[-1]
        indices = attention.sum(1) - 1
        states.append(
            hidden[torch.arange(len(chunk), device=model.device), indices].float().cpu()
        )
    return torch.cat(states)


def intervention_rows(rows, tokenizer):
    contributors = [_largest_target_overlap_component(row) for row in rows]
    contributor_sizes = [_heavy_atoms(component) for _, component in contributors]
    source_lengths = tokenizer(
        [row["src"] for row in rows], add_special_tokens=False
    )["input_ids"]
    source_lengths = [len(value) for value in source_lengths]
    source_atoms = [_heavy_atoms(row["src"]) for row in rows]
    identities = [row["target_identity"] for row in rows]
    lengths = dict(zip(identities, source_lengths))
    atoms = dict(zip(identities, source_atoms))
    _, unrelated_map, unrelated_cost = identity_mappings(identities, lengths, atoms, SEED + 17)
    by_identity = {row["target_identity"]: row for row in rows}

    removed, replaced, unrelated = [], [], []
    for row_index, row in enumerate(rows):
        components = row["src"].split(".")
        contributor_index, contributor = contributors[row_index]
        removed_source = ".".join(
            component for index, component in enumerate(components)
            if index != contributor_index
        )
        alternatives = [
            index for index, other in enumerate(contributors)
            if index != row_index and other[1] != contributor
        ]
        replacement_index = min(
            alternatives,
            key=lambda index: (
                abs(contributor_sizes[row_index] - contributor_sizes[index]),
                contributors[index][1],
            ),
        )
        replacement_components = list(components)
        replacement_components[contributor_index] = contributors[replacement_index][1]
        common = {"tgt": row["tgt"]}
        removed.append({"src": removed_source, **common})
        replaced.append({"src": ".".join(replacement_components), **common})
        unrelated_row = by_identity[unrelated_map[row["target_identity"]]]
        unrelated.append({"src": unrelated_row["src"], **common})
    return contributors, removed, replaced, unrelated, unrelated_cost


def target_logits(logits: torch.Tensor, labels: torch.Tensor):
    shifted_logits = logits[:, :-1].float()
    shifted_labels = labels[:, 1:]
    result = []
    for row_logits, row_labels in zip(shifted_logits, shifted_labels):
        mask = row_labels.ne(IGNORE_INDEX)
        result.append((row_logits[mask], row_labels[mask]))
    return result


@torch.inference_mode()
def decoder_interventions(model, collator, views, batch_size: int):
    names = list(views)
    rows = views[names[0]]
    output = {name: {"ce": [], "nll_sum": [], "kl_from_original": []} for name in names}
    for start in range(0, len(rows), batch_size):
        stop = min(start + batch_size, len(rows))
        distributions = {}
        for name in names:
            batch = collator(views[name][start:stop])
            tensors = {
                key: value.to(model.device)
                for key, value in batch.items() if torch.is_tensor(value)
            }
            model_output = model(
                input_ids=tensors["input_ids"],
                attention_mask=tensors["attention_mask"],
                labels=tensors["labels"],
            )
            distributions[name] = target_logits(model_output.logits, tensors["labels"])
        originals = distributions["original"]
        for name in names:
            for (original_logits, original_labels), (view_logits, view_labels) in zip(
                originals, distributions[name]
            ):
                if not torch.equal(original_labels, view_labels):
                    raise RuntimeError("intervention changed the target token sequence")
                nll = F.cross_entropy(view_logits, view_labels, reduction="none")
                output[name]["ce"].append(float(nll.mean()))
                output[name]["nll_sum"].append(float(nll.sum()))
                if name == "original":
                    kl = 0.0
                else:
                    original_logp = F.log_softmax(original_logits, dim=-1)
                    view_logp = F.log_softmax(view_logits, dim=-1)
                    original_p = original_logp.exp()
                    kl = float((original_p * (original_logp - view_logp)).sum(-1).mean())
                output[name]["kl_from_original"].append(kl)
    return output


def representation_inference(args) -> None:
    panel = read_analysis_panel(args.panel_reference, args.panel_limit)
    if args.output.exists():
        existing = json.loads(args.output.read_text(encoding="utf-8"))
        if len(existing.get("reactions", [])) == len(panel):
            print(json.dumps({"status": "complete", "condition": args.condition}))
            return
        raise RuntimeError("partial representation output exists; refusing to overwrite")
    rows = [
        {"src": row["source"], "tgt": row["target"], **row}
        for row in panel
    ]
    tokenizer, predictor_ids, model = load_model(args.checkpoint)
    collator = ReactionCollator(tokenizer, task="forward")
    contributors, removed, replaced, unrelated, unrelated_cost = intervention_rows(rows, tokenizer)
    original = [{"src": row["src"], "tgt": row["tgt"]} for row in rows]
    views = {
        "original": original,
        "removed": removed,
        "replaced": replaced,
        "unrelated": unrelated,
    }
    started = time.perf_counter()
    decoder = decoder_interventions(model, collator, views, args.batch_size)

    def source_token_rows(view):
        batch = collator(view)
        tensors = {key: value for key, value in batch.items() if torch.is_tensor(value)}
        sources, targets = extract_source_and_target(tensors)
        suffix = list(reversed(predictor_ids[:args.k]))
        return [torch.cat((source, source.new_tensor(suffix))) for source in sources], targets

    original_sources, targets = source_token_rows(original)
    removed_sources, _ = source_token_rows(removed)
    replaced_sources, _ = source_token_rows(replaced)
    unrelated_sources, _ = source_token_rows(unrelated)
    target_states = encode_position(model, targets, tokenizer.pad_token_id, args.state_batch_size)
    source_states = encode_position(model, original_sources, tokenizer.pad_token_id, args.state_batch_size)
    removed_states = encode_position(model, removed_sources, tokenizer.pad_token_id, args.state_batch_size)
    replaced_states = encode_position(model, replaced_sources, tokenizer.pad_token_id, args.state_batch_size)
    unrelated_states = encode_position(model, unrelated_sources, tokenizer.pad_token_id, args.state_batch_size)

    identities = [row["target_identity"] for row in rows]
    target_lengths = tokenizer(
        [row["tgt"] for row in rows], add_special_tokens=False
    )["input_ids"]
    token_lengths = dict(zip(identities, map(len, target_lengths)))
    heavy_atoms = {row["target_identity"]: _heavy_atoms(row["tgt"]) for row in rows}
    _, matched_map, matched_cost = identity_mappings(
        identities, token_lengths, heavy_atoms, SEED
    )
    index = {identity: position for position, identity in enumerate(identities)}
    matched = torch.tensor([index[matched_map[identity]] for identity in identities])

    source_states = source_states.float()
    target_states = target_states.float()
    normalized_sources = F.normalize(source_states, dim=-1)
    normalized_targets = F.normalize(target_states, dim=-1)
    raw_correct = (normalized_sources * normalized_targets).sum(-1)
    raw_matched = (normalized_sources * normalized_targets[matched]).sum(-1)

    joint = torch.cat((source_states, target_states)).cuda()
    joint_mean = joint.mean(0, keepdim=True)
    centered = joint - joint_mean
    torch.manual_seed(SEED)
    _, _, components = torch.pca_lowrank(centered, q=8, center=False, niter=6)
    basis = components[:, :2]

    def residual(values):
        values = values.cuda() - joint_mean
        return F.normalize(values - (values @ basis) @ basis.T, dim=-1).cpu()

    residual_sources = residual(source_states)
    residual_targets = residual(target_states)
    residual_removed = residual(removed_states.float())
    residual_replaced = residual(replaced_states.float())
    residual_unrelated = residual(unrelated_states.float())
    residual_correct = (residual_sources * residual_targets).sum(-1)
    residual_matched = (residual_sources * residual_targets[matched]).sum(-1)

    def raw_sensitivity(other):
        return 1.0 - F.cosine_similarity(source_states, other.float(), dim=-1)

    raw_sensitivities = {
        "removed": raw_sensitivity(removed_states),
        "replaced": raw_sensitivity(replaced_states),
        "unrelated": raw_sensitivity(unrelated_states),
    }
    residual_sensitivities = {
        "removed": 1.0 - (residual_sources * residual_removed).sum(-1),
        "replaced": 1.0 - (residual_sources * residual_replaced).sum(-1),
        "unrelated": 1.0 - (residual_sources * residual_unrelated).sum(-1),
    }

    reaction_results = []
    for row_index, row in enumerate(rows):
        record = {
            "panel_index": int(row["selection_rank"]),
            "reaction_identity": row["reaction_identity"],
            "target_token_count": len(target_lengths[row_index]),
            "raw_pair_margin": float(raw_correct[row_index] - raw_matched[row_index]),
            "residual_pc2_pair_margin": float(
                residual_correct[row_index] - residual_matched[row_index]
            ),
            "contributor_index": contributors[row_index][0],
            "contributor": contributors[row_index][1],
            "interventions": {},
        }
        for name in views:
            intervention = {
                "target_ce": decoder[name]["ce"][row_index],
                "target_nll_sum": decoder[name]["nll_sum"][row_index],
                "target_kl_from_original": decoder[name]["kl_from_original"][row_index],
            }
            if name != "original":
                intervention["raw_predictor_sensitivity"] = float(
                    raw_sensitivities[name][row_index]
                )
                intervention["residual_pc2_predictor_sensitivity"] = float(
                    residual_sensitivities[name][row_index]
                )
                intervention["target_ce_change"] = (
                    decoder[name]["ce"][row_index]
                    - decoder["original"]["ce"][row_index]
                )
            record["interventions"][name] = intervention
        reaction_results.append(record)

    token_total = sum(record["target_token_count"] for record in reaction_results)
    aggregate_native_loss = sum(
        record["interventions"]["original"]["target_nll_sum"]
        for record in reaction_results
    ) / token_total
    output = {
        "condition": args.condition,
        "checkpoint": str(args.checkpoint.resolve()),
        "manifest": str(PANEL_PATH.resolve()),
        "manifest_sha256": file_sha256(PANEL_PATH),
        "panel_reference": (
            None if args.panel_reference is None else str(args.panel_reference.resolve())
        ),
        "identities": len(panel),
        "k": args.k,
        "matched_assignment_cost": matched_cost,
        "unrelated_source_assignment_cost": unrelated_cost,
        "aggregate_native_target_token_ce": aggregate_native_loss,
        "mean_reaction_target_ce": float(np.mean(decoder["original"]["ce"])),
        "inference_seconds": time.perf_counter() - started,
        "reactions": reaction_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in output.items() if key != "reactions"}))
    del model
    torch.cuda.empty_cache()


def first_exact_rank(record: dict) -> int:
    for rank, exact in enumerate(record["exact"], 1):
        if exact:
            return rank
    return 11


def bootstrap_mean(values, rng, repetitions=5000):
    values = np.asarray(values, dtype=float)
    indices = rng.integers(0, len(values), size=(repetitions, len(values)))
    samples = values[indices].mean(1)
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def bootstrap_spearman(x, y, rng, repetitions=2000):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    estimates = []
    for _ in range(repetitions):
        indices = rng.integers(0, len(x), size=len(x))
        rho = spearmanr(x[indices], y[indices]).statistic
        if math.isfinite(rho):
            estimates.append(rho)
    if not estimates:
        return [None, None]
    return [float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))]


def association(x, y, rng):
    if np.ptp(np.asarray(x, dtype=float)) == 0 or np.ptp(np.asarray(y, dtype=float)) == 0:
        return {
            "spearman_rho": None,
            "p_value": None,
            "bootstrap_95_ci": [None, None],
            "reason": "association is undefined because one input is constant",
        }
    result = spearmanr(x, y)
    return {
        "spearman_rho": float(result.statistic),
        "p_value": float(result.pvalue),
        "bootstrap_95_ci": bootstrap_spearman(x, y, rng),
    }


def quartile_contrast(signal, outcome, rng):
    signal = np.asarray(signal)
    outcome = np.asarray(outcome)
    order = np.argsort(signal)
    count = len(signal) // 4
    difference = outcome[order[-count:]].mean() - outcome[order[:count]].mean()
    boot = []
    low, high = order[:count], order[-count:]
    for _ in range(5000):
        boot.append(
            outcome[rng.choice(high, count, replace=True)].mean()
            - outcome[rng.choice(low, count, replace=True)].mean()
        )
    return {
        "top_minus_bottom_quartile": float(difference),
        "bootstrap_95_ci": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
    }


def summarize(args) -> None:
    rows = read_analysis_panel(args.panel_reference, args.panel_limit)
    panel_size = len(rows)
    native_generation = sorted(read_jsonl(args.native_generation), key=lambda row: row["panel_index"])
    clm_generation = sorted(read_jsonl(args.clm_generation), key=lambda row: row["panel_index"])
    native_diag = json.loads(args.native_diagnostics.read_text(encoding="utf-8"))
    clm_diag = json.loads(args.clm_diagnostics.read_text(encoding="utf-8"))
    expected = [row["reaction_identity"] for row in rows]

    def select_expected(values):
        by_identity = {row["reaction_identity"]: row for row in values}
        missing = [identity for identity in expected if identity not in by_identity]
        if missing:
            raise RuntimeError(f"artifact is missing {len(missing)} panel identities")
        return [by_identity[identity] for identity in expected]

    native_generation = select_expected(native_generation)
    clm_generation = select_expected(clm_generation)
    native_diag["reactions"] = select_expected(native_diag["reactions"])
    clm_diag["reactions"] = select_expected(clm_diag["reactions"])
    if not all(len(value) == panel_size for value in (
        native_generation, clm_generation, native_diag["reactions"], clm_diag["reactions"]
    )):
        raise RuntimeError("all selected artifacts must match the analysis panel")

    eval_rows = [{"src": row["source"], "tgt": row["target"]} for row in rows]
    native_metrics, _ = score_candidates(
        eval_rows, [row["candidates"] for row in native_generation], "forward"
    )
    clm_metrics, _ = score_candidates(
        eval_rows, [row["candidates"] for row in clm_generation], "forward"
    )
    native_ranks = np.array([first_exact_rank(row) for row in native_generation])
    clm_ranks = np.array([first_exact_rank(row) for row in clm_generation])
    rank_improvement = native_ranks - clm_ranks
    native_top1 = native_ranks == 1
    clm_top1 = clm_ranks == 1
    rng = np.random.default_rng(SEED)

    native_reactions = native_diag["reactions"]
    clm_reactions = clm_diag["reactions"]

    def aggregate_target_ce(reactions):
        token_total = sum(row["target_token_count"] for row in reactions)
        nll_total = sum(
            row["interventions"]["original"]["target_nll_sum"] for row in reactions
        )
        return nll_total / token_total, token_total

    native_aggregate_ce, target_tokens = aggregate_target_ce(native_reactions)
    clm_aggregate_ce, clm_target_tokens = aggregate_target_ce(clm_reactions)
    if target_tokens != clm_target_tokens:
        raise RuntimeError("native and comparison diagnostics have different target token totals")
    native_ce = np.array([
        row["interventions"]["original"]["target_ce"] for row in native_reactions
    ])
    clm_ce = np.array([
        row["interventions"]["original"]["target_ce"] for row in clm_reactions
    ])
    ce_improvement = native_ce - clm_ce
    raw_signal = np.array([row["raw_pair_margin"] for row in clm_reactions])
    residual_signal = np.array([
        row["residual_pc2_pair_margin"] for row in clm_reactions
    ])

    top1_delta = clm_top1.astype(float) - native_top1.astype(float)
    discordant = int((native_top1 != clm_top1).sum())
    native_only = int((native_top1 & ~clm_top1).sum())
    clm_only = int((clm_top1 & ~native_top1).sum())
    paired = {
        "both_top1": int((native_top1 & clm_top1).sum()),
        f"{args.baseline_label}_only_top1": native_only,
        f"{args.comparison_label}_only_top1": clm_only,
        "neither_top1": int((~native_top1 & ~clm_top1).sum()),
        "top1_difference": float(top1_delta.mean()),
        "top1_difference_bootstrap_95_ci": bootstrap_mean(top1_delta, rng),
        "mcnemar_exact_two_sided_p": (
            float(binomtest(clm_only, discordant, 0.5).pvalue) if discordant else 1.0
        ),
        "rank_improved": int((rank_improvement > 0).sum()),
        "rank_worsened": int((rank_improvement < 0).sum()),
        "rank_tied": int((rank_improvement == 0).sum()),
        "mean_rank_improvement": float(rank_improvement.mean()),
        "mean_rank_improvement_bootstrap_95_ci": bootstrap_mean(rank_improvement, rng),
    }

    cutoff_transitions = {}
    transition_associations = {}
    for cutoff in (1, 3, 5, 10):
        native_hit = native_ranks <= cutoff
        clm_hit = clm_ranks <= cutoff
        transition = (~native_hit & clm_hit).astype(float)
        cutoff_transitions[str(cutoff)] = {
            "both": int((native_hit & clm_hit).sum()),
            f"{args.baseline_label}_only": int((native_hit & ~clm_hit).sum()),
            f"{args.comparison_label}_only": int((~native_hit & clm_hit).sum()),
            "neither": int((~native_hit & ~clm_hit).sum()),
        }
        transition_associations[str(cutoff)] = {
            "raw_pair_signal": association(raw_signal, transition, rng),
            "residual_pc2_pair_signal": association(residual_signal, transition, rng),
        }

    coupling = {}
    for label, signal in (("raw_pair_signal", raw_signal), ("residual_pc2_pair_signal", residual_signal)):
        coupling[label] = {
            "ce_improvement": association(signal, ce_improvement, rng),
            "rank_improvement": association(signal, rank_improvement, rng),
            "ce_quartile_contrast": quartile_contrast(signal, ce_improvement, rng),
            "rank_quartile_contrast": quartile_contrast(signal, rank_improvement, rng),
        }

    intervention_summary = {}
    for condition, diagnostic in (
        (args.baseline_label, native_diag), (args.comparison_label, clm_diag)
    ):
        intervention_summary[condition] = {}
        for name in ("removed", "replaced", "unrelated"):
            pred_raw = np.array([
                row["interventions"][name]["raw_predictor_sensitivity"]
                for row in diagnostic["reactions"]
            ])
            pred_residual = np.array([
                row["interventions"][name]["residual_pc2_predictor_sensitivity"]
                for row in diagnostic["reactions"]
            ])
            ce_change = np.array([
                row["interventions"][name]["target_ce_change"]
                for row in diagnostic["reactions"]
            ])
            kl = np.array([
                row["interventions"][name]["target_kl_from_original"]
                for row in diagnostic["reactions"]
            ])
            intervention_summary[condition][name] = {
                "mean_raw_predictor_sensitivity": float(pred_raw.mean()),
                "mean_residual_pc2_predictor_sensitivity": float(pred_residual.mean()),
                "mean_target_ce_change": float(ce_change.mean()),
                "mean_target_kl": float(kl.mean()),
                "raw_predictor_vs_ce_change": association(pred_raw, ce_change, rng),
                "raw_predictor_vs_target_kl": association(pred_raw, kl, rng),
                "residual_predictor_vs_ce_change": association(pred_residual, ce_change, rng),
                "residual_predictor_vs_target_kl": association(pred_residual, kl, rng),
            }

    ce_wilcoxon_p = (
        1.0 if np.allclose(ce_improvement, 0.0)
        else float(wilcoxon(ce_improvement, alternative="two-sided", zero_method="wilcox").pvalue)
    )
    ce_summary = {
        "target_tokens": target_tokens,
        f"{args.baseline_label}_aggregate_target_token_ce": native_aggregate_ce,
        f"{args.comparison_label}_aggregate_target_token_ce": clm_aggregate_ce,
        "relative_aggregate_improvement": (
            native_aggregate_ce - clm_aggregate_ce
        ) / native_aggregate_ce,
        "mean_reaction_ce_improvement": float(ce_improvement.mean()),
        "mean_reaction_ce_improvement_bootstrap_95_ci": bootstrap_mean(ce_improvement, rng),
        "fraction_reactions_improved": float((ce_improvement > 0).mean()),
        "fraction_reactions_worsened": float((ce_improvement < 0).mean()),
        "wilcoxon_two_sided_p": ce_wilcoxon_p,
    }

    output = {
        "protocol": {
            "manifest": str(PANEL_PATH.resolve()),
            "manifest_sha256": file_sha256(PANEL_PATH),
            "identities": panel_size,
            "panel_reference": (
                None if args.panel_reference is None else str(args.panel_reference.resolve())
            ),
            "enumerations_per_identity": 1,
            "beam_size": 10,
            "primary_metric": "exact_top1",
            "baseline_label": args.baseline_label,
            "analysis_only_residual": "joint mean centering followed by shared top-2-PC removal",
        },
        "generation": {
            args.baseline_label: native_metrics,
            args.comparison_label: clm_metrics,
            "paired": paired,
            "cutoff_transitions": cutoff_transitions,
        },
        "cross_entropy": ce_summary,
        "coupling": coupling,
        "cutoff_transition_associations": transition_associations,
        "interventions": intervention_summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        f"{args.baseline_label}_exact_top1": native_metrics["exact_top1"],
        f"{args.comparison_label}_exact_top1": clm_metrics["exact_top1"],
        "relative_ce_improvement": ce_summary["relative_aggregate_improvement"],
    }))


def parse_args():
    parser = argparse.ArgumentParser(description="Frozen-checkpoint decoder coupling diagnostic")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare")

    generation = subparsers.add_parser("generate")
    generation.add_argument("--condition", required=True)
    generation.add_argument("--checkpoint", type=Path, required=True)
    generation.add_argument("--output", type=Path, required=True)
    generation.add_argument("--max-new", type=int, default=0)
    generation.add_argument("--generation-batch-size", type=int, default=1)
    generation.add_argument("--panel-reference", type=Path)
    generation.add_argument("--panel-limit", type=int)
    generation.add_argument("--shard-count", type=int, default=1)
    generation.add_argument("--shard-index", type=int, default=0)

    representation = subparsers.add_parser("represent")
    representation.add_argument("--condition", required=True)
    representation.add_argument("--checkpoint", type=Path, required=True)
    representation.add_argument("--output", type=Path, required=True)
    representation.add_argument("--batch-size", type=int, default=4)
    representation.add_argument("--state-batch-size", type=int, default=16)
    representation.add_argument("--panel-reference", type=Path)
    representation.add_argument("--panel-limit", type=int)
    representation.add_argument("--k", type=int, choices=(0, 1), default=1)

    summary = subparsers.add_parser("summarize")
    summary.add_argument("--native-generation", type=Path, required=True)
    summary.add_argument("--clm-generation", type=Path, required=True)
    summary.add_argument("--native-diagnostics", type=Path, required=True)
    summary.add_argument("--clm-diagnostics", type=Path, required=True)
    summary.add_argument("--output", type=Path, required=True)
    summary.add_argument("--panel-reference", type=Path)
    summary.add_argument("--panel-limit", type=int)
    summary.add_argument("--comparison-label", default="clm_jepa")
    summary.add_argument("--baseline-label", default="native")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        print(json.dumps(prepare_panel(), sort_keys=True))
    elif args.command == "generate":
        generate_panel(args)
    elif args.command == "represent":
        representation_inference(args)
    else:
        summarize(args)


if __name__ == "__main__":
    main()
