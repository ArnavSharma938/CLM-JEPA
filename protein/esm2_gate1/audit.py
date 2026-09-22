from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from .config import CANONICAL_AA, DATASET_REVISION, MODEL_REVISION
from .manifests import load_manifest


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(f"Gate-1 audit failed: {message}")


def _validate_sequences(rows: list[dict], label: str, *, synthetic: bool = True) -> dict:
    ids = [str(row["backbone_id"]) for row in rows]
    _require(len(ids) == len(set(ids)), f"{label} repeats a backbone")
    hashes = []
    for row in rows:
        sequence = row["sequence"]
        _require(sequence == sequence.upper(), f"{label} contains a lower-case sequence")
        _require(set(sequence) <= set(CANONICAL_AA), f"{label} contains a noncanonical residue")
        _require(40 <= len(sequence) <= 512, f"{label} contains a sequence outside 40-512")
        _require(len(sequence) == row["length"], f"{label} length field mismatch")
        digest = hashlib.sha256(sequence.encode()).hexdigest()
        _require(digest == row["sequence_sha256"], f"{label} sequence hash mismatch")
        if synthetic:
            _require(row["brn_member"] is True, f"{label} is not BRn-novel")
        hashes.append(digest)
    return {
        "backbones": len(ids),
        "residues": sum(int(row["length"]) for row in rows),
        "minimum_length": min(int(row["length"]) for row in rows),
        "maximum_length": max(int(row["length"]) for row in rows),
        "sequence_hashes": hashes,
    }


def _target_audit(preflight: dict) -> dict:
    targets = preflight["target_modules"]
    _require(len(targets) == 198, "ESM-2 target family does not contain 6 matrices x 33 layers")
    expected = {
        "attention.self.query": [1280, 1280],
        "attention.self.key": [1280, 1280],
        "attention.self.value": [1280, 1280],
        "attention.output.dense": [1280, 1280],
        "intermediate.dense": [5120, 1280],
        "output.dense": [1280, 5120],
    }
    seen_layers: dict[int, set[str]] = {layer: set() for layer in range(33)}
    for target in targets:
        name = target["name"]
        layer_tail = name.split(".layer.", 1)[1]
        layer_text, suffix = layer_tail.split(".", 1)
        layer = int(layer_text)
        _require(suffix in expected, f"unexpected dense target: {name}")
        _require(target["shape"] == expected[suffix], f"wrong target shape for {name}")
        seen_layers[layer].add(suffix)
    _require(all(values == set(expected) for values in seen_layers.values()), "target family differs by layer")
    _require(preflight["target_scalar_count"] == 648_806_400, "dense target scalar count changed")
    # Count unique nn.Parameter objects, as PyTorch/Hugging Face do for the
    # actual optimizer parameter set.  Counting tied tensors through duplicate
    # state-dict aliases inflates this value and is not an FT trainable count.
    _require(preflight["total_pretrained_parameters"] == 651_043_254, "ESM-2 total parameter count changed")
    _require(preflight["ft_trainable_count"] == 651_043_254, "FT trainable count changed")
    _require(preflight["dtft_trainable_count"] == 648_806_400, "DTFT trainable count changed")
    _require(preflight["lora_r8_trainable_count"] == 6_082_560, "LoRA-r8 count changed")
    _require(preflight["lora_r64_trainable_count"] == 48_660_480, "LoRA-r64 count changed")
    return {"layers": len(seen_layers), "matrices": len(targets), "suffixes": sorted(expected)}


def audit(manifests: Path, runs: Path, output: Path) -> dict:
    corpus = _json(manifests / "corpus_audit.json")
    distance = _json(manifests / "distance_audit.json")
    preflight = _json(runs / "preflight.json")
    parity = _json(runs / "fair_esm_parity.json")
    backend = _json(runs / "backend_benchmark.json")
    profile = _json(runs / "execution_profile.json")

    source_digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        source_digest.update(path.name.encode())
        source_digest.update(path.read_bytes())
    _require(
        preflight["esm2_gate1_source_sha256"] == source_digest.hexdigest(),
        "preflight source hash is stale",
    )

    _require(corpus["dataset_revision"] == DATASET_REVISION, "dataset revision is not pinned")
    _require(preflight["source"]["resolved_model_revision"] == MODEL_REVISION, "model revision is not pinned")
    _require(corpus["brn_unique_backbones"] == 138_044, "BRn accession count differs from release")
    _require(corpus["brq_unique_backbones"] == 127_633, "BRq accession count differs from release")
    _require(corpus["brn_scan"]["rows_scanned"] == 10_000_000, "BRn row count differs from release")
    _require(corpus["brq_rows"] == 10_000_000, "BRq row count differs from release")
    _require(corpus["brn_scan"]["rows_per_accession"]["minimum"] > 1, "BRn accession is not a grouping key")
    _require(corpus["brq_rows_per_accession_min"] > 1, "BRq accession is not a grouping key")
    _require(corpus["malformed_records_removed"] == 0, "unexpected malformed BRn rows")
    _require(corpus.get("invalid_length_records_removed", 0) == 0, "unexpected out-of-range BRn rows")
    _require("composition_complexity_comparison" in corpus, "natural/synthetic composition audit missing")
    _require(
        corpus["composition_complexity_comparison"]["natural_length_matched_uniref50"]["sequences"] >= 10_000,
        "length-matched natural reference is too small",
    )
    natural_lengths = corpus["composition_complexity_comparison"][
        "natural_length_matched_uniref50"
    ]["length"]
    synthetic_lengths = corpus["composition_complexity_comparison"][
        "synthetic_before_low_complexity_filter"
    ]["length"]
    for statistic in ("q01", "q25", "median", "q75", "q99", "mean"):
        _require(
            abs(natural_lengths[statistic] - synthetic_lengths[statistic]) <= 32,
            f"natural reference is not length matched at {statistic}",
        )

    names = (
        "train_10k", "train_50k", "heldout_candidates", "natural_test",
        "validation", "standard_test", "remote_ood_test",
    )
    loaded = {name: load_manifest(manifests / f"{name}.jsonl") for name in names}
    checks = {
        name: _validate_sequences(rows, name, synthetic=name != "natural_test")
        for name, rows in loaded.items()
    }
    ids = {name: {row["backbone_id"] for row in rows} for name, rows in loaded.items()}
    _require(len(ids["train_10k"]) == 10_000, "10k load size is not exact")
    _require(len(ids["train_50k"]) == 50_000, "50k load size is not exact")
    _require(ids["train_10k"] <= ids["train_50k"], "10k load is not nested in 50k")
    _require(not (ids["train_50k"] & ids["heldout_candidates"]), "heldout candidate leakage")
    eval_union = ids["validation"] | ids["standard_test"] | ids["remote_ood_test"]
    _require(not (ids["train_50k"] & eval_union), "train/evaluation backbone leakage")
    _require(not (ids["validation"] & ids["standard_test"]), "validation/standard overlap")
    _require(not (ids["validation"] & ids["remote_ood_test"]), "validation/OOD overlap")
    _require(not (ids["standard_test"] & ids["remote_ood_test"]), "standard/OOD overlap")
    synthetic_hashes = (
        checks["train_50k"]["sequence_hashes"] + checks["heldout_candidates"]["sequence_hashes"]
    )
    duplicate_hashes = sum(count - 1 for count in Counter(synthetic_hashes).values() if count > 1)
    _require(duplicate_hashes == 0, "exact sequence duplicates remain across synthetic backbones")
    residue_ratio = checks["train_10k"]["residues"] / checks["train_50k"]["residues"]
    _require(abs(residue_ratio - 0.2) <= 0.01, "nested load residue budgets are disproportionate")
    length10 = np.asarray([row["length"] for row in loaded["train_10k"]])
    length50 = np.asarray([row["length"] for row in loaded["train_50k"]])
    quantile_difference = float(
        np.max(np.abs(np.quantile(length10, np.linspace(0, 1, 101)) - np.quantile(length50, np.linspace(0, 1, 101))))
    )
    _require(quantile_difference <= 5, "10k/50k length distributions are not closely matched")

    _require(distance["structure_zip_sha256"] == "5c75396fe6e0229f4e4a8dbeab7b88b95cd8e89c47137cf83f9e33d3fb40270a", "structure archive hash changed")
    _require(distance["intersection_structure_coverage"]["missing"] == 0, "intersection structures are incomplete")
    for row in loaded["standard_test"]:
        _require(row["structural_similarity_10k"] >= 0.5, "standard test violates TM threshold")
    for row in loaded["remote_ood_test"]:
        _require(row["structural_similarity_50k"] < 0.5, "remote OOD violates TM threshold")
    for label in ("validation", "standard_test", "remote_ood_test"):
        for row in loaded[label]:
            _require(
                row["structural_similarity_50k"] + 1e-8 >= row["structural_similarity_10k"],
                f"{label} has max-TM nonmonotonicity between nested loads",
            )
            for load in ("10k", "50k"):
                identity = row[f"sequence_identity_{load}"]
                coverage = row[f"sequence_coverage_{load}"]
                _require(identity is not None and 0 <= identity <= 100, f"invalid MMseqs identity in {label}")
                _require(coverage is not None and 0 <= coverage <= 1, f"invalid MMseqs coverage in {label}")

    _require(parity["passed"], "Hugging Face/Fair-ESM FP32 parity did not pass")
    selected_backend = profile["backend"]
    selected_backend_row = next(row for row in backend["benchmarks"] if row["backend"] == selected_backend)
    _require(selected_backend_row["passed"], "selected attention backend failed parity")
    _require(
        {"eager", "sdpa"} <= set(profile["profiled_backends"]),
        "full-path optimization did not compare eager and SDPA",
    )
    train_metadata = _json(manifests / "train_10k.jsonl.metadata.json")
    _require(
        profile["manifest_sha256"] == train_metadata["manifest_sha256"],
        "full-path profile did not use the final 10k manifest",
    )
    for key in ("selected_training", "selected_evaluation"):
        _require(profile[key]["passed"], f"{key} did not pass")
        _require(
            profile[key]["peak_vram_bytes"] <= preflight["environment"]["gpu_vram_bytes"] * 0.90,
            f"{key} lacks registered VRAM headroom",
        )
    target_check = _target_audit(preflight)

    payload = {
        "passed": True,
        "dataset_revision": DATASET_REVISION,
        "model_revision": MODEL_REVISION,
        "manifest_checks": {name: {k: v for k, v in value.items() if k != "sequence_hashes"} for name, value in checks.items()},
        "nested_residue_ratio": residue_ratio,
        "maximum_101_quantile_length_difference": quantile_difference,
        "cross_backbone_exact_duplicates": duplicate_hashes,
        "target_check": target_check,
        "selected_backend": selected_backend,
        "selected_training": profile["selected_training"],
        "selected_evaluation": profile["selected_evaluation"],
        "checked_contracts": [
            "pinned revisions and released row/accession counts",
            "accession grouping semantics and one-sequence-per-backbone manifests",
            "canonical alphabet, length, sequence hashes, exact duplicate removal",
            "BRn novelty, nested loads, residue/length balance, split leakage",
            "structure archive/accession correspondence and 0.5 TM-score definitions",
            "MMseqs identity/coverage fields for both nested loads",
            "Fair-ESM parity, attention backend parity, profiling VRAM headroom",
            "exact ESM-2 dense target family and trainable counts",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-root", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    audit(args.manifest_root, args.run_root, args.output)


if __name__ == "__main__":
    main()
