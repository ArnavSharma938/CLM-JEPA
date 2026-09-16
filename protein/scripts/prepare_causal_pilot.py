#!/usr/bin/env python
"""Prepare and homology-decontaminate the locked causal-pilot datasets."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.constants import CANONICAL_AA_SET
from src.data import manifest_sha256, write_jsonl
from src.pilot import deterministic_orientation_order, load_protocol

UNIREf_URL = ("https://huggingface.co/datasets/alejoacelas/uniref50-2025-10/"
              "resolve/refs%2Fconvert%2Fparquet/default/validation/0000.parquet")
SWISS_URLS = {
    split: ("https://huggingface.co/datasets/lightonai/SwissProt-EC-leaf/"
            f"resolve/refs%2Fconvert%2Fparquet/default/{split}/0000.parquet")
    for split in ("train", "dev", "test")
}
PROTEINGYM_URL = (
    "https://zenodo.org/api/records/15293562/files/"
    "DMS_ProteinGym_substitutions.zip/content"
)
TAPE_URLS = {
    "secondary": "https://s3.amazonaws.com/songlabdata/proteindata/data_raw_pytorch/secondary_structure.tar.gz",
    "proteinnet": "https://s3.amazonaws.com/songlabdata/proteindata/data_raw_pytorch/proteinnet.tar.gz",
}
PG_SALT = "rita-causal-pilot-pg-v1"
EC_SALT = "rita-causal-pilot-ec-v1"
LM_SALT = "rita-causal-pilot-lm-v1"


def stable(value: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}\0{value}".encode()).hexdigest()


def canonical(sequence: str, low: int = 64, high: int = 512) -> bool:
    return low <= len(sequence) <= high and set(sequence) <= CANONICAL_AA_SET


def download(url: str, path: Path):
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    request = urllib.request.Request(url, headers={"User-Agent": "CLM-JEPA-causal-pilot/1"})
    with urllib.request.urlopen(request, timeout=120) as source, temporary.open("wb") as target:
        shutil.copyfileobj(source, target, length=1 << 20)
    temporary.replace(path)


def safe_extract(path: Path, destination: Path):
    complete = destination / ".complete"
    if complete.exists():
        return
    destination.mkdir(parents=True)
    with tarfile.open(path, "r:gz") as archive:
        root = destination.resolve()
        for member in archive.getmembers():
            resolved = (destination / member.name).resolve()
            if resolved != root and root not in resolved.parents:
                raise RuntimeError(f"unsafe archive member {member.name}")
        archive.extractall(destination)
    complete.write_text("complete\n", encoding="ascii")


def download_inputs(raw: Path):
    download(UNIREf_URL, raw / "uniref_validation.parquet")
    for split, url in SWISS_URLS.items():
        download(url, raw / f"swissprot_ec_{split}.parquet")
    download(PROTEINGYM_URL, raw / "DMS_ProteinGym_substitutions.zip")
    for name, url in TAPE_URLS.items():
        archive = raw / f"tape_{name}.tar.gz"
        download(url, archive)
        safe_extract(archive, raw / f"tape_{name}")


def lock_proteingym(reference: Path, archive: Path, output: Path, variants: int):
    table = pd.read_csv(reference)
    eligible = table[
        table.seq_len.between(64, 512)
        & table.target_seq.astype(str).map(canonical)
        & (table.DMS_number_single_mutants >= 100)
    ].copy()
    eligible["rank"] = eligible.DMS_id.map(lambda value: stable(str(value), PG_SALT))
    eligible = eligible.sort_values("rank").drop_duplicates("UniProt_ID")
    categories = sorted(eligible.coarse_selection_type.dropna().unique())
    chosen = []
    while len(chosen) < 24:
        category = categories[len(chosen) % len(categories)]
        available = eligible[
            (eligible.coarse_selection_type == category)
            & ~eligible.DMS_id.isin([row.DMS_id for row in chosen])
        ]
        chosen.append(next(available.itertuples()))
    panel_dir = output / "proteingym"
    panel_dir.mkdir(parents=True, exist_ok=True)
    names = {}
    with zipfile.ZipFile(archive) as bundle:
        for name in bundle.namelist():
            names.setdefault(Path(name).name, []).append(name)
        assays = []
        for row in chosen:
            matches = names.get(row.DMS_filename, [])
            if len(matches) != 1:
                raise RuntimeError(f"ProteinGym member mismatch: {row.DMS_filename}")
            with bundle.open(matches[0]) as handle:
                variants_table = pd.read_csv(handle)
            variants_table = variants_table[
                variants_table.mutant.astype(str).str.count(":").eq(0)
                & variants_table.mutated_sequence.astype(str).map(canonical)
            ].copy()
            variants_table["rank"] = variants_table.mutant.map(
                lambda value: stable(f"{row.DMS_id}\0{value}", "rita-causal-pilot-variants-v1")
            )
            variants_table = variants_table.sort_values("rank").head(variants).drop(columns="rank")
            target = panel_dir / row.DMS_filename
            variants_table.to_csv(target, index=False)
            assays.append({
                "DMS_id": row.DMS_id,
                "DMS_filename": row.DMS_filename,
                "UniProt_ID": row.UniProt_ID,
                "target_seq": row.target_seq,
                "seq_len": int(row.seq_len),
                "coarse_selection_type": row.coarse_selection_type,
                "selection_type": None if pd.isna(row.selection_type) else row.selection_type,
                "variant_count": len(variants_table),
                "rank": row.rank,
                "file_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            })
    payload = {
        "locked_before_training": True,
        "selection_salt": PG_SALT,
        "selection": "balanced category round-robin after phenotype-blind SHA256 rank",
        "variant_selection": "phenotype-blind SHA256 rank",
        "variant_cap": variants,
        "assays": assays,
    }
    (output / "proteingym_panel.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return assays


def lock_swissprot(raw: Path, output: Path, protocol: dict):
    frames = {}
    for split in ("train", "dev", "test"):
        frame = pd.read_parquet(raw / f"swissprot_ec_{split}.parquet")
        frame = frame[
            frame.seq.astype(str).map(canonical)
            & frame.labels.map(lambda labels: len(labels) == 1)
        ].copy()
        frame["label"] = frame.labels.map(lambda labels: int(labels[0]))
        frames[split] = frame
    counts = {split: Counter(frame.label) for split, frame in frames.items()}
    requirement = {
        "train": protocol["evaluation"]["swissprot_ec_train_per_class"],
        "dev": protocol["evaluation"]["swissprot_ec_dev_per_class"],
        "test": protocol["evaluation"]["swissprot_ec_test_per_class"],
    }
    labels = [
        label for label in counts["train"]
        if all(counts[split][label] >= requirement[split] for split in requirement)
    ]
    labels = sorted(labels, key=lambda label: stable(str(label), EC_SALT))[
        :protocol["evaluation"]["swissprot_ec_classes"]
    ]
    rows = []
    for split, frame in frames.items():
        frame = frame[frame.label.isin(labels)].copy()
        frame["rank"] = frame.id.map(lambda value: stable(str(value), f"{EC_SALT}-{split}"))
        for label in labels:
            selected = frame[frame.label == label].sort_values("rank").head(requirement[split])
            for row in selected.itertuples():
                rows.append({
                    "id": str(row.id), "sequence": str(row.seq), "label": int(row.label),
                    "split": split, "source": "lightonai/SwissProt-EC-leaf",
                })
    write_jsonl(output / "swissprot_ec.jsonl", rows)
    return rows


def iter_json_array(path: Path, chunk_size: int = 1 << 20):
    """Stream a top-level JSON array without loading multi-GB TAPE files."""
    decoder = json.JSONDecoder()
    with path.open(encoding="utf-8") as handle:
        buffer, position, started, ended = "", 0, False, False
        while not ended:
            chunk = handle.read(chunk_size)
            if not chunk and position >= len(buffer):
                break
            buffer = buffer[position:] + chunk
            position = 0
            while True:
                while position < len(buffer) and buffer[position].isspace():
                    position += 1
                if not started:
                    if position >= len(buffer):
                        break
                    if buffer[position] != "[":
                        raise ValueError(f"expected JSON array in {path}")
                    position += 1
                    started = True
                    continue
                while position < len(buffer) and (buffer[position].isspace()
                                                   or buffer[position] == ","):
                    position += 1
                if position < len(buffer) and buffer[position] == "]":
                    ended = True
                    break
                try:
                    item, position = decoder.raw_decode(buffer, position)
                except json.JSONDecodeError:
                    break
                yield item
            if not chunk and not ended:
                raise ValueError(f"truncated JSON array in {path}")


def lock_tape(raw: Path, output: Path):
    secondary_root = raw / "tape_secondary" / "secondary_structure"
    secondary = []
    limits = {"train": 256, "valid": 64, "casp12": 512, "cb513": 512, "ts115": 512}
    for split, limit in limits.items():
        path = secondary_root / f"secondary_structure_{split}.json"
        candidates = []
        for item in iter_json_array(path):
            sequence = item["primary"].upper()
            if canonical(sequence):
                candidates.append({
                    "id": f"ss_{split}_{item['id']}", "sequence": sequence,
                    "split": split, "labels": [int(x) for x in item["ss3"][:len(sequence)]],
                    "valid": [bool(x) for x in item["valid_mask"][:len(sequence)]],
                })
        secondary.extend(sorted(candidates, key=lambda row: stable(row["id"], "pilot-ss-v1"))[:limit])
    write_jsonl(output / "tape_secondary.jsonl", secondary)

    contact_root = raw / "tape_proteinnet" / "proteinnet"
    contacts = []
    for split, limit in (("valid", 128), ("test", 64)):
        candidates = []
        for item in iter_json_array(contact_root / f"proteinnet_{split}.json"):
            sequence = item["primary"].upper()
            if not canonical(sequence):
                continue
            coords = np.asarray(item["tertiary"], dtype=np.float32)[:len(sequence)]
            valid = np.asarray(item["valid_mask"], dtype=bool)[:len(sequence)]
            candidates.append({
                "id": f"contact_{split}_{item['id']}", "sequence": sequence,
                "split": split, "coords": coords.tolist(), "valid": valid.tolist(),
            })
        contacts.extend(sorted(candidates, key=lambda row: stable(row["id"], "pilot-contact-v1"))[:limit])
    write_jsonl(output / "proteinnet_contact.jsonl", contacts)
    return secondary, contacts


def load_uniref(path: Path):
    frame = pd.read_parquet(path, columns=["sequence_id", "sequence", "description"])
    rows = []
    for index, row in enumerate(frame.itertuples(index=False)):
        sequence = str(row.sequence).upper()
        if canonical(sequence):
            rows.append({
                "id": str(row.sequence_id), "sequence": sequence,
                "description": str(row.description), "source_row": index,
            })
    return rows


def write_fasta(path: Path, rows):
    seen = set()
    with path.open("w", encoding="ascii", newline="\n") as handle:
        for row in rows:
            identifier = row["id"].replace(" ", "_")
            if identifier in seen:
                identifier += "_" + stable(row["sequence"], "duplicate")[:12]
            seen.add(identifier)
            handle.write(f">{identifier}\n{row['sequence']}\n")


def prepare_inputs(raw: Path, output: Path, reference: Path, protocol: dict):
    output.mkdir(parents=True, exist_ok=True)
    pg = lock_proteingym(
        reference, raw / "DMS_ProteinGym_substitutions.zip", output,
        protocol["evaluation"]["proteingym_variant_cap"],
    )
    ec = lock_swissprot(raw, output, protocol)
    secondary, contacts = lock_tape(raw, output)
    candidates = load_uniref(raw / "uniref_validation.parquet")
    write_jsonl(output / "uniref_candidates.jsonl", candidates)
    external = []
    external.extend({"id": f"pg_{row['DMS_id']}", "sequence": row["target_seq"]} for row in pg)
    external.extend({"id": f"ec_{row['split']}_{row['id']}", "sequence": row["sequence"]} for row in ec)
    external.extend({"id": row["id"], "sequence": row["sequence"]} for row in secondary)
    external.extend({"id": row["id"], "sequence": row["sequence"]} for row in contacts)
    write_fasta(output / "uniref_candidates.fasta", candidates)
    write_fasta(output / "external_evaluation.fasta", external)
    return candidates, external


def run_mmseqs(output: Path, threads: int = 6):
    executable = shutil.which("mmseqs")
    if not executable:
        raise RuntimeError("MMseqs2 is required on the preparation host")
    # Emit candidate-candidate alignments and apply shorter-sequence coverage
    # ourselves. This avoids cov-mode directionality/symmetry ambiguity.
    subprocess.run([
        executable, "easy-search", str(output / "uniref_candidates.fasta"),
        str(output / "uniref_candidates.fasta"), str(output / "cluster_hits.tsv"),
        str(output / "tmp-cluster-search"), "--min-seq-id", "0.30", "-c", "0.0",
        "-s", "7.5", "--max-seqs", "5000", "--threads", str(threads),
        "--format-output", "query,target,fident,qcov,tcov",
    ], check=True)
    subprocess.run([
        executable, "easy-search", str(output / "uniref_candidates.fasta"),
        str(output / "external_evaluation.fasta"), str(output / "external_hits.tsv"),
        str(output / "tmp-search"), "--min-seq-id", "0.30", "-c", "0.0",
        "-s", "7.5", "--max-seqs", "5000", "--threads", str(threads),
        "--format-output", "query,target,fident,qcov,tcov",
    ], check=True)


def normalize_fraction(value: str) -> float:
    parsed = float(value)
    return parsed / 100.0 if parsed > 1.0 else parsed


def finalize(output: Path, protocol: dict):
    candidates = {
        row["id"]: row
        for row in (json.loads(line) for line in
                    (output / "uniref_candidates.jsonl").read_text().splitlines())
    }
    parent = {identifier: identifier for identifier in candidates}
    def find(identifier):
        while parent[identifier] != identifier:
            parent[identifier] = parent[parent[identifier]]
            identifier = parent[identifier]
        return identifier
    def union(left, right):
        left, right = find(left), find(right)
        if left != right:
            parent[max(left, right)] = min(left, right)
    qualifying_cluster_hits = 0
    with (output / "cluster_hits.tsv").open(encoding="utf-8") as handle:
        for line in handle:
            query, target_id, identity, qcov, tcov = line.rstrip().split("\t")[:5]
            identity = normalize_fraction(identity)
            qcov, tcov = normalize_fraction(qcov), normalize_fraction(tcov)
            if identity >= 0.30 and max(qcov, tcov) >= 0.80:
                union(query, target_id)
                qualifying_cluster_hits += 1
    cluster = {identifier: find(identifier) for identifier in candidates}
    excluded = set()
    hit_records = []
    with (output / "external_hits.tsv").open(encoding="utf-8") as handle:
        for line in handle:
            query, target, identity, qcov, tcov = line.rstrip().split("\t")[:5]
            identity = normalize_fraction(identity)
            qcov, tcov = normalize_fraction(qcov), normalize_fraction(tcov)
            if identity >= 0.30 and max(qcov, tcov) >= 0.80:
                excluded.add(query)
                hit_records.append({
                    "query": query, "target": target, "identity": identity,
                    "qcov": qcov, "tcov": tcov,
                })
    write_jsonl(output / "decontamination_hits.jsonl", hit_records)
    by_cluster = defaultdict(list)
    for identifier, row in candidates.items():
        if identifier not in excluded:
            by_cluster[cluster[identifier]].append(row)
    representatives = [
        sorted(rows, key=lambda row: row["id"])[0] for rows in by_cluster.values()
    ]
    representatives.sort(key=lambda row: stable(row["id"], LM_SALT))
    heldout = representatives[:protocol["evaluation"]["heldout_lm_proteins"]]
    heldout_clusters = {cluster[row["id"]] for row in heldout}
    heldout_rows = [
        {**row, "cluster_id": cluster[row["id"]], "split": "heldout_lm"}
        for row in heldout
    ]
    write_jsonl(output / "heldout_lm.jsonl", heldout_rows)
    write_jsonl(output / "mechanism_set.jsonl", heldout_rows[:protocol["evaluation"]["mechanism_gradient_proteins"]])
    write_jsonl(output / "generation_prompts.jsonl", [
        {"id": row["id"], "prompt": row["sequence"][:protocol["evaluation"]["generation_prompt_residues"]]}
        for row in heldout_rows[:protocol["evaluation"]["generation_samples_per_model"]]
    ])

    available = [
        {**row, "cluster_id": cluster[row["id"]]}
        for row in representatives
        if cluster[row["id"]] not in heldout_clusters
    ]
    used_clusters = set()
    manifest_hashes = {}
    target = protocol["data"]["primary_residue_target_per_replicate"]
    for replicate, seed in enumerate(protocol["replicates"]["seeds"]):
        ranked = sorted(
            (row for row in available if row["cluster_id"] not in used_clusters),
            key=lambda row: stable(row["id"], f"pilot-train-{seed}"),
        )
        chosen, residues = [], 0
        for row in ranked:
            chosen.append({
                **row, "replicate": replicate,
                "orientation_order": deterministic_orientation_order(row["id"], seed),
            })
            residues += len(row["sequence"])
            if residues >= target and len(chosen) % 8 == 0:
                break
        if residues < target:
            raise RuntimeError(f"insufficient data for replicate {replicate}")
        used_clusters.update(row["cluster_id"] for row in chosen)
        path = output / f"train_replicate_{replicate}.jsonl"
        manifest_hashes[path.name] = write_jsonl(path, chosen)
    artifact_names = [
        "heldout_lm.jsonl", "mechanism_set.jsonl", "generation_prompts.jsonl",
        "proteingym_panel.json", "swissprot_ec.jsonl", "tape_secondary.jsonl",
        "proteinnet_contact.jsonl", "decontamination_hits.jsonl",
        *[f"train_replicate_{index}.jsonl" for index in range(8)],
    ]
    commit_process = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT.parent,
        capture_output=True, text=True,
    )
    repository_commit = (commit_process.stdout.strip() if commit_process.returncode == 0
                         else os.environ.get("CLM_JEPA_BASE_COMMIT", "unavailable"))
    mmseqs_version = subprocess.run(
        [shutil.which("mmseqs"), "version"], check=True, capture_output=True, text=True,
    ).stdout.strip()
    summary = {
        "protocol_sha256": hashlib.sha256(
            json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "candidate_count": len(candidates),
        "external_exclusion_sequence_count": len(excluded),
        "external_qualifying_hit_count": len(hit_records),
        "heldout_lm_count": len(heldout_rows),
        "training_clusters_unique_across_replicates": len(used_clusters),
        "training_manifest_sha256": manifest_hashes,
        "artifact_sha256": {
            name: hashlib.sha256((output / name).read_bytes()).hexdigest()
            for name in artifact_names
        },
        "repository_commit_at_preparation": repository_commit,
        "deployed_source_bundle_sha256": os.environ.get("CLM_JEPA_BUNDLE_SHA256"),
        "pilot_code_sha256": {
            str(path.relative_to(ROOT.parent)).replace("\\", "/"):
                hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [
                ROOT / "configs/causal_pilot.json",
                ROOT / "src/pilot.py", ROOT / "src/pilot_eval.py",
                ROOT / "src/pilot_stats.py", ROOT / "src/nextlat.py",
                ROOT / "scripts/prepare_causal_pilot.py",
                ROOT / "scripts/run_causal_pilot.py",
                ROOT / "scripts/evaluate_causal_pilot.py",
                ROOT / "scripts/evaluate_causal_mechanism.py",
                ROOT / "scripts/run_esmfold_pilot.py",
                ROOT / "scripts/summarize_causal_pilot.py",
                ROOT / "scripts/validate_causal_pilot_parity.py",
            ]
        },
        "mmseqs_version": mmseqs_version,
        "candidate_cluster_qualifying_hit_count": qualifying_cluster_hits,
        "decontamination_rule": "MMseqs local identity>=0.30 and max(qcov,tcov)>=0.80; transitive union-find clusters",
    }
    (output / "manifest.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("download", "prepare", "mmseqs", "finalize", "all"))
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/causal_pilot.json")
    parser.add_argument("--raw", type=Path, default=ROOT / "data/causal_pilot_raw")
    parser.add_argument("--output", type=Path, default=ROOT / "data/causal_pilot")
    parser.add_argument("--proteingym-reference", type=Path,
                        default=ROOT / "data/ProteinGym_DMS_substitutions_reference.csv")
    args = parser.parse_args()
    protocol = load_protocol(args.protocol)
    if args.stage in {"download", "all"}:
        download_inputs(args.raw)
    if args.stage in {"prepare", "all"}:
        prepare_inputs(args.raw, args.output, args.proteingym_reference, protocol)
    if args.stage in {"mmseqs", "all"}:
        run_mmseqs(args.output)
    if args.stage in {"finalize", "all"}:
        finalize(args.output, protocol)


if __name__ == "__main__":
    main()
