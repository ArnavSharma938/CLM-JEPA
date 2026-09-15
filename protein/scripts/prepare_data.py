#!/usr/bin/env python
"""Prepare deterministic UniRef50 and TAPE diagnostic manifests."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.parse
import urllib.request
import urllib.error
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.constants import UNIREF50_ID, UNIREF50_REVISION
from src.data import assign_clusters, filter_sequence, validate_split_integrity, write_jsonl

VIEWER = "https://datasets-server.huggingface.co/rows"
TAPE_URLS = {
    "secondary_structure": "https://s3.amazonaws.com/songlabdata/proteindata/data_raw_pytorch/secondary_structure.tar.gz",
    "proteinnet": "https://s3.amazonaws.com/songlabdata/proteindata/data_raw_pytorch/proteinnet.tar.gz",
}


def fetch_uniref(output: Path, target: int, scan_limit: int = 20000) -> list[dict]:
    rows: list[dict] = []
    offset = 0
    while len(rows) < target and offset < scan_limit:
        query = urllib.parse.urlencode({
            "dataset": UNIREF50_ID, "config": "default", "split": "validation",
            "offset": offset, "length": 100,
        })
        url = f"{VIEWER}?{query}"
        for attempt in range(6):
            try:
                request = urllib.request.Request(
                    url, headers={"User-Agent": "CLM-JEPA-RITA-diagnostic/1.0"}
                )
                payload = json.load(urllib.request.urlopen(request, timeout=60))
                break
            except urllib.error.HTTPError as error:
                if error.code != 429 or attempt == 5:
                    raise
                retry_after = error.headers.get("Retry-After")
                delay = float(retry_after) if retry_after else min(60.0, 2.0 ** (attempt + 1))
                time.sleep(delay)
        else:  # pragma: no cover - loop either breaks or raises
            raise RuntimeError("unreachable viewer retry state")
        for wrapper in payload["rows"]:
            row = wrapper["row"]
            row["sequence"] = row["sequence"].upper()
            if filter_sequence(row):
                rows.append({
                    "id": row["sequence_id"], "sequence": row["sequence"],
                    "length": len(row["sequence"]), "description": row["description"],
                    "source": UNIREF50_ID, "source_revision": UNIREF50_REVISION,
                    "source_split": "validation", "source_row": wrapper["row_idx"],
                })
                if len(rows) == target:
                    break
        offset += len(payload["rows"])
        time.sleep(0.25)
        if not payload["rows"]:
            break
    if len(rows) != target:
        raise RuntimeError(f"found {len(rows)} eligible UniRef50 rows, wanted {target}")
    write_jsonl(output, rows)
    return rows


def write_fasta(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="ascii", newline="\n") as handle:
        for row in rows:
            handle.write(f">{row['id']}\n{row['sequence']}\n")


def finalize_clusters(rows: list[dict], data_dir: Path) -> list[dict]:
    """Validate an existing MMseqs TSV and create the locked split manifest."""
    mapping: dict[str, str] = {}
    with (data_dir / "mmseqs30_cluster.tsv").open(encoding="utf-8") as handle:
        for line in handle:
            representative, member = line.rstrip("\n").split("\t")[:2]
            mapping[member] = representative
    missing = {row["id"] for row in rows} - mapping.keys()
    if missing:
        raise RuntimeError(f"MMseqs output omitted {len(missing)} proteins")
    assigned = assign_clusters(rows, mapping)
    validate_split_integrity(assigned)
    write_jsonl(data_dir / "diagnostic_pool.jsonl", assigned)
    return assigned


def cluster_uniref(rows: list[dict], data_dir: Path, min_identity: float = 0.30) -> list[dict]:
    executable = shutil.which("mmseqs")
    if not executable:
        raise RuntimeError("MMseqs2 is required; no random-split fallback is permitted")
    fasta = data_dir / "uniref50_candidates.fasta"
    prefix = data_dir / "mmseqs30"
    tmp = data_dir / "mmseqs_tmp"
    write_fasta(fasta, rows)
    subprocess.run([
        executable, "easy-cluster", str(fasta), str(prefix), str(tmp),
        "--min-seq-id", str(min_identity), "-c", "0.8", "--cov-mode", "0",
        "--cluster-mode", "2", "--threads", "6",
    ], check=True)
    return finalize_clusters(rows, data_dir)


def safe_extract_tar(url: str, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination.parent / (destination.name + ".tar.gz")
    if not archive.exists():
        with urllib.request.urlopen(url) as source, archive.open("wb") as target:
            shutil.copyfileobj(source, target)
    with tarfile.open(archive, "r:gz") as tar:
        root = destination.resolve()
        for member in tar.getmembers():
            resolved = (destination / member.name).resolve()
            if root not in resolved.parents and resolved != root:
                raise RuntimeError("unsafe path in TAPE archive")
        tar.extractall(destination)


def download_tape(data_dir: Path) -> dict[str, str]:
    result = {}
    for name, url in TAPE_URLS.items():
        destination = data_dir / "tape" / name
        safe_extract_tar(url, destination)
        archive = destination.parent / (destination.name + ".tar.gz")
        result[name] = hashlib.sha256(archive.read_bytes()).hexdigest()
    (data_dir / "tape" / "archive_hashes.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    build_tape_manifests(data_dir)
    return result


def _load_json(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def build_tape_manifests(data_dir: Path) -> None:
    """Create compact, canonical-only structural annotation manifests."""
    tape = data_dir / "tape"
    secondary_root = tape / "secondary_structure" / "secondary_structure"
    secondary_rows = []
    for split in ("casp12", "cb513", "ts115"):
        path = secondary_root / f"secondary_structure_{split}.json"
        for raw in _load_json(path):
            sequence = raw["primary"].upper()
            if not (64 <= len(sequence) <= 512 and filter_sequence({"sequence": sequence})):
                continue
            labels = [int(x) for x in raw["ss3"][:len(sequence)]]
            valid = [bool(x) and label >= 0 for x, label in zip(raw["valid_mask"], labels)]
            secondary_rows.append({
                "id": f"tape_ss_{split}_{raw['id']}", "sequence": sequence,
                "length": len(sequence), "source": "TAPE secondary_structure",
                "source_split": split, "ss3": labels, "structure_valid": valid,
                "ss3_mapping": {"0": "helix", "1": "beta", "2": "coil"},
            })
    write_jsonl(data_dir / "tape_secondary_structure.jsonl", secondary_rows)

    contact_root = tape / "proteinnet" / "proteinnet"
    contact_rows = []
    for split in ("valid", "test"):
        path = contact_root / f"proteinnet_{split}.json"
        for raw in _load_json(path):
            sequence = raw["primary"].upper()
            if not (64 <= len(sequence) <= 512 and filter_sequence({"sequence": sequence})):
                continue
            coords = np.asarray(raw["tertiary"], dtype=np.float32)[:len(sequence)]
            valid = np.asarray(raw["valid_mask"], dtype=bool)[:len(sequence)]
            distance = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
            separation = np.abs(np.arange(len(sequence))[:, None] - np.arange(len(sequence))[None, :])
            eligible = valid[:, None] & valid[None, :] & (separation >= 24)
            contacts = eligible & (distance < 8.0)
            denominator = eligible.sum(axis=1)
            density = np.divide(
                contacts.sum(axis=1), denominator, out=np.full(len(sequence), np.nan),
                where=denominator > 0,
            )
            contact_rows.append({
                "id": f"tape_contact_{split}_{raw['id']}", "sequence": sequence,
                "length": len(sequence), "source": "TAPE ProteinNet",
                "source_split": split, "contact_valid": valid.tolist(),
                "long_range_contact_density": [None if np.isnan(x) else float(x) for x in density],
                "contact_definition": "CA distance <8A and sequence separation >=24",
            })
    write_jsonl(data_dir / "tape_contacts.jsonl", contact_rows)

    combined = secondary_rows + contact_rows
    write_jsonl(data_dir / "tape_structural_manifest.jsonl", combined)
    json_dump = {
        "secondary_count": len(secondary_rows), "contact_count": len(contact_rows),
        "combined_count": len(combined),
    }
    (tape / "manifest_summary.json").write_text(
        json.dumps(json_dump, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("uniref", "cluster", "finalize", "tape", "all"))
    parser.add_argument("--target", type=int, default=3000)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    args = parser.parse_args()
    args.data_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = args.data_dir / "uniref50_candidates.jsonl"
    if args.stage in ("uniref", "all"):
        rows = fetch_uniref(candidate_path, args.target)
    else:
        rows = [json.loads(line) for line in candidate_path.read_text().splitlines()]
    if args.stage in ("cluster", "all"):
        cluster_uniref(rows, args.data_dir)
    elif args.stage == "finalize":
        finalize_clusters(rows, args.data_dir)
    if args.stage in ("tape", "all"):
        download_tape(args.data_dir)


if __name__ == "__main__":
    main()
