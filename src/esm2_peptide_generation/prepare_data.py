from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from .config import (
    CANONICAL_AA,
    DATASET_FILE,
    DATASET_RELEASE,
    DATASET_SHA256,
    DATASET_URL,
    MAX_MODEL_TOKENS,
)


_CHAIN_SEPARATORS = re.compile(r"[,&/+;|\s]")


@dataclass(frozen=True)
class Pair:
    pair_id: str
    receptor_id: str
    receptor_sequence: str
    peptide_sequence: str
    pdb_id: str
    receptor_chain: str
    peptide_chain: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sequence_id(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def pair_id(receptor: str, peptide: str) -> str:
    return hashlib.sha256((receptor + "\0" + peptide).encode("ascii")).hexdigest()


def normalize_sequence(value: object) -> str | None:
    sequence = str(value or "").strip().upper()
    if not sequence or any(residue not in CANONICAL_AA for residue in sequence):
        return None
    return sequence


def _unambiguous_chain(value: object) -> str | None:
    chain = str(value or "").strip()
    if not chain or _CHAIN_SEPARATORS.search(chain):
        return None
    return chain


def read_pairs(csv_path: Path) -> tuple[list[Pair], dict]:
    digest = sha256_file(csv_path)
    if digest != DATASET_SHA256:
        raise RuntimeError(f"Propedia release checksum mismatch: {digest}")

    counters: Counter[str] = Counter()
    pairs: dict[tuple[str, str], Pair] = {}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter=";")
        required = {
            "PDB_ID", "PEPTIDE_CHAIN", "PEPTIDE_SEQ", "PROTEIN_CHAIN", "PROTEIN_SEQ"
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise RuntimeError("Propedia CSV is missing required chain/sequence fields")
        for row in reader:
            counters["rows"] += 1
            receptor = normalize_sequence(row["PROTEIN_SEQ"])
            peptide = normalize_sequence(row["PEPTIDE_SEQ"])
            receptor_chain = _unambiguous_chain(row["PROTEIN_CHAIN"])
            peptide_chain = _unambiguous_chain(row["PEPTIDE_CHAIN"])
            if receptor is None or peptide is None:
                counters["noncanonical_sequence"] += 1
                continue
            if receptor_chain is None or peptide_chain is None:
                counters["ambiguous_or_missing_assignment"] += 1
                continue
            if len(receptor) + len(peptide) + 2 > MAX_MODEL_TOKENS:
                counters["exceeds_model_context"] += 1
                continue

            counters["valid_rows"] += 1
            key = (receptor, peptide)
            record = Pair(
                pair_id=pair_id(receptor, peptide),
                receptor_id=sequence_id(receptor),
                receptor_sequence=receptor,
                peptide_sequence=peptide,
                pdb_id=str(row["PDB_ID"]).strip(),
                receptor_chain=receptor_chain,
                peptide_chain=peptide_chain,
            )
            previous = pairs.get(key)
            if previous is None or (record.pdb_id, record.receptor_chain, record.peptide_chain) < (
                previous.pdb_id, previous.receptor_chain, previous.peptide_chain
            ):
                pairs[key] = record

    unique = sorted(pairs.values(), key=lambda item: item.pair_id)
    counters["duplicate_pair_rows_collapsed"] = counters["valid_rows"] - len(unique)
    audit = {
        "source": DATASET_URL,
        "release": DATASET_RELEASE,
        "source_file": csv_path.name,
        "source_sha256": digest,
        "raw_rows": counters["rows"],
        "valid_rows_before_pair_collapse": counters["valid_rows"],
        "usable_unique_pair_count": len(unique),
        "unique_receptor_sequence_count": len({item.receptor_id for item in unique}),
        "unique_peptide_sequence_count": len({item.peptide_sequence for item in unique}),
        "filters_and_collapses": dict(counters),
    }
    return unique, audit


def _write_jsonl(path: Path, rows: Iterable[dict]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            line = json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            stream.write(line)
            digest.update(line.encode("utf-8"))
    return digest.hexdigest()


def write_sequences(pairs: list[Pair], output_dir: Path, audit: dict) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    pairs_path = output_dir / "unique_pairs.jsonl"
    pair_hash = _write_jsonl(pairs_path, (asdict(pair) for pair in pairs))
    receptors = sorted({pair.receptor_id: pair.receptor_sequence for pair in pairs}.items())
    fasta_path = output_dir / "receptors.fasta"
    with fasta_path.open("w", encoding="ascii", newline="\n") as stream:
        for identifier, sequence in receptors:
            stream.write(f">r{identifier}\n{sequence}\n")
    audit.update({
        "unique_pairs_manifest": pairs_path.name,
        "unique_pairs_manifest_sha256": pair_hash,
        "receptor_fasta": fasta_path.name,
        "receptor_fasta_sha256": sha256_file(fasta_path),
    })
    (output_dir / "stage1_source_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit and collapse official Propedia v26 pairs")
    parser.add_argument("--csv", type=Path, default=Path("data/protein/propedia26") / DATASET_FILE)
    parser.add_argument("--output-dir", type=Path, default=Path("data/protein/propedia26"))
    args = parser.parse_args()
    pairs, audit = read_pairs(args.csv)
    print(json.dumps(write_sequences(pairs, args.output_dir, audit), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
