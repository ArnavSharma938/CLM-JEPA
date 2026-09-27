from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import zipfile
from dataclasses import replace
from pathlib import Path

from .manifests import BackboneSequence, load_manifest, stable_hash, write_manifest
from .modeling import sha256_file


STRUCTURE_ZIP_SHA256 = "5c75396fe6e0229f4e4a8dbeab7b88b95cd8e89c47137cf83f9e33d3fb40270a"
SEARCH_THREADS = 6
FOLDSEEK_ALIGNMENT_TYPE = "1"
FOLDSEEK_MAX_SEQS = "1000"


def require_tools() -> dict[str, str]:
    found = {name: shutil.which(name) for name in ("foldseek", "mmseqs")}
    missing = [name for name, path in found.items() if path is None]
    if missing:
        raise RuntimeError(f"required executable(s) unavailable: {', '.join(missing)}")
    versions = {}
    for name, path in found.items():
        result = subprocess.run([path, "version"], check=True, text=True, capture_output=True)
        versions[name] = result.stdout.strip() or result.stderr.strip()
    return versions


def _records(path: Path) -> list[BackboneSequence]:
    return [BackboneSequence(**{key: value for key, value in row.items() if key != "split"}) for row in load_manifest(path)]


def extract_structures(archive: Path, accessions: set[str], output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    members: dict[str, str] = {}
    with zipfile.ZipFile(archive) as handle:
        for info in handle.infolist():
            path = Path(info.filename)
            if "__MACOSX" in path.parts or path.suffix.lower() != ".pdb":
                continue
            if path.stem in accessions:
                if path.stem in members:
                    raise RuntimeError(f"duplicate PDB member for accession {path.stem}")
                members[path.stem] = info.filename
        missing = accessions - set(members)
        if missing:
            raise RuntimeError(f"official structure ZIP lacks {len(missing)} selected accessions")
        for accession in sorted(accessions):
            destination = output / f"{accession}.pdb"
            if not destination.exists():
                with handle.open(members[accession]) as source, destination.open("wb") as target:
                    shutil.copyfileobj(source, target)
    return {"requested": len(accessions), "resolved": len(members), "missing": 0}


def structure_archive_accessions(archive: Path) -> set[str]:
    with zipfile.ZipFile(archive) as handle:
        return {
            Path(info.filename).stem
            for info in handle.infolist()
            if "__MACOSX" not in Path(info.filename).parts and Path(info.filename).suffix.lower() == ".pdb"
        }


def _run(command: list[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        subprocess.run(command, check=True, stdout=handle, stderr=subprocess.STDOUT, text=True)


def foldseek_maxima(
    query_dir: Path, target_dir: Path, output: Path, work: Path,
    *, alignment_type: str = FOLDSEEK_ALIGNMENT_TYPE,
    exact_tmscore: bool = True, max_seqs: str = FOLDSEEK_MAX_SEQS,
) -> dict[str, float]:
    executable = shutil.which("foldseek")
    if executable is None:
        raise RuntimeError("foldseek is unavailable")
    cache = output.with_suffix(".maxima.json")
    if cache.exists():
        return {key: float(value) for key, value in json.loads(cache.read_text(encoding="utf-8")).items()}
    result = output.with_suffix(".tsv")
    result.unlink(missing_ok=True)
    if work.exists():
        shutil.rmtree(work)
    _run(
        [
            executable, "easy-search", str(query_dir), str(target_dir), str(result), str(work),
            "--alignment-type", alignment_type, "--exact-tmscore", str(int(exact_tmscore)), "-s", "9.5",
            "--max-seqs", max_seqs, "--threads", str(SEARCH_THREADS),
            "--format-output", "query,target,qtmscore",
        ],
        output.with_suffix(".log"),
    )
    maxima: dict[str, float] = {}
    with result.open(encoding="utf-8") as handle:
        for query, _target, score in csv.reader(handle, delimiter="\t"):
            maxima[query] = max(maxima.get(query, 0.0), float(score))
    cache.write_text(json.dumps(maxima, sort_keys=True), encoding="utf-8")
    return maxima


def _lookup(mapping: dict[str, float], accession: str) -> float:
    for alias in (accession, f"{accession}_A", f"{accession}.pdb", f"{accession}.pdb_A"):
        if alias in mapping:
            return mapping[alias]
    return 0.0


def write_fasta(path: Path, records: list[BackboneSequence]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(f">{record.backbone_id}\n{record.sequence}\n")


def mmseqs_nearest(query: Path, target: Path, output: Path, work: Path) -> dict[str, tuple[float, float]]:
    executable = shutil.which("mmseqs")
    if executable is None:
        raise RuntimeError("mmseqs is unavailable")
    cache = output.with_suffix(".nearest.json")
    if cache.exists():
        return {
            key: (float(value[0]), float(value[1]))
            for key, value in json.loads(cache.read_text(encoding="utf-8")).items()
        }
    output.unlink(missing_ok=True)
    if work.exists():
        shutil.rmtree(work)
    _run(
        [
            executable, "easy-search", str(query), str(target), str(output), str(work),
            "-s", "7.5", "--max-seqs", "100", "--alignment-mode", "3",
            "--threads", str(SEARCH_THREADS),
            "--format-output", "query,target,fident,qcov,tcov,bits",
        ],
        output.with_suffix(".log"),
    )
    best: dict[str, tuple[float, float, float]] = {}
    with output.open(encoding="utf-8") as handle:
        for query_id, _target, identity, qcov, tcov, bits in csv.reader(handle, delimiter="\t"):
            candidate = (float(bits), float(identity), min(float(qcov), float(tcov)))
            if query_id not in best or candidate > best[query_id]:
                best[query_id] = candidate
    nearest = {key: (value[1], value[2]) for key, value in best.items()}
    cache.write_text(json.dumps(nearest, sort_keys=True), encoding="utf-8")
    return nearest


def construct_splits(root: Path, structure_zip: Path) -> dict:
    versions = require_tools()
    archive_hash = sha256_file(structure_zip)
    if archive_hash != STRUCTURE_ZIP_SHA256:
        raise RuntimeError(f"structure ZIP hash mismatch: {archive_hash}")
    expected_intersection = set(
        line.strip() for line in (root / "intersection_accessions.txt").read_text(encoding="utf-8").splitlines() if line
    )
    archive_accessions = structure_archive_accessions(structure_zip)
    missing_intersection = expected_intersection - archive_accessions
    if missing_intersection:
        raise RuntimeError(
            f"official structure archive lacks {len(missing_intersection)} BRn/BRq intersection accessions"
        )
    train10 = _records(root / "train_10k.jsonl")
    train50 = _records(root / "train_50k.jsonl")
    heldout = _records(root / "heldout_candidates.jsonl")
    distance_digest = hashlib.sha256()
    distance_digest.update(archive_hash.encode())
    distance_digest.update(json.dumps(versions, sort_keys=True).encode())
    for name in ("train_10k", "train_50k", "heldout_candidates"):
        metadata = json.loads(
            (root / f"{name}.jsonl.metadata.json").read_text(encoding="utf-8")
        )
        distance_digest.update(metadata["manifest_sha256"].encode())
    distance_key = distance_digest.hexdigest()[:16]
    workspace = root / "distance_cache" / distance_key
    structures = workspace / "structures"
    all_records = train50 + heldout
    extract_audit = extract_structures(
        structure_zip, {record.backbone_id for record in all_records}, structures / "all"
    )
    for label, records in (("train10", train10), ("train50", train50), ("heldout", heldout)):
        directory = structures / label
        directory.mkdir(exist_ok=True)
        for record in records:
            source = structures / "all" / f"{record.backbone_id}.pdb"
            destination = directory / source.name
            if not destination.exists():
                try:
                    destination.hardlink_to(source)
                except OSError:
                    shutil.copy2(source, destination)
    # The exact 10k search both defines the standard population and proves that
    # any qTM10 >= .5 item cannot be remote from the nested 50k set.  Therefore
    # only qTM10 < .5 candidates need comparison against the additional 40k
    # structures to establish the 50k remote-OOD set.
    qtm10 = foldseek_maxima(
        structures / "heldout", structures / "train10", workspace / "foldseek_10k", workspace / "tmp_foldseek_10k"
    )
    remote10 = [row for row in heldout if _lookup(qtm10, row.backbone_id) < 0.5]
    train10_ids = {row.backbone_id for row in train10}
    train40 = [row for row in train50 if row.backbone_id not in train10_ids]
    for label, records in (("remote10", remote10), ("train40", train40)):
        directory = structures / label
        directory.mkdir(exist_ok=True)
        for record in records:
            source = structures / "all" / f"{record.backbone_id}.pdb"
            destination = directory / source.name
            if not destination.exists():
                try:
                    destination.hardlink_to(source)
                except OSError:
                    shutil.copy2(source, destination)
    qtm40_remote = foldseek_maxima(
        structures / "remote10", structures / "train40",
        workspace / "foldseek_remote10_vs_train40", workspace / "tmp_foldseek_remote10_vs_train40",
    )
    qtm50 = dict(qtm10)
    for row in remote10:
        qtm50[row.backbone_id] = max(
            _lookup(qtm10, row.backbone_id), _lookup(qtm40_remote, row.backbone_id)
        )
    scored = [
        replace(
            record,
            structural_similarity_10k=_lookup(qtm10, record.backbone_id),
            structural_similarity_50k=_lookup(qtm50, record.backbone_id),
        )
        for record in heldout
    ]
    standard = [x for x in scored if x.structural_similarity_10k >= 0.5]
    remote = [x for x in scored if x.structural_similarity_50k < 0.5]
    standard.sort(key=lambda x: stable_hash(x.backbone_id, seed=31))
    remote.sort(key=lambda x: stable_hash(x.backbone_id, seed=37))
    validation_standard_count = min(500, len(standard) // 3)
    validation_remote_count = min(500, len(remote) // 3)
    validation = standard[:validation_standard_count] + remote[:validation_remote_count]
    validation_ids = {x.backbone_id for x in validation}
    standard_test = [x for x in standard if x.backbone_id not in validation_ids][:2000]
    remote_test = [x for x in remote if x.backbone_id not in validation_ids][:1000]
    if not standard_test or not remote_test:
        raise RuntimeError("the fixed 0.5 threshold produced an empty standard or remote test population")
    evaluation = validation + standard_test + remote_test

    sequence_results = {}
    query_fasta = workspace / "evaluation.fasta"
    write_fasta(query_fasta, evaluation)
    for label, training in (("10k", train10), ("50k", train50)):
        target_fasta = workspace / f"train_{label}.fasta"
        write_fasta(target_fasta, training)
        sequence_results[label] = mmseqs_nearest(
            query_fasta, target_fasta, workspace / f"mmseqs_{label}.tsv", workspace / f"tmp_mmseqs_{label}"
        )
    enriched = []
    for record in evaluation:
        identity10, coverage10 = sequence_results["10k"].get(record.backbone_id, (0.0, 0.0))
        identity50, coverage50 = sequence_results["50k"].get(record.backbone_id, (0.0, 0.0))
        enriched.append(replace(
            record,
            sequence_identity_10k=identity10, sequence_coverage_10k=coverage10,
            sequence_identity_50k=identity50, sequence_coverage_50k=coverage50,
        ))
    enriched_by_id = {x.backbone_id: x for x in enriched}
    outputs = {}
    for name, records, split in (
        ("validation", validation, "validation"),
        ("standard_test", standard_test, "standard_test"),
        ("remote_ood_test", remote_test, "remote_ood_test"),
    ):
        outputs[name] = write_manifest(
            root / f"{name}.jsonl", [enriched_by_id[x.backbone_id] for x in records], split=split,
            extra={"structural_threshold": 0.5, "structural_metric": "global query-normalized TM-score over Foldseek 3Di shortlist"},
        )
    audit = {
        "tool_versions": versions,
        "foldseek_search": {
            "alignment_type": int(FOLDSEEK_ALIGNMENT_TYPE),
            "exact_tmscore": True,
            "sensitivity": 9.5,
            "max_seqs": int(FOLDSEEK_MAX_SEQS),
            "threads": SEARCH_THREADS,
            "metric": "global query-normalized TM-score; nested-set pruning restricts the 40k increment search to qTM10 < 0.5 candidates",
            "remote10_candidates_searched_against_train40": len(remote10),
            "train40_structures": len(train40),
            "similarity_50k_semantics": (
                "exact maximum for qTM10<0.5 candidates; qTM10 certified lower bound "
                "for already-standard backbones, sufficient for the fixed 0.5 split"
            ),
        },
        "distance_cache_key": distance_key,
        "distance_workspace": str(workspace),
        "structure_mapping": extract_audit,
        "structure_zip_sha256": archive_hash,
        "released_structure_accessions": len(archive_accessions),
        "intersection_structure_coverage": {
            "expected": len(expected_intersection), "missing": len(missing_intersection)
        },
        "heldout_candidates": len(heldout),
        "standard_candidates": len(standard),
        "remote_candidates": len(remote),
        "outputs": outputs,
    }
    (root / "distance_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-root", required=True, type=Path)
    parser.add_argument("--structure-zip", required=True, type=Path)
    args = parser.parse_args()
    construct_splits(args.manifest_root, args.structure_zip)


if __name__ == "__main__":
    main()
