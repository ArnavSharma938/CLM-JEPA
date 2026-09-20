from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import importlib.metadata
import json
import os
import random
import re
import shutil
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

import pandas as pd
import requests
import numpy as np


DATASET_URLS = {
    "processed": (
        "https://zenodo.org/api/records/7992926/files/Processed_K50_dG_datasets.zip/content",
        "f7e8c553efee734cf161ee6f2b0a09cf",
    ),
    "pdbs": (
        "https://zenodo.org/api/records/7992926/files/AlphaFold_model_PDBs.zip/content",
        "c29317a75f28d008b2718316995902ef",
    ),
}
SUBSTITUTION_PATTERN = re.compile(r"^([ACDEFGHIKLMNPQRSTVWY])(\d+)([ACDEFGHIKLMNPQRSTVWY])$")
MUTATION = SUBSTITUTION_PATTERN
_TM_STRUCTURES: list[tuple[str, np.ndarray, str]] = []


def _domain_key(value: str) -> str:
    # The published CSV uses "|" where the PDB archive uses ":". Colons are
    # NTFS alternate-stream delimiters, so normalize both before touching disk.
    normalized = value.replace(":", "__SEP__").replace("|", "__SEP__")
    name = PurePosixPath(normalized).name
    for suffix in (".mmcif", ".pdb", ".cif"):
        if name.lower().endswith(suffix):
            return name[: -len(suffix)]
    return name


def _structure_key(value: str) -> str:
    """Map a CSV WT domain (including derived WT suffixes) to its PDB key."""
    normalized = value.replace(":", "__SEP__").replace("|", "__SEP__")
    name = PurePosixPath(normalized).name
    lower = name.lower()
    for marker in (".mmcif", ".pdb", ".cif"):
        position = lower.find(marker)
        if position >= 0:
            return name[:position]
    return _domain_key(name)


def _md5(path: Path) -> str:
    digest = hashlib.md5()  # nosec B324 - required only to verify the published Zenodo checksum
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download_published_data(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for label, (url, expected) in DATASET_URLS.items():
        destination = output_dir / ("Processed_K50_dG_datasets.zip" if label == "processed" else "AlphaFold_model_PDBs.zip")
        if destination.exists() and _md5(destination) == expected:
            continue
        temporary = destination.with_suffix(destination.suffix + ".part")
        offset = temporary.stat().st_size if temporary.exists() else 0
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        with requests.get(url, headers=headers, stream=True, timeout=120) as response:
            response.raise_for_status()
            append = offset > 0 and response.status_code == 206
            with temporary.open("ab" if append else "wb") as handle:
                for chunk in response.iter_content(8 << 20):
                    if chunk:
                        handle.write(chunk)
        if _md5(temporary) != expected:
            raise RuntimeError(f"Published checksum mismatch for {destination.name}")
        temporary.replace(destination)


def extract_published_data(download_dir: Path, output_dir: Path) -> dict[str, str | int]:
    """Extract only the calibration CSV and structures from verified archives."""
    processed = download_dir / "Processed_K50_dG_datasets.zip"
    structures = download_dir / "AlphaFold_model_PDBs.zip"
    for archive, expected in (
        (processed, DATASET_URLS["processed"][1]),
        (structures, DATASET_URLS["pdbs"][1]),
    ):
        if not archive.is_file() or _md5(archive) != expected:
            raise RuntimeError(f"Missing or checksum-invalid published archive: {archive}")
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_name = "Tsuboyama2023_Dataset2_Dataset3_20230416.csv"
    csv_path = output_dir / csv_name
    with zipfile.ZipFile(processed) as archive:
        matches = [name for name in archive.namelist() if Path(name).name == csv_name]
        if len(matches) != 1:
            raise RuntimeError(f"Expected exactly one {csv_name} in the processed archive")
        with archive.open(matches[0]) as source, csv_path.open("wb") as destination:
            shutil.copyfileobj(source, destination, length=8 << 20)
    pdb_dir = output_dir / "pdbs"
    count = 0
    # Extract into a sibling and swap only after the complete archive has been
    # validated. This makes a rerun replace an incomplete/legacy extraction
    # without ever mixing stale files with the published structure inventory.
    with tempfile.TemporaryDirectory(prefix="pdbs.extracting-", dir=output_dir) as temporary:
        # Windows keeps the TemporaryDirectory root open, so rename a child
        # payload rather than the managed root during the atomic swap.
        staged_pdb_dir = Path(temporary) / "payload"
        staged_pdb_dir.mkdir()
        with zipfile.ZipFile(structures) as archive:
            for member in archive.infolist():
                archive_path = PurePosixPath(member.filename)
                basename = archive_path.name
                if (
                    member.is_dir()
                    or "__MACOSX" in archive_path.parts
                    or basename.startswith("._")
                    or archive_path.suffix.lower() not in {".pdb", ".cif", ".mmcif"}
                ):
                    continue
                safe_basename = basename.replace(":", "__SEP__").replace("|", "__SEP__")
                destination = staged_pdb_dir / safe_basename
                if destination.exists():
                    raise RuntimeError(f"Duplicate structure basename in archive: {destination.name}")
                with archive.open(member) as source, destination.open("wb") as target:
                    shutil.copyfileobj(source, target, length=8 << 20)
                count += 1
        if count == 0:
            raise RuntimeError("The published structure archive contained no structures")
        backup = output_dir / "pdbs.replacing"
        if backup.exists():
            raise RuntimeError(f"Stale extraction backup requires inspection: {backup}")
        if pdb_dir.exists():
            pdb_dir.replace(backup)
        try:
            staged_pdb_dir.replace(pdb_dir)
        except BaseException:
            if backup.exists() and not pdb_dir.exists():
                backup.replace(pdb_dir)
            raise
        if backup.exists():
            shutil.rmtree(backup)
    structure_keys = {_domain_key(path.name) for path in pdb_dir.iterdir() if path.is_file()}
    if len(structure_keys) != count:
        raise RuntimeError("Extracted structure names collide after normalization")
    csv_domains = {
        _domain_key(str(value))
        for value in pd.read_csv(csv_path, usecols=["WT_name"])["WT_name"].dropna().unique()
    }
    csv_structure_keys = {_structure_key(value) for value in csv_domains}
    missing_structures = sorted(csv_structure_keys - structure_keys)
    if missing_structures:
        raise RuntimeError(f"CSV domains reference missing structures: {missing_structures[:5]}")
    inventory = {
        "source_csv": str(csv_path.resolve()),
        "source_csv_sha256": _sha256(csv_path),
        "pdb_dir": str(pdb_dir.resolve()),
        "structure_count": count,
        "csv_domain_count": len(csv_domains),
        "csv_structure_count": len(csv_structure_keys),
        "csv_domains_missing_structure_count": 0,
        "unreferenced_structure_count": len(structure_keys - csv_structure_keys),
    }
    (output_dir / "source_inventory.json").write_text(
        json.dumps(inventory, indent=2, sort_keys=True), encoding="utf-8"
    )
    return inventory


def foldseek_commands(pdb_dir: Path, work_dir: Path, foldseek: str = "foldseek") -> list[list[str]]:
    """Commands pinned to the FoldSeek procedure disclosed by ProteinDPO.

    Clustering uses global TMalign and the paper's 0.5 structural-similarity
    threshold. The held-out search emits query-normalized TM-score explicitly.
    """
    clusters = work_dir / "clusters"
    cluster_tmp = work_dir / "cluster_tmp"
    heldout = work_dir / "heldout_pdbs"
    train = work_dir / "train_pdbs"
    qtm = work_dir / "heldout_to_train.tsv"
    search_tmp = work_dir / "search_tmp"
    return [
        [
            foldseek, "easy-cluster", str(pdb_dir), str(clusters), str(cluster_tmp),
            "--alignment-type", "1", "--tmscore-threshold", "0.5",
        ],
        [
            foldseek, "easy-search", str(heldout), str(train), str(qtm), str(search_tmp),
            "--alignment-type", "1", "--exhaustive-search", "1",
            "--format-output", "query,target,qtmscore",
        ],
    ]


def write_foldseek_plan(pdb_dir: Path, work_dir: Path, output: Path, foldseek: str = "foldseek") -> None:
    commands = foldseek_commands(pdb_dir.resolve(), work_dir.resolve(), foldseek)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"foldseek_version": "8.ef4e960", "commands": commands}, indent=2), encoding="utf-8")


def assign_cluster_splits(cluster_tsv: Path, seed: int = 42) -> dict[str, str]:
    members: dict[str, set[str]] = {}
    for line in cluster_tsv.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        representative, member = (_domain_key(value) for value in line.split("\t")[:2])
        members.setdefault(representative, set()).add(member)
    clusters = sorted(members)
    random.Random(seed).shuffle(clusters)
    train_end = int(0.90 * len(clusters))
    validation_end = train_end + int(0.05 * len(clusters))
    labels = (["train"] * train_end + ["validation"] * (validation_end - train_end)
              + ["test"] * (len(clusters) - validation_end))
    result: dict[str, str] = {}
    for representative, split in zip(clusters, labels):
        for member in members[representative]:
            if member in result:
                raise RuntimeError(f"FoldSeek member occurs in multiple clusters: {member}")
            result[member] = split
    return result


def materialize_search_directories(
    pdb_dir: Path, cluster_tsv: Path, work_dir: Path, *, seed: int = 42
) -> dict[str, int]:
    """Create zero-copy FoldSeek train/held-out inputs from the fixed cluster split."""
    split_by_domain = assign_cluster_splits(cluster_tsv, seed)
    structures: dict[str, Path] = {}
    for path in sorted(pdb_dir.iterdir()):
        if path.suffix.lower() not in {".pdb", ".cif", ".mmcif"}:
            continue
        key = _domain_key(path.name)
        if key in structures:
            raise RuntimeError(f"Ambiguous structure basename after normalization: {key}")
        structures[key] = path.resolve()
    missing = sorted(set(split_by_domain) - set(structures))
    if missing:
        raise RuntimeError(f"FoldSeek clusters reference missing structures: {missing[:5]}")
    counts = {"train": 0, "heldout": 0}
    for label in counts:
        (work_dir / f"{label}_pdbs").mkdir(parents=True, exist_ok=True)
    for domain, split in split_by_domain.items():
        label = "train" if split == "train" else "heldout"
        source = structures[domain]
        destination = work_dir / f"{label}_pdbs" / source.name
        if destination.exists():
            if not os.path.samefile(destination, source):
                raise RuntimeError(f"Refusing to replace existing search input: {destination}")
        else:
            try:
                os.link(source, destination)
            except OSError:
                destination.symlink_to(source)
        counts[label] += 1
    return counts


def read_query_tm(path: Path) -> dict[str, float]:
    scores: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        query, _target, value = line.split("\t")[:3]
        query = _domain_key(query)
        scores[query] = max(scores.get(query, 0.0), float(value))
    return scores


def _structure_inventory_sha256(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(_sha256(path)))
    return digest.hexdigest()


def _load_tm_structures(pdb_dir: Path) -> list[tuple[str, np.ndarray, str]]:
    from esm.inverse_folding.util import extract_coords_from_structure, load_structure

    result = []
    for path in sorted(pdb_dir.iterdir()):
        if path.suffix.lower() not in {".pdb", ".cif", ".mmcif"}:
            continue
        structure = load_structure(str(path), "A")
        coords, sequence = extract_coords_from_structure(structure)
        ca = np.asarray(coords[:, 1, :], dtype=np.float64)
        if len(ca) != len(sequence) or not np.isfinite(ca).all():
            raise ValueError(f"TM-align input is incomplete: {path}")
        result.append((_domain_key(path.name), ca, sequence))
    return result


def seed_tmalign_subset_cache(
    pdb_dir: Path,
    work_dir: Path,
    parent_pdb_dir: Path,
    parent_work_dir: Path,
) -> dict[str, int | str]:
    """Reuse exact independent pairwise scores for a strict population subset."""
    paths = sorted(
        path for path in pdb_dir.iterdir()
        if path.suffix.lower() in {".pdb", ".cif", ".mmcif"}
    )
    parent_paths = sorted(
        path for path in parent_pdb_dir.iterdir()
        if path.suffix.lower() in {".pdb", ".cif", ".mmcif"}
    )
    names = [_domain_key(path.name) for path in paths]
    parent_names = [_domain_key(path.name) for path in parent_paths]
    parent_index = {name: index for index, name in enumerate(parent_names)}
    if len(parent_index) != len(parent_names) or not set(names) <= set(parent_names):
        raise RuntimeError("Eligible structures are not an unambiguous subset of parent TM-align inputs")
    indices = np.asarray([parent_index[name] for name in names], dtype=np.int64)
    parent_scores_path = parent_work_dir / "tmalign_query_normalized_scores.npy"
    parent_coverage_path = parent_work_dir / "tmalign_minimum_span_coverage.npy"
    parent_scores = np.load(parent_scores_path, allow_pickle=False, mmap_mode="r")
    parent_coverage = np.load(parent_coverage_path, allow_pickle=False, mmap_mode="r")
    expected_shape = (len(parent_names), len(parent_names))
    if parent_scores.shape != expected_shape or parent_coverage.shape != expected_shape:
        raise RuntimeError("Parent TM-align matrices disagree with parent structure inventory")
    scores = np.asarray(parent_scores[np.ix_(indices, indices)], dtype=np.float32)
    coverage = np.asarray(parent_coverage[np.ix_(indices, indices)], dtype=np.float32)
    work_dir.mkdir(parents=True, exist_ok=True)
    rows_dir = work_dir / "tmalign_rows_v2"
    rows_dir.mkdir(exist_ok=True)
    state = {
        "structure_inventory_sha256": _structure_inventory_sha256(paths),
        "structure_count": len(paths),
        "tmtools_version": importlib.metadata.version("tmtools"),
        "row_schema": 2,
    }
    (rows_dir / "state.json").write_text(
        json.dumps(state, indent=2, sort_keys=True), encoding="utf-8"
    )
    for index in range(len(paths) - 1):
        temporary = rows_dir / f"{index:04}.npz.tmp"
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                forward=scores[index, index + 1 :],
                reverse=scores[index + 1 :, index],
                minimum_coverage=coverage[index, index + 1 :],
            )
        os.replace(temporary, rows_dir / f"{index:04}.npz")
    provenance = {
        "method": "exact principal submatrix of independently computed all-pairs TM-align scores",
        "parent_structure_count": len(parent_names),
        "eligible_structure_count": len(names),
        "parent_score_matrix_sha256": _sha256(parent_scores_path),
        "parent_coverage_matrix_sha256": _sha256(parent_coverage_path),
        "eligible_structure_inventory_sha256": state["structure_inventory_sha256"],
    }
    (work_dir / "pairwise_subset_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8"
    )
    return provenance


def _init_tm_workers(structures: list[tuple[str, np.ndarray, str]]) -> None:
    global _TM_STRUCTURES
    _TM_STRUCTURES = structures


def _alignment_span_coverage(left: str, right: str) -> tuple[float, float]:
    matches = [
        position
        for position, pair in enumerate(zip(left, right))
        if pair[0] != "-" and pair[1] != "-"
    ]
    if not matches:
        return 0.0, 0.0
    start, end = matches[0], matches[-1] + 1
    return (
        sum(value != "-" for value in left[start:end]) / sum(value != "-" for value in left),
        sum(value != "-" for value in right[start:end]) / sum(value != "-" for value in right),
    )


def _align_tm_row(index: int) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    from tmtools import tm_align

    _, coords, sequence = _TM_STRUCTURES[index]
    forward = np.empty(len(_TM_STRUCTURES) - index - 1, dtype=np.float32)
    reverse = np.empty_like(forward)
    minimum_coverage = np.empty_like(forward)
    for offset, (_, other_coords, other_sequence) in enumerate(_TM_STRUCTURES[index + 1 :]):
        result = tm_align(coords, other_coords, sequence, other_sequence)
        forward[offset] = result.tm_norm_chain1
        reverse[offset] = result.tm_norm_chain2
        minimum_coverage[offset] = min(_alignment_span_coverage(result.seqxA, result.seqyA))
    return index, forward, reverse, minimum_coverage


def _set_cover_clusters(
    scores: np.ndarray,
    threshold: float,
    names: list[str],
    minimum_coverage: np.ndarray | None = None,
    coverage_threshold: float = 0.0,
) -> list[list[int]]:
    """Deterministic exhaustive analogue of FoldSeek/MMseqs set-cover mode."""
    count = len(scores)
    neighbors = [
        {
            other
            for other in range(count)
            if max(float(scores[index, other]), float(scores[other, index])) >= threshold
            and (
                minimum_coverage is None
                or float(minimum_coverage[index, other]) >= coverage_threshold
            )
        }
        | {index}
        for index in range(count)
    ]
    uncovered = set(range(count))
    representatives: list[int] = []
    while uncovered:
        # FoldSeek's set-cover implementation selects the largest remaining
        # set. A stable lexical tie break makes the replacement reproducible.
        representative = max(
            range(count), key=lambda index: (len(neighbors[index] & uncovered), names[index])
        )
        covered = neighbors[representative] & uncovered
        if not covered:
            raise RuntimeError("Set-cover clustering failed to cover an input structure")
        representatives.append(representative)
        uncovered -= covered
    groups: dict[int, list[int]] = {representative: [] for representative in representatives}
    for member in range(count):
        candidates = [representative for representative in representatives if member in neighbors[representative]]
        representative = max(
            candidates,
            key=lambda index: (float(scores[index, member]), names[index]),
        )
        groups[representative].append(member)
    return list(groups.values())


def tmalign_fallback_cluster(
    pdb_dir: Path,
    work_dir: Path,
    *,
    workers: int,
    threshold: float = 0.5,
    coverage_threshold: float = 0.8,
    seed: int = 42,
) -> dict[str, int | str]:
    """Exhaustive Windows fallback using TM-align, explicitly not FoldSeek."""
    paths = sorted(
        path for path in pdb_dir.iterdir()
        if path.suffix.lower() in {".pdb", ".cif", ".mmcif"}
    )
    inventory_hash = _structure_inventory_sha256(paths)
    work_dir.mkdir(parents=True, exist_ok=True)
    rows_dir = work_dir / "tmalign_rows_v2"
    rows_dir.mkdir(exist_ok=True)
    state_path = rows_dir / "state.json"
    state = {
        "structure_inventory_sha256": inventory_hash,
        "structure_count": len(paths),
        "tmtools_version": importlib.metadata.version("tmtools"),
        "row_schema": 2,
    }
    if state_path.exists() and json.loads(state_path.read_text(encoding="utf-8")) != state:
        raise RuntimeError("TM-align row cache does not match the current inputs")
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    names = [_domain_key(path.name) for path in paths]
    pending = [index for index in range(len(paths) - 1) if not (rows_dir / f"{index:04}.npz").exists()]
    if pending:
        structures = _load_tm_structures(pdb_dir)
        if [item[0] for item in structures] != names:
            raise RuntimeError("TM-align structure order disagrees with the hashed inventory")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=workers, initializer=_init_tm_workers, initargs=(structures,)
        ) as executor:
            for index, forward, reverse, minimum_coverage in executor.map(
                _align_tm_row, pending, chunksize=1
            ):
                temporary = rows_dir / f"{index:04}.npz.tmp"
                with temporary.open("wb") as handle:
                    np.savez_compressed(
                        handle,
                        forward=forward,
                        reverse=reverse,
                        minimum_coverage=minimum_coverage,
                    )
                os.replace(temporary, rows_dir / f"{index:04}.npz")
    count = len(paths)
    scores = np.eye(count, dtype=np.float32)
    minimum_coverage = np.eye(count, dtype=np.float32)
    for index in range(count - 1):
        with np.load(rows_dir / f"{index:04}.npz", allow_pickle=False) as row:
            scores[index, index + 1 :] = row["forward"]
            scores[index + 1 :, index] = row["reverse"]
            minimum_coverage[index, index + 1 :] = row["minimum_coverage"]
            minimum_coverage[index + 1 :, index] = row["minimum_coverage"]
    score_path = work_dir / "tmalign_query_normalized_scores.npy"
    np.save(score_path, scores, allow_pickle=False)
    coverage_path = work_dir / "tmalign_minimum_span_coverage.npy"
    np.save(coverage_path, minimum_coverage, allow_pickle=False)
    components = _set_cover_clusters(
        scores, threshold, names, minimum_coverage, coverage_threshold
    )
    cluster_path = work_dir / "clusters_cluster.tsv"
    lines = []
    for component in sorted(components, key=lambda values: min(names[index] for index in values)):
        representative = min(names[index] for index in component)
        lines.extend(f"{representative}.pdb\t{names[index]}.pdb\n" for index in sorted(component, key=names.__getitem__))
    cluster_path.write_text("".join(lines), encoding="utf-8")
    split_by_domain = assign_cluster_splits(cluster_path, seed)
    train_indices = [index for index, name in enumerate(names) if split_by_domain[name] == "train"]
    heldout_indices = [index for index, name in enumerate(names) if split_by_domain[name] != "train"]
    qtm_path = work_dir / "heldout_to_train.tsv"
    qtm_lines = []
    for query in heldout_indices:
        target = max(train_indices, key=lambda index: (float(scores[query, index]), names[index]))
        qtm_lines.append(f"{names[query]}.pdb\t{names[target]}.pdb\t{float(scores[query, target]):.9g}\n")
    qtm_path.write_text("".join(qtm_lines), encoding="utf-8")
    metadata = {
        **state,
        "backend": "tmtools-tmalign-fallback",
        "tmalign_version": "20210224",
        "threshold": threshold,
        "coverage_threshold": coverage_threshold,
        "edge_rule": "max(query-normalized-TM-score-in-either-direction)>=threshold",
        "cluster_rule": "deterministic-greedy-set-cover; closest exhaustive analogue of FoldSeek 8 default SET_COVER",
        "seed": seed,
        "cluster_count": len(components),
        "cluster_tsv_sha256": _sha256(cluster_path),
        "qtm_tsv_sha256": _sha256(qtm_path),
        "score_matrix_sha256": _sha256(score_path),
        "coverage_matrix_sha256": _sha256(coverage_path),
        "limitation": "Strong Windows fallback; not FoldSeek 8.ef4e960 and not method-identical to ProteinDPO.",
    }
    subset_provenance = work_dir / "pairwise_subset_provenance.json"
    if subset_provenance.exists():
        metadata["pairwise_score_provenance"] = json.loads(
            subset_provenance.read_text(encoding="utf-8")
        )
    cluster_path.with_suffix(cluster_path.suffix + ".metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    return {"structures": count, "clusters": len(components), "heldout": len(heldout_indices), "backend": metadata["backend"]}


def _parse_substitutions(value: str) -> list[re.Match[str]] | None:
    parts = str(value).split(":")
    if len(parts) not in (1, 2):
        return None
    matches = [SUBSTITUTION_PATTERN.fullmatch(part) for part in parts]
    return matches if all(matches) else None  # type: ignore[return-value]


_mutations = _parse_substitutions


def curate_eligible_rows(source_csv: Path, pdb_dir: Path) -> pd.DataFrame:
    """Apply every study eligibility rule before structural operations."""
    required = {"WT_name", "aa_seq", "mut_type", "ddG_ML"}
    frame = pd.read_csv(source_csv, usecols=sorted(required), low_memory=False)
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"MegaScale table is missing official columns: {sorted(missing)}")
    native = (
        frame.loc[frame["mut_type"].astype(str).str.lower().eq("wt"), ["WT_name", "aa_seq"]]
        .drop_duplicates("WT_name")
        .set_index("WT_name")["aa_seq"]
        .to_dict()
    )
    native_by_domain = {_domain_key(str(key)): str(value) for key, value in native.items()}
    if len(native_by_domain) != len(native):
        raise RuntimeError("Native domain identifiers collide after filename normalization")
    frame["ddG_ML"] = pd.to_numeric(frame["ddG_ML"], errors="coerce")
    frame = frame.dropna(subset=["WT_name", "aa_seq", "mut_type", "ddG_ML"]).copy()
    frame["parsed_substitutions"] = frame["mut_type"].map(_parse_substitutions)
    frame = frame[frame["parsed_substitutions"].notna()].copy()
    frame = frame.drop_duplicates(subset=["WT_name", "aa_seq"], keep="first")
    structures: dict[str, Path] = {}
    for path in pdb_dir.iterdir():
        if path.suffix.lower() not in {".pdb", ".cif", ".mmcif"}:
            continue
        key = _domain_key(path.name)
        if key in structures:
            raise RuntimeError(f"Ambiguous structure basename after normalization: {key}")
        structures[key] = path.resolve()
    rows: list[dict[str, object]] = []
    for item in frame.itertuples(index=False):
        raw_domain = str(item.WT_name)
        domain = _domain_key(raw_domain)
        structure_domain = _structure_key(raw_domain)
        if domain not in native_by_domain:
            continue
        variant = str(item.aa_seq)
        wildtype = native_by_domain[domain]
        matches = item.parsed_substitutions
        if len(variant) != len(wildtype) or any(match is None for match in matches):
            continue
        observed = sum(a != b for a, b in zip(wildtype, variant))
        if observed != len(matches) or observed not in (1, 2):
            continue
        substitution_positions: set[int] = set()
        valid_substitutions = True
        for match in matches:
            assert match is not None
            position = int(match.group(2)) - 1
            if (
                position < 0
                or position >= len(wildtype)
                or wildtype[position] != match.group(1)
                or variant[position] != match.group(3)
            ):
                valid_substitutions = False
                break
            substitution_positions.add(position)
        changed_positions = {
            index for index, (left, right) in enumerate(zip(wildtype, variant)) if left != right
        }
        if not valid_substitutions or substitution_positions != changed_positions:
            continue
        structure = structures.get(structure_domain)
        if structure is None:
            raise FileNotFoundError(f"Could not uniquely resolve structure for {structure_domain}")
        rows.append({
            "domain_id": domain,
            "structure_domain": structure_domain,
            "structure_path": structure.resolve(),
            "native_sequence": wildtype,
            "target_sequence": variant,
            "ddg": float(item.ddG_ML),
            "substitution_count": observed,
        })
    if not rows:
        raise RuntimeError("Curation produced no valid single/double-substitution records")
    return pd.DataFrame(rows)


def materialize_eligible_structures(
    source_csv: Path, pdb_dir: Path, output_dir: Path
) -> dict[str, int | str]:
    """Create the exact structure population that may influence clustering."""
    eligible = curate_eligible_rows(source_csv, pdb_dir)
    structures = (
        eligible[["structure_domain", "structure_path"]]
        .drop_duplicates()
        .sort_values("structure_domain")
    )
    if structures["structure_domain"].duplicated().any():
        raise RuntimeError("Eligible structure identifiers are ambiguous")
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_names: set[str] = set()
    for row in structures.itertuples(index=False):
        source = Path(row.structure_path)
        destination = output_dir / source.name
        expected_names.add(source.name)
        if destination.exists():
            if not os.path.samefile(destination, source):
                raise RuntimeError(f"Refusing to replace eligible structure input: {destination}")
        else:
            try:
                os.link(source, destination)
            except OSError:
                destination.symlink_to(source)
    unexpected = sorted(
        path.name for path in output_dir.iterdir()
        if path.name not in expected_names and path.name != "eligibility.json"
    )
    if unexpected:
        raise RuntimeError(f"Eligible structure directory contains excluded entries: {unexpected[:5]}")
    structure_paths = sorted(output_dir / name for name in expected_names)
    result: dict[str, int | str] = {
        "eligible_record_count": len(eligible),
        "eligible_domain_count": int(eligible["domain_id"].nunique()),
        "eligible_structure_count": len(structure_paths),
        "eligible_structure_inventory_sha256": _structure_inventory_sha256(structure_paths),
        "source_csv_sha256": _sha256(source_csv),
    }
    (output_dir / "eligibility.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    return result


def build_manifest(
    source_csv: Path,
    pdb_dir: Path,
    cluster_tsv: Path,
    qtm_tsv: Path,
    output: Path,
    *,
    seed: int = 42,
) -> dict[str, int]:
    frame = curate_eligible_rows(source_csv, pdb_dir)
    splits = assign_cluster_splits(cluster_tsv, seed)
    eligible_structures = set(frame["structure_domain"])
    if set(splits) != eligible_structures:
        raise RuntimeError(
            "Structural clusters must contain exactly the final eligible population; "
            f"excluded={sorted(set(splits) - eligible_structures)[:5]}, "
            f"missing={sorted(eligible_structures - set(splits))[:5]}"
        )
    qtm = read_query_tm(qtm_tsv)
    heldout = {name for name, split in splits.items() if split != "train"}
    if set(qtm) != heldout:
        raise RuntimeError(
            "Held-out structural distances must contain exactly eligible held-out structures; "
            f"extra={sorted(set(qtm) - heldout)[:5]}, missing={sorted(heldout - set(qtm))[:5]}"
        )
    rows = []
    for item in frame.itertuples(index=False):
        split = splits[item.structure_domain]
        maximum = 1.0 if split == "train" else qtm.get(item.structure_domain)
        if maximum is None:
            raise RuntimeError(f"Missing query-normalized train similarity for held-out domain {item.domain_id}")
        identifier = hashlib.sha256(f"{item.domain_id}\0{item.target_sequence}".encode()).hexdigest()[:24]
        rows.append({
            "domain_id": item.domain_id,
            "sequence_id": identifier,
            "split": split,
            "backbone_path": str(Path(os.path.relpath(item.structure_path, output.parent))),
            "chain_id": "A",
            "native_sequence": item.native_sequence,
            "target_sequence": item.target_sequence,
            "variant_sequence": item.target_sequence,
            "ddg": item.ddg,
            "substitution_count": item.substitution_count,
            "mutation_count": item.substitution_count,
            "foldseek_qtm_max_to_train": float(maximum),
        })
    if not rows:
        raise RuntimeError("Curation produced no valid single/double-substitution records")
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(output, index=False)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    counts = pd.DataFrame(rows).groupby("split").size().astype(int).to_dict()
    clustering_sidecar = cluster_tsv.with_suffix(cluster_tsv.suffix + ".metadata.json")
    clustering = (
        json.loads(clustering_sidecar.read_text(encoding="utf-8"))
        if clustering_sidecar.exists()
        else {"backend": "foldseek", "foldseek_version": "8.ef4e960"}
    )
    fallback = clustering.get("backend") == "tmtools-tmalign-fallback"
    metadata = {
        "dataset": "ProteinDPO curated Megascale v2 2023-04-20",
        "source": "Zenodo 7992926; ProteinDPO Methods reconstruction",
        "split_provenance": (
            "reconstructed-tmalign-fallback" if fallback else "reconstructed-proteindpo-methods"
        ),
        "split_seed": seed,
        "split_fractions": [0.90, 0.05, 0.05],
        "foldseek_version": None if fallback else "8.ef4e960",
        "clustering_backend": clustering,
        "foldseek_metric": "query-normalized-tm-score",
        "split_algorithm": "lexicographic-cluster-ids; python-random-v1-seed-42-shuffle; floor-90%-floor-5%-remainder",
        "manifest_sha256": digest,
        "source_csv_sha256": _sha256(source_csv),
        "cluster_tsv_sha256": _sha256(cluster_tsv),
        "heldout_to_train_qtm_tsv_sha256": _sha256(qtm_tsv),
        "counts": counts,
        "eligible_record_count": len(frame),
        "eligible_domain_count": int(frame["domain_id"].nunique()),
        "eligible_structure_count": len(eligible_structures),
        "population_order": "final eligibility filtering precedes clustering, splitting, and heldout-to-train search",
        "limitation": "The authors did not release their curated split; this is a deterministic reconstruction.",
    }
    output.with_suffix(output.suffix + ".metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download")
    download.add_argument("--output-dir", type=Path, required=True)
    extract = commands.add_parser("extract")
    extract.add_argument("--download-dir", type=Path, required=True)
    extract.add_argument("--output-dir", type=Path, required=True)
    eligible = commands.add_parser("eligible")
    eligible.add_argument("--source-csv", type=Path, required=True)
    eligible.add_argument("--pdb-dir", type=Path, required=True)
    eligible.add_argument("--output-dir", type=Path, required=True)
    plan = commands.add_parser("foldseek-plan")
    plan.add_argument("--pdb-dir", type=Path, required=True)
    plan.add_argument("--work-dir", type=Path, required=True)
    plan.add_argument("--output", type=Path, required=True)
    search_dirs = commands.add_parser("materialize-search-dirs")
    search_dirs.add_argument("--pdb-dir", type=Path, required=True)
    search_dirs.add_argument("--cluster-tsv", type=Path, required=True)
    search_dirs.add_argument("--work-dir", type=Path, required=True)
    fallback = commands.add_parser("tmalign-fallback")
    fallback.add_argument("--pdb-dir", type=Path, required=True)
    fallback.add_argument("--work-dir", type=Path, required=True)
    fallback.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    subset = commands.add_parser("tmalign-subset-cache")
    subset.add_argument("--pdb-dir", type=Path, required=True)
    subset.add_argument("--work-dir", type=Path, required=True)
    subset.add_argument("--parent-pdb-dir", type=Path, required=True)
    subset.add_argument("--parent-work-dir", type=Path, required=True)
    manifest = commands.add_parser("manifest")
    manifest.add_argument("--source-csv", type=Path, required=True)
    manifest.add_argument("--pdb-dir", type=Path, required=True)
    manifest.add_argument("--cluster-tsv", type=Path, required=True)
    manifest.add_argument("--qtm-tsv", type=Path, required=True)
    manifest.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "download":
        download_published_data(args.output_dir)
    elif args.command == "extract":
        print(extract_published_data(args.download_dir, args.output_dir))
    elif args.command == "eligible":
        print(materialize_eligible_structures(args.source_csv, args.pdb_dir, args.output_dir))
    elif args.command == "foldseek-plan":
        write_foldseek_plan(args.pdb_dir, args.work_dir, args.output)
    elif args.command == "materialize-search-dirs":
        print(materialize_search_directories(args.pdb_dir, args.cluster_tsv, args.work_dir))
    elif args.command == "tmalign-fallback":
        print(tmalign_fallback_cluster(args.pdb_dir, args.work_dir, workers=args.workers))
    elif args.command == "tmalign-subset-cache":
        print(seed_tmalign_subset_cache(
            args.pdb_dir, args.work_dir, args.parent_pdb_dir, args.parent_work_dir
        ))
    else:
        print(build_manifest(args.source_csv, args.pdb_dir, args.cluster_tsv, args.qtm_tsv, args.output))


if __name__ == "__main__":
    main()
