from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import defaultdict
from pathlib import Path

from .config import CLUSTER_COVERAGE, CLUSTER_IDENTITY, DIAMOND_VERSION
from .prepare_data import sha256_file


class UnionFind:
    def __init__(self, values: list[str]) -> None:
        self.parent = {value: value for value in values}
        self.rank = dict.fromkeys(values, 0)

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1

    def groups(self) -> list[list[str]]:
        members: dict[str, list[str]] = defaultdict(list)
        for value in self.parent:
            members[self.find(value)].append(value)
        return sorted((sorted(group) for group in members.values()), key=lambda group: group[0])


def run_diamond(fasta: Path, output_dir: Path, threads: int = 6) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    database = output_dir / "receptors.dmnd"
    matches = output_dir / "receptor_all_vs_all.tsv"
    temp_dir = output_dir / "diamond_tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    makedb_command = ["diamond", "makedb", "--in", str(fasta), "--db", str(database)]
    blast_command = [
        "diamond", "blastp", "--query", str(fasta), "--db", str(database),
        "--out", str(matches), "--outfmt", "6", "qseqid", "sseqid", "pident",
        "qcovhsp", "scovhsp", "evalue", "--very-sensitive", "--evalue", "1000",
        "--max-target-seqs", "0", "--threads", str(threads), "--tmpdir", str(temp_dir),
    ]
    makedb = subprocess.run(makedb_command, check=True, text=True, capture_output=True)
    align = subprocess.run(blast_command, check=True, text=True, capture_output=True)
    return {
        "version": DIAMOND_VERSION,
        "fasta_sha256": sha256_file(fasta),
        "makedb_command": makedb_command,
        "blastp_command": blast_command,
        "mode": "very-sensitive all-vs-all candidates; apply identity and reciprocal coverage thresholds before connected components",
        "candidate_evalue_max": 1000,
        "thresholds": {
            "identity_percent": CLUSTER_IDENTITY,
            "query_coverage_percent": CLUSTER_COVERAGE,
            "subject_coverage_percent": CLUSTER_COVERAGE,
        },
        "makedb_stderr": makedb.stderr.strip(),
        "blastp_stderr": align.stderr.strip(),
        "match_file": matches.name,
        "match_file_sha256": sha256_file(matches),
    }


def connected_components(fasta: Path, matches: Path) -> list[list[str]]:
    identifiers = []
    with fasta.open(encoding="ascii") as stream:
        for line in stream:
            if line.startswith(">"):
                identifiers.append(line[1:].strip())
    union_find = UnionFind(identifiers)
    with matches.open(encoding="utf-8") as stream:
        for line in stream:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 5:
                raise RuntimeError("DIAMOND output is missing identity or bidirectional coverage")
            query, subject = fields[:2]
            identity, query_coverage, subject_coverage = map(float, fields[2:5])
            if (
                identity >= CLUSTER_IDENTITY
                and query_coverage >= CLUSTER_COVERAGE
                and subject_coverage >= CLUSTER_COVERAGE
            ):
                union_find.union(query, subject)
    return union_find.groups()


def write_clusters(groups: list[list[str]], path: Path) -> dict:
    rows = []
    for members in groups:
        cluster_id = "cluster_" + hashlib.sha256("\0".join(members).encode()).hexdigest()
        rows.extend({"receptor_id": member[1:], "receptor_cluster_id": cluster_id} for member in members)
    rows.sort(key=lambda row: row["receptor_id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    return {
        "receptor_sequence_count": len(rows),
        "receptor_cluster_count": len(groups),
        "cluster_size_min": min(map(len, groups)),
        "cluster_size_median": sorted(map(len, groups))[(len(groups) - 1) // 2],
        "cluster_size_max": max(map(len, groups)),
        "clusters_file": path.name,
        "clusters_sha256": sha256_file(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="DIAMOND all-vs-all receptor family clustering")
    parser.add_argument("--fasta", type=Path, default=Path("data/protein/propedia26/receptors.fasta"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/protein/propedia26/clustering"))
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()
    version = subprocess.run(["diamond", "version"], check=True, text=True, capture_output=True).stdout.strip()
    if not version.endswith(DIAMOND_VERSION):
        raise RuntimeError(f"Expected DIAMOND {DIAMOND_VERSION}, found {version}")
    provenance = run_diamond(args.fasta, args.output_dir, args.threads)
    groups = connected_components(args.fasta, args.output_dir / provenance["match_file"])
    summary = {**provenance, **write_clusters(groups, args.output_dir / "receptor_clusters.jsonl")}
    (args.output_dir / "clustering_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
