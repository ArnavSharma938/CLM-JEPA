from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exact parallel ChemFM evaluation using independent batch-1 workers"
    )
    parser.add_argument("--condition", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--panel-reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")

    panel = sorted(read_jsonl(args.panel_reference), key=lambda row: row["panel_index"])
    existing = read_jsonl(args.output)
    completed = {row["reaction_identity"] for row in existing}
    missing = [row for row in panel if row["reaction_identity"] not in completed]
    if not missing:
        print(json.dumps({"status": "complete", "rows": len(existing)}))
        return

    work = args.output.parent / f".{args.output.stem}_parallel"
    work.mkdir(parents=True, exist_ok=True)
    processes = []
    shard_outputs = []
    for worker in range(min(args.workers, len(missing))):
        shard = missing[worker::args.workers]
        reference = work / f"reference_{worker}.jsonl"
        output = work / f"output_{worker}.jsonl"
        log = work / f"worker_{worker}.log"
        write_jsonl(reference, [
            {"panel_index": row["panel_index"], "reaction_identity": row["reaction_identity"]}
            for row in shard
        ])
        if output.exists():
            output.unlink()
        command = [
            sys.executable, str(Path(__file__).with_name("decoder_coupling.py")), "generate",
            "--condition", args.condition,
            "--checkpoint", str(args.checkpoint),
            "--panel-reference", str(reference),
            "--generation-batch-size", "1",
            "--output", str(output),
        ]
        handle = log.open("w", encoding="utf-8")
        processes.append((subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT), handle))
        shard_outputs.append(output)
    failures = []
    for process, handle in processes:
        returncode = process.wait()
        handle.close()
        if returncode:
            failures.append(returncode)
    if failures:
        raise RuntimeError(f"parallel generation workers failed: {failures}")

    combined = existing + [row for path in shard_outputs for row in read_jsonl(path)]
    by_identity = {row["reaction_identity"]: row for row in combined}
    expected = {row["reaction_identity"] for row in panel}
    if len(by_identity) != len(combined) or set(by_identity) != expected:
        raise RuntimeError("parallel outputs do not exactly cover the frozen panel")
    ordered = [by_identity[row["reaction_identity"]] for row in panel]
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    write_jsonl(temporary, ordered)
    temporary.replace(args.output)
    print(json.dumps({
        "status": "complete", "rows": len(ordered), "workers": len(processes),
        "generation_batch_size_per_worker": 1, "reused_rows": len(existing),
    }))


if __name__ == "__main__":
    main()
