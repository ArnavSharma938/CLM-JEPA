#!/usr/bin/env python
"""Resumable, sequential execution of the locked causal pilot on one GPU."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_stage(name: str, command: list[str], expected: Path, log_dir: Path,
              ledger: list[dict]):
    if expected.exists():
        ledger.append({"stage": name, "status": "reused", "output": str(expected)})
        return
    log_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    log_path = log_dir / f"{name}.log"
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    record = {
        "stage": name, "status": "complete" if completed.returncode == 0 else "failed",
        "returncode": completed.returncode, "elapsed_seconds": time.time() - started,
        "output": str(expected), "log": str(log_path), "command": command,
    }
    ledger.append(record)
    if completed.returncode or not expected.exists():
        raise RuntimeError(record)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--data", type=Path, default=ROOT / "data/causal_pilot")
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/causal_pilot.json")
    parser.add_argument("--skip-esmfold", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    logs = root / "logs"
    ledger_path = root / "execution_ledger.json"
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else []
    py = sys.executable

    def stage(name, command, expected):
        run_stage(name, command, expected, logs, ledger)
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        ledger_path.write_text(json.dumps(ledger, indent=2) + "\n")

    stage("parity", [py, str(ROOT / "scripts/validate_causal_pilot_parity.py"),
                     str(root / "parity_gpu.json")], root / "parity_gpu.json")
    stage("benchmark", [py, str(ROOT / "scripts/run_causal_pilot.py"),
                        "benchmark", str(args.data / "train_replicate_0.jsonl"),
                        str(root / "benchmark.json"), "--steps", "200"],
          root / "benchmark.json")
    for replicate in range(8):
        order = ("native", "nextlat") if replicate % 2 == 0 else ("nextlat", "native")
        for arm in order:
            expected = root / "training" / f"replicate_{replicate}" / arm / "training.json"
            stage(
                f"train_r{replicate}_{arm}",
                [py, str(ROOT / "scripts/run_causal_pilot.py"), "train",
                 str(args.data / f"train_replicate_{replicate}.jsonl"),
                 str(root / "training"), "--replicate", str(replicate), "--arm", arm],
                expected,
            )
    for replicate in range(8):
        for arm in ("native", "nextlat"):
            for fraction in (25, 50, 100):
                adapter = root / "training" / f"replicate_{replicate}" / arm / f"checkpoint_{fraction:03d}" / "adapter"
                output = root / "evaluation" / f"replicate_{replicate}" / arm / f"checkpoint_{fraction:03d}_primary.json"
                stage(
                    f"primary_r{replicate}_{arm}_{fraction}",
                    [py, str(ROOT / "scripts/evaluate_causal_pilot.py"), "primary",
                     str(adapter), str(output), "--data", str(args.data)], output,
                )
            adapter = root / "training" / f"replicate_{replicate}" / arm / "checkpoint_100" / "adapter"
            for evaluation in ("representations", "generation"):
                output = root / "evaluation" / f"replicate_{replicate}" / arm / f"checkpoint_100_{evaluation}.json"
                stage(
                    f"{evaluation}_r{replicate}_{arm}",
                    [py, str(ROOT / "scripts/evaluate_causal_pilot.py"), evaluation,
                     str(adapter), str(output), "--data", str(args.data)], output,
                )
        for fraction in (0, 25, 50, 100):
            output = root / "mechanism" / f"replicate_{replicate}" / f"checkpoint_{fraction:03d}.json"
            stage(
                f"mechanism_r{replicate}_{fraction}",
                [py, str(ROOT / "scripts/evaluate_causal_mechanism.py"),
                 str(root / "training" / f"replicate_{replicate}"), str(output),
                 "--replicate", str(replicate), "--fraction", str(fraction),
                 "--data", str(args.data)], output,
            )
    esmfold = root / "evaluation" / "esmfold.json"
    if not args.skip_esmfold:
        generation_files = [
            str(root / "evaluation" / f"replicate_{replicate}" / arm /
                "checkpoint_100_generation.json")
            for replicate in range(8) for arm in ("native", "nextlat")
        ]
        stage("esmfold", [py, str(ROOT / "scripts/run_esmfold_pilot.py"),
                          str(esmfold), *generation_files], esmfold)
    summary = root / "causal_pilot_summary.json"
    command = [py, str(ROOT / "scripts/summarize_causal_pilot.py"), str(root), str(summary)]
    if esmfold.exists():
        command.extend(["--esmfold", str(esmfold)])
    stage("summary", command, summary)


if __name__ == "__main__":
    main()
