from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

LRS = {"ft": 1e-7, "dtft": 1e-7, "lora8": 3e-5, "lora64": 3e-5}
COMPACT_EVIDENCE_SEEDS = (11, 23)


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def _run(command: list[str], ledger: list[dict], ledger_path: Path, outputs: list[Path]) -> None:
    expected = [str(path.resolve()) for path in outputs]
    if any(event.get("command") == command and event.get("status") == "complete" and event.get("outputs") == expected for event in ledger) and all(path.is_file() for path in outputs):
        return
    event = {"command": command, "outputs": expected, "status": "running", "started_unix": time.time()}
    ledger.append(event); _write(ledger_path, ledger)
    try:
        subprocess.run(command, check=True)
    except BaseException as error:
        event.update(status="failed", finished_unix=time.time(), error=repr(error)); _write(ledger_path, ledger); raise
    missing = [str(path) for path in outputs if not path.is_file()]
    if missing:
        event.update(status="failed", finished_unix=time.time(), error=f"missing outputs: {missing}"); _write(ledger_path, ledger); raise RuntimeError(event["error"])
    event.update(status="complete", finished_unix=time.time(), duration_seconds=time.time() - event["started_unix"]); _write(ledger_path, ledger)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compact balanced Gate-1 queue")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--cluster-tsv", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--model-checkpoint", required=True, type=Path)
    args = parser.parse_args()
    root = args.output_root / "gate1_compact"
    root.mkdir(parents=True, exist_ok=True)
    ledger_path = root / "execution_ledger.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8")) if ledger_path.is_file() else []
    provenance = args.data_dir / "compact_gate1_provenance.json"
    _run([sys.executable, "-m", "protein.calibration.prepare_compact_benchmark", "--manifest", str(args.manifest), "--cluster-tsv", str(args.cluster_tsv), "--output-dir", str(args.data_dir)], ledger, ledger_path, [provenance])
    manifest = args.data_dir / "megascale_gate1_compact_seed42.parquet"
    preflight = root / "preflight.json"
    _run([sys.executable, "-m", "protein.calibration.preflight", "--manifest", str(manifest), "--model-checkpoint", str(args.model_checkpoint), "--output", str(preflight)], ledger, ledger_path, [preflight])
    base_eval = root / "evaluation" / "base"
    _run([sys.executable, "-m", "protein.calibration.evaluate", "--mode", "base", "--seed", "11", "--manifest", str(manifest), "--model-checkpoint", str(args.model_checkpoint), "--output-dir", str(base_eval), "--batch-size", "64"], ledger, ledger_path, [base_eval / "summary.json"])
    for mode, lr in LRS.items():
        for seed in COMPACT_EVIDENCE_SEEDS:
            output = root / "evidence" / mode / f"seed_{seed}"
            _run([sys.executable, "-m", "protein.calibration.training", "--mode", mode, "--seed", str(seed), "--lr", str(lr), "--manifest", str(manifest), "--output-dir", str(output), "--checkpoint", str(args.model_checkpoint), "--epochs", "10", "--no-diagnostics", "--early-stopping-patience-checks", "4", "--minimum-epochs", "2"], ledger, ledger_path, [output / "result.json", output / "history.json", output / "checkpoints" / "best.pt"])
            evaluation = root / "evaluation" / mode / f"seed_{seed}"
            _run([sys.executable, "-m", "protein.calibration.evaluate", "--mode", mode, "--seed", str(seed), "--manifest", str(manifest), "--checkpoint", str(output / "checkpoints" / "best.pt"), "--model-checkpoint", str(args.model_checkpoint), "--output-dir", str(evaluation), "--batch-size", "64"], ledger, ledger_path, [evaluation / "summary.json"])
    summary = root / "gate1_summary.json"
    _run([sys.executable, "-m", "protein.calibration.summarize_compact_gate1", "--root", str(root), "--output", str(summary)], ledger, ledger_path, [summary])
    (root / "GATE1_COMPLETE").write_text("complete\n", encoding="utf-8")


if __name__ == "__main__":
    main()
