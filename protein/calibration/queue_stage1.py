from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from .config import EVIDENCE_SEEDS, HPO_LRS


def _write_ledger(path: Path, ledger: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(ledger, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _run(arguments: list[str], ledger: list[dict], ledger_path: Path, outputs: list[Path]) -> None:
    expected = [str(path.resolve()) for path in outputs]
    completed = any(
        event.get("command") == arguments
        and event.get("status") == "complete"
        and event.get("outputs") == expected
        for event in ledger
    )
    if completed and all(path.is_file() for path in outputs):
        return
    event = {"command": arguments, "outputs": expected, "status": "running"}
    ledger.append(event)
    _write_ledger(ledger_path, ledger)
    try:
        subprocess.run(arguments, check=True)
    except BaseException:
        event["status"] = "failed"
        _write_ledger(ledger_path, ledger)
        raise
    missing = [str(path) for path in outputs if not path.is_file()]
    if missing:
        event["status"] = "failed"
        event["error"] = f"command completed without required outputs: {missing}"
        _write_ledger(ledger_path, ledger)
        raise RuntimeError(event["error"])
    event["status"] = "complete"
    _write_ledger(ledger_path, ledger)


def _train_command(
    mode: str,
    seed: int,
    lr: float,
    manifest: Path,
    output: Path,
    model_checkpoint: Path | None,
    panel: Path | None,
    *,
    diagnostics: bool,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "protein.calibration.training",
        "--mode",
        mode,
        "--seed",
        str(seed),
        "--lr",
        str(lr),
        "--manifest",
        str(manifest),
        "--output-dir",
        str(output),
    ]
    if diagnostics:
        if panel is None:
            raise ValueError("Diagnostic training requires a fixed panel")
        command += ["--diagnostic-panel", str(panel)]
    else:
        command.append("--no-diagnostics")
    if model_checkpoint:
        command += ["--checkpoint", str(model_checkpoint)]
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description="Sequential Stage-1 queue; one GPU, no polling")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--model-checkpoint", type=Path, required=True)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    ledger_path = args.output_root / "execution_ledger.json"
    ledger: list[dict] = (
        json.loads(ledger_path.read_text(encoding="utf-8")) if ledger_path.is_file() else []
    )

    preflight = [
        sys.executable,
        "-m",
        "protein.calibration.preflight",
        "--manifest",
        str(args.manifest),
        "--output",
        str(args.output_root / "preflight.json"),
    ]
    if args.model_checkpoint:
        preflight += ["--model-checkpoint", str(args.model_checkpoint)]
    _run(preflight, ledger, ledger_path, [args.output_root / "preflight.json"])
    panel_path = args.output_root / "fixed_panels.json"
    _run(
        [sys.executable, "-m", "protein.calibration.panels", "--manifest", str(args.manifest), "--output", str(panel_path)],
        ledger,
        ledger_path,
        [panel_path],
    )
    selected_lrs: dict[str, float] = {}
    for mode, candidates in HPO_LRS.items():
        best: tuple[float, float] | None = None
        for lr in candidates:
            output = args.output_root / "hpo" / mode / f"lr_{lr:g}"
            _run(
                _train_command(
                    mode,
                    7,
                    lr,
                    args.manifest,
                    output,
                    args.model_checkpoint,
                    panel_path,
                    diagnostics=True,
                ),
                ledger,
                ledger_path,
                [output / "result.json"],
            )
            result = json.loads((output / "result.json").read_text(encoding="utf-8"))
            candidate = (float(result["best_validation_nll"]), lr)
            best = candidate if best is None or candidate < best else best
        assert best is not None
        selected_lrs[mode] = best[1]
    (args.output_root / "selected_lrs.json").write_text(json.dumps(selected_lrs, indent=2, sort_keys=True), encoding="utf-8")

    # Base is evaluated once; it has no adaptation seed uncertainty.
    base_eval = args.output_root / "evaluation" / "base"
    base_command = [
        sys.executable,
        "-m",
        "protein.calibration.evaluate",
        "--mode",
        "base",
        "--seed",
        "7",
        "--manifest",
        str(args.manifest),
        "--output-dir",
        str(base_eval),
    ]
    if args.model_checkpoint:
        base_command += ["--model-checkpoint", str(args.model_checkpoint)]
    _run(base_command, ledger, ledger_path, [base_eval / "summary.json"])
    for seed in EVIDENCE_SEEDS:
        generation = args.output_root / "generation" / "base" / f"seed_{seed}" / "best.jsonl"
        command = [
            sys.executable,
            "-m",
            "protein.calibration.generate",
            "--mode",
            "base",
            "--seed",
            str(seed),
            "--manifest",
            str(args.manifest),
            "--panel",
            str(panel_path),
            "--checkpoint-label",
            "best",
            "--output",
            str(generation),
        ]
        if args.model_checkpoint:
            command += ["--model-checkpoint", str(args.model_checkpoint)]
        _run(command, ledger, ledger_path, [generation])

    for mode, lr in selected_lrs.items():
        for seed in EVIDENCE_SEEDS:
            output = args.output_root / "evidence" / mode / f"seed_{seed}"
            _run(
                _train_command(
                    mode, seed, lr, args.manifest, output, args.model_checkpoint, panel_path, diagnostics=True
                ),
                ledger,
                ledger_path,
                [output / "result.json", output / "checkpoints" / "best.pt"],
            )
            evaluation = args.output_root / "evaluation" / mode / f"seed_{seed}"
            command = [
                sys.executable,
                "-m",
                "protein.calibration.evaluate",
                "--mode",
                mode,
                "--seed",
                str(seed),
                "--manifest",
                str(args.manifest),
                "--checkpoint",
                str(output / "checkpoints" / "best.pt"),
                "--output-dir",
                str(evaluation),
            ]
            if args.model_checkpoint:
                command += ["--model-checkpoint", str(args.model_checkpoint)]
            _run(command, ledger, ledger_path, [evaluation / "summary.json"])
            for label in ("best", "epoch_1", "epoch_10"):
                generation = args.output_root / "generation" / mode / f"seed_{seed}" / f"{label}.jsonl"
                generation_command = [
                    sys.executable,
                    "-m",
                    "protein.calibration.generate",
                    "--mode",
                    mode,
                    "--seed",
                    str(seed),
                    "--manifest",
                    str(args.manifest),
                    "--panel",
                    str(panel_path),
                    "--checkpoint",
                    str(output / "checkpoints" / f"{label}.pt"),
                    "--checkpoint-label",
                    label,
                    "--output",
                    str(generation),
                ]
                if args.model_checkpoint:
                    generation_command += ["--model-checkpoint", str(args.model_checkpoint)]
                _run(generation_command, ledger, ledger_path, [generation])

    _run(
        [
            sys.executable,
            "-m",
            "protein.calibration.generation_metrics",
            "prepare",
            "--generation-root",
            str(args.output_root / "generation"),
            "--output-dir",
            str(args.output_root / "external_generation_evaluation"),
        ],
        ledger,
        ledger_path,
        [args.output_root / "external_generation_evaluation" / "generation_manifest.parquet"],
    )
    _run(
        [
            sys.executable,
            "-m",
            "protein.calibration.summarize_stage1",
            "--root",
            str(args.output_root),
            "--output",
            str(args.output_root / "stage1_summary.json"),
        ],
        ledger,
        ledger_path,
        [args.output_root / "stage1_summary.json"],
    )
    (args.output_root / "STAGE1_COMPLETE").write_text("complete\n", encoding="utf-8")


if __name__ == "__main__":
    main()
