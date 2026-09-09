"""Write and verify the compact immutable manifest for Report 07."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/faithful_nextlat/artifact_manifest.json"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    files = [
        "runs/faithful_nextlat/preflight.json",
        "runs/faithful_nextlat/behavioral_summary.json",
        "runs/faithful_nextlat/architecture/results.json",
        "runs/faithful_nextlat/architecture/complete.json",
        "runs/faithful_nextlat/diagnostics/seed_533.json",
        "runs/faithful_nextlat/diagnostics/seed_917.json",
        "runs/faithful_nextlat/seed_533/training/checkpoints/epoch_4/USPTO-MIT-Synthesis/adapter_model.safetensors",
        "runs/faithful_nextlat/seed_533/training/checkpoints/epoch_4/auxiliary_training_state.pt",
        "runs/faithful_nextlat/seed_533/evaluation/predictions.jsonl",
        "runs/faithful_nextlat/seed_533/comparison.json",
        "runs/faithful_nextlat/seed_917/training/checkpoints/epoch_4/USPTO-MIT-Synthesis/adapter_model.safetensors",
        "runs/faithful_nextlat/seed_917/training/checkpoints/epoch_4/auxiliary_training_state.pt",
        "runs/faithful_nextlat/seed_917/evaluation/predictions.jsonl",
        "runs/faithful_nextlat/seed_917/comparison.json",
        "runs/faithful_nextlat/seed_533/extension_corrected/checkpoints/step_480/USPTO-MIT-Synthesis/adapter_model.safetensors",
        "runs/faithful_nextlat/seed_533/extension_corrected/checkpoints/step_480/auxiliary_training_state.pt",
        "src/faithful_nextlat.py",
        "src/architecture_analysis.py",
        "scripts/run_faithful_nextlat.py",
        "scripts/diagnose_faithful_nextlat_checkpoints.py",
        "scripts/extend_faithful_nextlat.py",
        "scripts/extract_architecture_states.py",
        "scripts/analyze_faithful_nextlat_architecture.py",
        "scripts/summarize_faithful_nextlat.py",
        "scripts/finalize_faithful_nextlat.py",
        "tests/test_faithful_nextlat.py",
        "tests/test_architecture_analysis.py",
        "docs/reports/07_FAITHFUL_NEXTLAT_AND_CORRECTED_ARCHITECTURE_ANALYSIS.md",
    ]
    entries = {}
    for relative in files:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(relative)
        entries[relative] = {"bytes": path.stat().st_size, "sha256": digest(path)}

    for seed in (533, 917):
        diagnostics = json.loads((ROOT / f"runs/faithful_nextlat/diagnostics/seed_{seed}.json").read_text())
        checkpoint = next(row for row in diagnostics["checkpoints"] if row["step"] == 320)
        prefix = f"runs/faithful_nextlat/seed_{seed}/training/checkpoints/epoch_4"
        if entries[f"{prefix}/USPTO-MIT-Synthesis/adapter_model.safetensors"]["sha256"] != checkpoint["adapter_sha256"]:
            raise RuntimeError(f"seed {seed} adapter hash mismatch")
        if entries[f"{prefix}/auxiliary_training_state.pt"]["sha256"] != checkpoint["auxiliary_sha256"]:
            raise RuntimeError(f"seed {seed} auxiliary hash mismatch")

    payload = {
        "status": "complete",
        "scope": "development seeds 533/917; no 2027/3163 NextLat confirmation",
        "official_nextlat_commit": "3770be6009cea2b3c455a9ce7f2ca88b504bb955",
        "excluded_artifact": "runs/faithful_nextlat/seed_533/extension (invalid clipping semantics)",
        "files": entries,
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"{OUT}: {digest(OUT)}")


if __name__ == "__main__":
    main()
