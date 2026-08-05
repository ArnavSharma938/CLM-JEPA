from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clm_jepa.paths import MODEL_DIR  # noqa: E402


REPOSITORY = "ChemFM/ChemFM-1B"
REVISION = "f99dc2e89726539bb9cf31b2e2b4360650bac6a8"
MODEL_SHA256 = "24686705d779db6876acc09c81d64d432262ef8b5dbfccc385212587079ce419"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    snapshot_download(
        repo_id=REPOSITORY,
        revision=REVISION,
        local_dir=MODEL_DIR,
        allow_patterns=["*.json", "model.safetensors"],
    )
    model_file = MODEL_DIR / "model.safetensors"
    observed = sha256(model_file)
    if observed != MODEL_SHA256:
        raise RuntimeError(f"ChemFM-1B SHA-256 mismatch: {observed}")
    provenance = {
        "repository": REPOSITORY,
        "revision": REVISION,
        "model_sha256": observed,
        "model_bytes": model_file.stat().st_size,
    }
    (MODEL_DIR / "PROVENANCE.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance))


if __name__ == "__main__":
    main()
