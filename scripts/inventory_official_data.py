from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GROUPS = {
    "uspto_mit_synthesis": ("data/official/uspto_mit_synthesis/*", "https://clemson.app.box.com/s/kct8hy0pc0i7iyjlpmrxng8cyoj12i9v"),
    "uspto_50k": ("data/official/uspto_50k/*", "https://clemson.app.box.com/s/kct8hy0pc0i7iyjlpmrxng8cyoj12i9v"),
    "uspto_stereo_mixed": ("data/official/uspto_stereo/data/STEREO_mixed/*", "https://ibm.box.com/v/MolecularTransformerData"),
    "orderly": ("data/official/orderly/*.parquet", "https://figshare.com/articles/dataset/23298467"),
    "orderly_non_uspto": ("data/official/orderly/non_uspto_data/*.parquet", "https://figshare.com/articles/dataset/23502372"),
    "metatrans": ("data/official/metatrans/repo/datasets/**/*", "https://github.com/KavrakiLab/MetaTrans"),
    "retro_extrapolation": ("data/official/retro_extrapolation/*", "https://doi.org/10.6084/m9.figshare.30843134"),
    "nl_rx_synth": ("_references/llm-jepa/datasets/synth_*.jsonl", "https://github.com/Extensive-AI/llm-jepa"),
    "llama_checkpoint": ("data/models/llama-3.2-1b-instruct/model.safetensors", "https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct/tree/9213176726f574b556790deb65791e0c5aa438b6a"),
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    groups = {}
    for name, (pattern, source) in GROUPS.items():
        files = []
        for path in sorted(ROOT.glob(pattern)):
            if path.is_file() and not path.name.startswith((".~lock", ".DS_Store")):
                files.append({
                    "path": str(path.relative_to(ROOT)),
                    "bytes": path.stat().st_size,
                    "sha256": digest(path),
                })
        groups[name] = {"source": source, "files": files, "total_bytes": sum(item["bytes"] for item in files)}
    output = {
        "full_data_versioned": False,
        "checkpoint_revision": "9213176726f574b556790deb65791e0c5aa438b6a",
        "groups": groups,
    }
    path = ROOT / "artifacts" / "data" / "download_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({name: {"files": len(value["files"]), "bytes": value["total_bytes"]} for name, value in groups.items()}, indent=2))


if __name__ == "__main__":
    main()
