#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PHASE=${1:-setup}
PYTHON=${PYTHON:-.venv/bin/python}
ROOT=runs/stp/a6000

export PYTHONPATH=src
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$ROOT"/logs

case "$PHASE" in
  setup)
    if [[ ! -x .venv/bin/python ]] || ! .venv/bin/python -m pip --version >/dev/null 2>&1; then
      if ! dpkg-query -W -f='${Status}' python3.10-venv 2>/dev/null \
          | grep -q 'install ok installed'; then
        sudo apt-get update
        sudo DEBIAN_FRONTEND=noninteractive apt-get install -y python3.10-venv
      fi
      python3.10 -m venv --clear .venv
    fi
    "$PYTHON" -m pip install --upgrade pip
    "$PYTHON" -m pip install -r requirements.txt
    "$PYTHON" - <<'PY'
import hashlib
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path

import peft
import torch
import transformers

def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
payload = {
    "implementation_commit": os.environ.get("IMPLEMENTATION_COMMIT"),
    "deployment_archive_sha256": os.environ.get("DEPLOYMENT_ARCHIVE_SHA256"),
    "platform": platform.platform(), "python": platform.python_version(),
    "torch": torch.__version__, "transformers": transformers.__version__,
    "peft": peft.__version__, "cuda_runtime": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(), "gpu": gpu,
    "cpu_count": os.cpu_count(), "disk_total_bytes": shutil.disk_usage(".").total,
    "disk_free_bytes_after_setup": shutil.disk_usage(".").free,
    "nvidia_smi": subprocess.check_output([
        "nvidia-smi", "--query-gpu=name,driver_version,memory.total",
        "--format=csv,noheader",
    ], text=True).strip(),
    "manifests": {
        path: sha256(path) for path in (
            "data/clm_jepa_uspto_mit_pilot_1280/uspto_mit_train.csv",
            "data/clm_jepa_uspto_mit_validation_256/uspto_mit_validation_length_stratified_256.csv",
            "data/clm_jepa_uspto_mit_official_endpoint/prespecified_stage1_256.jsonl",
        )
    },
}
if not torch.cuda.is_available() or "A6000" not in gpu:
    raise SystemExit("the locked experiment requires one NVIDIA A6000")
root = Path("runs/stp/a6000")
root.mkdir(parents=True, exist_ok=True)
(root / "environment.json").write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, sort_keys=True))
PY
    "$PYTHON" -m pytest -q
    ;;
  download-model)
    "$PYTHON" scripts/download_chemfm_model.py
    ;;
  preflight)
    out="$ROOT/preflight"
    "$PYTHON" -u src/train.py \
      --gate 5 --dataset uspto_mit_synthesis --condition stp --seed 533 \
      --learning-rate 1e-4 --stp-lambda 0.02 --k 0 --lambda-eff 1.0 \
      --dropout 0.5 --epochs 1 --stop-after-epoch 1 --evaluation-epochs 1 \
      --batch-size 4 --gradient-accumulation-steps 4 \
      --no-gradient-checkpointing --fused-adamw --attention-implementation sdpa \
      --pin-memory --eval-generation-batch-size 1 \
      --train-manifest data/clm_jepa_uspto_mit_pilot_1280/uspto_mit_train.csv \
      --validation-manifest data/clm_jepa_uspto_mit_validation_256/uspto_mit_validation_length_stratified_256.csv \
      --max-train-rows 64 --max-validation-rows 2 \
      --checkpoint-dir "$out/checkpoints" --no-wandb --output "$out/result.json"
    "$PYTHON" - <<'PY'
import json
from pathlib import Path
p=json.loads(Path("runs/stp/a6000/preflight/result.json").read_text())
assert p["compute"]["optimizer_steps"] == 4
assert p["config"]["stp_upstream_commit"] == "ea0017c654ad917066ff32afc88276bea8ca5f7e"
assert p["config"]["stp_lambda"] == 0.02
assert p["diagnostics"]["final_epoch_mean_stp_loss"] == p["diagnostics"]["final_epoch_mean_stp_loss"]
print(json.dumps({"updates": 4, "seconds": p["compute"]["wall_time_seconds"], "peak_vram_bytes": p["compute"]["peak_vram_bytes"]}))
PY
    ;;
  run)
    "$PYTHON" -u scripts/run_stp_local.py \
      --output-root "$ROOT/results" --training-concurrency 3 \
      --batch-size 4 --gradient-accumulation-steps 4 \
      --workers 4 --teacher-batch-size 16 --phase all
    ;;
  *)
    echo "usage: $0 {setup|download-model|preflight|run}" >&2
    exit 2
    ;;
esac
