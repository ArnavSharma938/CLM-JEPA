#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PHASE=${1:-supermini}
PYTHON=${PYTHON:-.venv/bin/python}
ROOT=runs/vjepa2_1/a6000
TRAIN=data/clm_jepa_uspto_mit_pilot_1280/uspto_mit_train.csv
VALIDATION=data/clm_jepa_uspto_mit_validation_256/uspto_mit_validation_length_stratified_256.csv

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
import json
import torch
print(json.dumps({
    "torch": torch.__version__,
    "cuda_runtime": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
}))
PY
    "$PYTHON" -m pytest -q
    ;;
  supermini)
    out="$ROOT/supermini"
    mkdir -p "$out"
    "$PYTHON" src/train.py \
      --gate 4 \
      --dataset uspto_mit_synthesis \
      --condition clm_jepa_vjepa2_1 \
      --seed 533 \
      --learning-rate 0.0001 \
      --k 0 \
      --lambda-eff 1.0 \
      --dropout 0.5 \
      --epochs 1 \
      --stop-after-epoch 1 \
      --batch-size 2 \
      --gradient-accumulation-steps 1 \
      --evaluation-epochs 1 \
      --no-gradient-checkpointing \
      --fused-adamw \
      --attention-implementation sdpa \
      --pin-memory \
      --dataloader-workers 0 \
      --eval-generation-batch-size 1 \
      --data-fraction 0.0125 \
      --train-manifest "$TRAIN" \
      --validation-manifest "$VALIDATION" \
      --checkpoint-dir "$out/checkpoints" \
      --max-train-rows 16 \
      --max-validation-rows 8 \
      --no-wandb \
      --output "$out/result.json"
    ;;
  pilot)
    out="$ROOT/pilot"
    mkdir -p "$out"
    "$PYTHON" src/train.py \
      --gate 4 \
      --dataset uspto_mit_synthesis \
      --condition clm_jepa_vjepa2_1 \
      --seed 533 \
      --learning-rate 0.0001 \
      --k 0 \
      --lambda-eff 1.0 \
      --dropout 0.5 \
      --epochs 4 \
      --stop-after-epoch 4 \
      --batch-size 4 \
      --gradient-accumulation-steps 4 \
      --evaluation-epochs 4 \
      --no-gradient-checkpointing \
      --fused-adamw \
      --attention-implementation sdpa \
      --pin-memory \
      --dataloader-workers 0 \
      --eval-generation-batch-size 1 \
      --data-fraction 1.0 \
      --train-manifest "$TRAIN" \
      --validation-manifest "$VALIDATION" \
      --checkpoint-dir "$out/checkpoints" \
      --max-train-rows 1280 \
      --max-validation-rows 256 \
      --no-wandb \
      --output "$out/result.json"
    ;;
  *)
    echo "usage: $0 {setup|supermini|pilot}" >&2
    exit 2
    ;;
esac
