#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PHASE=${1:-preflight}
PYTHON=${PYTHON:-.venv/bin/python}
ROOT=runs/pair_residual/a6000
TRAIN=data/clm_jepa_uspto_mit_pilot_1280/uspto_mit_train.csv
VALIDATION=data/clm_jepa_uspto_mit_validation_256/uspto_mit_validation_length_stratified_256.csv
ENDPOINT=data/clm_jepa_uspto_mit_official_endpoint/prespecified_stage1_256.jsonl
EQUIVALENCE=data/clm_jepa_uspto_mit_official_endpoint/equivalence_24.jsonl

export PYTHONPATH=src
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$ROOT"/logs

exact_decode_environment() {
  export OMP_NUM_THREADS=1
  export MKL_NUM_THREADS=1
  export OPENBLAS_NUM_THREADS=1
  export NUMEXPR_NUM_THREADS=1
  export RAYON_NUM_THREADS=1
  export CHEMFM_EXACT_PREALLOCATED_CACHE=1
  export CHEMFM_EXACT_BEAM_SCORER=1
  export CHEMFM_DECODE_BATCH_SIZE=10
  export CHEMFM_EXACT_LORA_FASTPATH=1
  export CHEMFM_EXACT_LORA_CUDAGRAPH=1
  export CHEMFM_EXACT_LAYER_CUDAGRAPH=1
  export CHEMFM_EXACT_RMSNORM_CUDAGRAPH=1
  export CHEMFM_EXACT_ROPE_CUDAGRAPH=1
}

train_preflight() {
  local label=$1
  local condition=$2
  local out="$ROOT/preflight/$label"
  if [[ -f "$out/result.json" ]]; then
    return
  fi
  mkdir -p "$out"
  extra=()
  if [[ "$label" == residual ]]; then
    extra+=(--sigreg-batch-size 16)
  fi
  "$PYTHON" -u src/train.py \
    --gate 5 \
    --dataset uspto_mit_synthesis \
    --condition "$condition" \
    --seed 533 \
    --learning-rate 1e-4 \
    --k 0 \
    --lambda-eff 1.0 \
    --dropout 0.5 \
    --epochs 1 \
    --stop-after-epoch 1 \
    --evaluation-epochs 1 \
    --batch-size 4 \
    --gradient-accumulation-steps 4 \
    --no-gradient-checkpointing \
    --fused-adamw \
    --attention-implementation sdpa \
    --pin-memory \
    --dataloader-workers 0 \
    --eval-generation-batch-size 1 \
    --train-manifest "$TRAIN" \
    --validation-manifest "$VALIDATION" \
    --max-train-rows 256 \
    --max-validation-rows 2 \
    --checkpoint-dir "$out/checkpoints" \
    --no-wandb \
    --output "$out/result.json" \
    "${extra[@]}"
}

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
if not torch.cuda.is_available() or "A6000" not in torch.cuda.get_device_name(0):
    raise SystemExit("the locked experiment requires one NVIDIA A6000")
PY
    "$PYTHON" -m pytest -q
    ;;
  download-model)
    "$PYTHON" scripts/download_chemfm_model.py
    ;;
  preflight)
    train_preflight native native
    train_preflight residual clm_jepa_pair_residual
    "$PYTHON" - <<'PY'
import json
from pathlib import Path

root = Path("runs/pair_residual/a6000/preflight")
native = json.loads((root / "native/result.json").read_text())
residual = json.loads((root / "residual/result.json").read_text())
assert native["config"]["initial_trainable_sha256"] == residual["config"]["initial_trainable_sha256"]
assert native["compute"]["optimizer_steps"] == residual["compute"]["optimizer_steps"] == 16
active = [row for row in residual["curves"] if row["jepa_active"]]
assert active and all(row["gradient_interaction"] is not None for row in active)
for left, right in zip(native["curves"], residual["curves"]):
    if not right["jepa_active"]:
        assert left["native_loss"] == right["native_loss"]
print(json.dumps({
    "initial_trainable_sha256": native["config"]["initial_trainable_sha256"],
    "updates": 16,
    "active_residual_updates": len(active),
    "native_wall_seconds": native["compute"]["wall_time_seconds"],
    "residual_wall_seconds": residual["compute"]["wall_time_seconds"],
    "residual_active_seconds_mean": sum(
        row["native_forward_backward_seconds"] + row["pair_residual_statistics_vjp_seconds"]
        for row in active
    ) / len(active),
}, sort_keys=True))
PY
    ;;
  beam-benchmark)
    exact_decode_environment
    "$PYTHON" -u src/eval_uspto_mit_five_view_a6000.py benchmark \
      --checkpoint "$ROOT/preflight/native/checkpoints/epoch_1" \
      --manifest "$EQUIVALENCE" \
      --configurations 1x1xleft-pad,3x1xleft-pad,4x1xleft-pad \
      --output-dir "$ROOT/beam_benchmark"
    ;;
  run)
    exact_decode_environment
    benchmark="$ROOT/beam_benchmark/benchmark.json"
    test -f "$benchmark"
    workers=$("$PYTHON" -c \
      'import json,sys; print(int(json.load(open(sys.argv[1]))["winning_configuration"].split("_")[0][1:]))' \
      "$benchmark")
    "$PYTHON" -u scripts/run_pair_residual_local.py \
      --output-root "$ROOT/results" \
      --workers "$workers" \
      --threads-per-worker 1 \
      --teacher-batch-size 16 \
      --batch-size 4 \
      --gradient-accumulation-steps 4 \
      --no-gradient-checkpointing \
      --pin-memory \
      --phase all
    ;;
  *)
    echo "usage: $0 {setup|download-model|preflight|beam-benchmark|run}" >&2
    exit 2
    ;;
esac
