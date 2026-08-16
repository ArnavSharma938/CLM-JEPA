#!/usr/bin/env bash
set -euo pipefail

mkdir -p runs/pcsf/benchmark
nvidia-smi \
  --query-gpu=timestamp,utilization.gpu,memory.used,power.draw \
  --format=csv,noheader,nounits -l 1 \
  > runs/pcsf/benchmark/gpu_frontier.csv &
sampler=$!
trap 'kill "$sampler" 2>/dev/null || true' EXIT

common=(
  --train-manifest data/clm_jepa_uspto_mit_pilot_1280/uspto_mit_train.csv
  --reference-cache runs/pcsf/reference/native_epoch4_pair_centers.pt
  --rho 0.8 --beta 4.2 --updates 8 --no-optimized --profile-phases
)

run_candidate() {
  local output=$1
  shift
  PYTHONPATH=src .venv/bin/python scripts/benchmark_pcsf_training.py \
    "${common[@]}" "$@" --output "runs/pcsf/benchmark/${output}.json"
}

run_candidate baseline_eager_gc_p4 \
  --physical-batch 4 --attention eager --gradient-checkpointing \
  --no-fused-adamw --no-pin-memory --workers 0
run_candidate sdpa_gc_p4 \
  --physical-batch 4 --attention sdpa --gradient-checkpointing \
  --no-fused-adamw --no-pin-memory --workers 0
run_candidate sdpa_nogc_p4 \
  --physical-batch 4 --attention sdpa --no-gradient-checkpointing \
  --no-fused-adamw --no-pin-memory --workers 0
run_candidate sdpa_nogc_fused_w2_p4 \
  --physical-batch 4 --attention sdpa --no-gradient-checkpointing \
  --fused-adamw --pin-memory --workers 2
run_candidate sdpa_nogc_w2_p8 \
  --physical-batch 8 --attention sdpa --no-gradient-checkpointing \
  --no-fused-adamw --pin-memory --workers 2
run_candidate sdpa_nogc_w2_p16 \
  --physical-batch 16 --attention sdpa --no-gradient-checkpointing \
  --no-fused-adamw --pin-memory --workers 2

kill "$sampler" 2>/dev/null || true
trap - EXIT

PYTHONPATH=src .venv/bin/python - <<'PY'
import glob
import json
import os

for filename in sorted(glob.glob("runs/pcsf/benchmark/*.json")):
    payload = json.load(open(filename, encoding="utf-8"))
    benchmark = payload.get("benchmark", {})
    print(
        os.path.basename(filename),
        benchmark.get("step_seconds"),
        benchmark.get("examples_per_second"),
        benchmark.get("peak_allocated_bytes"),
        benchmark.get("phase_seconds_per_update"),
    )
PY
