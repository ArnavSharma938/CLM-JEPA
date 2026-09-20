#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/../.."
run_root="protein/runs/megascale_25pct_proportional"
mkdir -p "$run_root"
rm -f "$run_root/pilot25.exit" "$run_root/PILOT25_FAILED"
set +e
CUBLAS_WORKSPACE_CONFIG=:4096:8 .venv/bin/python -m protein.calibration.queue_pilot \
  --manifest protein/data/megascale_seed42.parquet \
  --pilot-data-dir protein/data/pilots \
  --output-root "$run_root" \
  --model-checkpoint /home/ubuntu/models/esm_if1_gvp4_t16_142M_UR50.pt
rc=$?
set -e
echo "$rc" > "$run_root/pilot25.exit"
if [[ "$rc" -ne 0 ]]; then touch "$run_root/PILOT25_FAILED"; fi
exit "$rc"
