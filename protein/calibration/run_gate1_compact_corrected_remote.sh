#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/../.."
run_root="protein/runs/megascale_gate1_compact_corrected_sign"
mkdir -p "$run_root"
rm -f "$run_root/gate1.exit" "$run_root/GATE1_FAILED"
set +e
CUBLAS_WORKSPACE_CONFIG=:4096:8 .venv/bin/python -m protein.calibration.queue_pilot \
  --manifest protein/data/megascale_seed42_corrected_sign.parquet \
  --cluster-tsv protein/data/external/foldseek_eligible/clusters_cluster.tsv \
  --data-dir protein/data/compact_gate1_corrected_sign \
  --output-root "$run_root" \
  --model-checkpoint /home/ubuntu/models/esm_if1_gvp4_t16_142M_UR50.pt
rc=$?
set -e
echo "$rc" > "$run_root/gate1.exit"
if [[ "$rc" -ne 0 ]]; then touch "$run_root/GATE1_FAILED"; fi
exit "$rc"
