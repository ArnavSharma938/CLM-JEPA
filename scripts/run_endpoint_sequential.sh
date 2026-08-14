#!/usr/bin/env bash
set -euo pipefail

cd /home/ubuntu/CLM-JEPA

ROOT=runs/official_five_view_endpoint
STAGE_MANIFEST=data/official_five_view_endpoint/prespecified_stage1_1280.jsonl
FULL_MANIFEST=data/official_five_view_endpoint/prespecified_3300_sequential.jsonl
NATIVE_CHECKPOINT=runs/sigreg_batch16_pilot/matched_b4/native_checkpoints/epoch_4
CLM_CHECKPOINT=runs/mse_ablation/stage1/mse_sigreg_checkpoints/epoch_4
COMMON_ENV=(
  OMP_NUM_THREADS=1
  MKL_NUM_THREADS=1
  OPENBLAS_NUM_THREADS=1
  NUMEXPR_NUM_THREADS=1
  RAYON_NUM_THREADS=1
  TOKENIZERS_PARALLELISM=false
  PYTHONPATH=src
  CHEMFM_EXACT_LORA_FASTPATH=1
  CHEMFM_EXACT_LORA_CUDAGRAPH=1
  CHEMFM_EXACT_RMSNORM_CUDAGRAPH=1
  CHEMFM_EXACT_ROPE_CUDAGRAPH=1
  CHEMFM_EXACT_APPLY_ROPE_CUDAGRAPH=1
)

run_endpoint() {
  local checkpoint="$1"
  local manifest="$2"
  local output_dir="$3"
  env "${COMMON_ENV[@]}" .venv-457/bin/python \
    src/official_five_view_evaluation.py run \
    --checkpoint "$checkpoint" \
    --manifest "$manifest" \
    --workers 4 \
    --prompt-batch-size 1 \
    --threads-per-worker 1 \
    --batch-mode left-pad \
    --output-dir "$output_dir"
}

# The native stage was launched interactively after the inference path and
# sequential design were frozen. Wait for its atomic summary before starting
# the matched endpoint. A dead launch without a summary is an explicit error.
while [[ ! -f "$ROOT/stage1_native/summary.json" ]]; do
  if ! pgrep -f "official_five_view_evaluation.py.*stage1_native" >/dev/null; then
    printf '%s native stage terminated without summary\n' "$(date -Iseconds)" \
      > "$ROOT/sequential_supervisor.error"
    exit 1
  fi
  sleep 60
done

printf '%s native stage complete; launching cLM-JEPA\n' "$(date -Iseconds)" \
  >> "$ROOT/sequential_supervisor.log"
mkdir -p "$ROOT/stage1_clm"
run_endpoint "$CLM_CHECKPOINT" "$STAGE_MANIFEST" "$ROOT/stage1_clm" \
  > "$ROOT/stage1_clm.launch.log" 2>&1

env PYTHONPATH=src .venv-457/bin/python src/endpoint_sequential_design.py interim \
  --manifest "$STAGE_MANIFEST" \
  --native-predictions "$ROOT/stage1_native/predictions.jsonl" \
  --clm-predictions "$ROOT/stage1_clm/predictions.jsonl" \
  --output "$ROOT/interim_1280.json" \
  > "$ROOT/interim_1280.log" 2>&1

decision="$({ python3 - <<'PY'
import json
print(json.load(open("runs/official_five_view_endpoint/interim_1280.json"))["decision"])
PY
} | tr -d '\r\n')"
printf '%s interim decision: %s\n' "$(date -Iseconds)" "$decision" \
  >> "$ROOT/sequential_supervisor.log"

if [[ "$decision" == "CONTINUE_TO_3300" ]]; then
  run_endpoint "$NATIVE_CHECKPOINT" "$FULL_MANIFEST" "$ROOT/stage1_native" \
    > "$ROOT/full_native_resume.log" 2>&1
  run_endpoint "$CLM_CHECKPOINT" "$FULL_MANIFEST" "$ROOT/stage1_clm" \
    > "$ROOT/full_clm_resume.log" 2>&1
  env PYTHONPATH=src .venv-457/bin/python src/official_five_view_evaluation.py summarize \
    --manifest "$FULL_MANIFEST" \
    --native-predictions "$ROOT/stage1_native/predictions.jsonl" \
    --clm-predictions "$ROOT/stage1_clm/predictions.jsonl" \
    --output "$ROOT/final_paired_summary_3300.json" \
    > "$ROOT/final_paired_summary_3300.log" 2>&1
else
  env PYTHONPATH=src .venv-457/bin/python src/official_five_view_evaluation.py summarize \
    --manifest "$STAGE_MANIFEST" \
    --native-predictions "$ROOT/stage1_native/predictions.jsonl" \
    --clm-predictions "$ROOT/stage1_clm/predictions.jsonl" \
    --output "$ROOT/stage1_descriptive_summary_1280.json" \
    > "$ROOT/stage1_descriptive_summary_1280.log" 2>&1
fi

printf '%s sequential endpoint evaluation complete\n' "$(date -Iseconds)" \
  >> "$ROOT/sequential_supervisor.log"
touch "$ROOT/SEQUENTIAL_EVALUATION_COMPLETE"
