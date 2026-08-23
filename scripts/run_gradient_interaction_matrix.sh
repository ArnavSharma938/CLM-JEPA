#!/usr/bin/env bash
set -euo pipefail

# Controlled A6000 runner for the gradient-interaction experiment.  PHASE may
# be one of: all, train, diagnostics, generation.  Completed atomic artifacts
# are skipped, so the runner can be restarted after an interrupted instance.

cd "$(dirname "$0")/.."

PHASE=${PHASE:-all}
PYTHON=${PYTHON:-.venv/bin/python}
LABEL_FILTER=${LABEL_FILTER:-}
ROOT=runs/gradient_interaction/a6000
TRAIN_MANIFEST=data/clm_jepa_uspto_mit_pilot_1280/uspto_mit_train.csv
VALIDATION_MANIFEST=data/clm_jepa_uspto_mit_validation_256/uspto_mit_validation_length_stratified_256.csv
PANEL_REFERENCE=runs/mse_ablation/references/native_epoch4_generation.jsonl
NATIVE_DIAGNOSTICS=runs/mse_ablation/references/native_epoch4_diagnostics.json
OFFICIAL_MANIFEST=data/clm_jepa_uspto_mit_official_endpoint/prespecified_stage1_1280.jsonl
OFFICIAL_NATIVE=runs/official_five_view_endpoint/stage1_native/predictions.jsonl

labels=(lambda_025 lambda_05 lambda_10 lambda_20 pcgrad cagrad aux_similarity)
lambdas=(0.25 0.5 1.0 2.0 1.0 1.0 1.0)
methods=(weighted_sum weighted_sum weighted_sum weighted_sum pcgrad cagrad aux_similarity)

selected_label() {
  [[ -z "$LABEL_FILTER" || ",$LABEL_FILTER," == *",$1,"* ]]
}

export PYTHONPATH=src
export TOKENIZERS_PARALLELISM=false

mkdir -p "$ROOT"/{training,representation,decoder,official,logs}

checkpoint_for() {
  "$PYTHON" - "$ROOT/training/$1/result.json" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
checkpoint = pathlib.Path(payload["selected_checkpoint"])
if not checkpoint.exists():
    raise SystemExit(f"selected checkpoint does not exist: {checkpoint}")
print(checkpoint)
PY
}

validate_result() {
  "$PYTHON" - "$1" <<'PY'
import json
import math
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
if payload.get("selected_epoch") != 4:
    raise SystemExit(f"{path}: selected_epoch is not 4")
active = [row for row in payload.get("curves", []) if row.get("jepa_active")]
if not active:
    raise SystemExit(f"{path}: no active JEPA updates")
required = [
    "cosine",
    "raw_auxiliary_to_main_norm_ratio",
    "modification_relative_to_raw_sum",
]
for key in required:
    values = [float(row["gradient_interaction"][key]) for row in active]
    if not all(math.isfinite(v) for v in values):
        raise SystemExit(f"{path}: missing/non-finite {key}")
print(f"validated {path}")
PY
}

run_training() {
  for i in "${!labels[@]}"; do
    label=${labels[$i]}
    selected_label "$label" || continue
    lambda=${lambdas[$i]}
    method=${methods[$i]}
    out="$ROOT/training/$label"
    mkdir -p "$out"
    if [[ ! -f "$out/result.json" ]]; then
      resume=()
      for resume_epoch in 4 3 2 1; do
        if [[ -f "$out/checkpoints/epoch_${resume_epoch}/training_state.pt" ]]; then
          resume=(--resume-from "$out/checkpoints/epoch_${resume_epoch}")
          break
        fi
      done
      "$PYTHON" src/train.py \
        --gate 4 \
        --dataset uspto_mit_synthesis \
        --condition clm_jepa_mse_sigreg \
        --seed 533 \
        --learning-rate 0.0001 \
        --k 0 \
        --lambda-eff "$lambda" \
        --gradient-interaction "$method" \
        --dropout 0.5 \
        --epochs 4 \
        --stop-after-epoch 4 \
        --batch-size 4 \
        --gradient-accumulation-steps 4 \
        --sigreg-batch-size 16 \
        --sigreg-tradeoff 0.01 \
        --evaluation-epochs 4 \
        --no-gradient-checkpointing \
        --fused-adamw \
        --optimized-jepa-forward \
        --attention-implementation sdpa \
        --pin-memory \
        --dataloader-workers 0 \
        --eval-generation-batch-size 1 \
        --train-manifest "$TRAIN_MANIFEST" \
        --validation-manifest "$VALIDATION_MANIFEST" \
        --checkpoint-dir "$out/checkpoints" \
        --max-validation-rows 2 \
        "${resume[@]}" \
        --no-wandb \
        --output "$out/result.json" \
        > "$ROOT/logs/train_${label}.log" 2>&1
    fi
    validate_result "$out/result.json"
  done
  touch "$ROOT/TRAINING_COMPLETE"
}

run_diagnostics() {
  run_jsons=()
  for i in "${!labels[@]}"; do
    label=${labels[$i]}
    selected_label "$label" || continue
    run_json="$ROOT/training/$label/result.json"
    run_jsons+=("$run_json")
    checkpoint=$(checkpoint_for "$label")
    rep="$ROOT/representation/${label}.json"
    if [[ ! -f "$rep" ]]; then
      extra=()
      if [[ "$label" == "lambda_025" ]]; then
        extra+=(--include-pretrained)
      fi
      "$PYTHON" src/representation_eval.py \
        --dataset uspto_mit_synthesis \
        --validation-manifest "$VALIDATION_MANIFEST" \
        --run-json "$run_json" \
        --seed 533 \
        --k 0 \
        --diagnostic-limit 256 \
        --diagnostic-batch-size 16 \
        --output "$rep" \
        "${extra[@]}" \
        > "$ROOT/logs/representation_${label}.log" 2>&1
    fi

    diag="$ROOT/decoder/${label}_diagnostics.json"
    if [[ ! -f "$diag" ]]; then
      "$PYTHON" scripts/decoder_coupling.py represent \
        --condition "$label" \
        --checkpoint "$checkpoint" \
        --output "$diag" \
        --batch-size 16 \
        --state-batch-size 64 \
        --panel-reference "$PANEL_REFERENCE" \
        --panel-limit 256 \
        --k 0 \
      > "$ROOT/logs/decoder_represent_${label}.log" 2>&1
    fi
  done
  if [[ ! -f "$ROOT/gradient_checkpoints.json" ]]; then
    "$PYTHON" scripts/audit_gradient_interaction_checkpoints.py \
      --run-json "${run_jsons[@]}" \
      --validation-manifest "$VALIDATION_MANIFEST" \
      --dataset uspto_mit_synthesis \
      --epochs 1 2 4 \
      --logical-batch-size 16 \
      --physical-batch-size 4 \
      --seed 533 \
      --output "$ROOT/gradient_checkpoints.json" \
      > "$ROOT/logs/gradient_checkpoints.log" 2>&1
  fi
  touch "$ROOT/DIAGNOSTICS_COMPLETE"
}

run_decoder_generation() {
  local label=$1
  local checkpoint=$2
  local out="$ROOT/decoder/$label"
  local merged="$out/${label}_generation.jsonl"
  if [[ ! -f "$merged" ]]; then
    bash scripts/run_generation_shards.sh "$label" "$checkpoint" "$out" \
      > "$ROOT/logs/decoder_generate_${label}.log" 2>&1
  fi
  if [[ ! -f "$ROOT/decoder/${label}_summary.json" ]]; then
    "$PYTHON" scripts/decoder_coupling.py summarize \
      --native-generation "$PANEL_REFERENCE" \
      --clm-generation "$merged" \
      --native-diagnostics "$NATIVE_DIAGNOSTICS" \
      --clm-diagnostics "$ROOT/decoder/${label}_diagnostics.json" \
      --output "$ROOT/decoder/${label}_summary.json" \
      --panel-reference "$PANEL_REFERENCE" \
      --panel-limit 256 \
      --comparison-label "$label" \
      --baseline-label native \
      > "$ROOT/logs/decoder_summarize_${label}.log" 2>&1
  fi
}

run_official_generation() {
  local label=$1
  local checkpoint=$2
  local out="$ROOT/official/$label"
  if [[ ! -f "$out/summary.json" ]]; then
    env \
      OMP_NUM_THREADS=1 \
      MKL_NUM_THREADS=1 \
      OPENBLAS_NUM_THREADS=1 \
      NUMEXPR_NUM_THREADS=1 \
      RAYON_NUM_THREADS=1 \
      TOKENIZERS_PARALLELISM=false \
      PYTHONPATH=src \
      CHEMFM_ATTENTION_IMPLEMENTATION=sdpa \
      CHEMFM_DECODE_BATCH_SIZE=10 \
      CHEMFM_EXACT_BEAM_SCORER=1 \
      CHEMFM_EXACT_PREALLOCATED_CACHE=1 \
      CHEMFM_EXACT_LORA_FASTPATH=1 \
      CHEMFM_EXACT_LORA_CUDAGRAPH=1 \
      CHEMFM_EXACT_LAYER_CUDAGRAPH=1 \
      CHEMFM_EXACT_RMSNORM_CUDAGRAPH=1 \
      CHEMFM_EXACT_ROPE_CUDAGRAPH=1 \
      "$PYTHON" src/eval_uspto_mit_five_view_a6000.py run \
        --checkpoint "$checkpoint" \
        --manifest "$OFFICIAL_MANIFEST" \
        --workers 3 \
        --prompt-batch-size 1 \
        --threads-per-worker 1 \
        --batch-mode left-pad \
        --output-dir "$out" \
        > "$ROOT/logs/official_${label}.log" 2>&1
  fi
  if [[ ! -f "$ROOT/official/${label}_paired.json" ]]; then
    "$PYTHON" src/eval_uspto_mit_five_view_a6000.py summarize \
      --manifest "$OFFICIAL_MANIFEST" \
      --native-predictions "$OFFICIAL_NATIVE" \
      --clm-predictions "$out/predictions.jsonl" \
      --output "$ROOT/official/${label}_paired.json" \
      --seed 533 \
      > "$ROOT/logs/official_summarize_${label}.log" 2>&1
  fi
}

run_generation() {
  [[ -f "$ROOT/DIAGNOSTICS_COMPLETE" ]] || run_diagnostics
  for label in "${labels[@]}"; do
    selected_label "$label" || continue
    checkpoint=$(checkpoint_for "$label")
    run_decoder_generation "$label" "$checkpoint"
    run_official_generation "$label" "$checkpoint"
  done
  touch "$ROOT/GENERATION_COMPLETE"
}

case "$PHASE" in
  all)
    run_training
    run_diagnostics
    run_generation
    touch "$ROOT/EXPERIMENT_COMPLETE"
    ;;
  train) run_training ;;
  diagnostics) run_diagnostics ;;
  generation) run_generation ;;
  *) echo "unknown PHASE=$PHASE" >&2; exit 2 ;;
esac
