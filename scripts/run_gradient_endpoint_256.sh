#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-.venv/bin/python}
ROOT=runs/gradient_interaction/a6000
OUT="$ROOT/endpoint_256"
MANIFEST_1280=data/clm_jepa_uspto_mit_official_endpoint/prespecified_stage1_1280.jsonl
MANIFEST_256=data/clm_jepa_uspto_mit_official_endpoint/prespecified_stage1_256.jsonl
NATIVE_FULL=runs/official_five_view_endpoint/stage1_native/predictions.jsonl
DIRECT_FULL=runs/official_five_view_endpoint/stage1_clm/predictions.jsonl
LAMBDA025_FULL="$ROOT/official/lambda_025/predictions.jsonl"
NATIVE="$OUT/references/native_predictions.jsonl"
DIRECT="$OUT/references/direct_mse_sigreg_predictions.jsonl"
LAMBDA025="$OUT/references/lambda_025_predictions.jsonl"
PANEL_REFERENCE=runs/mse_ablation/references/native_epoch4_generation.jsonl
NATIVE_DIAGNOSTICS=runs/mse_ablation/references/native_epoch4_diagnostics.json

labels=(pcgrad cagrad aux_similarity)

export PYTHONPATH=src
export TOKENIZERS_PARALLELISM=false
mkdir -p "$OUT"/{references,decoder,official,paired,logs}

[[ -f "$ROOT/TRAINING_COMPLETE" ]] || { echo "training is incomplete" >&2; exit 2; }
[[ -f "$ROOT/DIAGNOSTICS_COMPLETE" ]] || { echo "diagnostics are incomplete" >&2; exit 2; }

"$PYTHON" scripts/slice_official_panel.py \
  --manifest "$MANIFEST_1280" \
  --output-manifest "$MANIFEST_256" \
  --limit 256 \
  --predictions "$NATIVE_FULL" "$NATIVE" \
  --predictions "$DIRECT_FULL" "$DIRECT" \
  --predictions "$LAMBDA025_FULL" "$LAMBDA025" \
  --metadata "$OUT/references/panel_256_metadata.json" \
  > "$OUT/logs/prepare_references.log" 2>&1

summarize_pair() {
  local baseline=$1
  local candidate=$2
  local output=$3
  if [[ ! -f "$output" ]]; then
    "$PYTHON" src/eval_uspto_mit_five_view_a6000.py summarize \
      --manifest "$MANIFEST_256" \
      --native-predictions "$baseline" \
      --clm-predictions "$candidate" \
      --output "$output" \
      --seed 533
  fi
}

summarize_pair "$NATIVE" "$DIRECT" "$OUT/paired/direct_vs_native.json"
summarize_pair "$NATIVE" "$LAMBDA025" "$OUT/paired/lambda_025_vs_native.json"
summarize_pair "$DIRECT" "$LAMBDA025" "$OUT/paired/lambda_025_vs_direct.json"

checkpoint_for() {
  "$PYTHON" - "$ROOT/training/$1/result.json" <<'PY'
import json
import pathlib
import sys
payload = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
checkpoint = pathlib.Path(payload["selected_checkpoint"])
if not checkpoint.exists():
    raise SystemExit(f"selected checkpoint does not exist: {checkpoint}")
print(checkpoint)
PY
}

run_decoder() {
  local label=$1
  local checkpoint=$2
  local out="$OUT/decoder/$label"
  local merged="$out/${label}_generation.jsonl"
  if [[ ! -f "$merged" ]]; then
    bash scripts/run_generation_shards.sh "$label" "$checkpoint" "$out" \
      > "$OUT/logs/decoder_generate_${label}.log" 2>&1
  fi
  if [[ ! -f "$OUT/decoder/${label}_summary.json" ]]; then
    "$PYTHON" scripts/decoder_coupling.py summarize \
      --native-generation "$PANEL_REFERENCE" \
      --clm-generation "$merged" \
      --native-diagnostics "$NATIVE_DIAGNOSTICS" \
      --clm-diagnostics "$ROOT/decoder/${label}_diagnostics.json" \
      --output "$OUT/decoder/${label}_summary.json" \
      --panel-reference "$PANEL_REFERENCE" \
      --panel-limit 256 \
      --comparison-label "$label" \
      --baseline-label native \
      > "$OUT/logs/decoder_summarize_${label}.log" 2>&1
  fi
}

run_official() {
  local label=$1
  local checkpoint=$2
  local out="$OUT/official/$label"
  if [[ ! -f "$out/summary.json" ]]; then
    env \
      OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
      NUMEXPR_NUM_THREADS=1 RAYON_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
      PYTHONPATH=src CHEMFM_ATTENTION_IMPLEMENTATION=sdpa \
      CHEMFM_DECODE_BATCH_SIZE=10 CHEMFM_EXACT_BEAM_SCORER=1 \
      CHEMFM_EXACT_PREALLOCATED_CACHE=1 CHEMFM_EXACT_LORA_FASTPATH=1 \
      CHEMFM_EXACT_LORA_CUDAGRAPH=1 CHEMFM_EXACT_LAYER_CUDAGRAPH=1 \
      CHEMFM_EXACT_RMSNORM_CUDAGRAPH=1 CHEMFM_EXACT_ROPE_CUDAGRAPH=1 \
      "$PYTHON" src/eval_uspto_mit_five_view_a6000.py run \
        --checkpoint "$checkpoint" \
        --manifest "$MANIFEST_256" \
        --workers 3 \
        --prompt-batch-size 1 \
        --threads-per-worker 1 \
        --batch-mode left-pad \
        --output-dir "$out" \
        > "$OUT/logs/official_${label}.log" 2>&1
  fi
  summarize_pair "$NATIVE" "$out/predictions.jsonl" "$OUT/paired/${label}_vs_native.json"
  summarize_pair "$DIRECT" "$out/predictions.jsonl" "$OUT/paired/${label}_vs_direct.json"
}

for label in "${labels[@]}"; do
  checkpoint=$(checkpoint_for "$label")
  run_decoder "$label" "$checkpoint"
  run_official "$label" "$checkpoint"
done

touch "$OUT/ENDPOINT_256_COMPLETE"
touch "$ROOT/EXPERIMENT_COMPLETE"
