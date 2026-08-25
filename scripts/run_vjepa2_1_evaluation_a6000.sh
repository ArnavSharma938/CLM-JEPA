#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-.venv/bin/python}
ROOT=runs/vjepa2_1/a6000
PILOT_RESULT="$ROOT/pilot/result.json"
PANEL=data/clm_jepa_uspto_mit_validation_256/uspto_mit_validation_length_stratified_256.csv
NATIVE_RESULT=runs/sigreg_batch16_pilot/matched_b4/native.json
ENDPOINT_RESULT=runs/mse_ablation/stage1/mse_sigreg.json

export PYTHONPATH=src
export TOKENIZERS_PARALLELISM=false

pilot_pid=$(cat "$ROOT/pilot.pid")
while kill -0 "$pilot_pid" 2>/dev/null; do
  sleep 20
done
test -f "$PILOT_RESULT"

dense_checkpoint=$("$PYTHON" - "$PILOT_RESULT" <<'PY'
import json
import pathlib
import sys
print(json.loads(pathlib.Path(sys.argv[1]).read_text())["selected_checkpoint"])
PY
)

mkdir -p "$ROOT/evaluation"
"$PYTHON" src/representation_eval.py \
  --dataset uspto_mit_synthesis \
  --validation-manifest "$PANEL" \
  --run-json "$NATIVE_RESULT" \
  --run-json "$ENDPOINT_RESULT" \
  --run-json "$PILOT_RESULT" \
  --seed 533 \
  --k 0 \
  --diagnostic-limit 256 \
  --diagnostic-batch-size 4 \
  --output "$ROOT/evaluation/global_representation.json"

"$PYTHON" scripts/audit_vjepa2_1_feasibility.py \
  --dense-checkpoint "$dense_checkpoint" \
  --dense-result "$PILOT_RESULT" \
  --panel "$PANEL" \
  --limit 64 \
  --batch-size 4 \
  --max-tokens 4096 \
  --output "$ROOT/evaluation/local_target_tokens.json"

touch "$ROOT/evaluation/evaluation.done"
