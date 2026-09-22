#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 /absolute/path/to/backboneref-structure-archive.zip" >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
structure_zip="$1"
data_root="$repo_root/protein/data/esm2_gate1"
run_root="$repo_root/protein/runs/esm2_gate1"
report_path="$repo_root/protein/docs/reports/ESM2_650M_BACKBONEREF_GATE1.md"
status_log="$run_root/logs/queue_status.log"

mkdir -p "$run_root/logs" "$(dirname "$report_path")"
rm -f "$run_root/queue.exit"
exec > >(tee -a "$run_root/logs/queue.log") 2>&1

stage() {
  printf '%s STAGE %s\n' "$(date --iso-8601=seconds)" "$1" | tee -a "$status_log"
}

finish() {
  code=$?
  printf '%s EXIT %s\n' "$(date --iso-8601=seconds)" "$code" | tee -a "$status_log"
  printf '%s\n' "$code" > "$run_root/queue.exit"
}
trap finish EXIT

cd "$repo_root"
source .venv-esm2/bin/activate
export PATH="/home/ubuntu/.local/bin:$PATH"

if [[ ! -f "$data_root/corpus_audit.json" ]]; then
  stage prepare_data
  python -m protein.esm2_gate1.prepare_data --output-dir "$data_root"
fi

stage model_metadata
python -m protein.esm2_gate1.preflight metadata --output "$run_root/preflight.json"

if [[ ! -f "$run_root/fair_esm_parity.json" ]]; then
  stage fair_esm_parity
  python -m protein.esm2_gate1.preflight parity --output "$run_root/fair_esm_parity.json"
fi

if [[ ! -f "$run_root/backend_benchmark.json" ]]; then
  stage attention_backend_benchmark
  python -m protein.esm2_gate1.preflight benchmark --output "$run_root/backend_benchmark.json"
fi

backend="$(python -c 'import json; print(json.load(open("protein/runs/esm2_gate1/backend_benchmark.json"))["selected_backend"])')"
compile_flag="$(python -c 'import json; print(int(json.load(open("protein/runs/esm2_gate1/backend_benchmark.json"))["use_torch_compile"]))')"

if [[ ! -f "$run_root/execution_profile.json" ]]; then
  stage end_to_end_training_and_evaluation_profile
  python -m protein.esm2_gate1.profile_execution \
    --manifest "$data_root/train_10k.jsonl" \
    --output "$run_root/execution_profile.json" \
    --attention-backend "$backend"
fi

if [[ ! -f "$data_root/distance_audit.json" ]]; then
  stage structural_and_sequence_distances
  python -m protein.esm2_gate1.distances \
    --manifest-root "$data_root" \
    --structure-zip "$structure_zip"
fi

stage pre_lr_scientific_audit
python -m protein.esm2_gate1.audit \
  --manifest-root "$data_root" \
  --run-root "$run_root" \
  --output "$run_root/pre_lr_audit.json"

microbatch_tokens="$(python -c 'import json; print(json.load(open("protein/runs/esm2_gate1/execution_profile.json"))["selected_training"]["token_budget"])')"
num_workers="$(python -c 'import json; print(json.load(open("protein/runs/esm2_gate1/execution_profile.json"))["selected_training"]["workers"])')"

stage gate1_screen
screen_args=(
  --manifest-root "$data_root"
  --output-root "$run_root"
  --attention-backend "$backend"
  --microbatch-tokens "$microbatch_tokens"
  --num-workers "$num_workers"
)
if [[ "$compile_flag" == "1" ]]; then
  screen_args+=(--compile-model)
fi
python -m protein.esm2_gate1.run_screen "${screen_args[@]}"

stage report
python -m protein.esm2_gate1.report \
  --manifest-root "$data_root" \
  --run-root "$run_root" \
  --output "$report_path"

stage complete
