#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 CONDITION CHECKPOINT OUTPUT_DIR" >&2
  exit 2
fi

condition=$1
checkpoint=$2
output_dir=$3
workers=3
panel_reference=runs/mse_ablation/references/native_epoch4_generation.jsonl
mkdir -p "$output_dir"

pids=()
for ((worker=0; worker<workers; worker++)); do
  env \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
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
    .venv/bin/python scripts/decoder_coupling.py generate \
    --condition "$condition" \
    --checkpoint "$checkpoint" \
    --output "$output_dir/shard_${worker}.jsonl" \
    --generation-batch-size 1 \
    --panel-reference "$panel_reference" \
    --panel-limit 256 \
    --shard-count "$workers" \
    --shard-index "$worker" \
    > "$output_dir/shard_${worker}.log" 2>&1 &
  pids+=("$!")
done

for pid in "${pids[@]}"; do
  wait "$pid"
done

PYTHONPATH=src .venv/bin/python - "$condition" "$output_dir" <<'PY'
import json
import pathlib
import sys

condition = sys.argv[1]
output_dir = pathlib.Path(sys.argv[2])
records = []
for shard in sorted(output_dir.glob("shard_*.jsonl")):
    with shard.open(encoding="utf-8") as handle:
        records.extend(json.loads(line) for line in handle if line.strip())
records.sort(key=lambda row: row["panel_index"])
identities = [row["reaction_identity"] for row in records]
if len(records) != 256 or len(set(identities)) != 256:
    raise RuntimeError(
        f"expected 256 unique reactions after merging, got "
        f"{len(records)} rows/{len(set(identities))} identities"
    )
if any(row["condition"] != condition for row in records):
    raise RuntimeError("condition mismatch in generation shards")
merged = output_dir / f"{condition}_generation.jsonl"
with merged.open("w", encoding="utf-8") as handle:
    for record in records:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
print(json.dumps({"output": str(merged), "identities": len(records)}))
PY
