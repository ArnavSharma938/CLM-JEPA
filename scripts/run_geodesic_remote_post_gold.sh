#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
main="runs/geodesic_mechanism_audit"
candidate_final="runs/geodesic_candidate_final"
started=$SECONDS
fail() {
  local code=$?
  jobs -pr | xargs -r kill 2>/dev/null || true
  wait 2>/dev/null || true
  printf '{"stage":"post_gold_failed","exit_code":%d,"seconds":%d}\n' \
    "$code" "$((SECONDS-started))"
  exit "$code"
}
trap fail ERR

printf '{"stage":"candidate_start"}\n'
rm -rf "$candidate_final"
mkdir -p "$candidate_final"
run_candidate_group() {
  local name="$1"
  local keys="$2"
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python scripts/run_geodesic_audit.py \
    analyze-candidates --output "$candidate_final/$name" --batch-size 32 \
    --analysis-workers 2 --keys "$keys" > "$candidate_final/$name.log" 2>&1
}
run_candidate_group a "native_r8_s533,native_r8_s917,native_r8_s1301" & pid_a=$!
run_candidate_group b "released_r8_l0.02_s533,released_r8_l0.02_s917,released_r8_l0.02_s1301" & pid_b=$!
run_candidate_group c "paper_r8_l0.02_s533,paper_r8_l0.02_s917" & pid_c=$!
wait "$pid_a"; wait "$pid_b"; wait "$pid_c"
cp "$candidate_final/a/raw/gold_wrong_candidate_geometry.jsonl.gz" \
  "$candidate_final/gold_wrong_candidate_geometry.jsonl.gz"
cat "$candidate_final/b/raw/gold_wrong_candidate_geometry.jsonl.gz" \
  "$candidate_final/c/raw/gold_wrong_candidate_geometry.jsonl.gz" >> \
  "$candidate_final/gold_wrong_candidate_geometry.jsonl.gz"
gzip -t "$candidate_final/gold_wrong_candidate_geometry.jsonl.gz"
mv "$main/raw/gold_wrong_candidate_geometry.jsonl.gz" \
  "$main/raw/gold_wrong_candidate_geometry.jsonl.gz.pre_exact_semantic_path"
cp "$candidate_final/gold_wrong_candidate_geometry.jsonl.gz" \
  "$main/raw/gold_wrong_candidate_geometry.jsonl.gz"
printf '{"stage":"candidate_complete","seconds":%d}\n' "$((SECONDS-started))"

for stage in matched intrinsic candidate_intrinsic cones final_operations
do
  printf '{"stage":"%s_start"}\n' "$stage"
  case "$stage" in
    matched)
      OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python scripts/run_geodesic_audit.py analyze-matched --output "$main" ;;
    intrinsic)
      OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python scripts/run_geodesic_audit.py analyze-intrinsic --output "$main" --intrinsic-queries 256 ;;
    candidate_intrinsic)
      OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python scripts/run_geodesic_audit.py analyze-candidate-intrinsic --output "$main" --batch-size 32 ;;
    cones)
      OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python scripts/run_geodesic_audit.py analyze-cones --output "$main" ;;
    final_operations)
      OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python scripts/run_geodesic_audit.py analyze-final-operations --output "$main" ;;
  esac
  printf '{"stage":"%s_complete","seconds":%d}\n' "$stage" "$((SECONDS-started))"
done

printf '{"stage":"summary_start"}\n'
OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 .venv/bin/python scripts/summarize_geodesic_audit.py --root "$main"
printf '{"stage":"post_gold_complete","seconds":%d}\n' "$((SECONDS-started))"
