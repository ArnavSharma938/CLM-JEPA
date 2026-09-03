#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
main="runs/geodesic_mechanism_audit"
candidate_final="runs/geodesic_candidate_final"
started=$SECONDS
trap 'code=$?; printf "{\"stage\":\"post_gold_failed\",\"exit_code\":%d,\"seconds\":%d}\n" "$code" "$((SECONDS-started))"; exit "$code"' ERR

printf '{"stage":"candidate_start"}\n'
rm -rf "$candidate_final"
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python scripts/run_geodesic_audit.py \
  analyze-candidates --output "$candidate_final" --batch-size 32 --analysis-workers 6
gzip -t "$candidate_final/raw/gold_wrong_candidate_geometry.jsonl.gz"
mv "$main/raw/gold_wrong_candidate_geometry.jsonl.gz" \
  "$main/raw/gold_wrong_candidate_geometry.jsonl.gz.pre_exact_semantic_path"
cp "$candidate_final/raw/gold_wrong_candidate_geometry.jsonl.gz" \
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
