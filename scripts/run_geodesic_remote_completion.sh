#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
main="runs/geodesic_mechanism_audit"
gold_resume="runs/geodesic_gold_resume"
candidate_remaining="runs/geodesic_candidate_remaining"
benchmark="runs/geodesic_candidate_benchmark"
started=$SECONDS

trap 'code=$?; printf "{\"stage\":\"audit_sequence_failed\",\"exit_code\":%d,\"seconds\":%d}\n" "$code" "$((SECONDS-started))"; exit "$code"' ERR

printf '{"stage":"gold_resume_start"}\n'
rm -rf "$gold_resume"
mkdir -p "$gold_resume"
ln -s "../geodesic_mechanism_audit/cache" "$gold_resume/cache"
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python scripts/run_geodesic_audit.py \
  analyze-gold --output "$gold_resume" \
  --keys released_r8_l0.02_s917,released_r8_l0.08_s533,released_r8_l0.08_s917

for name in \
  tube_scale_space_by_reaction.jsonl.gz trajectory_metrics.jsonl.gz \
  tangent_persistence.jsonl.gz multiscale_turning.jsonl.gz \
  signal_noise_interventions.jsonl.gz released_objective_anatomy.jsonl.gz
do
  cp "$main/raw/$name" "$main/raw/$name.combined"
  cat "$gold_resume/raw/$name" >> "$main/raw/$name.combined"
  gzip -t "$main/raw/$name.combined"
  mv "$main/raw/$name" "$main/raw/$name.pre_resume"
  mv "$main/raw/$name.combined" "$main/raw/$name"
done
printf '{"stage":"gold_resume_complete","seconds":%d}\n' "$((SECONDS-started))"

printf '{"stage":"candidate_remaining_start"}\n'
rm -rf "$candidate_remaining"
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python scripts/run_geodesic_audit.py \
  analyze-candidates --output "$candidate_remaining" --batch-size 32 --analysis-workers 6 \
  --keys native_r8_s917,native_r8_s1301,released_r8_l0.02_s533,released_r8_l0.02_s917,released_r8_l0.02_s1301,paper_r8_l0.02_s533,paper_r8_l0.02_s917
cp "$benchmark/raw/gold_wrong_candidate_geometry.jsonl.gz" \
  "$main/raw/gold_wrong_candidate_geometry.jsonl.gz.combined"
cat "$candidate_remaining/raw/gold_wrong_candidate_geometry.jsonl.gz" >> \
  "$main/raw/gold_wrong_candidate_geometry.jsonl.gz.combined"
gzip -t "$main/raw/gold_wrong_candidate_geometry.jsonl.gz.combined"
mv "$main/raw/gold_wrong_candidate_geometry.jsonl.gz" \
  "$main/raw/gold_wrong_candidate_geometry.jsonl.gz.pre_batched"
mv "$main/raw/gold_wrong_candidate_geometry.jsonl.gz.combined" \
  "$main/raw/gold_wrong_candidate_geometry.jsonl.gz"
printf '{"stage":"candidate_complete","seconds":%d}\n' "$((SECONDS-started))"

printf '{"stage":"matched_start"}\n'
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python scripts/run_geodesic_audit.py analyze-matched --output "$main"
printf '{"stage":"matched_complete","seconds":%d}\n' "$((SECONDS-started))"

printf '{"stage":"intrinsic_start"}\n'
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python scripts/run_geodesic_audit.py \
  analyze-intrinsic --output "$main" --intrinsic-queries 256
printf '{"stage":"intrinsic_complete","seconds":%d}\n' "$((SECONDS-started))"

printf '{"stage":"candidate_intrinsic_start"}\n'
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python scripts/run_geodesic_audit.py \
  analyze-candidate-intrinsic --output "$main" --batch-size 32
printf '{"stage":"candidate_intrinsic_complete","seconds":%d}\n' "$((SECONDS-started))"

printf '{"stage":"cones_start"}\n'
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python scripts/run_geodesic_audit.py \
  analyze-cones --output "$main"
printf '{"stage":"cones_complete","seconds":%d}\n' "$((SECONDS-started))"

printf '{"stage":"final_operations_start"}\n'
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python scripts/run_geodesic_audit.py \
  analyze-final-operations --output "$main"
printf '{"stage":"final_operations_complete","seconds":%d}\n' "$((SECONDS-started))"

printf '{"stage":"summary_start"}\n'
OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 .venv/bin/python scripts/summarize_geodesic_audit.py --root "$main"
printf '{"stage":"audit_sequence_complete","seconds":%d}\n' "$((SECONDS-started))"
