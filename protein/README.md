# Protein diagnostic

Frozen RITA-M analyses for Semantic Tube Prediction and NextLat. All
protein-specific manifests, caches, outputs, and reports live below this
directory. No script in this directory performs RITA backbone optimization.

See `docs/PROTOCOL.md` for the locked scope and causal/token conventions,
`docs/REPRODUCE.md` for exact commands, and
`docs/reports/RITA_STP_NEXTLAT_DIAGNOSTIC_REPORT.md` for the completed analysis.
The scripts are deliberately separated by scientific stage:

- `prepare_data.py` and `prepare_proteingym.py`: locked manifests;
- `run_diagnostic.py`: audit, caching, geometry, stratification, reversal, and
  orthogonal attenuation;
- `run_nextlat_probes.py`, `run_gradient_audit.py`, and
  `stratify_nextlat.py`: probes, predictor-only warmup, and gradient safety;
- `analyze_behavior_coupling.py`, `score_proteingym.py`, and
  `generation_replay.py`: behavior coupling;
- `summarize_results.py` and `record_reproducibility.py`: report inputs and
  provenance.
