# Reproducing the frozen RITA diagnostic

Run from the repository root with `.venv/Scripts/python.exe`. All stochastic
stages use seed `20260914`. RITA and UniRef revisions are pinned in
`protein/src/constants.py`.

## Data and caches

```powershell
.venv/Scripts/python.exe protein/scripts/prepare_data.py uniref --target 3000
.venv/Scripts/python.exe protein/scripts/prepare_data.py cluster
.venv/Scripts/python.exe protein/scripts/prepare_data.py tape
.venv/Scripts/python.exe protein/scripts/run_diagnostic.py audit
.venv/Scripts/python.exe protein/scripts/run_diagnostic.py extract protein/data/diagnostic_pool.jsonl protein/runs/uniref/cache
.venv/Scripts/python.exe protein/scripts/run_diagnostic.py extract protein/data/tape_structural_manifest.jsonl protein/runs/tape/cache
.venv/Scripts/python.exe protein/scripts/run_diagnostic.py reverse-manifest protein/data/tape_structural_manifest.jsonl protein/data/tape_reverse_128.jsonl --limit 128
.venv/Scripts/python.exe protein/scripts/run_diagnostic.py extract protein/data/tape_reverse_128.jsonl protein/runs/tape_reverse/cache
```

The clustering stage requires MMseqs2 13-45111+ds-2 and records the exact
command in `protein/runs/reproducibility.json`. Raw TAPE URLs are declared in
`prepare_data.py`.

## STP and NextLat

```powershell
.venv/Scripts/python.exe protein/scripts/run_diagnostic.py geometry protein/runs/uniref/cache protein/runs/uniref/geometry.json
.venv/Scripts/python.exe protein/scripts/run_diagnostic.py geometry protein/runs/tape/cache protein/runs/tape/geometry.json
.venv/Scripts/python.exe protein/scripts/run_diagnostic.py geometry protein/runs/tape_reverse/cache protein/runs/tape_reverse/geometry.json
.venv/Scripts/python.exe protein/scripts/run_diagnostic.py stratify protein/runs/tape/cache protein/runs/tape/stratified_geometry.json
.venv/Scripts/python.exe protein/scripts/run_diagnostic.py orthogonal protein/runs/tape/cache protein/runs/tape/orthogonal.json
.venv/Scripts/python.exe protein/scripts/run_nextlat_probes.py protein/runs/uniref/cache protein/data/diagnostic_pool.jsonl protein/runs/uniref/nextlat_probes.json
.venv/Scripts/python.exe protein/scripts/run_gradient_audit.py warmup protein/runs/uniref/cache protein/data/diagnostic_pool.jsonl protein/runs/uniref/faithful_nextlat_predictor.pt
.venv/Scripts/python.exe protein/scripts/stratify_nextlat.py protein/runs/tape/cache protein/runs/uniref/faithful_nextlat_predictor.pt protein/runs/tape/nextlat_stratified.json
.venv/Scripts/python.exe protein/scripts/run_gradient_audit.py audit protein/data/diagnostic_pool.jsonl protein/runs/uniref/stp_gradient.json --objective stp --batches 6
.venv/Scripts/python.exe protein/scripts/run_gradient_audit.py audit protein/data/diagnostic_pool.jsonl protein/runs/uniref/nextlat_gradient.json --objective nextlat --predictor protein/runs/uniref/faithful_nextlat_predictor.pt --batches 6
.venv/Scripts/python.exe protein/scripts/analyze_behavior_coupling.py protein/runs/uniref/cache protein/runs/uniref/geometry.json protein/runs/tape_reverse/geometry.json protein/runs/uniref/behavior_coupling.json --reverse-forward-geometry protein/runs/tape/geometry.json
```

## ProteinGym and generation

Download the current official ProteinGym v1.3 substitution archive and its
reference CSV from the URLs recorded in the final report, then run:

```powershell
.venv/Scripts/python.exe protein/scripts/prepare_proteingym.py lock protein/data/ProteinGym_DMS_substitutions_reference.csv protein/data/proteingym_panel.json
.venv/Scripts/python.exe protein/scripts/prepare_proteingym.py materialize protein/data/proteingym_panel.json protein/data/DMS_ProteinGym_substitutions.zip protein/data/proteingym_panel
.venv/Scripts/python.exe protein/scripts/score_proteingym.py protein/data/proteingym_panel.json protein/data/proteingym_panel protein/runs/uniref/faithful_nextlat_predictor.pt protein/runs/proteingym/scores.json --batch-size 8
.venv/Scripts/python.exe protein/scripts/generation_replay.py protein/data/diagnostic_pool.jsonl protein/runs/uniref/faithful_nextlat_predictor.pt protein/runs/generation/native_replay.json --count 100 --batch-size 10
.venv/Scripts/python.exe protein/scripts/verify_official_fitness.py protein/data/official_rita_compute_fitness.py protein/runs/audit/fitness_parity.json
.venv/Scripts/python.exe protein/scripts/summarize_results.py
.venv/Scripts/python.exe protein/scripts/record_reproducibility.py
```

Bulk archives, hidden-state shards, and the warmed predictor checkpoint are
ignored. Their source hashes, compact manifests, output JSON, and all IDs needed
to reconstruct them are retained.
