# CLM-JEPA

This repository contains two deliberately separated research surfaces:

- `protein/`: the completed frozen RITA-M diagnostic and paired rank-32
  Native-versus-NextLat causal pilot.
- `docs/reports/`: the consolidated ChemFM experimental record. The reusable
  ChemFM model, objective, and training code remains in `src/`.

## Repository map

| Path | Contents |
|---|---|
| `data/` | Preserved chemistry datasets, locked panels, and split metadata |
| `docs/reports/` | Consolidated ChemFM reports 00-08 |
| `protein/` | Protein source, scripts, tests, manifests, compact results, and final report |
| `references/` | Pinned ChemFM reference implementation and tokenizer |
| `scripts/download_chemfm_model.py` | The remaining reusable chemistry utility |
| `src/` | ChemFM loading, training, STP, NextLat, decoder-projected, and metric primitives |
| `tests/` | Focused tests for the retained chemistry core |

Large model weights, checkpoints, hidden-state caches, downloaded archives,
temporary outputs, and local experiment logs are intentionally ignored and
must be regenerated when needed.

## Protein diagnostic (`protein/`)

Frozen RITA-M analyses for Semantic Tube Prediction and NextLat. All protein-specific manifests, caches, outputs, and reports live below `protein/`. No script performs RITA backbone optimization.

- **Scope & protocol**: [protein/docs/PROTOCOL.md](file:///c:/Users/arnav.DHEERAJACER/CLM-JEPA/protein/docs/PROTOCOL.md) (locked scope and causal/token conventions)
- **Reproduction**: [protein/docs/REPRODUCE.md](file:///c:/Users/arnav.DHEERAJACER/CLM-JEPA/protein/docs/REPRODUCE.md) (exact regeneration commands)
- **Primary report**: [protein/docs/reports/RITA_STP_NEXTLAT_DIAGNOSTIC_REPORT.md](file:///c:/Users/arnav.DHEERAJACER/CLM-JEPA/protein/docs/reports/RITA_STP_NEXTLAT_DIAGNOSTIC_REPORT.md) (completed diagnostic and causal pilot analysis)

### Scientific stage scripts (`protein/scripts/`)

The protein scripts are separated by scientific stage:

- `prepare_data.py` and `prepare_proteingym.py`: locked manifests
- `run_diagnostic.py`: audit, caching, geometry, stratification, reversal, and orthogonal attenuation
- `run_nextlat_probes.py`, `run_gradient_audit.py`, and `stratify_nextlat.py`: probes, predictor-only warmup, and gradient safety
- `analyze_behavior_coupling.py`, `score_proteingym.py`, and `generation_replay.py`: behavior coupling
- `summarize_results.py` and `record_reproducibility.py`: report inputs and provenance

