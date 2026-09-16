# CLM-JEPA

This repository contains two deliberately separated research surfaces:

- `protein/`: the completed frozen RITA-M diagnostic and paired rank-32
  Native-versus-NextLat causal pilot. Start with
  `protein/docs/reports/RITA_STP_NEXTLAT_DIAGNOSTIC_REPORT.md`; exact
  regeneration commands are in `protein/docs/REPRODUCE.md`.
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
