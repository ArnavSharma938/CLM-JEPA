# CLM-JEPA

This repository contains two deliberately separated research surfaces:

- `protein/`: protein adaptation studies: the ESM-IF1 MegaScale calibration
  and the separate ESM-2 BackboneRef Gate-1 screen.
- `docs/reports/`: the consolidated ChemFM experimental record. The reusable
  ChemFM model, objective, and training code remains in `src/`.

## Repository map

| Path | Contents |
|---|---|
| `data/` | Preserved chemistry datasets, locked panels, and split metadata |
| `docs/reports/` | Consolidated ChemFM reports 00-08 |
| `protein/` | ESM-IF1 calibration code, tests, frozen split metadata, and launch plan |
| `protein/esm2_gate1/` | ESM-2 BackboneRef Gate-1 data, MLM, adaptation, evaluation, and reporting code |
| `references/` | Pinned ChemFM reference implementation and tokenizer |
| `scripts/download_chemfm_model.py` | The remaining reusable chemistry utility |
| `src/` | ChemFM loading, training, STP, NextLat, decoder-projected, and metric primitives |
| `tests/` | Focused tests for the retained chemistry core |

Large model weights, checkpoints, hidden-state caches, downloaded archives,
temporary outputs, and local experiment logs are intentionally ignored and
must be regenerated when needed.

## Protein calibration (`protein/`)

The protein surface studies ESM-IF1 adaptation and MegaScale likelihood
calibration. Its implementation is in `protein/calibration/`, focused tests are
in `protein/tests/`, and the frozen scientific and launch contract is recorded
in `protein/PLAN.md`. Full Stage-1 training remains gated on target-instance
preflight and explicit authorization.
