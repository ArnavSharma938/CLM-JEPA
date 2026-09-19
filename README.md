# CLM-JEPA

This repository contains two deliberately separated research surfaces:

- `protein/`: the ESM-IF1 parameter-efficient calibration study, including
  MegaScale preparation, adaptation, evaluation, and launch tooling.
- `docs/reports/`: the consolidated ChemFM experimental record. The reusable
  ChemFM model, objective, and training code remains in `src/`.

## Repository map

| Path | Contents |
|---|---|
| `data/` | Preserved chemistry datasets, locked panels, and split metadata |
| `docs/reports/` | Consolidated ChemFM reports 00-08 |
| `protein/` | ESM-IF1 calibration code, tests, frozen split metadata, and launch plan |
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
