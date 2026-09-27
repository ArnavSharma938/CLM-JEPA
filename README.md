# CLM-JEPA

This repository has two explicitly separated research surfaces:

- Protein adaptation code lives directly under `src/`; retained evidence lives in `data/protein/`, `runs/protein/`, and `docs/protein/`.
- Retained chemistry datasets and reports live only in `data/chemistry/` and `docs/chemistry/`. The legacy ChemFM/NextLat executable surface has been removed.

## Repository map

| Path | Contents |
|---|---|
| `src/esm_if1/` | Reusable ESM-IF1 core, with one-shot workflows and profilers isolated in subpackages |
| `src/esm2_gate1/` | ESM-2 BackboneRef Gate-1 implementation |
| `src/esm2_peptide_generation/` | Leakage-controlled Propedia target-conditioned ESM-2 peptide-generation Gate-0 implementation |
| `scripts/` | Thin executable launchers only |
| `tests/` | Focused protein tests |
| `data/protein/esm_if1/` | Canonical corrected-sign MegaScale manifest and compact Gate-1 data |
| `data/protein/esm2_gate1/` | BackboneRef manifests and structural-distance metadata |
| `runs/protein/` | Retained final protein summaries and evaluation evidence; generated additions are ignored |
| `docs/protein/` | Protein plans, handoff ledger, and final reports |
| `data/chemistry/` | Preserved chemistry datasets |
| `docs/chemistry/` | Preserved chemistry reports |

## Current protein results

- ESM-2-650M BackboneRef Gate-1: **No meaningful LoRA gap through 50k**. See `docs/protein/ESM2_650M_BACKBONEREF_GATE1.md` and `HANDOFF.md`.
- The retained ESM-IF1 result is the corrected-sign compact Gate-1 branch under `runs/protein/esm_if1_gate1/`; the superseded wrong-sign and pilot branches were removed.

Run the focused local suite from the repository root:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Large checkpoints, downloaded archives, generated run state, caches, and logs are intentionally excluded from version control.

All tested dependencies are pinned in the single root `requirements.txt`.
