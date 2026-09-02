# cLM-JEPA

cLM-JEPA adds a training-only joint-embedding prediction objective to ChemFM-1B while retaining ordinary autoregressive target loss and unchanged generation. The study covers forward reaction prediction, metabolism, and retrosynthesis; completed training experiments are currently limited to the USPTO-MIT forward pilot plus a reduced GSM8K LLM-JEPA reference.

The authoritative design is [docs/CLM_JEPA_Plan.md](docs/CLM_JEPA_Plan.md). Current results are indexed in [docs/reports/README.md](docs/reports/README.md), and executable entrypoints are mapped in [docs/CODE_LAYOUT.md](docs/CODE_LAYOUT.md).

## Recorded experiment status

| Stage | Status | Result |
|---|---|---|
| Gates 0–3 | Passed | Backend checks, JEPA-core parity, and frozen representation-position assay completed |
| Gate 4 | Passed for reduced USPTO-MIT pilot | Fixed reliable configurations trained; no broad HPO |
| Original Gate 5 | Completed | Native and symmetric cosine cLM-JEPA tied at 2/32 exact top-1; the frozen selector threshold was not met |
| Geometry/coupling diagnosis | Completed | Symmetric cosine JEPA produced 495× lower target variance than native while retaining residual pair retrieval; pair strength did not predict decoder improvement |
| Endpoint objective studies | Completed | Stop-gradient and SIGReg comparisons completed; MSE+SIGReg epoch-4 top-1 tied native at 6/256 and CE was 3.36% above native |
| Projection-space MSE+SIGReg | Completed | Shared `2048->2048->2048->64` head produced projected rank `3.22/3.29`; CE was 3.10% above direct MSE+SIGReg and matched-512 top-1 was 4 versus 15 |
| Dense causal V-JEPA 2.1-style | Completed | Four-depth EMA supervision had target CE 3.98% above native, top-1 6/256, top-10 39/256, and global retrieval 33.59% |
| Official endpoint | Stopped for prespecified futility | On 1,280 unique five-view reactions, native/cLM-JEPA top-1 was 3.906%/3.125%; difference -0.781 pp, 95% CI [-1.719,+0.156], McNemar p=0.1433 |

The official endpoint's prespecified 99% futility upper bound was `+0.458 pp`, below the frozen `+1 pp` threshold. The projection experiment used a separate budget-bounded 512-reaction prefix. Neither comparison estimates multi-seed variability.

## Method organization

One shared trainer exposes three explicit method families:

| Family | Condition | Owned implementation | Training-only state | Generation path |
|---|---|---|---|---|
| Native ChemFM | `native` | `src/train.py`, `src/chemfm.py` | None beyond ordinary optimizer state | Historical extended-vocabulary ChemFM control |
| Original endpoint cLM-JEPA | `clm_jepa`, `clm_jepa_mse_sigreg`, and gradient-interaction variants | `src/jepa.py` | Endpoint objective state; the family retains the historical predictor-token vocabulary | Same causal LM; no auxiliary loss at inference |
| Dense causal V-JEPA 2.1-style | `clm_jepa_vjepa2_1` | `src/vjepa2_1.py` | Dense predictor, level norms, EMA encoder state, mask/schedule state | Ordinary ChemFM; predictor and EMA are omitted |

`src/train.py` owns shared data order, NTP, optimizer/scheduler, checkpointing,
and dispatch. `src/chemfm.py` owns shared serialization, collation, model loading,
and generation. Endpoint and dense auxiliary implementations do not import or
compose one another. Native and endpoint conditions retain their historical
extended vocabulary for checkpoint/control parity. Dense V-JEPA uses ChemFM's
base vocabulary because its predictor is latent-only.
PCSF and projection-head experiments remain historical scripts/reports and are
not active trainer conditions.

## Repository map

| Path | Purpose |
|---|---|
| `src/train.py` | Canonical ChemFM native/cLM-JEPA trainer for both RTX 4050 and A6000 |
| `src/chemfm.py` | Tokenization, task collation, LoRA loading, generation, canonicalization |
| `src/jepa.py` | Original endpoint cLM-JEPA readouts, predictor-token objective, MSE, and exact SIGReg |
| `src/vjepa2_1.py` | Dense causal token prediction, deep supervision, latent predictor, and EMA target state |
| `src/metrics.py` | Generative and representation metrics |
| `src/representation_eval.py` | Standard frozen representation evaluation |
| `src/eval_uspto_mit_five_view_a6000.py` | Official five-view endpoint evaluation optimized for one A6000 |
| `scripts/` | Maintained setup, feasibility-evaluation, and hardware-specific execution wrappers |
| `docs/reports/` | Numbered experiment reports and report index |
| `references/` | Pinned upstream ChemFM and LLM-JEPA source |
| `runs/` | Compact retained summaries plus selected local comparison checkpoints; new outputs are ignored |

There is one ChemFM trainer. A6000 experiments use the same `src/train.py` with verified physical-batch/checkpointing settings. See [docs/CODE_LAYOUT.md](docs/CODE_LAYOUT.md) for method boundaries and evaluation entrypoints.

## Setup and tests

Create a Python environment compatible with `requirements.txt`, then run:

```powershell
python -m pip install -r requirements.txt
python -m pytest -q
```

The base ChemFM checkpoint is expected at `models/ChemFM-1B` unless `CHEMFM_MODEL_PATH` is set. Environments, full datasets, most checkpoints, and routine run outputs are ignored because they are large or machine-specific.

## Main entrypoints

```powershell
python src/train.py --help
python src/representation_eval.py --help
python scripts/audit_vjepa2_1_feasibility.py --help
python src/eval_uspto_mit_five_view_a6000.py --help
```

W&B credentials belong only in environment variables. The tracker is called from the canonical training loop and records losses, gradient norm, learning rate, auxiliary activity, throughput, VRAM, validation metrics, seed, condition, and resolved configuration.

For continuation, read [docs/context.md](docs/context.md), the research plan, [the consolidated result index](docs/reports/README.md), and [the pre-STP evidence record](docs/reports/00_PRE_STP_JEPA_CONSOLIDATED.md) in that order.
