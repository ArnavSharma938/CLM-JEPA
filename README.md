# cLM-JEPA

cLM-JEPA adds a training-only joint-embedding prediction objective to ChemFM-1B while retaining ordinary autoregressive target loss and unchanged generation. The study covers forward reaction prediction, metabolism, and retrosynthesis; completed training experiments are currently limited to the USPTO-MIT forward pilot plus a reduced GSM8K LLM-JEPA reference.

The authoritative design is [docs/CLM_JEPA_Plan.md](docs/CLM_JEPA_Plan.md). Current results are indexed in [docs/reports/README.md](docs/reports/README.md), and executable entrypoints are mapped in [docs/CODE_LAYOUT.md](docs/CODE_LAYOUT.md).

## Current result

| Stage | Status | Result |
|---|---|---|
| Gates 0–3 | Passed | Backend checks, JEPA-core parity, and frozen representation-position assay completed |
| Gate 4 | Passed for reduced USPTO-MIT pilot | Fixed reliable configurations trained; no broad HPO |
| Original Gate 5 | Failed strict selector | Native and symmetric cosine cLM-JEPA tied at 2/32 exact top-1 |
| Geometry/coupling diagnosis | Completed | Symmetric cosine JEPA produced 495× lower target variance than native while retaining residual pair retrieval; pair strength did not predict decoder improvement |
| Rescue/regularizer studies | Completed | Stop-gradient and SIGReg studies did not establish a generation gain; MSE+SIGReg restored endpoint geometry but tied native at 6/256 top-1 and had 3.36% worse CE |
| Official endpoint | Stopped for prespecified futility | On 1,280 unique five-view reactions, native/cLM-JEPA top-1 was 3.906%/3.125%; difference -0.781 pp, 95% CI [-1.719,+0.156], McNemar p=0.1433 |

The official endpoint result excludes the prespecified +1 percentage-point benefit at the frozen futility boundary. It does not estimate multi-seed variability or establish a universal result for other JEPA objectives, tasks, or training scales.

## Repository map

| Path | Purpose |
|---|---|
| `src/train.py` | Canonical ChemFM native/cLM-JEPA trainer for both RTX 4050 and A6000 |
| `src/chemfm.py` | Tokenization, task collation, LoRA loading, generation, canonicalization |
| `src/jepa.py` | JEPA readouts, losses, and exact streamed SIGReg |
| `src/metrics.py` | Generative and representation metrics |
| `src/representation_eval.py` | Standard frozen representation evaluation |
| `src/eval_uspto_mit_five_view_a6000.py` | Official five-view endpoint evaluation optimized for one A6000 |
| `scripts/` | Experiment diagnostics, setup utilities, and hardware-specific execution wrappers |
| `docs/reports/` | Numbered experiment reports and report index |
| `references/` | Pinned upstream ChemFM and LLM-JEPA source |
| `runs/` | Generated checkpoints, logs, diagnostics, and candidate outputs; mostly ignored |

There is one ChemFM trainer. A6000 experiments use the same `src/train.py` with verified physical-batch/checkpointing settings; they do not use a separate scientific training implementation. See [docs/CODE_LAYOUT.md](docs/CODE_LAYOUT.md) for distinctions between normal validation, one-view diagnostics, and official five-view evaluation.

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
python scripts/decoder_coupling.py --help
python src/eval_uspto_mit_five_view_a6000.py --help
```

W&B credentials belong only in environment variables. The tracker is called from the canonical training loop and records losses, gradient norm, learning rate, auxiliary activity, throughput, VRAM, validation metrics, seed, condition, and resolved configuration.

For continuation, read [docs/context.md](docs/context.md), the research plan, [the consolidated result index](docs/reports/README.md), and [method/protocol fidelity](docs/reports/00_METHOD_AND_PROTOCOL_FIDELITY.md) in that order.
