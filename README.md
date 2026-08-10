# cLM-JEPA

cLM-JEPA asks whether joint-embedding predictive learning can improve a
chemistry language model without changing its inference interface. The project
adapts the training-only representation objective from LLM-JEPA to ChemFM-1B:
ChemFM keeps its native autoregressive target loss, while an auxiliary cosine
loss aligns source-reaction and target-molecule hidden states. At generation
time, the auxiliary rows and predictor tokens disappear; the model remains a
standard chemical language model.

The study covers forward reaction prediction, metabolism/product prediction,
and retrosynthesis. Its hypotheses, controls, stopping rules, datasets,
metrics, and staged compute permissions are defined in
[CLM_JEPA_Plan.md](docs/CLM_JEPA_Plan.md). The plan is authoritative; this README is
the repository map and current-status guide.

## Current status

| Gate | Status | Evidence |
|---|---|---|
| 0 | Passed historically | Official ChemFM and LLM-JEPA paths inspected |
| 1 | Passed | 32- and 128-example overfit results in `gates/gate1/results/` |
| 2 | Passed for the JEPA core | Exact native equivalence, real-model smoke evidence, and parity tests |
| 3 | Passed after the documented correction | Seven retained datasets, 7,168 frozen examples; retained `k=0` and `k=1` |
| 4 | Passed for the reduced USPTO-MIT pilot | Fixed reliable configuration; no broad HPO |
| 5 | Failed on the frozen primary rule | Native and cLM-JEPA tied at exact top-1 on seed 533 |

Gate 3 is a relationship diagnostic, not evidence that cLM-JEPA improves
generation. See [the Gate 3 report](gates/gate3/README.md) and the
[LLM-JEPA fidelity audit](docs/LLM_JEPA_FIDELITY.md) before interpreting or
continuing the experiments.

## Repository map

| Path | Purpose |
|---|---|
| `src/chemfm.py` | ChemFM tokenizer, task collation, LoRA loading, generation, and canonicalization |
| `src/jepa.py` | Predictor tokens, auxiliary row construction, hidden-state selection, and JEPA loss |
| `src/metrics.py` | Task evaluation and representation/relationship diagnostics |
| `src/train.py` | Main fine-tuning path, validation, diagnostics, and W&B logging |
| `src/experiments.py` | Gate 4/5 experiment orchestration |
| `src/download_model.py` | Explicit ChemFM checkpoint bootstrap utility |
| `gates/gate1/` | Compact Gate 1 evidence; obsolete one-off runner removed |
| `gates/gate2/` | Compact Gate 2 evidence; behavior is protected by tests |
| `gates/gate3/` | Reusable preparation/assay/summary code plus the consolidated result |
| `data/gate3/` | Versioned deterministic 1,024-row Gate 3 samples |
| `data/` | Ignored full datasets, one clearly named directory per dataset and one canonical CSV per real split |
| `references/` | Pinned official LLM-JEPA and ChemFM source used for fidelity comparison |
| `tests/` | Native-behavior, JEPA, metric, tracking, and exact upstream-parity tests |
| `runs/` | Ignored generated checkpoints, logs, W&B state, and experiment outputs |

There is intentionally no `assets/` directory: its tokenizer files were
byte-for-byte duplicates of the official tokenizer retained under
`references/chemfm/`. There is also no general `artifacts/` hierarchy;
decision-relevant gate evidence now lives with its gate, while reproducible
logs, GPU traces, prediction dumps, and audits are generated on demand.

## Setup and verification

Create a Python environment compatible with `requirements.txt`, then run:

```powershell
python -m pip install -r requirements.txt
python -m pytest -q
```

Environments, the public ChemFM base model, full datasets, and routine run
outputs are ignored because they are large or machine-specific. The two
selected seed-533 epoch-3 endpoints needed for the frozen Gate 5 diagnostic are
tracked with Git LFS. The local ChemFM checkpoint is expected at
`models/ChemFM-1B` unless `CHEMFM_MODEL_PATH` is set.

Full data uses `source` and `target` columns rather than ambiguous parallel
text filenames. The exact retained definitions and counts are in Section 3.4
of the research plan; `data/dataset_manifest.json` records provenance and
hashes. The small versioned Gate 3 samples remain under `data/gate3/` and use
the historical `src`/`tgt` assay schema only for compatibility with the frozen
result.

## Reproducing Gate 3

The checked-in samples are sufficient to rerun the frozen assay. Regenerating
samples requires the corresponding author-released full datasets under
`data/` in the locations consumed by `gates/gate3/prepare_data.py`.

```powershell
python gates/gate3/run.py --manifest data/gate3/uspto_50k_retro.csv --dataset uspto_50k_retro --task retro --output C:/tmp/uspto_50k_retro.json --batch-size 16
python gates/gate3/summarize.py C:/tmp/uspto_50k_retro.json --output C:/tmp/uspto_50k_summary.json
```

The canonical seven-dataset result is the single file
`gates/gate3/results.json`. Do not overwrite it with a partial-dataset summary.

## Fine-tuning and experiment tracking

`src/train.py` is the actual fine-tuning entrypoint, and W&B logging is called
from that loop—not from a standalone smoke script. Credentials belong only in
environment variables:

```powershell
$env:WANDB_API_KEY = "..."
$env:WANDB_PROJECT = "clm-jepa"
$env:WANDB_ENTITY = "..."
python src/train.py --help
python src/experiments.py --help
```

The tracker records native, JEPA, and total losses; gradient norm; learning
rate; JEPA activation; tokens, throughput, wall time, and peak VRAM; task and
validity metrics; seed, data fraction, condition, and resolved parameters.
Gate 4's reduced USPTO-MIT runs were completed with offline W&B records. Gate 5
failed its strict primary rule; the current frozen-checkpoint decoder-coupling
diagnostic is a mechanism analysis and cannot change checkpoint selection.

For a memoryless continuation, read [context.md](docs/context.md), the research
plan, the Gate 3 report, and the fidelity audit in that order.
