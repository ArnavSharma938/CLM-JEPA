# Code and hardware layout

## Canonical model and training code

There is one ChemFM training implementation for both GPUs:

| File | Role | Hardware scope |
|---|---|---|
| `src/train.py` | Native and cLM-JEPA fine-tuning, checkpoint/resume, validation, diagnostics, W&B | Hardware-agnostic; batch/checkpointing settings determine fit |
| `src/experiments.py` | Gate 4/5 condition orchestration | Hardware-agnostic |
| `src/chemfm.py` | ChemFM tokenizer, collation, LoRA loading, generation | Hardware-agnostic |
| `src/jepa.py` | JEPA readouts/losses and exact streamed SIGReg | Hardware-agnostic |
| `src/metrics.py` | Generative and representation metrics | Hardware-agnostic |

The A6000 experiments do not use a second ChemFM trainer. They invoke `src/train.py` with larger physical batches and without low-memory checkpointing/offload when those settings are verified to fit.

## Evaluation and diagnostics

| File | Purpose | Intended environment |
|---|---|---|
| `src/representation_eval.py` | Standard frozen representation diagnostics | Any compatible GPU |
| `src/geometry_diagnosis.py` | Base/native/cLM-JEPA geometry and residual PCA analysis | Any compatible GPU; initially run on RTX 4050 |
| `src/decoder_coupling.py` | One-view generation, per-reaction CE, coupling, and source interventions | Any compatible GPU |
| `src/diagnose_sigreg_batch16_rtx4050.py` | Exact streamed/direct SIGReg calibration and 16-update smoke test | RTX 4050-specific preflight |
| `src/diagnose_sigreg_gradients_rtx4050.py` | Frozen-checkpoint gradient-response assay | RTX 4050-specific assay |
| `src/prepare_uspto_mit_sigreg_panel.py` | Freeze the length-stratified 256-reaction panel | CPU |
| `src/design_uspto_mit_endpoint.py` | Freeze and evaluate the sequential endpoint stopping rule | CPU |
| `src/eval_uspto_mit_five_view_a6000.py` | Official five-view, beam-10 endpoint generation and paired statistics | A6000 exact-parity path |

The normal training-time validation in `src/train.py` and the one-view mechanism diagnostics are intentionally separate from the official five-view endpoint evaluator. They answer different questions and are not interchangeable.

## A6000 launch wrappers

`scripts/a6000/` contains execution wrappers, not alternative scientific implementations:

| File | Purpose |
|---|---|
| `run_uspto_mit_official_endpoint.sh` | Four-worker parity-verified ChemFM endpoint run and stopping decision |
| `train_llm_jepa_gsm8k.py` | A6000 execution wrapper around pinned upstream LLM-JEPA training |
| `eval_llm_jepa_gsm8k.py` | Batched wrapper with exact upstream-output verification |
| `diagnose_llm_jepa_geometry.py` | Frozen GSM8K representation diagnostics |

Pinned upstream code remains under `references/chemfm/` and `references/llm-jepa/`. Project-authored wrappers do not belong in `references/`.

## Removed one-off utilities

The cleanup removed the superseded batch-size benchmark, endpoint-forward probe, legacy parallel-beam wrapper, and two profiler scripts. Their selected optimization is implemented directly in the official A6000 evaluator, and their retained measurements are recorded in report 09.
