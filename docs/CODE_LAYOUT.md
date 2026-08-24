# Code and hardware layout

## Canonical model and training code

There is one ChemFM training implementation for both GPUs:

| File | Role | Hardware scope |
|---|---|---|
| `src/train.py` | Native and cLM-JEPA fine-tuning, checkpoint/resume, validation, diagnostics, W&B | Hardware-agnostic; batch/checkpointing settings determine fit |
| `src/chemfm.py` | ChemFM tokenizer, collation, LoRA loading, generation | Hardware-agnostic |
| `src/jepa.py` | Direct raw-endpoint JEPA readouts/losses and exact SIGReg | Hardware-agnostic |
| `src/gradient_interaction.py` | Weighted sum, asymmetric PCGrad, two-task CAGrad, and Du auxiliary-gradient similarity | Hardware-agnostic |
| `src/metrics.py` | Generative and vectorized representation/PCA metrics | Hardware-agnostic |
| `src/representation_eval.py` | Standard frozen representation diagnostics | Any compatible GPU |
| `src/eval_uspto_mit_five_view_a6000.py` | Official five-view, beam-10 endpoint generation and paired statistics | A6000 exact-parity path |

The A6000 experiments do not use a second ChemFM trainer. They invoke `src/train.py` with larger physical batches and without low-memory checkpointing/offload when those settings are verified to fit.

## Experiment and execution scripts

| File | Purpose | Intended environment |
|---|---|---|
| `scripts/geometry_diagnosis.py` | Base/native/cLM-JEPA geometry and residual PCA analysis | Any compatible GPU; initially run on RTX 4050 |
| `scripts/decoder_coupling.py` | One-view generation, per-reaction CE, coupling, and source interventions | Any compatible GPU |
| `scripts/diagnose_sigreg_batch16_rtx4050.py` | Exact streamed/direct SIGReg calibration and 16-update smoke test | RTX 4050-specific preflight |
| `scripts/diagnose_sigreg_gradients_rtx4050.py` | Frozen-checkpoint gradient-response assay | RTX 4050-specific assay |
| `scripts/audit_chemfm_mechanism.py` | Frozen NTP/MSE/SIGReg gradient decomposition and exact LoRA block-swap CE audit | RTX 4050-optimized diagnostic |
| `scripts/audit_sigreg_pair_specificity.py` | Frozen epoch-1/2/4 SIGReg true-vs-shuffled gradient-response audit with fresh projection draws | RTX 4050-optimized diagnostic |
| `scripts/audit_projected_mse_sigreg.py` | Raw/projected geometry plus projected MSE, SIGReg, and full-auxiliary alignment with disjoint held-out NTP | Any compatible GPU; physical chunking controls memory |
| `scripts/audit_gradient_interaction_checkpoints.py` | Evaluation-only epoch-1/2/4 MSE, SIGReg, full-auxiliary, and selected-combiner alignment with held-out NTP | Any compatible GPU; A6000 used for the reported matrix |
| `scripts/audit_generation_mechanism.py` | Frozen layerwise pathway comparison, activation patching, exact saved-state AdamW virtual steps, shortcut retrieval, and existing-generation chemistry rescoring | Local RTX 4050; no training |
| `scripts/profile_a6000_generation.py` | CUDA-event/model-forward and cProfile split for exact beam-10 generation | A6000 optimization assay |
| `scripts/benchmark_gpu_utilization.py` | Fixed-interval NVIDIA utilization, memory, power, and clock sampler | Hardware utilization assay |
| `scripts/summarize_training_timing.py` | Aggregate synchronized training phase timers from a saved checkpoint | Training optimization assay |
| `scripts/subset_endpoint_panel.py` | Deterministically order resumable worker shards against a frozen manifest prefix | CPU |
| `scripts/pcsf_experiment.py` | PCSF reference extraction, frozen spread trajectory, and gradient calibration | Any compatible GPU; A6000 used for the reported run |
| `scripts/benchmark_pcsf_training.py` | Exact objective/parity and A6000 throughput frontier for PCSF training | A6000 |
| `scripts/historical_pcsf.py` | Archived PCSF mathematics used only by prior read-only diagnostics | Historical; never imported by `src/` |
| `scripts/historical_projection.py` | Archived projection-head definition used only to reproduce the prior projection report | Historical; never imported by `src/` |
| `scripts/prepare_uspto_mit_sigreg_panel.py` | Freeze the length-stratified 256-reaction panel | CPU |
| `scripts/design_uspto_mit_endpoint.py` | Freeze and evaluate the sequential endpoint stopping rule | CPU |
| `scripts/download_chemfm_model.py` | Download and hash-check the pinned ChemFM-1B snapshot | CPU/network |

The report-specific PCSF and projection scripts and artifacts preserve historical evidence but are not imported by the maintained trainer. The active trainer has no PCSF reference cache, collation, statistic, VJP, configuration, or metric path, and no projection-head condition or checkpoint state. The normal training-time validation in `src/train.py`, one-view mechanism diagnostics, and official five-view endpoint evaluator answer different questions and are not interchangeable.

The A6000 wrappers are also in flat `scripts/`; they are not alternative scientific implementations:

| File | Purpose |
|---|---|
| `run_uspto_mit_official_endpoint.sh` | Four-worker parity-verified ChemFM endpoint run and stopping decision |
| `run_pcsf_a6000_benchmarks.sh` | Fixed A6000 execution frontier for the PCSF trainer |
| `run_pcsf_generation_shards.sh` | Four exact batch-1 generation shards with deterministic identity merge |
| `run_gradient_interaction_matrix.sh` | Restartable seven-condition training, diagnostics, and frozen generation matrix |
| `run_gradient_endpoint_256.sh` | Revised 256-reaction endpoint for the three gradient-interaction methods and aligned reference rescoring |
| `slice_official_panel.py` | Deterministic identity-checked official-manifest/prediction slicing |
| `run_generation_shards.sh` | Three-worker exact-parity one-view generation wrapper used by the active matrix |
| `train_llm_jepa_gsm8k.py` | A6000 execution wrapper around pinned upstream LLM-JEPA training |
| `eval_llm_jepa_gsm8k.py` | Batched wrapper with exact upstream-output verification |
| `diagnose_llm_jepa_geometry.py` | Frozen GSM8K representation diagnostics |

Pinned upstream code remains under `references/chemfm/` and `references/llm-jepa/`. Project-authored wrappers do not belong in `references/`.

## Data layout

| Path | Purpose |
|---|---|
| `data/clm_jepa_uspto_mit_pilot_1280/` | Frozen 1,280-row training and 160-row validation pilot manifests |
| `data/clm_jepa_uspto_mit_validation_1024/` | Frozen 1,024-identity representation and coupling panel |
| `data/clm_jepa_uspto_mit_validation_256/` | Frozen length-stratified 256-identity SIGReg/MSE panel |
| `data/clm_jepa_uspto_mit_official_endpoint/` | Official five-view parity, powered-sample, sequential-order, and stage-1 manifests |

Full released datasets and these frozen manifests are intentionally visible to Git. Historical run artifacts retain the paths used at execution time.

## Removed one-off utilities

The cleanup removed the superseded batch-size benchmark, endpoint-forward probe, legacy parallel-beam wrapper, and two profiler scripts. Their selected optimization is implemented directly in the official A6000 evaluator, and the retained measurements are recorded in `docs/reports/03_OFFICIAL_ENDPOINT_EVALUATION.md`.
