# Code and execution layout

## Maintained model and training code

There is one ChemFM trainer for local and cloud execution.

| File | Ownership |
|---|---|
| `src/train.py` | Shared native/endpoint/dense/STP training loop, method dispatch, optimizer, scheduler, checkpoint/resume, validation, and tracking |
| `src/chemfm.py` | ChemFM loading, tokenizer, reaction collation, canonicalization, LoRA configuration, and causal generation |
| `src/jepa.py` | Original endpoint cLM-JEPA readouts, predictor-token controls, MSE, and exact SIGReg |
| `src/vjepa2_1.py` | Dense causal suffix/context masking, four-depth fusion, latent predictor, and functional EMA target |
| `src/stp.py` | Faithful released random-span STP and separate paper-equation STP objectives, sampling, and final-state capture |
| `src/gradient_interaction.py` | Published weighted-sum, PCGrad, CAGrad, and Du gradient-combination rules retained by the endpoint trainer |
| `src/metrics.py` | Chemical candidate scoring and representation metrics |
| `src/representation_eval.py` | Standard frozen endpoint-representation evaluation |
| `src/frozen_geometry.py` | Frozen all-layer token curvature, ordered-span alignment, chemical-event matching, uncertainty, and plots |
| `src/stp_representation_analysis.py` | Frozen all-checkpoint layer/event/spectrum/relationship analysis |
| `src/geodesic_audit.py` | Tube scale-space, tangent/acceleration, Fisher--Rao, intervention, intrinsic-local-PCA, and cone primitives |
| `src/eval_uspto_mit_five_view_a6000.py` | Exact five-view beam-10 endpoint evaluation |

Cloud runs call the same `src/train.py`; batch and checkpoint settings are
execution parameters, not alternative scientific implementations.

## Method-family boundaries

| Family | Conditions | Auxiliary owner | Vocabulary and checkpoint state |
|---|---|---|---|
| Native | `native` | None | Historical extended vocabulary for control/checkpoint parity |
| Original endpoint cLM-JEPA | `clm_jepa`, `clm_jepa_target_sg`, `clm_jepa_mse`, `clm_jepa_mse_sigreg`, and controls | `src/jepa.py` | Historical predictor-token vocabulary and endpoint/SIGReg state where applicable |
| Dense causal V-JEPA 2.1-style | `clm_jepa_vjepa2_1` | `src/vjepa2_1.py` | Base ChemFM vocabulary plus train-only predictor, level norms, EMA encoder/count, sampler, and schedules |
| Semantic Tube Prediction | `stp_released`, `stp_paper` | `src/stp.py` | Configurable r8/r128 LoRA adapters; no predictor, EMA, masking, projection head, or stop-gradient |

All families share ChemFM serialization, label-shifted NTP, data order,
optimizer/scheduler construction, validation, and generation. Endpoint, dense,
and STP auxiliaries do not compose one another. Gradient-interaction selection
changes endpoint LoRA gradient combination; it is not another representation
objective.

## Retained scripts

| File | Role |
|---|---|
| `scripts/design_uspto_mit_endpoint.py` | Prespecified endpoint design and interim rule |
| `scripts/download_chemfm_model.py` | Pinned model acquisition |
| `scripts/run_uspto_mit_official_endpoint.sh` | Official endpoint wrapper |
| `scripts/run_vjepa2_1_a6000.sh`; `scripts/run_vjepa2_1_evaluation_a6000.sh` | Dense execution/evaluation wrappers |
| `scripts/run_stp_matrix.py`; `scripts/run_stp_completion.py` | Preregistered STP rank/formulation/lambda execution and completion |
| `scripts/analyze_stp_representations.py` | Frozen 22-checkpoint representation study |
| `scripts/run_geodesic_audit.py` | Frozen geodesic extraction and primary/candidate/intrinsic/cone analyses |
| `scripts/summarize_geodesic_audit.py` | Reaction-clustered reductions and plots |
| `scripts/analyze_candidate_length_controls.py`; `scripts/analyze_signal_uncertainty.py` | Focused length controls and reaction-clustered uncertainty supplements |
| `scripts/finalize_geodesic_audit.py` | Canonical gzip/hash/integrity manifest |

Historical code remains recoverable from Git history. Current protocols,
measurements, and artifact provenance are in the two consolidated reports.

## Retained local run state

The local `runs/` directory keeps:

- selected endpoint and dense checkpoints needed by the pre-STP evidence;
- compact JSON/TXT/MD summaries needed by the consolidated reports;
- completed STP matrices, ordered beam evidence, and frozen representation
  summaries; and
- the compact Geodesic Mechanism Audit archive plus derived analysis and plots.

Older epochs, superseded raw generation shards, W&B state, profiler outputs,
and duplicate transfer bundles were removed where they were not needed for
reproduction. The current report index is `docs/reports/README.md`.

## Data layout

| Path | Purpose |
|---|---|
| `data/clm_jepa_uspto_mit_pilot_1280/` | Frozen 1,280-row training and validation pilot |
| `data/clm_jepa_uspto_mit_validation_1024/` | Frozen representation/coupling identities |
| `data/clm_jepa_uspto_mit_validation_256/` | Frozen length-stratified diagnostic panel |
| `data/clm_jepa_uspto_mit_official_endpoint/` | Official five-view parity, power, order, and stopping-rule manifests |

Pinned upstream source remains under `references/chemfm/` and
`references/llm-jepa/`.
