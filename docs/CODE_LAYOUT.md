# Code and execution layout

## Maintained model and training code

There is one ChemFM trainer for local and A6000 execution.

| File | Ownership |
|---|---|
| `src/train.py` | Shared native/endpoint/dense training loop, method dispatch, optimizer, scheduler, checkpoint/resume, validation, and tracking |
| `src/chemfm.py` | ChemFM loading, tokenizer, reaction collation, canonicalization, and unchanged causal generation |
| `src/jepa.py` | Original endpoint cLM-JEPA readouts, predictor-token controls, MSE, and exact SIGReg |
| `src/vjepa2_1.py` | Dense causal suffix/context masking, four-depth fusion, latent predictor, and functional EMA target |
| `src/gradient_interaction.py` | Published weighted-sum, PCGrad, CAGrad, and Du gradient-combination rules retained by the endpoint trainer |
| `src/metrics.py` | Chemical candidate scoring and representation metrics |
| `src/representation_eval.py` | Standard frozen endpoint-representation evaluation |
| `src/eval_uspto_mit_five_view_a6000.py` | Exact five-view beam-10 endpoint evaluation |

A6000 runs call the same `src/train.py`; batch and checkpointing settings are execution parameters, not alternative scientific implementations.

## Method-family boundaries

| Family | Conditions | Auxiliary owner | Vocabulary and checkpoint state |
|---|---|---|---|
| Native | `native` | None | Historical extended vocabulary for control/checkpoint parity |
| Original endpoint cLM-JEPA | `clm_jepa`, `clm_jepa_target_sg`, `clm_jepa_mse`, `clm_jepa_mse_sigreg`, and controls | `src/jepa.py` | Historical predictor-token vocabulary and endpoint/SIGReg state where applicable |
| Dense causal V-JEPA 2.1-style | `clm_jepa_vjepa2_1` | `src/vjepa2_1.py` | Base ChemFM vocabulary plus train-only predictor, level norms, EMA encoder/count, sampler, and schedules |

All families share ChemFM serialization, label-shifted NTP, data order, optimizer/scheduler construction, validation, and generation. Endpoint and dense auxiliary modules do not compose one another. Gradient-interaction selection changes endpoint LoRA gradient combination; it is not a fourth representation objective.

## Retained scripts

| File | Role |
|---|---|
| `scripts/audit_vjepa2_1_feasibility.py` | Dense checkpoint feasibility/local-token comparison |
| `scripts/design_uspto_mit_endpoint.py` | Prespecified endpoint design and interim rule |
| `scripts/download_chemfm_model.py` | Pinned model acquisition |
| `scripts/run_uspto_mit_official_endpoint.sh` | Official endpoint wrapper |
| `scripts/run_vjepa2_1_a6000.sh` | Dense setup/super-mini/pilot wrapper |
| `scripts/run_vjepa2_1_evaluation_a6000.sh` | Dense evaluation wrapper |

Historical audit and intervention scripts were removed after report consolidation. Their source remains in commit `61fbc74`; their exact protocols, measurements, and artifact provenance are in reports 01–03.

## Retained local run state

The local `runs/` directory keeps only:

- selected epoch-4 native and direct endpoint checkpoints;
- the selected dense V-JEPA epoch-4 checkpoint and its four compact result JSONs;
- compact JSON/TXT/MD summaries needed by reports 01–03.

Older checkpoint epochs, completed alternative-condition checkpoints, raw generation shards, W&B state, logs, profiler outputs, duplicate archives, and transfer bundles were removed. The reports are the authoritative human-readable records; compact retained files support machine inspection.

## Data layout

| Path | Purpose |
|---|---|
| `data/clm_jepa_uspto_mit_pilot_1280/` | Frozen 1,280-row training and validation pilot |
| `data/clm_jepa_uspto_mit_validation_1024/` | Frozen representation/coupling identities |
| `data/clm_jepa_uspto_mit_validation_256/` | Frozen length-stratified diagnostic panel |
| `data/clm_jepa_uspto_mit_official_endpoint/` | Official five-view parity, power, order, and stopping-rule manifests |

Pinned upstream source remains under `references/chemfm/` and `references/llm-jepa/`.
