# Experiment and execution scripts

These are executable diagnostics, setup utilities, and hardware-specific wrappers. Maintained model, loss, metric, training, and official endpoint-evaluation code remains in `src/`.

## ChemFM diagnostics and setup

- `decoder_coupling.py`: one-view generation, per-reaction CE, coupling, and source interventions.
- `audit_generation_mechanism.py`: frozen layerwise JEPA/autoregressive comparison, cross-checkpoint activation patching, exact saved-state AdamW counterfactuals, shortcut-controlled retrieval, and chemistry-aware rescoring of existing predictions.
- `run_gradient_interaction_matrix.sh`: restartable, cadence-matched A6000 runner for the JEPA weight, PCGrad, CAGrad, and published auxiliary-gradient-similarity conditions plus their frozen evaluation pipelines.
- `run_gradient_endpoint_256.sh`: revised endpoint runner for the user-requested 256-reaction scope; evaluates PCGrad, CAGrad, and auxiliary similarity and rescores native, direct MSE+SIGReg, and lambda-0.25 references on the identical panel.
- `slice_official_panel.py`: identity-checked deterministic prefix-manifest and prediction slicer used to make all endpoint comparisons share the same 256 reactions.
- `run_generation_shards.sh`: neutral three-worker exact-parity wrapper for the frozen 256-reaction one-view generation panel.
- `audit_gradient_interaction_checkpoints.py`: evaluation-only held-out LoRA-gradient alignment for raw MSE, raw SIGReg, their full active-weighted auxiliary, and the selected published combiner at epochs 1, 2, and 4.
- `summarize_gradient_interaction.py`: deterministic merger of training, held-out-gradient, representation, one-view CE/generation, and official five-view artifacts into the report table source.
- `profile_a6000_generation.py`: CUDA-event and host-side profiler for one exact beam-10 ChemFM view; used to validate the segmented decoder graph, beam scorer, and preallocated dynamic-cache path.
- `benchmark_gpu_utilization.py`: interval sampler for A6000 utilization, memory, power, and SM clocks during a frozen workload.
- `summarize_training_timing.py`: aggregates the synchronized per-phase timings stored in a training checkpoint.
- `geometry_diagnosis.py`: base/native/cLM-JEPA geometry and residual-PCA analysis.
- `diagnose_sigreg_batch16_rtx4050.py`: exact SIGReg batch-16 calibration and smoke test.
- `diagnose_sigreg_gradients_rtx4050.py`: frozen-checkpoint regularizer-gradient assay.
- `audit_projected_mse_sigreg.py`: two-space geometry and projected-objective alignment with held-out NTP.
- `subset_endpoint_panel.py`: order resumable worker shards against a frozen manifest prefix for matched, budget-bounded panels.
- `prepare_uspto_mit_sigreg_panel.py`: freeze the length-stratified evaluation panel.
- `design_uspto_mit_endpoint.py`: prepare and evaluate the sequential endpoint design.
- `download_chemfm_model.py`: download and hash-check the pinned ChemFM-1B snapshot.

## A6000 wrappers

- `run_vjepa2_1_a6000.sh`: sequential setup, 16-reaction super-mini, and controlled 1,280-reaction dense causal V-JEPA 2.1 pilot.
- `run_uspto_mit_official_endpoint.sh`: official five-view ChemFM endpoint evaluation.
- `train_llm_jepa_gsm8k.py`: verified execution wrapper for upstream LLM-JEPA training.
- `eval_llm_jepa_gsm8k.py`: batch-optimized GSM8K evaluation with upstream string parity checks.
- `diagnose_llm_jepa_geometry.py`: frozen GSM8K representation diagnostics.

The wrappers do not replace `src/train.py` or the pinned upstream repositories in `references/`. See `docs/CODE_LAYOUT.md` for the full mapping.

PCSF scripts remain solely to reproduce reports 06-07. Their mathematics lives
in `historical_pcsf.py`; PCSF is not part of the maintained model, collator, or
training implementation.
