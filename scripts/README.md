# Experiment and execution scripts

These are executable diagnostics, setup utilities, and hardware-specific wrappers. Maintained model, loss, metric, training, and official endpoint-evaluation code remains in `src/`.

## ChemFM diagnostics and setup

- `decoder_coupling.py`: one-view generation, per-reaction CE, coupling, and source interventions.
- `geometry_diagnosis.py`: base/native/cLM-JEPA geometry and residual-PCA analysis.
- `diagnose_sigreg_batch16_rtx4050.py`: exact SIGReg batch-16 calibration and smoke test.
- `diagnose_sigreg_gradients_rtx4050.py`: frozen-checkpoint regularizer-gradient assay.
- `prepare_uspto_mit_sigreg_panel.py`: freeze the length-stratified evaluation panel.
- `design_uspto_mit_endpoint.py`: prepare and evaluate the sequential endpoint design.
- `download_chemfm_model.py`: download and hash-check the pinned ChemFM-1B snapshot.

## A6000 wrappers

- `run_uspto_mit_official_endpoint.sh`: official five-view ChemFM endpoint evaluation.
- `train_llm_jepa_gsm8k.py`: verified execution wrapper for upstream LLM-JEPA training.
- `eval_llm_jepa_gsm8k.py`: batch-optimized GSM8K evaluation with upstream string parity checks.
- `diagnose_llm_jepa_geometry.py`: frozen GSM8K representation diagnostics.

The wrappers do not replace `src/train.py` or the pinned upstream repositories in `references/`. See `docs/CODE_LAYOUT.md` for the full mapping.
