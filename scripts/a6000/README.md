# A6000 execution wrappers

These files contain A6000-specific launch or execution optimizations. They do not replace the canonical ChemFM trainer in `src/train.py` or the pinned upstream repositories in `references/`.

- `run_uspto_mit_official_endpoint.sh`: official five-view ChemFM endpoint evaluation.
- `train_llm_jepa_gsm8k.py`: verified execution wrapper for upstream LLM-JEPA training.
- `eval_llm_jepa_gsm8k.py`: batch-optimized GSM8K evaluation with upstream string parity checks.
- `diagnose_llm_jepa_geometry.py`: frozen GSM8K representation diagnostics.

See `docs/CODE_LAYOUT.md` for the full mapping and `docs/reports/` for the measured configurations.
