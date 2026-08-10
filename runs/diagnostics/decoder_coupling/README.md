# Frozen decoder-coupling continuation

This directory contains the frozen 1,024-identity USPTO-MIT decoder-coupling
diagnostic requested after Gate 5. It is analysis only: it must not be used to
select checkpoints or tune hyperparameters.

Selected endpoints:

- native: `runs/gate5/checkpoints/native-s533/epoch_3`
- cLM-JEPA: `runs/gate4_v2/reliable/clm_jepa-s533-checkpoints/epoch_3`

Frozen panel:
`data/gate5_decoder_coupling/uspto_mit_validation_1024.csv`, SHA-256
`abc9654a2d5a854266807ab1975557f80ea38c3a93ff226f76dd640071f89b41`.

`native_diagnostics.json` and `clm_jepa_diagnostics.json` are complete for all
1,024 identities. `native_generation.jsonl` is resumable and contains 36
completed single-view, beam-10 native reactions at the cloud handoff point.
The rejected batched-generation smoke artifact is intentionally excluded
because batching changed lower-beam ordering.

Resume the native pass with protocol-faithful batch size one:

```bash
uv run python src/decoder_coupling.py generate \
  --condition native \
  --checkpoint runs/gate5/checkpoints/native-s533/epoch_3 \
  --output runs/diagnostics/decoder_coupling/native_generation.jsonl \
  --generation-batch-size 1
```

Then run the cLM-JEPA endpoint to
`runs/diagnostics/decoder_coupling/clm_jepa_generation.jsonl` with the same
generation settings. Run `uv run python src/decoder_coupling.py summarize
--help` for the final paired analysis interface.
