# Frozen decoder-coupling continuation

This directory contains the completed frozen 512-identity USPTO-MIT
decoder-coupling diagnostic requested after Gate 5. It is analysis only and
was not used to select checkpoints or tune hyperparameters.

Selected endpoints:

- native: `runs/gate5/checkpoints/native-s533/epoch_3`
- cLM-JEPA: `runs/gate4_v2/reliable/clm_jepa-s533-checkpoints/epoch_3`

Parent panel:
`data/gate5_decoder_coupling/uspto_mit_validation_1024.csv`, SHA-256
`abc9654a2d5a854266807ab1975557f80ea38c3a93ff226f76dd640071f89b41`.

The final analysis panel contains 512 paired unique identities. It preserves
510 of the first 512 identities frozen from the native deterministic
prompt-length traversal. Two late-unpaired identities were replaced before
metric inspection by the lowest panel-index identities available in both
streams (69 and 72). Generation used one official enumeration, beam 10, and
batch size one. Batched smoke outputs remain excluded because batch size two
changed lower-beam ordering.

Authoritative artifacts:

- `native_generation_512.jsonl`: SHA-256 `bafd0add3140c097cb30933cd8c9f1bbacd0e7ed95aec27a874db9c90b7259a0`
- `clm_jepa_generation_512.jsonl`: SHA-256 `dadbe56142f09cf4aaf371785e92021f8eea0333ccaec7a6c5af4dffb4c71211`
- `native_diagnostics.json` and `clm_jepa_diagnostics.json`: frozen
  per-reaction CE, pair-margin, and intervention evidence for the 1,024 parent
  identities
- `summary_512.json`: paired metrics, bootstrap intervals, correlations, and
  intervention summaries

Primary result: native exact top-1 is 0.046875 and cLM-JEPA exact top-1 is
0.033203. cLM-JEPA has a small 0.524% token-weighted CE advantage, but its
per-reaction confidence interval includes zero and pair strength does not
predict CE or rank improvement. The raw cLM `[PRED]` representation is nearly
source-invariant while the normal decoder remains strongly source-sensitive.
The report therefore supports auxiliary/decoder decoupling rather than
representation collapse.

A6000 execution used two concurrent batch-one endpoint workers. Across 1,346
five-second samples, GPU utilization averaged 47.1% and peaked at 79%; peak
VRAM was 5,189 MiB, peak temperature was
45 C, and peak power was 130.28 W. The detached full 1,024 streams were left
running after the 512 paired artifacts were frozen, per user instruction.
