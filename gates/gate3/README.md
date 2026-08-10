# Gate 3 — frozen relationship assay

## Decision: PASS

The pinned ChemFM-1B checkpoint retains pair-specific source/target signal for
every tested position on all seven retained chemistry datasets. Under the prespecified
cross-dataset rank rule, retain `k=0` and `k=1` for Gate 4. This is a frozen
representation diagnostic, not a fine-tuning or task-performance result.

## Corrected protocol

- Checkpoint: `ChemFM/ChemFM-1B`, revision
  `f99dc2e89726539bb9cf31b2e2b4360650bac6a8`.
- No fine-tuning.
- Seed 533; batch size 16.
- Seven datasets; 1,024 deterministic unique-target examples each; 7,168
  examples total. External-test datasets are included only as frozen
  representation diagnostics, not as fine-tuning data.
- Positions: `k in {-1,0,1,2,3}`.
- `k=-1` means the second-to-last active source token and replaces the plan's
  original `k=4` at the user's direction.
- `k=0` uses source `<eos>`; `k>0` uses the final token of the descending
  `<|predictor_k|> ... <|predictor_1|>` suffix.
- Every target representation uses target `<eos>` from an independently
  encoded target-only row.
- Pair-signal criterion: positive correct-minus-random and
  correct-minus-matched margins, matched retrieval top-1 above candidate
  chance, and target effective rank greater than 1.5.
- The permitted mean-centering/normalization rescue was not needed.

## Correction note

The originally reported retrosynthesis values were invalid because the first
assays used forward reaction markers. USPTO-50K also had a second sampling
problem: its direction was corrected only after deterministic target-keyed
sampling. Swapping an already selected sample did not reproduce the intended
selection.

Corrections applied:

1. Reran `uspto_50k_retro`, `uspto_480k_template_heldout`, and
   `non_uspto_retro` using `<prostart>product<eos>` as source and
   `<rstart>precursors<eos>` as target.
2. Downloaded and verified the complete official USPTO-50K
   `train_single.csv`: 160,012 rows, 16,208,464 bytes, SHA-256
   `fbf510afb9d66dec5f005408173a4a9621f38e675b333eb53a679d322ff6b738`.
3. Regenerated USPTO-50K before sampling in its released retrosynthesis
   direction. Corrected 1,024-row manifest SHA-256:
   `8a94fc9d8d02b85cf7f98fe9ab2027ca3e3d26047c6c51d55e664ce5cd5bd6df`.
4. Reran USPTO-50K and recomputed the retained-dataset aggregate.

The audit found no additional Gate 3 functional correction requirement.

## Aggregate result

| k | datasets passing | correct−matched | correct−random | retrieval top-1 | ridge explained variance | primary rank |
|---:|---:|---:|---:|---:|---:|---:|
| -1 | 7/7 | 0.038634 | 0.043656 | 0.353376 | -0.256964 | 4.857143 |
| 0 | 7/7 | 0.090445 | 0.105838 | 0.512835 | -0.199884 | 1.000000 |
| 1 | 7/7 | 0.059604 | 0.068240 | 0.438616 | -0.182612 | 2.071429 |
| 2 | 7/7 | 0.052280 | 0.060617 | 0.411412 | -0.138871 | 3.214286 |
| 3 | 7/7 | 0.048759 | 0.058271 | 0.400949 | -0.113158 | 3.857143 |

All five positions passed the minimum relationship criterion. `k=0` ranked
first and `k=1` second under the equal-weight mean of per-dataset
correct-minus-matched and retrieval rankings. Only those two advance because
Gate 3 may retain at most two positions.

## Per-dataset evidence for selected positions

The table reports correct-minus-matched margin / matched retrieval top-1.

| Dataset | Task | k=0 | k=1 |
|---|---|---:|---:|
| MetaTrans full | metabolism | 0.166880 / 0.682617 | 0.110875 / 0.535156 |
| non-USPTO forward | forward | 0.059172 / 0.452148 | 0.032140 / 0.354492 |
| non-USPTO retro | retro | 0.072440 / 0.478516 | 0.045755 / 0.415039 |
| ORDerly forward | forward | 0.030868 / 0.351562 | 0.025254 / 0.340820 |
| USPTO-480K template-held-out | retro | 0.113440 / 0.589844 | 0.073845 / 0.504883 |
| USPTO-50K retro | retro | 0.154514 / 0.660156 | 0.104439 / 0.570312 |
| USPTO-MIT synthesis | forward | 0.035800 / 0.375000 | 0.024919 / 0.349609 |

The strongest relationship signal occurs for MetaTrans and the two large
retrosynthesis datasets. The weakest retained signal occurs for ORDerly
forward, but its matched margins remain positive and its matched retrieval
remains above the 0.25 candidate chance rate.

## Representation diagnostics

- Target effective rank ranged from 24.46 to 33.39, well above the collapse
  threshold of 1.5.
- Correct-pair margins were positive without the centering rescue.
- Retrieval was consistently strongest at `k=0` and generally second strongest
  at `k=1`.
- Mean ridge explained variance is negative for every aggregate position,
  although individual cases such as MetaTrans are positive. Therefore Gate 3
  supports a pair-specific nonlinear/geometric relationship, not a claim that
  a strong globally linear source-to-target map has already been learned.
- These diagnostics cannot establish that JEPA fine-tuning improves molecular
  generation; Gate 5 controls are required for that conclusion.

## Leakage and validity boundary

The released-split audit recorded 7 ORDerly forward overlaps, 3 MetaTrans train/validation parent
overlaps, 16 USPTO-50K train/test overlaps, and 0 USPTO-480K train/test
overlaps. Gate 3 uses training data only. Published splits were not silently
rewritten.

Gate 3 uses author-released strings and excludes exact source-target component
overlap. Full RDKit sanitization remains a task-preprocessing responsibility;
invalid generations must be counted as failures in generative evaluation.

## Commands

Representative commands used for the corrected run:

```powershell
.\.venv\Scripts\python.exe gates\gate3\prepare_data.py --dataset uspto_50k_retro --skip-leakage-checks
.\.venv\Scripts\python.exe gates\gate3\run.py --manifest data\gate3\uspto_50k_retro.csv --dataset uspto_50k_retro --task retro --output C:\tmp\uspto_50k_retro.json --batch-size 16
.\.venv\Scripts\python.exe gates\gate3\summarize.py
```

The final corrected USPTO-50K assay completed in 121.2 seconds locally without
changing batch size, checkpoint behavior, model operations, or assay settings.

## Retained artifacts

- `gates/gate3/results.json`: decision, correction provenance, hashes,
  per-dataset metrics, selection rule, and aggregate in one file.
- `data/gate3/*.csv`: the seven deterministic samples used by the assay.
- This report: the human-readable protocol, correction note, results, and
  interpretation.

The old aggregate CSV, duplicated per-dataset JSONs, preparation audit, console
logs, and GPU trace were removed after their decision-relevant contents were
consolidated here and in `results.json`. They can be regenerated on demand;
they are not additional scientific evidence.
