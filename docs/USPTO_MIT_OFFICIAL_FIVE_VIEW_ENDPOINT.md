# USPTO-MIT official five-view endpoint evaluation

## Scope and conclusion

This evaluation compared only the existing, matched epoch-4 endpoints:

- cadence-matched native NTP;
- selected MSE+SIGReg-16 cLM-JEPA.

No model was retrained. The primary endpoint was exact top-1 product accuracy under the official ChemFM five-view USPTO-MIT procedure. On the prespecified first-stage sample of 1,280 unique reactions, native reached 3.906% and cLM-JEPA reached 3.125%, a paired difference of -0.781 percentage points. The prespecified futility boundary was crossed, so evaluation stopped without extending to 3,300 reactions. This result rules out the prespecified +1 percentage-point benefit at the futility confidence level; it does not establish exact equivalence or prove that every smaller benefit is absent.

## Inference optimization and equivalence

The reference was the sequential batch-1 ChemFM implementation. The selected A6000 path used four independent single-model workers, physical prompt batch 1, one CPU thread per worker, left padding, SDPA, and exact CUDA-graph fast paths for the LoRA and repeated Llama kernels. Tokenization/candidate handling and the official reciprocal-rank aggregation across five R-SMILES views were unchanged.

On the fixed 24-reaction equivalence panel, every ordered raw, canonical, and aggregated candidate list and every exact-match flag agreed with the reference. The complete ordered-candidate digest was identical (`8ded7c06...`). The selected path achieved 0.15496 complete five-view reactions/s in the parity benchmark versus 0.01877 reactions/s for the sequential reference, an 8.25-fold speedup. Configurations with five or six replicas and larger prompt batches were measured but were slower or failed the exact-output selection criterion. The completed cLM-JEPA endpoint ran at 0.1752 reactions/s, with 69.5% mean GPU utilization, 89% maximum utilization, and 16.4 GiB peak observed GPU memory across workers.

## Prespecified sample and stopping design

The test population contained 40,000 official reaction groups, each with five R-SMILES views. A seeded simple random sample of 3,300 unique canonical reactions was frozen before endpoint inference (seed 533), then put in a separately frozen deterministic random order.

The planning assumptions were:

- paired, two-sided exact McNemar test at alpha 0.05;
- minimum effect of interest: +1 percentage point exact top-1;
- at least 80% power;
- conservative discordance 3.952%, the upper 95% Clopper-Pearson bound from the pre-existing 256-reaction pilot rather than its 1.563% point estimate.

The exact fixed-sample requirement was 3,253 reactions; 3,300 gave estimated power 80.62%. To avoid unnecessary compute, a futility analysis at 1,280 reactions was also frozen before inference: stop if the two-sided 99% Wald upper confidence bound for the paired JEPA-minus-native difference was below +1 percentage point. Simulation estimated 80.46% overall power for this sequential design and only a 0.62% probability of stopping at stage 1 when the true benefit was exactly +1 point.

## Official five-view results

All values below operate on 1,280 unique reactions, not on the 6,400 augmented input rows. Each identity used all five views, beam width 10, ten returned candidates per view, RDKit canonical product handling, and the official ChemFM reciprocal-rank aggregation.

| Endpoint | Native | cLM-JEPA | JEPA - native |
|---|---:|---:|---:|
| Exact top-1 (primary) | 3.906% | 3.125% | -0.781 pp |
| Exact top-3 | 17.891% | 16.172% | -1.719 pp |
| Exact top-5 | 26.797% | 23.516% | -3.281 pp |
| Exact top-10 | 37.656% | 35.625% | -2.031 pp |
| Official per-view candidate validity | 98.905% | 97.348% | -1.556 pp |
| Aggregated ranked-candidate validity | 99.969% | 99.977% | +0.008 pp |

For top-1, 26 reactions were correct for both models, 24 for native only, 14 for cLM-JEPA only, and 1,216 for neither. The exact two-sided McNemar p-value was 0.1433. The unique-reaction bootstrap 95% confidence interval for the paired difference was [-1.719, +0.156] percentage points. The prespecified 99% Wald interval was [-2.021, +0.458] points, whose upper limit crossed the futility boundary because it remained below +1 point.

The secondary exact McNemar tests were top-3 p=0.0448, top-5 p=0.000359, and top-10 p=0.0369. After Holm correction across the three secondary cutoffs, only top-5 remained significant (adjusted p=0.00108), in the direction favoring native. Top-3 and top-10 each had adjusted p=0.0737.

## Interpretation

The benchmark-faithful evidence does not support a +1 percentage-point generative advantage for the selected cLM-JEPA endpoint. Its point estimates were lower at every beam cutoff, and the top-5 paired disadvantage survived the prespecified multiplicity correction. Consequently, the earlier small-panel uncertainty was not concealing a useful endpoint improvement of the targeted size.

This establishes a behavioral result for these two fixed epoch-4 checkpoints, this pilot-training regime, and this official test sample. It does not establish that JEPA objectives can never help ChemFM, isolate which training mechanism caused the deficit, or estimate multi-seed training variability. Teacher-forced CE and representation geometry are intentionally not substituted for the primary generation conclusion here.

## Evidence paths

- Frozen design: `data/official_five_view_endpoint/sequential_design_metadata.json`
- Frozen 1,280-reaction manifest: `data/official_five_view_endpoint/prespecified_stage1_1280.jsonl`
- Paired statistics: `runs/official_five_view_endpoint/stage1_paired_summary.json`
- Prespecified futility decision: `runs/official_five_view_endpoint/interim_1280.json`
- Complete native candidates: `runs/official_five_view_endpoint/stage1_native/predictions.jsonl`
- Complete cLM-JEPA candidates: `runs/official_five_view_endpoint/stage1_clm/predictions.jsonl`

