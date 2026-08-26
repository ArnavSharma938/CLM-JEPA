# Official five-view endpoint evaluation

## Measured summary

On 1,280 prespecified unique reactions, the fixed epoch-4 native endpoint had
3.906% exact top-1 and the fixed epoch-4 MSE+SIGReg cLM-JEPA endpoint had
3.125%: `-0.781` percentage points, 95% paired bootstrap CI
`[-1.719,+0.156]`, exact McNemar `p=0.1433`.

The prespecified 99% futility interval had upper bound `+0.458` pp, below the
minimum effect of interest of `+1` pp. Under the frozen stopping rule,
evaluation stopped at 1,280 rather than extending to 3,300.

## Fixed comparison and official semantics

No model was retrained. The endpoints were cadence-matched native NTP epoch 4 and the selected MSE+exact-SIGReg-16 cLM-JEPA epoch 4 from [report 02](02_MSE_SIGREG_EXPERIMENT.md).

Each unique reaction used all five official R-SMILES views, beam width 10, ten returned candidates per view, RDKit canonical product handling, and ChemFM reciprocal-rank aggregation across views. Accuracy was computed per unique reaction, not per augmented row. Exact top-1 was primary.

## Inference parity and speed

The selected A6000 path used four independent batch-1 model workers with one CPU thread each, left padding, SDPA, and exact CUDA-graph fast paths. Tokenization, candidate handling, and five-view aggregation were unchanged.

On a fixed 24-reaction panel, every ordered raw, canonical, and aggregated candidate list and every exact-match flag matched the sequential reference; the complete digest was identical (`8ded7c06...`). Throughput increased from `0.01877` to `0.15496` complete five-view reactions/s, an `8.25x` speedup. The completed cLM-JEPA pass ran at `0.1752` reactions/s, 69.5% mean GPU utilization, 89% maximum utilization, and 16.4 GiB peak VRAM. Larger prompt batches were slower; five/six replicas either were slower or changed exact outputs, so neither candidate was used.

## Prespecified sample and power

The official test population contained 40,000 five-view reaction groups. Before inference, seed 533 froze a simple random sample and deterministic order of 3,300 unique reactions.

- Paired two-sided exact McNemar test, alpha 0.05.
- Minimum effect of interest: +1 pp exact top-1.
- Required power: at least 80%.
- Conservative discordance: 3.952%, the upper 95% Clopper-Pearson bound from the earlier 256-reaction panel.
- Fixed-sample requirement: 3,253; 3,300 gave 80.62% estimated power.
- Frozen interim: stop at 1,280 if the two-sided 99% Wald upper bound for cLM-JEPA minus native was below +1 pp.
- Simulated sequential-design power: 80.46%; probability of an interim stop when the true benefit is +1 pp: 0.62%.

## Results

| Endpoint | Native | MSE+SIGReg cLM-JEPA | Difference |
|---|---:|---:|---:|
| Exact top-1 | 3.906% (50/1,280) | 3.125% (40/1,280) | -0.781 pp |
| Top-3 | 17.891% | 16.172% | -1.719 pp |
| Top-5 | 26.797% | 23.516% | -3.281 pp |
| Top-10 | 37.656% | 35.625% | -2.031 pp |
| Official per-view validity | 98.905% | 97.348% | -1.556 pp |
| Aggregated-candidate validity | 99.969% | 99.977% | +0.008 pp |

Top-1 paired counts were 26 both correct, 24 native-only, 14 cLM-JEPA-only, and 1,216 neither. The 99% Wald interval was `[-2.021,+0.458]` pp. Secondary exact McNemar p-values were 0.0448/0.000359/0.0369 for top-3/5/10. Holm-adjusted p-values were 0.0737/0.00108/0.0737; only top-5 remained significant, favoring native.

## Measurement scope

For these two seed-533 epoch-4 checkpoints, reduced pilot exposure, and the
frozen official test sample, the cLM-JEPA-minus-native 99% interval upper bound
was below the prespecified +1 pp threshold. Native point estimates were higher
at every top-k cutoff.

The evaluation does not estimate multi-seed training variability, test larger
training exposure, cover MetaTrans or retrosynthesis, or measure untested JEPA
objectives. Teacher-forced CE and representation geometry were not endpoint
substitutes in this evaluation.

## Evidence paths

- Design metadata: `data/clm_jepa_uspto_mit_official_endpoint/sequential_design_metadata.json`
- Frozen manifest: `data/clm_jepa_uspto_mit_official_endpoint/prespecified_stage1_1280.jsonl`
- Paired statistics: `runs/official_five_view_endpoint/stage1_paired_summary.json`
- Futility decision: `runs/official_five_view_endpoint/interim_1280.json`
- Native candidates: `runs/official_five_view_endpoint/stage1_native/predictions.jsonl`
- cLM-JEPA candidates: `runs/official_five_view_endpoint/stage1_clm/predictions.jsonl`
