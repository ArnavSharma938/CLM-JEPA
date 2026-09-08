# Decoder-projected mechanism and confirmation (Report 05)

> Panel-provenance correction (2026-09-08): the four archived evaluations in
> section 3 actually used the **first** 640 reactions, not rows 641–1280.
> Their ordered reaction identities and official group indices match
> `untouched_1280.jsonl[0:640]` exactly, with zero overlap with its second half.
> The evaluator command, logs, and manifest hash agree on the first panel
> (`3655e58404c3509c04b15cd4ffcdf15723f9be62e76c48c676ccb7decf9e2945`).
> The historical counts below are first-640 results and **are not the
> second-640 confirmation result**. The claimed untouched-second-640
> confirmation conclusion and its associated decision are invalid as such.
> This historical text is retained for provenance; the separate actual
> second-640 evaluation uses the existing frozen checkpoints without retraining.

The frozen Report-04 decoder-projected method was evaluated without changing its
training or inference path. This report has three evidence blocks.

## 1. Development generation and training evidence

Report 04 contains the complete development comparison for seeds 533/917,
including NTP, projected/full-state auxiliary losses, calibration (`g_N`,
unscaled `g_A`, locked alpha and 0.05 ratio), epoch-1/4 means, and the corrected
projected-vs-full-state endpoint labeling. The 69/68/375 and 67/73/372 counts
are ordinal gold-rank comparisons; they are not binary top-1 wins/losses. The
actual binary projected-vs-full-state top-1 discordances are 6 projected-only /
5 full-only (533) and 5/4 (917).

## 2. Focused post-training representation evidence

Treatment final post-RMSNorm product states were extracted once and compacted in
`runs/decoder_projected/representation/cache/`. Native Report-03 caches were
reused. The efficient functional probe is implemented by
`scripts/analyze_decoder_projected_mechanism.py` as the requested standardized
ridge `[z_t,a_(t+1)] -> z_(t+1)` with fixed executed width `r=352`, using the
640/192/192 reaction split and product k=1 rows. Results are in
`runs/decoder_projected/representation/mechanism_analysis.json`, including
visible decoder JS, top-1 agreement, gold rank and margin, and per-arm R2,
normalized MSE and cosine. No new ChemFM inference was used.

The compact caches do not contain the orthogonal hidden coordinates, so the
visible/null target split, head-vs-representation margin decomposition, and
singular-spectrum/token-class enrichment requiring those additional quantities
remain explicitly marked incomplete rather than inferred. The final-head
retention calculation likewise requires a separate weights-only extraction and
is not used to alter the frozen confirmation decision.

## 3. Untouched confirmation evidence

The locked untouched panel is the exact second half (rows 641-1280) of the
original manifest: 640 reactions, SHA-256
`f454d1f12e8035dc63c3992ce59a21809ecf3745b049dc910aec4538ed3c36f8`, with zero
overlap with the first 640 or Report-03 latent-audit panel. Prerequisites are
recorded in `runs/decoder_projected/confirmation_prerequisites.json`.

Both matched Native and decoder-projected checkpoints were completed for seeds
2027 and 3163 (4 epochs/320 steps), then evaluated with the unchanged official
five-view endpoint (5 views, beam 10, 10 candidates/view, reciprocal-rank
aggregation, canonical identity). Paired summaries are in
`runs/decoder_projected/confirmation/seed_{2027,3163}/comparison.json`.

| seed | Native top-1 | Projected top-1 | effect (pp) | projected-only / Native-only | top-3 N/P | top-5 N/P | top-10 N/P |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2027 | 3.9063% | 3.5938% | -0.3125 | 8 / 10 | 18.125% / 16.7188% | 25.7813% / 25.1563% | 35.4688% / 36.2500% |
| 3163 | 4.8438% | 3.4375% | -1.4063 | 3 / 12 | 19.0625% / 18.1250% | 25.9375% / 23.5938% | 37.9688% / 36.0938% |

Official view-candidate validity was Native/projected 98.1813%/97.7000%
(seed 2027) and 98.3219%/97.7250% (seed 3163); aggregated ranked validity was
100.0%/100.0% and 99.9844%/100.0%, respectively. The mean top-1 effect is
**-0.8594 pp**. Crossed seed/reaction paired bootstrap (20,000 resamples) is
**[-1.7188, 0.0000] pp**; pooled binary discordances are 11 projected-only and
22 Native-only. Both initial confirmation effects are nonpositive, so the
prespecified rule stops this formulation and does not train expansion seeds
4211/5393.

All L40 checkpoints, predictor/training logs, evaluator shards, summaries,
comparisons, environment metadata, and the untouched manifest were copied into
this repository under `runs/decoder_projected/confirmation/` and the associated
artifact archive. The Thunder L40 instance (1 GPU, 6 vCPUs, 100 GB, base
software image) was deleted after local verification; `tnr status --json` is
empty.
