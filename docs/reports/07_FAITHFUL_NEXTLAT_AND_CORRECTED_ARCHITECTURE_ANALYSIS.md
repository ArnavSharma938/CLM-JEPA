# Report 07: Faithful NextLat and Corrected Architecture Analysis

Status: complete (2026-09-08). This report supersedes only the invalid
representation/mechanism section of Report 05. Report 06 remains the frozen
decoder-projected behavioral confirmation.

## Implementation audit

Reference: [`JaydenTeoh/NextLat`](https://github.com/JaydenTeoh/NextLat), commit
`3770be6009cea2b3c455a9ce7f2ca88b504bb955`, file
`models/model_nextlat.py`, and the 1B horizon-1 configuration
`config/fineweb/1B/nextlat_finewebedu_1b_100b_horizon1.yaml`.

The earlier `full_state_nextlat` condition is retained as a provenance-preserved matched ablation, not an exact NextLat reproduction. It used a 256-wide predictor with a zero-initialized final layer, detached token embeddings, a frozen initial decoder, per-row rather than global valid-position reductions, and decoder-projected 5% gradient calibration. Each changes a material upstream mechanic.

The new, separately named `faithful_nextlat` condition uses:

- input ordering `[e(x[t+1]), h[t]]`, biasless LayerNorm, and residual `h[t] + delta`;
- the upstream 1B `proj_factor=1.6`, giving `4096 -> 6528 -> 6528 -> 2048` for ChemFM-1B;
- bias-free linear layers, GELU, zero dropout, and independent normal `std=0.02` initialization for every linear layer;
- detached future states but live auxiliary gradients through both current-state and next-token-embedding paths;
- the current live native-vocabulary LM head, detached only as an auxiliary parameter path;
- globally reduced SmoothL1 and teacher-to-student KL, each with weight 1;
- no decoder-projected gradient calibration and no inference-time auxiliary module.

ChemFM-only adaptations are limited to ordinary ChemFM NTP training, the established product-transition eligibility mask, native-vocabulary slicing, and ChemFM reaction serialization.

## Development training and optimization-budget diagnostic

Both development cells used the matched ChemFM setup: 1,280 rows representing
256 reactions and five views, manifest SHA-256
`b5900bc7e4f1a858ecf3fdf3732da63e08fc0f955f1cb9ccf90534e2273c8dba`,
LoRA rank/alpha/dropout 8/8/0.1, the same seven attention/MLP target modules,
saved embeddings and LM head, AdamW (betas 0.9/0.999, epsilon 1e-8,
weight decay 0.01), LR 1e-4, 5% warmup, cosine decay to 1e-5, physical batch
4, accumulation 4, four epochs, and exactly 320 optimizer steps. The selected
checkpoint is epoch 4. Training used an L40 because no A6000 was available.

| seed | initial trainable SHA-256 | epoch-4 adapter SHA-256 | epoch-4 auxiliary SHA-256 | wall time | peak VRAM |
|---:|---|---|---|---:|---:|
| 533 | `5fe60189…ca7` | `3df25a1b…1aa7` | `f7aaccca…c983` | 247.9 s | 8.238 GB |
| 917 | `798e72c6…798c` | `ef3181b6…399` | `d4fa2aa6…87d7` | 252.2 s | 8.241 GB |

The frozen diagnostic used 2,048 reaction-balanced positions per split. Its
train and validation selected-key hashes are respectively `dd825a8c…ee56` and
`15bd5c18…6518`, identical for both seeds. All five held-out diagnostics
improved from step 80 through step 320 on both seeds, so the prespecified
underfitting check was triggered. Only seed 533 was continued.

| checkpoint | held-out SmoothL1 | teacher→student KL | gold rank | gold margin | ordinary NTP loss |
|---|---:|---:|---:|---:|---:|
| seed 533, step 320 | 0.04905 | 0.33619 | 1.427 | 3.274 | 0.22766 |
| seed 533, step 480 | 0.04678 | 0.33977 | 1.483 | 3.633 | 0.23157 |

The corrected continuation jointly clipped ChemFM and predictor parameters and
preserved optimizer/dataloader state. SmoothL1 continued to fall, but KL, gold
rank, and ordinary NTP loss all deteriorated. The evidence is a trade-off or
plateau, not consistent held-out improvement, so 320 steps are not a defensible
explanation for the weak generation result. An earlier continuation that
accidentally clipped ChemFM alone is retained as invalid provenance and excluded.

## Development behavioral comparison

The endpoint is the unchanged 512-reaction, five-view panel (SHA-256
`a2e6202a…a6057d`), beam 10 per view, reciprocal-rank aggregation, and canonical
chemical identity scoring.

| seed | arm | top-1 | top-3 | top-5 | top-10 | candidate validity | ranked validity |
|---:|---|---:|---:|---:|---:|---:|---:|
| 533 | Native | 2.539% | 15.234% | 22.852% | 35.352% | 98.988% | 99.922% |
| 533 | faithful NextLat | 0.977% | 13.086% | 21.289% | 31.445% | 98.125% | 99.941% |
| 533 | decoder-projected | 3.711% | 16.992% | 24.609% | 34.961% | 98.555% | 100.000% |
| 917 | Native | 2.148% | 14.648% | 21.875% | 30.273% | 98.699% | 100.000% |
| 917 | faithful NextLat | 0.977% | 13.672% | 21.875% | 32.617% | 97.359% | 100.000% |
| 917 | decoder-projected | 3.711% | 16.992% | 24.805% | 36.328% | 98.277% | 100.000% |

For faithful NextLat minus Native, seed 533 was -1.5625 pp (1 faithful-only,
9 Native-only, 502 ties; exact McNemar p=0.02148; paired bootstrap 95% CI
[-2.734, -0.391] pp), and seed 917 was -1.1719 pp (2/8/502; p=0.10938;
CI [-2.344, 0.000] pp). Mean effect was -1.3672 pp. Gold-rank comparisons were
54/97/361 and 73/80/359 (faithful better/Native better/tie).

Decoder-projected minus faithful NextLat was +2.7344 pp in each seed: 15/1/496
(p=0.000519, CI [1.367, 4.297] pp) for 533 and 16/2/494 (p=0.001312,
CI [1.172, 4.492] pp) for 917. Candidate validity fell more under faithful
NextLat than under decoder projection in both seeds. Ranked validity did not
explain the top-1 loss.

## Corrected architecture-level analysis

The former Report-05 representation metrics are invalidated and were not reused.
The common pipeline explicitly slices `H[product_indices]`, asserts absolute
token/state relationships, uses each arm's own active embedding and live head,
and loads each seed's exact projected-run `projection=ΣVᵀ`, `u`, frozen decoder,
singular values, and rank. Hidden states are never decoder-mean shifted.

The frozen Report-03 reaction split is used throughout. Training rows use the
exact reaction-balanced selector capped at 16,384 positions; all arms have the
same train-key hash `3915c627…7210` and test-key hash `a91e104b…ee0`.
Training-only standardization/PCA and the Report-03 ridge convention are used;
test SST uses the training-target mean. Null targets use rank-128 train-only PCA.
Synthetic tests cover the former indexing, coordinate, JS, and predicted-logit
rank/margin failures. Both projected bases have numerical rank 352; their exact
auxiliary hashes are `14812a44…34f2` (533) and `ec777797…6303` (917).

### Transition predictability

The table gives the mean across development seeds. `z=ΣVᵀh` is the
singular-value-weighted decoder-functional coordinate, `q=Vᵀh` is the
orthonormal coordinate, and null is the rank-128 PCA target. Full-state and null
PCA train coverage was 98.0–98.7% and 85.8–86.9%, respectively.

| arm | full h R² | z R² | q R² | null R² | decoder JS | predicted top-1 agreement | gold rank |
|---|---:|---:|---:|---:|---:|---:|---:|
| Native | 0.5379 | 0.5300 | 0.5490 | 0.4079 | 0.2063 | 0.6824 | 5.268 |
| faithful NextLat | 0.6043 | 0.5992 | 0.6216 | 0.4626 | 0.1523 | 0.7627 | 3.725 |
| decoder-projected | 0.5441 | 0.5299 | 0.5570 | 0.4133 | 0.1958 | 0.7013 | 5.015 |

Faithful minus Native reaction-level ΔR² was positive for every target and both
seeds: full h +0.0636/+0.0676, weighted functional +0.0591/+0.0749,
orthonormal visible +0.0723/+0.0704, and null +0.0570/+0.0514 (533/917).
Every paired 95% CI excludes zero. Decoder-projected changes were much smaller:
full h +0.0042/+0.0075, q +0.0072/+0.0068, and null +0.0028/+0.0073; weighted
functional R² was -0.0115 for seed 533 and +0.0099 for seed 917.

Mean transition magnitudes `(parallel RMS, null RMS)` were (12.699, 4.232) for
Native, (10.744, 4.372) for faithful NextLat, and (11.713, 4.181) for projected.
Thus faithful NextLat makes both decoder-visible and decoder-null dynamics more
linearly predictable while contracting visible transition magnitude; it is not
merely learning decoder-null regularity.

### Functional spectrum and decoder drift

Weighted-coordinate effective dimensionality remained extremely concentrated.
Participation ratios were 1.236/1.254 Native, 1.328/1.326 faithful, and
1.294/1.321 projected; 95% variance required 4/4, 5/5, and 4/5 dimensions.
Prediction-error improvement was concentrated in high-singular-value axes for
both methods (sigma/improvement correlations 0.929/0.928 faithful and
0.926/0.927 projected), although faithful's improvement was substantially larger.

The initial functional subspace remains representative of all final decoders:

| arm | retained energy ρ | median angle | maximum angle | projection overlap | chordal distance |
|---|---:|---:|---:|---:|---:|
| Native | 0.99543 | 5.53° | 86.72° | 0.82665 | 7.811 |
| faithful NextLat | 0.99253 | 6.32° | 86.76° | 0.81701 | 8.026 |
| decoder-projected | 0.99397 | 6.13° | 86.76° | 0.81425 | 8.086 |

These values use only the native 392 decoder rows; reserved added-token rows are
excluded. Faithful NextLat causes modestly more energy drift, but no wholesale
rotation explains its behavioral result.

### Representation versus head contributions

For Native's strongest competing token, reaction-balanced total gold-margin
change is dominated by the hidden-state term. Values below are means with paired
reaction-bootstrap 95% CIs; the algebraic decomposition was checked per row.

| seed | treatment | representation | head | interaction | total |
|---:|---|---:|---:|---:|---:|
| 533 | faithful | -0.961 [-1.003,-0.918] | +0.066 [+0.062,+0.069] | +0.002 [+0.001,+0.003] | -0.894 [-0.936,-0.852] |
| 917 | faithful | -0.841 [-0.884,-0.796] | +0.075 [+0.071,+0.078] | +0.002 [+0.001,+0.003] | -0.764 [-0.807,-0.719] |
| 533 | projected | -0.201 [-0.223,-0.179] | -0.002 [-0.004,-0.001] | +0.0003 [+0.00005,+0.00062] | -0.203 [-0.224,-0.181] |
| 917 | projected | -0.151 [-0.190,-0.112] | +0.038 [+0.034,+0.042] | -0.007 [-0.008,-0.006] | -0.120 [-0.159,-0.081] |

The correctness strata behave coherently: positive total shifts accompany fixes
and negative shifts accompany breaks, but the overwhelmingly common unchanged
stratum has negative mean margin shifts. Faithful's live-head change offsets
only a small fraction of its adverse representation contribution.

## Scientific conclusion and decision

This faithful implementation succeeds at its proximal objective: it produces a
large, reproducible improvement in token-conditioned transition predictability
in full, decoder-functional, and decoder-null geometry, with better predicted
state distributions and gold-token ranks. That improvement does **not** transfer
to useful chemical generation: top-1 falls in both seeds, lower cutoffs mostly
fall, and raw candidate validity falls more than for decoder projection. The
failure is therefore architectural/objective-alignment evidence, not evidence
that the predictor failed to learn, and the short continuation does not support
insufficient optimization as the explanation.

**Decision: do not run frozen faithful-NextLat confirmation on seeds 2027/3163.**
Report 06 already answers decoder-projected confirmation, and no decoder-projected
checkpoint was retrained or re-evaluated here; its existing development artifacts
were reused only for matched analysis.
