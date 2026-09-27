# ChemFM JEPA experiments before STP: consolidated evidence record

## Scope and reading rules

This report replaces former Reports 00--05. It retains the distinct objectives,
controls, generation endpoints, uncertainty estimates, mechanism measurements,
limitations, and artifact locations while removing repeated narrative,
superseded engineering detail, and literature discussion that did not affect an
executed condition.

The sections are not a single factorial experiment. They record independent
trajectories and frozen assays with different panels and, in some cases,
different readouts. Results are compared only where the same panel and control
are stated. Exact generated top-1 is the behavioral endpoint; teacher-forced,
gradient, retrieval, and geometry measurements are diagnostics.

Historical source paths removed from the maintained tree are recoverable at
commit `61fbc74`. Compact outputs and selected checkpoints remain at the paths
listed below.

## 1. Method fidelity and shared system

| Component | Pinned source |
|---|---|
| LLM-JEPA | `references/llm-jepa/` at `ea0017c654ad917066ff32afc88276bea8ca5f7e` |
| ChemFM | `references/chemfm/` at `ee35b23d03de1a8e97b8e04dcdfb1d579de70f02` |
| ChemFM-1B | revision `f99dc2e89726539bb9cf31b2e2b4360650bac6a8` |
| LeJEPA/SIGReg | `c293d291ca87cd4fddee9d3fffe4e914c7272052` |
| V-JEPA 2.1 | `facebookresearch/vjepa2@204698b45b3712590f06245fbfba32d3be539812` |

| Repository milestone | Commit |
|---|---|
| Endpoint objective/official-evaluation record | `78c169d` |
| Gradient and block-swap audit | `e85dd86` |
| SIGReg pair-specificity and PCSF record | `fd04c0d` |
| Contraction/held-out-NTP audit | `90f2e8c` |
| Projection-space experiment | `3db02fa` |
| Gradient-combination implementation/results | `07fac75`; `e3572a1` |
| Generation-pathway audit | `a63e251` |
| Dense causal V-JEPA implementation/report | `07293be`; `8e059c2` |
| Persistent residual implementation/report | `29ebc15`; `0946d6b` |

The endpoint LLM-JEPA implementation retained native ChemFM NTP and the native
reaction serialization. Source and target rows were evaluated together. The
source readout was either an appended predictor token (`k=1`) or the source EOS
(`k=0`); the target readout was target EOS. Symmetric conditions backpropagated
through both endpoints. The stop-gradient control detached only the target in
the JEPA term. Auxiliary activity was sampled with probability `.5` and an
active-step outer coefficient of `2`, unless a section states otherwise.

The verified endpoint losses were:

```text
cosine:  mean(1 - cos(h_source, h_target))
raw MSE: mean((h_source - h_target)^2)
MSE+SIGReg auxiliary:
  MSE + [4 * .01 / .99] * SIGReg({h_source, h_target})
total: NTP + active * 2 * auxiliary
```

The factor four converts LeJEPA's two-view centered prediction term to raw
pairwise MSE. SIGReg was evaluated on the complete logical batch of 16. Its
streamed sufficient-statistic/VJP path matched materialization exactly in the
reference test: loss `1.6730396748` in both paths, representation-gradient
relative L2 error `1.75e-8`, parameter-gradient relative L2 error `0`, and
parameter-gradient cosine `1.0`.

The parity suite covered serialization, predictor-token insertion, k=0/k=1
readouts, target masks, symmetric/detached gradients, auxiliary cadence,
cosine/MSE values, and preservation of NTP. Fidelity is limited to the paths
actually tested; it is not a claim of byte-identical upstream execution or of
coverage for unused upstream branches.

Three evaluation protocols recur:

1. Small training-time checks verified completion.
2. One-view panels measured mechanism and were not official ChemFM endpoints.
3. Official evaluation used five R-SMILES views, beam width 10, ten candidates
   per view, canonical product handling, and reciprocal-rank aggregation per
   unique reaction.

## 2. Experiment inventory and primary generation results

| Condition | Scope | Generated result |
|---|---|---|
| Symmetric cosine endpoint, k=1 | seed 533, selected epoch 3 | `17/512` top-1 vs Native `24/512` |
| Target stop-gradient cosine, k=1 | seed 533 | `26/512` vs Native `24/512` |
| Cosine+SIGReg, batch 2, k=0/k=1 | seed 533 | `2/256`, `3/256` vs Native `7/256` |
| Cosine+SIGReg, batch 128, k=0 | 20 updates, not budget matched | `0/256` |
| Cadence-matched cosine+SIGReg-16, k=0 | 320 updates | `3/256` vs Native `6/256` |
| Raw endpoint MSE, k=0 | stopped at epoch-2 gate | no generation endpoint |
| Endpoint MSE+SIGReg-16, k=0 | seed 533, epoch 4 | one-view `6/256` vs `6/256`; official five-view `40/1280` vs `50/1280` |
| PCSF+MSE | seed 533, epoch 4 | one-view `4/256` vs Native `6/256` |
| Projection-space MSE+SIGReg | seed 533, epoch 4 | five-view `4/512` vs direct `15/512`, Native `18/512` |
| Direct-objective PCGrad/CAGrad/Du | seed 533, epoch 4 | five-view `3.91%/2.34%/3.91%` vs Native `4.30%` on 256 |
| Dense causal V-JEPA 2.1-style | seed 533, epoch 4 | one-view top-1 `6/256`; top-10 `39/256` vs Native `52/256` |
| Persistent pair residual | seeds 533/917/1301 | mean five-view top-1 effect `-1.04` pp; crossed CI `[-3.52,+1.17]` |
| Reduced DeepSeek/GSM8K reference | seed 533, two epochs | NTP `36/300`; JEPA `28/300`, `p=.229` |

Panel size, view protocol, seeds, and training budget differ across rows; the
table is an inventory, not a pooled comparison.

## 3. Endpoint cosine and MSE objective development

### 3.1 Symmetric cosine and early controls

On 512 reactions, the selected seed-533 k=1 checkpoints gave:

| Metric | Native | Symmetric cosine |
|---|---:|---:|
| Exact top-1 | 24/512 (4.688%) | 17/512 (3.320%) |
| Top-3 | 14.063% | 13.672% |
| Top-5 | 20.508% | 20.313% |
| Top-10 | 26.563% | 26.563% |
| Target-token CE | .239942 | .238685 |

Top-1 had 12 Native-only and 5 JEPA-only successes: difference `-1.367` pp,
95% paired bootstrap CI `[-2.93,+.20]`, exact McNemar `p=.143`. The
per-reaction CE interval included zero; Wilcoxon `p=.367`.

On 1,024 fixed identities:

| Checkpoint | Target variance | Effective rank | Mean-direction energy | Pair margin | Retrieval |
|---|---:|---:|---:|---:|---:|
| Base | .022171 | 26.16 | .674849 | .025028 | 34.38% |
| Native e3 | .015871 | 3.05 | .779364 | .016019 | 23.63% |
| Cosine e3 | .00003206 | 56.50 | .999617 | .00009372 | 72.27% |

For the cosine checkpoint, analysis-only mean centering and removal of the top
one, two, or four PCs gave pair margin/retrieval `.213060/76.27%`,
`.251651/81.64%`, `.285808/84.77%`, and `.273486/85.94%`, versus raw
`.0000937/72.27%`. Pair-margin correlations with CE or rank improvement ranged
from `-.056` to `+.006`; all bootstrap intervals included zero.

Early controls were:

| Condition | Generation | Other measurement |
|---|---|---|
| Target stop-gradient k=1 | `26/512` vs `24/512`; `+.39` pp, CI `[-1.37,+2.15]`, `p=.8318` | CE 7.69% above Native; target variance 12.03x symmetric and 46.3x below Native |
| SIGReg batch 2, k=0/k=1 | `2/256`, `3/256` vs `7/256` | statistic used only two samples/view |
| SIGReg batch 128, k=0 | `0/256`; CE `1.080978` | 20 rather than 320 updates; 15/20 auxiliary-active |
| SIGReg-16 preflight | not evaluated | applied SIGReg norm 1.20% of NTP; streamed/direct parity passed |
| Cadence-matched cosine+SIGReg-16 | `3/256` vs `6/256`; CE 1.30% higher | source/target variance 16.9x/14.8x below Native |

The cadence-matched cosine loss changed from `.19853` at epoch 1 to `.00200`
at epoch 4; SIGReg changed from `7.0744` to `7.4345`. Source/target
mean-direction energy was `.9988/.9978`; residual-PC2 retrieval was 73.0%.

The reduced official-repository DeepSeek-1.5B/GSM8K reference used one seed,
rank-16 LoRA, and two epochs. NTP/JEPA accuracy was `36/300` and `28/300`
(`p=.229`). JEPA target variance was 1.45x below NTP; target effective rank was
73.35; 300-way/four-way retrieval was 67.67%/94.00%.

### 3.2 Raw MSE and MSE+SIGReg

Both conditions used k=0, symmetric gradients, the same 1,280-row pilot,
seed 533, LoRA, LR `1e-4`, BF16, physical batch 4, accumulation 4, and 80
updates/epoch. Both reached epoch 2; only MSE+SIGReg passed the prespecified
geometry gate and continued to epoch 4. Plain MSE drew cadence per physical
microbatch, whereas exact SIGReg drew once per logical batch, so the epoch-2
comparison does not isolate cadence variance exactly.

Epoch-2 measurements:

| Condition | Variance S/T | Mean energy S/T | Rank S/T | Margin/retrieval | Target CE |
|---|---:|---:|---:|---:|---:|
| Native | .002028/.003203 | .9742/.9577 | 36.58/19.61 | .007600/44.9% | .251414 |
| Cosine+SIGReg-16 | .0001655/.0003277 | .9980/.9959 | 41.45/11.27 | .000486/58.6% | .255941 |
| MSE | .0005147/.0008114 | .9935/.9899 | 43.43/20.27 | .001650/64.5% | .257012 |
| MSE+SIGReg | .001607/.001513 | .9665/.9679 | 37.61/33.07 | .011916/84.0% | .261326 |

Training summaries:

| Condition | Epoch | NTP | MSE | SIGReg | Active updates |
|---|---:|---:|---:|---:|---:|
| MSE | 1 | 1.06584 | .032813 | - | 76/80 |
| MSE | 2 | .30591 | .003569 | - | 72/80 |
| MSE+SIGReg | 1 | 1.17629 | .035154 | 6.6591 | 47/80 |
| MSE+SIGReg | 2 | .30665 | .004796 | 6.9490 | 43/80 |
| MSE+SIGReg | 3 | .19851 | .002470 | 6.9976 | - |
| MSE+SIGReg | 4 | .15981 | .002154 | 6.9992 | - |

Epoch-4 one-view comparison:

| Endpoint | Native | MSE+SIGReg |
|---|---:|---:|
| Exact top-1 | 6/256 | 6/256 |
| Top-3/5/10 | 26/40/52 | 24/34/49 |
| Valid candidates | 78.20% | 86.60% |
| Historical aggregate CE | .240683 | .248779 |
| Label-normalized local CE | .229702 | .237275 |

Top-1 difference was `0` pp, CI `[-1.56,+1.56]`, McNemar `p=1`. Paired
outcomes were 4 both, 2 Native-only, 2 JEPA-only, 248 neither. Product rank
improved/worsened/tied on 18/35/203 reactions. Mean per-reaction Native-minus-
JEPA CE was `-.00773`, CI `[-.01386,-.00188]`; Wilcoxon `p=.0221`.

Epoch-4 geometry:

| Condition | Variance S/T | Mean energy S/T | Rank S/T | Raw margin/retrieval | Residual-PC2 margin/retrieval |
|---|---:|---:|---:|---:|---:|
| Native | .001431/.002320 | .9818/.9695 | 41.00/22.61 | .005479/43.4% | .2588/76.6% |
| MSE+SIGReg | .001293/.001281 | .9709/.9698 | 38.36/34.06 | .011634/85.9% | .4360/93.8% |

Pair-margin correlations with CE and rank changes were `.028/-.086` raw and
`.090/-.110` after PC2 removal; every bootstrap interval included zero.

### 3.3 Official five-view endpoint

The fixed epoch-4 checkpoints were evaluated without retraining on 1,280
prespecified unique reactions. The frozen sequential rule stopped at 1,280
because the 99% Wald upper bound was below the minimum effect of interest of
`+1` pp.

| Endpoint | Native | MSE+SIGReg | Difference |
|---|---:|---:|---:|
| Top-1 | 50/1280 (3.906%) | 40/1280 (3.125%) | -.781 pp |
| Top-3 | 17.891% | 16.172% | -1.719 pp |
| Top-5 | 26.797% | 23.516% | -3.281 pp |
| Top-10 | 37.656% | 35.625% | -2.031 pp |
| Per-view validity | 98.905% | 97.348% | -1.556 pp |
| Aggregate validity | 99.969% | 99.977% | +.008 pp |

Top-1 paired counts were 26 both, 24 Native-only, 14 JEPA-only, and 1,216
neither. The paired bootstrap 95% CI was `[-1.719,+.156]`; exact McNemar
`p=.1433`; the 99% Wald interval was `[-2.021,+.458]`. Holm-adjusted
top-3/5/10 p-values were `.0737/.00108/.0737`; only top-5 excluded the null,
in the Native-favored direction.

This endpoint concerns one seed, the fixed reduced-pilot exposure, and these
two checkpoints. It does not estimate training-seed variance.

## 4. Frozen mechanism audits of direct endpoint MSE+SIGReg

### 4.1 Gradient structure and block swaps

At the epoch-4 JEPA checkpoint, the active weighted auxiliary was `.212x` the
LoRA NTP-gradient norm with cosine `-.042`; 99.82% of auxiliary squared norm
was outside the one-dimensional NTP projection. Global measurements were:

| State | MSE ratio/cosine | SIGReg ratio/cosine | Active ratio/cosine |
|---|---:|---:|---:|
| Base | .195/+.360 | 2.952/-.341 | .276/+.215 |
| Native e4 | .184/-.048 | 5.270/+.022 | .359/-.023 |
| MSE+SIGReg e4 | .029/-.045 | 2.807/-.028 | .212/-.042 |

At the JEPA checkpoint, source-only and target-only MSE ratios/cosines were
`.028/-.026` and `.034/-.017`. True and length-matched shuffled active
gradients had cosine `.902`; the residual was `.096x` NTP with cosine `+.032`.
For depth groups 0--5, 6--11, 12--16, and 17--21, active-auxiliary/early-NTP
cosines were `+.060, -.014, -.042, -.107`; layers 20/21 were `-.272/-.213`.

Frozen block substitutions on 256 reactions used aggregate CE `.240753`
Native and `.248691` JEPA:

| Hybrid | CE | Delta vs Native | Mean paired delta | Better/worse |
|---|---:|---:|---:|---:|
| Native + JEPA layers 0--5 | .240394 | -.000359 | -.000780 | 137/119 |
| Native + JEPA layers 6--11 | .240293 | -.000460 | -.001050 | 133/123 |
| Native + JEPA layers 12--16 | .237306 | -.003447 | -.003457 | 148/108 |
| Native + JEPA layers 17--21 | .256360 | +.015607 | +.017361 | 9/247 |
| JEPA + Native layers 0--5 | .246895 | +.006142 | +.005755 | 115/141 |
| JEPA + Native layers 6--11 | .253085 | +.012332 | +.012561 | 103/153 |
| JEPA + Native layers 12--16 | .252334 | +.011581 | +.011293 | 94/162 |
| JEPA + Native layers 17--21 | .244320 | +.003567 | +.002794 | 126/130 |
| Native + JEPA token I/O | .255083 | +.014331 | +.014777 | 88/168 |
| JEPA + Native token I/O | .267329 | +.026577 | +.028449 | 40/216 |

For Native + JEPA layers 17--21, the reaction-level CI for mean CE change was
`[+.014308,+.017021]`; for Native + JEPA layers 12--16 it was
`[-.005278,-.001631]`. These off-trajectory hybrids mix co-adapted checkpoints;
their effects are not additive parameter attributions.

### 4.2 SIGReg pair-specificity

This frozen assay used MSE+SIGReg epochs 1/2/4, four disjoint batches of 16,
four fresh 1,024-slice SIGReg draws per batch, and LoRA A/B tensors only.

| Epoch | MSE true/shuffle cosine | Pair residual/NTP | Applied SIGReg/NTP | SIGReg/MSE | Pair-specific fraction of full aux |
|---:|---:|---:|---:|---:|---:|
| 1 | .918 +/- .023 | .040 +/- .018 | .184 +/- .073 | 1.51 +/- .59 | .468 +/- .069 |
| 2 | .938 +/- .015 | .058 +/- .018 | .302 +/- .062 | 3.43 +/- .53 | .418 +/- .061 |
| 4 | .937 +/- .012 | .042 +/- .010 | .223 +/- .032 | 4.06 +/- .53 | .398 +/- .060 |

SIGReg/NTP cosines were `-.056, -.047, -.047`. Under infinitesimal descent,
SIGReg increased squared-distance pair discrimination, cosine margin,
reaction-center separation, and joint variance in all 48 draws. MSE decreased
pair discrimination and cosine margin in every batch. At epochs 2/4, the full
auxiliary increased discrimination in 15/16 and 15/16 draws while also
increasing true-pair distance in most measurements. These are local directional
derivatives, not finite optimizer trajectories.

### 4.3 Contraction and held-out NTP directions

Three fixed train/held-out batch pairs of 16 were measured over Native,
MSE-only, PCSF, and MSE+SIGReg checkpoints. Only LoRA A/B tensors were included.
No optimizer step was taken.

| State | Pair-center sigma | Ratio to same-epoch Native |
|---|---:|---:|
| Native e1/e2/e4 | .05229/.04213/.03609 | 1/1/1 |
| MSE e1/e2 | .04471/.02219 | .847/.526 |
| PCSF e1/e2/e4 | .04604/.02583/.01963 | .873/.611/.543 |
| MSE+SIGReg e1/e2/e4 | .05651/.03584/.03329 | 1.078/.855/.925 |

MSE descent reduced pair-center spread in `33/33` checkpoint/batch
measurements; `2(g_true-g_shuffle)` increased it in `33/33`. At MSE+SIGReg
epochs 2/4, the pair residual reduced held-out NTP in `6/6`, total MSE increased
it in `6/6`, applied SIGReg increased it in `5/6`, and the full auxiliary
increased it in `5/6`. The measured mean held-out changes per unit learning
rate, multiplied by `1e3`, were:

| State | MSE | Pair residual | Regularizer | Full auxiliary |
|---|---:|---:|---:|---:|
| MSE e2 | +.144 | -.115 | - | +.144 |
| PCSF e2 | +.108 | -.084 | -.038 | +.071 |
| PCSF e4 | -.019 | -.016 | -.046 | -.065 |
| MSE+SIGReg e2 | +.469 | -.527 | +.409 | +.877 |
| MSE+SIGReg e4 | +.209 | -.474 | +1.068 | +1.277 |

The sign consistency is based on three batch pairs and does not estimate a
population effect precisely. The later persistent residual experiment in
Section 7 tested whether the local direction survives optimization.

### 4.4 Generation pathway

On 256 reactions, final-layer source-to-target-only CKA was `.128` Native and
`.547` JEPA; source-to-teacher-forced autoregressive-product CKA was `.182`
and `.278`. Source-to-target retrieval was `10.55%/47.27%`; source-to-AR-product
retrieval was `.39%/3.52%`. At layer 16, the corresponding retrieval was
`25.78%/65.23%` and `4.30%/1.95%`.

Frozen residual-stream patching on 64 reactions gave:

| Intervention | Aggregate CE change | Reaction-mean 95% CI |
|---|---:|---:|
| Native + JEPA state after layer 11 | +.00834 | [-.00586,+.02025] |
| Native + JEPA state after layer 16 | +.00690 | [-.00643,+.01888] |
| Native + JEPA state after layer 21 | +.03069 | [+.02149,+.04750] |
| JEPA + Native state after layer 11 | -.00722 | [-.01775,+.00907] |
| JEPA + Native state after layer 16 | -.00330 | [-.01474,+.01117] |
| JEPA + Native state after layer 21 | -.00613 | [-.01807,+.00793] |

At layer 16, separating positions gave Native-recipient changes `+.00071` for
context and `+.00635` for target-prediction positions. Patching measures whole
checkpoint states, not a JEPA subspace.

One exact saved-state AdamW counterfactual at step 320 used 16 train and 16
held-out reactions. Raw NTP/JEPA gradient cosine was `-.02228` over all
trainables (`-.03682` LoRA-only); AdamW-update cosine was `+.81434`.

| Virtual update | First-order held-out CE change | Observed change |
|---|---:|---:|
| NTP only | +.000258 | +.000982 |
| JEPA only | +.000127 | +.000244 |
| NTP+JEPA | +.000283 | +.000541 |

Hard-four-way retrieval was `42.19%` Native and `76.95%` JEPA; after
independent canonicalization/component sorting it was `45.31%/72.27%`.
Full-256 canonicalized retrieval was `10.94%/42.58%`.

Existing 1,280 five-view predictions were rescored chemically:

| Endpoint | Native | JEPA | Paired change, 95% CI |
|---|---:|---:|---:|
| Top-1 Morgan Tanimoto | .47346 | .47281 | -.00066 [-.01185,+.01061] |
| Best-top-3 | .63968 | .63057 | -.00911 [-.01723,-.00091] |
| Best-top-5 | .70068 | .68245 | -.01823 [-.02598,-.01047] |
| Best-top-10 | .76050 | .74765 | -.01285 [-.02038,-.00546] |
| Top-1 scaffold match | 40.94% | 40.86% | -.08 pp [-1.48,+1.33] |
| Any-top-3 scaffold | 54.84% | 52.97% | -1.88 pp [-3.36,-.47] |
| Any-top-5 scaffold | 60.55% | 58.20% | -2.34 pp [-3.91,-.78] |

## 5. Later endpoint interventions

These conditions were separate seed-533 trajectories, not continuations of
one another. Their code was removed from the maintained trainer after the
experiments. Report 03's lengthy PCSF and performance-engineering narrative is
not retained here; the executed definition, calibration, outcomes, limits,
and provenance are retained.

### 5.1 Pair-Center Spread Floor (PCSF)

PCSF added a reference-relative hinge on the standard deviation of pair
midpoints to raw endpoint MSE:

```text
m_i = (s_i+t_i)/2
sigma_PC = sqrt(sum_i ||m_i-mean(m)||^2 / ((B-1)D) + epsilon)
L_PCSF = relu(.80 * sigma_native,e4,batch - sigma_PC)^2
L_total = L_NTP + active * 2 * (L_MSE + 4.2 * L_PCSF)
```

The coefficient `4.2` was fixed from failed-MSE-e2 all-trainable gradient
calibration; no outcome-based sweep was run. The four-epoch trajectory used the
same 1,280 rows and 320-update schedule as the direct condition. The measured
frozen spread ratio to Native e4 was `1.192, .688, .560, .535` at epochs
1--4, so the `.80` floor was not maintained.

| Epoch-4 metric | Native | Direct MSE+SIGReg | PCSF |
|---|---:|---:|---:|
| Pair-center ratio | 1 | .935 | .535 |
| Rank S/T | 49.50/26.27 | 42.73/36.29 | 56.31/25.90 |
| Raw retrieval | 43.4% | 85.9% | 69.5% |
| Residual-PC2 retrieval | 76.6% | 93.8% | 84.0% |
| One-view top-1 | 6/256 | 6/256 | 4/256 |
| Top-3/5/10 | 26/40/52 | 24/34/49 | 22/34/45 |
| Target CE | .240683 | .248779 | .246664 |

PCSF-minus-Native top-1 was `-.781` pp, CI `[-2.344,+.781]`, McNemar
`p=.625`. Mean per-reaction Native-minus-PCSF CE was `-.00616`, CI
`[-.01241,-.00011]`; Wilcoxon `p=.245`. Four pair-margin correlations with CE
or beam-rank change ranged from `-.057` to `+.095`, all `p>=.128`.

### 5.2 Projection-space MSE+SIGReg

This condition applied both terms after one shared
`2048 -> 2048 -> 2048 -> 64` Linear-BatchNorm-ReLU/Linear projector. There was
no raw-space auxiliary. It used the matched four-epoch schedule and completed
320 updates in 32.33 minutes.

| Metric | Raw ChemFM space | Projected space |
|---|---:|---:|
| Variance S/T | .004884/.012639 | .243582/.263016 |
| Rank S/T | 12.58/5.22 | 3.22/3.29 |
| Pair-center spread | .080046 | .491247 |
| Mean-direction energy S/T | .9391/.8389 | .00147/.00392 |
| Pair margin | .04971 | .81588 |
| Retrieval top-1/MRR | 85.55%/90.53% | 76.56%/86.52% |

At epoch 4, projected auxiliary/held-out-NTP LoRA gradient cosine was `-.0300`
and its applied norm ratio was `2.36x`. On the matched 256 panel, CE was
`.256497`, versus `.248779` direct and `.240683` Native.

Five-view generation was budget-bounded before outcome inspection to the first
512 reactions of the frozen manifest:

| Endpoint | Native | Direct | Projected |
|---|---:|---:|---:|
| Top-1 | 18/512 | 15/512 | 4/512 |
| Top-3 | 88 | 83 | 33 |
| Top-5 | 130 | 119 | 70 |
| Top-10 | 187 | 176 | 135 |
| Per-view validity | 99.09% | 97.45% | 96.68% |

Projected-minus-direct top-1 was `-2.148` pp, CI `[-3.516,-.781]`, McNemar
`p=.00342`; projected-minus-Native was `-2.734` pp, CI
`[-4.297,-1.367]`, `p=.000122`. This 512 prefix was a descriptive,
budget-bounded panel rather than the 1,280-reaction endpoint.

### 5.3 Gradient-combination matrix

Seven four-epoch trajectories retained the direct MSE+SIGReg objective and
changed either its effective coefficient (`.25/.5/1/2`) or LoRA-only gradient
combination (PCGrad, CAGrad, or Du auxiliary similarity at effective 1). The
same 172/320 updates were auxiliary-active. Token-I/O tensors retained ordinary
weighted-sum gradients for the combiner conditions.

| Condition | Mean train cosine | Conflict | Raw ratio | Combiner modification |
|---|---:|---:|---:|---:|
| lambda .25 | +.0035 | 54.1% | .0983 | 0% |
| lambda .5 | +.0047 | 64.5% | .0991 | 0% |
| lambda 1 | -.0264 | 80.8% | .0899 | 0% |
| lambda 2 | -.0377 | 83.7% | .0689 | 0% |
| PCGrad | -.0300 | 82.6% | .0912 | 1.14% |
| CAGrad | -.0706 | 86.6% | .0593 | 67.86% |
| Du | -.0050 | 52.3% | .1188 | 21.63%; mean gate .0153 |

Across epochs 1/2/4, held-out full-auxiliary cosine was negative in `20/21`
audits; SIGReg was negative in `19/21`. Epoch-4 five-view evaluation was
restricted during execution to the first 256 frozen reactions:

| Condition | Variance S/T | Center spread | Rank S/T | Pair margin | Retrieval |
|---|---:|---:|---:|---:|---:|
| Native | .02488/.02233 | .11622 | 25.79/23.04 | .06211 | 40.6% |
| lambda .25 | .00167/.00201 | .03549 | 43.18/25.17 | .00792 | 74.2% |
| lambda .5 | .00133/.00338 | .03888 | 38.61/11.56 | .00870 | 73.8% |
| lambda 1 | .00188/.00192 | .03854 | 35.92/29.35 | .01881 | 85.2% |
| lambda 2 | .00098/.00082 | .02585 | 44.08/38.49 | .01237 | 85.2% |
| PCGrad | .00188/.00197 | .03817 | 40.27/34.48 | .01813 | 84.0% |
| CAGrad | .00071/.00064 | .02226 | 50.60/41.13 | .01751 | 82.4% |
| Du | .00166/.00153 | .03342 | 39.54/33.11 | .00628 | 66.4% |

| Condition | Top-1 | Top-3/5/10 | Target CE | CE delta vs Native |
|---|---:|---:|---:|---:|
| Native | 4.30% | 19.53/26.95/39.06% | .240683 | - |
| Historical direct | 3.52% | 18.75/26.17/37.11% | .248779 | +3.36% |
| lambda .25 | 1.95% | 15.23/24.22/36.72% | - | - |
| PCGrad | 3.91% | 16.41/25.00/35.55% | .257099 | +6.82% |
| CAGrad | 2.34% | 13.67/25.78/35.94% | .295247 | +22.67% |
| Du | 3.91% | 19.53/26.56/35.94% | .243893 | +1.33% |

Native-relative top-1 CIs were `[-3.13,+1.56]` direct,
`[-4.69,-.39]` lambda-.25, `[-2.73,+1.95]` PCGrad,
`[-4.30,0]` CAGrad, and `[-2.73,+1.95]` Du. This is a one-seed,
256-reaction screen.

## 6. Dense causal V-JEPA 2.1-style pilot

This was a distinct architecture family, not endpoint LLM-JEPA. It translated
the released V-JEPA 2.1 dense masked/context loss to causal product suffixes:
four target depths (6/11/17/22), a 24-block width-384 predictor, detached EMA
target (`.99925`), eight local `.15` and two global `.70` suffix masks, masked
L1, proximity-weighted visible-context L1, and a context coefficient ramp to
`.5`. Native NTP remained active. The mapping and deviations are documented in
`docs/VJEPA2_1_CHEMFM_MAPPING.md`.

The seed-533 pilot used the same 1,280 training reactions, 320 updates, and
fixed 256-reaction one-view panel. Training took 38.59 minutes. At step 72,
masked predictor/ChemFM component-gradient ratios were `6.84/4.73/3.07/3.15`
at depths 6/11/17/22; context ratios were `2.34/3.01/2.41/1.82`.

| Endpoint | Native | Direct endpoint | Dense causal V-JEPA |
|---|---:|---:|---:|
| One-view top-1 | 6/256 | 6/256 | 6/256 |
| Top-3/5/10 | 26/40/52 | 24/34/49 | 21/29/39 |
| Per-view valid candidates | 78.20% | 86.60% | 83.20% |
| Matched 64-reaction token CE | .243832 | .253330 | .253547 |
| Teacher-forced token top-1 | 92.394% | 92.431% | 92.249% |

Global k=0 geometry on 256 reactions:

| Condition | Variance S/T | Rank S/T | Center spread | Raw/residual retrieval | Ridge EV |
|---|---:|---:|---:|---:|---:|
| Native | .001432/.002317 | 41.02/22.70 | .03447 | 45.31/78.52% | .1366 |
| Direct | .001293/.001282 | 38.34/34.01 | .03125 | 83.98/93.75% | .3162 |
| Dense | .001255/.001988 | 40.68/27.64 | .03182 | 33.59/77.34% | .0819 |

Dense-vs-Native causal-token CKA at depths 6/11/17/22 and final norm was
`.9877/.9891/.9854/.9940/.9936`; relative RMS displacement was
`.1947/.2097/.2351/.1444/.1233`. This pilot did not use the later official
five-view endpoint and did not estimate seed variability.

## 7. Persistent pair-specific residual trajectory

This experiment directly tested the earlier frozen favorable residual
direction. For each logical batch of 16:

```text
g_pair = grad_LoRA MSE(source, true target)
       - grad_LoRA MSE(source, length-matched deranged target)
active LoRA update gradient = g_NTP + 2*g_pair
```

SIGReg cancels under the permutation contrast. The residual applied only to
the 308 LoRA A/B tensors (6,307,840 parameters); token-I/O tensors received NTP
only. Native and residual conditions used paired seeds 533/917/1301, the same
1,280-row data/order, 320 updates, `.5` cadence, and rank-8 ChemFM LoRA. The
primary endpoint was the prespecified fixed 256 reactions, all five views.

The preregistered rule labeled FAIL if the mean top-1 effect was nonpositive
and no more than one seed improved. PASS required all three seeds positive and
a crossed seed/reaction interval excluding zero; other outcomes were
INCONCLUSIVE.

| Seed | Aggregate top-1 Native -> residual | View effects 1--5 (pp) |
|---:|---:|---:|
| 533 | 2.73% -> 3.12% (`+.39`) | +1.95, 0, +1.17, +.39, -.78 |
| 917 | 2.34% -> 1.95% (`-.39`) | -1.56, -1.56, 0, +.39, +1.95 |
| 1301 | 5.08% -> 1.95% (`-3.12`) | -.78, -.39, -.78, 0, -.78 |

Across 768 seed-reaction pairs, Native/residual had 26/18 exact top-1. The
mean effect was `-1.04` pp, seed SD `1.85` pp, seed-level t interval
`[-5.63,+3.54]`, and crossed seed/reaction bootstrap interval
`[-3.52,+1.17]`. Only seed 1301's paired interval excluded zero
(`[-5.86,-.39]`, McNemar `p=.0386`); no across-seed significance was claimed.

Five-view token CE worsened in every seed by `+.01332,+.00659,+.00675`;
clustered mean `+.01075`, interval `[+.00586,+.01646]`. Mean correct-token
margin changed `-.0294` (`[-.0741,+.0385]`) and correct-token rate `-.0423`
pp (`[-.190,+.117]`).

Across training, mean residual/NTP cosine moved from about `-.004` in epoch 1
to `-.103` in epoch 4; applied residual/NTP norm ratio grew `.34 -> 1.79`;
the counterfactual AdamW update fraction grew `.13 -> .37`. True/shuffled
gradient cosine fell `.55 -> .39`, while residual/true norm grew `.98 -> 1.54`.
AdamW preconditioning amplification fell `.56 -> .26`.

The intervention changed endpoint diagnostics: correct-minus-shuffle cosine
`.0035 -> .4321`, retrieval `42.7% -> 80.9%`, replacement sensitivity
`.0048 -> .3201`, and mean source effective rank `36.5 -> 6.74`.

Under the preregistered rule the result is **FAIL** for the narrow tested
hypothesis: this exact residual, coefficient, cadence, LoRA scope, data, and
optimizer budget did not improve generation. The result does not contradict
the frozen local derivative, because that derivative describes selected
states and infinitesimal steps rather than the full adaptive trajectory. It
does not test other coefficients, optimizers, capacities, or architectures.

## 8. Factual boundaries of the combined record

- Endpoint JEPA conditions repeatedly changed source/product geometry and
  retrieval, but those diagnostics did not provide a consistent ordering of
  generated accuracy in the tested checkpoints.
- The official seed-533 MSE+SIGReg endpoint had lower top-k generation than its
  Native control; top-1 uncertainty included zero and the prespecified futility
  boundary excluded a `+1` pp benefit for that fixed comparison.
- Frozen residual gradients had favorable held-out-NTP signs on three batch
  pairs at selected states. Persistent residual training met its preregistered
  failure rule and worsened mean CE and generation in the three-seed screen.
- PCSF, projection-space loss, gradient-combination methods, and dense causal
  V-JEPA were each tested only in their recorded small, mostly one-seed pilots.
  No claim about the general objective family follows from those screens.
- Retrieval, CKA, curvature, effective rank, and local gradients are not
  substitutes for generated exact match.

## 9. Reproducibility map

| Evidence | Retained path |
|---|---|
| Cosine geometry/coupling | `runs/diagnostics/decoder_coupling/`; `runs/diagnostics/uspto_mit_geometry_diagnosis.json` |
| Stop-gradient/SIGReg controls | `runs/diagnostics/target_sg_rescue_512/`; `runs/diagnostics/sigreg_k_ablation_256/`; `runs/diagnostics/sigreg_k0_batch128_256/` |
| MSE/MSE+SIGReg training/evaluation | `runs/mse_ablation/` |
| Official 1,280 endpoint | `runs/official_five_view_endpoint/` |
| Gradient/block-swap audit | `runs/diagnostics/mse_sigreg_mechanistic_audit/` |
| SIGReg pair audit | `runs/diagnostics/sigreg_pair_specificity_audit/audit.json` |
| Contraction/NTP audit | `runs/diagnostics/contraction_ntp_directional_audit/audit.json` |
| Generation-pathway audit | `runs/diagnostics/generation_mechanism/` |
| PCSF compact results | `runs/pcsf/` |
| Projection-space results | `runs/projected_mse_sigreg/` |
| Gradient-combination matrix | `runs/gradient_interaction/a6000/endpoint_256/summary.json` |
| Dense causal V-JEPA pilot | `runs/vjepa2_1/a6000/` |
| Pair residual | `runs/pair_residual/a6000/` |

Maintained implementation paths are `src/train.py`, `src/chemfm.py`,
`src/jepa.py`, `src/vjepa2_1.py`, `src/representation_eval.py`, and
`src/eval_uspto_mit_five_view_a6000.py`. Historical scripts and removed bulky
artifacts remain recoverable from commit `61fbc74`.
