# RITA-M STP and NextLat diagnostic report

Date: 2026-09-14  
Checkpoint: `lightonai/RITA_m` at `d819157a3a96d278500232b2cbb2a02d9646bcf6`  
Status: frozen diagnostic complete and NextLat validity amendment incorporated; no RITA parameters were optimized

## Executive decision

Neither released STP nor faithful NextLat merits a rank-32 LoRA experiment on the present evidence.

RITA-M does not exhibit the straight, locally persistent native trajectory assumed by a literal tube/geodesic account of STP: mean adjacent tangent correlation is `-0.4045` (protein bootstrap 95% CI `[-0.4055,-0.4034]`). Structural differences exist but are small and internally inconsistent, the primary geometry survives sequence reversal, and removing only 10% of the chord-orthogonal component harms the correct-residue margin. Some local straightness correlates with native prediction quality, but longer-scale tube radius and path efficiency relate to ProteinGym fitness and free-running quality in the opposite direction from the STP premise. Released STP also requests a large and variable gradient (`2.99x` NTP, 95% CI `[1.74,4.51]`). These results do not overturn ChemFM's small, inconclusive positive behavioral signal; they say that RITA supplies no stronger mechanistic reason to repeat it.

The amended NextLat evidence is more precise. Direct full-1024D ridge prediction gives `R2=.3701` from a 256-D representation of `h_t`, `.3870` after adding the realized residue, and `.3631` with a matched shuffled residue. The actual amino acid therefore adds real but modest information. The latest state-sufficiency correction conditions on every coordinate of final `h_t`: the base reaches `R2=.4796`, and layer-12 information unique beyond it raises this to `.4912` (increment `+.01168`; 2.24% residual-error reduction). Decoder JS/KL do not improve and top-1 changes only `+.00308`. Some latent residual information remains distributed, but decoder-functional insufficiency of final `h_t` is not established.

Raw latent error was indeed scale-confounded. A denser predictor fitted on 130,682 train transitions reduces held-out latent loss, but does not change the faithful ordering. Most importantly, decoder error normalized by the actual decoder transition remains adverse on UniRef (`rho=+.528` versus correct probability, CI `[+.453,+.592]`) and ProteinGym (`macro rho=+.170` versus DMS, `[+.108,+.230]`); it is null in the generation replay. Absolute decoder JS and the complete faithful loss remain adverse on all three views. Thus “latent predictability is intrinsically adverse” remains withdrawn, but predictor underfitting and decoder-transition magnitude do not explain the behavior mismatch of the implemented faithful objective.

A dense predictor optimized under unit `SmoothL1 + KL` lowers validation latent/KL/total from `.1756/.5762/.7518` to `.04705/.1985/.2456` with RITA frozen. Relative to the earlier 24,504-example full-objective predictor, latent validation error improves by 8.37% and total by 0.98%, while KL worsens slightly; a large train/validation gap remains. The subsequent explicitly requested rank-32 LoRA audit finds ratio `.500` (CI `[.439,.564]`), cosine `+.213` (`[+.096,+.329]`), and retention `1.080` (`[1.021,1.138]`). Thus ChemFM-style adapter-gradient domination does not reappear, but no LoRA training was run and gradient compatibility does not repair adverse behavior coupling.

## 1. Scope and preregistered exclusions

This study characterized one frozen 300M-parameter autoregressive protein LM. It performed no STP fine-tuning, NextLat backbone fine-tuning, matched Native LoRA training, objective sweep, rank sweep, full continued pretraining, RITA-L confirmation, broad geometry suite, or large generation/structure campaign. Predictor-only fitting was used for faithful NextLat exactly so that its learned auxiliary gradient could be measured without changing RITA.

The independent resampling unit is a protein or MMseqs cluster, not a residue. Reported intervals use whole-cluster/protein bootstrap resampling. Paired biological contrasts use proteins present in both strata and sign-flip tests. ProteinGym uncertainty is across assays. Families of related structural and generation comparisons were interpreted with Benjamini-Hochberg correction.

### 1.1 NextLat validity amendment

The first diagnostic was incomplete in five specific ways. Raw SmoothL1 could covary with hidden-state or transition scale; the warmed predictor minimized only SmoothL1 before its `SmoothL1 + KL` backbone gradient was evaluated; concatenated final/layer-12 inputs competed for the same 256 PCA coordinates and therefore did not constitute a conditional-information test; target PCA retained only about 61.6% of train variance; and all gradient ratios were measured over the full backbone rather than the proposed rank-32 LoRA subspace. The first amendment then still conditioned layer 12 on only a 256-D PCA of final `h_t`, trained the faithful predictor on about 12 transitions per protein, and used absolute rather than decoder-transition-relative JS. These are limitations, not assumed fatal bugs. Original outputs and numbers are retained and labeled; successive amendment artifacts state which conclusions changed.

The real pinned checkpoint also closes the hidden-state/logit audit gap: applying the live tied LM head to the captured final state for the 20-residue canonical test sequence reproduced `outputs.logits` exactly in the observed FP16 run (maximum and mean absolute difference both `0.0`, tensor shape `1 x 21 x 26`).

## 2. RITA-M audit

The audit used the [RITA paper](https://arxiv.org/abs/2205.05789), [official RITA repository](https://github.com/lightonai/RITA), the pinned [RITA-M checkpoint](https://huggingface.co/lightonai/RITA_m), and the current [official fitness implementation](https://github.com/lightonai/RITA/blob/master/compute_fitness.py).

### Architecture and tokenization

RITA-M is a 24-layer decoder-only Transformer with hidden width 1024, 16 attention heads, feed-forward width 4096, vocabulary 26, and maximum sequence length 1024. The checkpoint's input embedding and LM-head weights are genuinely tied: they have the same storage pointer and exact values.

The tokenizer is not safely described by generic Hugging Face conventions:

- There is no BOS token.
- The tokenizer postprocessor appends `<EOS>` (id 2).
- `<PAD>` exists at id 1, but `bos_token_id`, `eos_token_id`, and `pad_token_id` are all `None` on the loaded tokenizer, and `special_tokens_map.json` is empty.
- Each of `ACDEFGHIKLMNPQRSTVWY` encodes to exactly one same-character token, followed by EOS. Thus a canonical protein of length L has L biological causal positions and L+1 encoded tokens.
- The tokenizer normalizer deterministically maps ambiguous symbols (`X->L`, `B->D`, `Z->E`, `J->L`). This differs from the paper's training-data description and is scientifically inappropriate for these diagnostics. All project paths therefore reject noncanonical sequences rather than silently normalizing them.

The official model implementation also does not expose hidden states in the modern generic form: `outputs.hidden_states` is the final tensor, and an intermediate layer must be captured with a hook. This study used the final state throughout and zero-based layer 11 only for the state-sufficiency control.

### Causal and fitness alignment

With no BOS state, `h_t` at residue t predicts residue `x_(t+1)`. STP paths and NextLat transitions exclude PAD, EOS, and every transition into EOS. The official fitness function instead encodes each direction, feeds `ids[:-1]`, scores `ids[1:]`, and sums negative mean cross-entropy for the forward and reversed sequence. It therefore omits the first residue of each direction and includes terminal EOS. A real RITA-M replay on `ACDEFGHIKLMNPQRSTVWY` matched the upstream function within `0.00195` absolute score under FP16 (tolerance `0.005`). The current source still explicitly evaluates both the sequence and its reversal.

The paper reports training on unclustered UniRef100, about 150B amino acids, in both sequence directions. It deliberately held out 20 Pfam families. Ordinary modern proteins in this report are not called pretraining-unseen.

## 3. Prior CLM-JEPA evidence that constrains this study

Reports 00--08 were reviewed before implementation.

For STP, the released and literal paper formulations are different objectives. Released rank-8 STP at lambda 0.02 produced an untouched four-seed mean top-1 change of `+0.547` percentage points, with all seed effects positive but a crossed 95% interval `[-0.195,+1.328]` and Holm-adjusted `p=.1025`. That is **inconclusive with a small positive signal**, not a clean ChemFM failure. The literal Euclidean/geodesic story was not supported: ChemFM product paths had `C(1)=-0.415`, wide normalized tubes, and chord-orthogonal components that were decoder-functional.

For NextLat, the faithful implementation succeeded proximally: full-state transition `R2` reached `0.6043` versus `0.538` Native, while generated top-1 fell in both seeds (mean about `-1.37` pp). Report 8 then found faithful auxiliary/LoRA NTP norm ratios of `2.8--4.6x`, systematic middle/late-layer conflict, and substantial loss of NTP-direction retention. Thus the important prior is not failure to learn transitions; it is that increased transition predictability did not produce better generation and could interfere strongly with NTP.

## 4. Data and leakage controls

### Ordinary sequence pool

The source is the documented [`alejoacelas/uniref50-2025-10`](https://huggingface.co/datasets/alejoacelas/uniref50-2025-10) validation split at revision `2b1724ac1b0adb362f6af3e80b87e675bf8f8933`. The first 3,000 eligible Dataset Viewer records after deterministic scanning were retained if they were canonical and length 64--512. Candidate-manifest SHA256 is `776459f5...19dd408`.

MMseqs2 13-45111+ds-2 used:

```text
easy-cluster --min-seq-id 0.3 -c 0.8 --cov-mode 0 --cluster-mode 2 --threads 6
```

It produced 2,996 clusters. Stable whole-cluster hashing yielded 2,042 train, 471 validation, and 487 test proteins. No protein or cluster crosses a split. Locked-manifest SHA256 is `9de1e2c7...1e0cb6`.

### Structural data

The [TAPE secondary-structure and ProteinNet tasks](https://github.com/songlab-cal/tape) supplied labels only. Canonical length filtering retained 466 CASP12/CB513/TS115 proteins with three-state secondary structure and 218 ProteinNet validation/test proteins with contacts. Long-range contact density is the fraction of valid residues at sequence separation at least 24 whose CA distance is below 8 A. The combined manifest contains 684 proteins and has SHA256 `769bc223...ea9a1`.

### Functional panel

The [ProteinGym v1.3 substitution benchmark](https://github.com/OATML-Markslab/ProteinGym) was locked before surrogate results were inspected. Eligibility required a canonical target of length 64--512, at least 100 single substitutions, and distinct UniProt IDs. SHA256 ranking and round-robin sampling across Activity, Binding, Expression, OrganismalFitness, and Stability selected 14 assays; each was capped at 256 variants by a phenotype-blind hash of assay and mutation. The panel includes D7PM05, YAP1, PTEN, KKA2, VILI, CASP3, CCR5, CP2C9 abundance, influenza NP, FECA, LGK, SPG1, GLPA, and HIS7 assays. The exact IDs and variant-file hashes are in `protein/data/proteingym_panel.json` and its manifest.

Project-created probe splits are homology-separated. TAPE and ProteinGym were not claimed to be absent from RITA's original UniRef100 corpus, nor was a retrospective pretraining decontamination claim attempted. Their labels are used only to compare quantities already present in the same frozen checkpoint. A clean fixed sequence source for all 20 historically withheld Pfam families was not established without disproportionate curation, so that optional OOD analysis was omitted rather than represented as clean when it was not.

## 5. Frozen Native STP characterization

### 5.1 Core geometry

The main 3,000-protein results are:

| Quantity | Mean | Protein bootstrap 95% CI |
|---|---:|---:|
| local curvature `1-cos(delta_t,delta_(t+1))` | 1.4045 | [1.4034, 1.4055] |
| adjacent tangent correlation C(1) | -0.4045 | [-0.4055, -0.4034] |
| normalized tube radius, span 4--8 | 0.7576 | [0.7538, 0.7627] |
| normalized tube radius, span 9--24 | 0.7605 | [0.7579, 0.7638] |
| normalized tube radius, span 25--64 | 0.7177 | [0.7149, 0.7206] |
| path efficiency, span 4--8 | 0.1636 | [0.1631, 0.1640] |
| path efficiency, span 9--24 | 0.06388 | [0.06361, 0.06416] |
| path efficiency, span 25--64 | 0.03021 | [0.02999, 0.03044] |

These values show backtracking, broad departure from endpoint chords, and rapidly declining chord/path efficiency. RITA therefore does not possess a native short-scale ballistic regime that a literal straight-tube mechanism could simply reinforce.

### 5.2 Biological stratification

The strongest paired results are small:

- Helix minus coil curvature is `-0.00666` (CI `[-0.01135,-0.00199]`, BH `q=.0113`), apparently straighter locally, but helix tube radius is **larger** by `+0.01917` (CI `[+0.01373,+0.02468]`, `q=.00013`) and efficiency is lower by `-0.01504` (`q=.00013`).
- Beta minus coil curvature is `+0.01472` (CI `[+0.00981,+0.01971]`, `q=.00013`): beta regions are less, not more, locally coherent by this measure.
- Secondary-structure transitions minus stable regions have curvature `+0.00642` (CI `[+0.00375,+0.00912]`, `q=.00013`), but tube and efficiency differences cross zero.
- High minus low within-protein contact density has curvature `+0.00526` (CI `[+0.00099,+0.00980]`, `q=.0275`); its tube and efficiency differences are not significant after correction.

The detectable effects are around 0.5--2% of their baseline quantities and disagree across metrics. RITA geometry senses biological context, but not as a unitary straight/coherent-tube organization.

### 5.3 Reversal

On 128 paired proteins processed independently in reverse order, forward-minus-reverse curvature is `+0.00076` (CI `[-0.00131,+0.00287]`, `p=.484`) and tangent C(1) is `-0.00076` with the mirrored interval. All tube comparisons cross zero; the largest is the 25--64 span (`+0.01296`, CI `[-0.00163,+0.02717]`, `p=.078`). Only short-span efficiency differs slightly (`-0.00124`, CI `[-0.00228,-0.00023]`, unadjusted `p=.018`).

Thus the dominant geometry is effectively direction-generic, as expected for a model trained on both directions. It is not evidence specific to natural N-to-C organization.

### 5.4 Orthogonal-component function

For 684 proteins and up to 24 decisions per protein, 10% of the local chord-orthogonal component was attenuated and the original hidden-state norm restored.

- Correct next-residue log-probability changes by `+0.00070` (CI `[-0.00048,+0.00190]`, `p=.458`).
- Correct-residue logit margin changes by `-0.02904` (CI `[-0.03393,-0.02414]`, `p<.0001`).

The mean probability effect is unresolved, but the reliably worse competitive margin shows that the orthogonal component is not safely expendable. This reproduces the important qualitative ChemFM warning.

### 5.5 Behavior coupling

At protein level, local tangent C(1) correlates modestly with higher correct-next-residue probability (`rho=+0.185`, CI `[+0.147,+0.222]`, permutation `p=.001`). This is the one result favorable to local straightness. Scale changes the conclusion: at spans 25--64, wider tubes correlate with *higher* correct probability (`rho=+0.339`, CI `[+0.305,+0.372]`), and greater path efficiency correlates with *lower* correct probability (`rho=-0.339`, CI `[-0.373,-0.304]`).

Across the 14 locked ProteinGym assays, trajectory curvature is not associated with DMS fitness (`macro rho=+0.049`, CI `[-0.012,+0.106]`), nor is tangent C(1) (`-0.049`, CI `[-0.106,+0.012]`). Wider tubes correlate with better fitness (`+0.143`, CI `[+0.059,+0.217]`, BH `q=.0138`), and path efficiency correlates negatively (`-0.140`, CI `[-0.206,-0.068]`, `q=.0138`).

In 100 Native samples generated from prompt `M` with temperature 1.0, top-p 0.9, canonical residues plus EOS, and length 64--128, a phenotype-blind quality composite was fixed from entropy, homopolymers, trimer diversity, and composition distance before surrogate inspection. The better half has larger tube radius by `+0.05397` (CI `[+0.01313,+0.09441]`, permutation `p=.0122`, family BH `q=.0305`) and nominally lower path efficiency by `-0.00587` (CI just excluding zero but permutation `p=.0704`). It is not more STP-organized.

### 5.6 Gradient safety and STP decision

Across six fixed 128-residue test prefixes, released STP has auxiliary/NTP norm ratio `2.986` (CI `[1.738,4.511]`). Global cosine is `+0.104` (CI `[-0.038,+0.234]`), masking marked batch and depth variation: mean early-layer cosine is `-0.067`, middle `-0.031`, and late `+0.319`. Unit weighting gives mean NTP-direction retention `1.43`, but individual values range from `0.56` to `2.66`; one early group reverses the NTP direction. A coefficient near `0.0100` would reduce mean backbone norm pressure to 3%.

**STP decision: no rank-32 LoRA experiment.** Native RITA has neither the proposed straight geometry nor a biologically specific forward geometry; useful orthogonal signal would be pressured; behavior coupling is mixed or reversed; and the released objective is large and unstable relative to NTP. This does not prove released STP cannot yield a small behavioral benefit. It says the protein modality has not supplied the missing mechanism or a sufficiently selective frozen surrogate to justify training.

## 6. Frozen Native NextLat characterization

### 6.1 Transition decomposition

Probes used 49,008 train, 11,304 validation, and 11,688 test transitions from 2,042/471/487 homology-separated proteins, capped at 24 stable positions per protein. The original analysis reduced both inputs and targets to 256 train-fitted PCA dimensions; target PCA retained 61.6% of train variance. Ridge was primary and a two-layer width-256 MLP was the sole nonlinear check.

**Original PCA-target result (preserved):**

| Input | Ridge R2 | nMSE | centered cosine | decoder JS | decoder top-1 | gold probability | gold rank | gold margin |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A: current `h_t` | .3527 | .6473 | .6021 | .05185 | .4961 | .07416 | 7.336 | -.7800 |
| B: realized residue only | .0274 | .9726 | .2042 | .07525 | .2931 | .06174 | 7.906 | -.6255 |
| C: `[h_t,e(x_(t+1))]` | .3697 | .6303 | .6314 | .05155 | .5035 | .07416 | 7.329 | -.7834 |
| C-shuffle | .3483 | .6518 | .5989 | .05204 | .4946 | .07397 | 7.344 | -.7810 |

C reduces protein-level raw state MSE relative to A by `-0.00255` (CI `[-0.00265,-0.00245]`, paired `p<.0001`) and relative to C-shuffle by `-0.00321` (CI `[-0.00330,-0.00313]`, `p<.0001`). C also reduces decoder probability-distribution disagreement relative to shuffle, but not detectably relative to A. The MLP reproduces the ordering: A/C/shuffle `R2=.3488/.3650/.3393` and decoder top-1 `.5027/.5151/.4802`.

The original realized-residue gain was genuine, but the target bottleneck made its absolute ceiling unclear. On untouched test states, projecting through and reconstructing from that PCA target has full-state `R2=.6097`, nMSE `.3903`, centered cosine `.7744`, decoder JS `.00144`, and decoder top-1 agreement `88.84%`. Thus 61.6% train variance retained is not itself the predictor's `R2`, but the representation discarded roughly 39% of standardized variance before prediction and imposed a measurable ceiling.

**Amended full-1024D target ridge result:** inputs retain the same capacity-controlled 256-D bottleneck, but each ridge predicts the unreduced 1024-D future state directly.

| Input | Full-state `R2` | nMSE | centered cosine | decoder JS | decoder top-1 |
|---|---:|---:|---:|---:|---:|
| A: current `h_t` | .3701 | .6299 | .6185 | .05177 | .4974 |
| B: realized residue only | .02968 | .9703 | .2130 | .07523 | .2931 |
| C: `[h_t,e(x_(t+1))]` | .3870 | .6130 | .6484 | .05145 | .5039 |
| C-shuffle | .3631 | .6369 | .6133 | .05197 | .4962 |

C gains `+.01689 R2` over A and `+.02385` over C-shuffle. Protein-paired C-minus-A MSE is `-.002527` (CI `[-.002642,-.002407]`, sign-flip `p<.0001`), decoder JS is `-.000318` (`[-.000418,-.000216]`, `p<.0001`), and top-1 agreement is `+.00647` (`[+.00069,+.01217]`, `p=.0270`). Against C-shuffle, the corresponding changes are MSE `-.003569` (`[-.003654,-.003486]`), JS `-.000516` (`[-.000574,-.000457]`), and top-1 `+.00776` (`[+.00244,+.01303]`; all `p<=.0056`).

Therefore the actual next amino acid still contributes meaningful out-of-sample information beyond `h_t`, and shuffle removes that advantage. Its magnitude remains modest: about 1.7 percentage points of full-state `R2` and 0.65 percentage points of decoder top-1 over A. B alone still has little explanatory power.

### 6.2 State sufficiency

**Original comparison (preserved, no longer interpreted conditionally):** final state plus layer 12 performed worse than final state alone under one shared 256-dimensional input PCA budget: ridge `R2=.3435` versus `.3527`, decoder top-1 `.4818` versus `.4961`, protein MSE difference `+.00138` (CI `[+.00111,+.00166]`, `p<.0001`), and decoder disagreement difference `+.0000617`. Because the concatenated 2048-D input had to compete for the same 256 components, this does not establish final-state sufficiency.

**First amendment (preserved, still PCA-conditioned):** a ridge first predicted full `h_(t+1)` from 256-D final-state features. Layer-12 features were separately reduced to 256 dimensions, linearly residualized against those PCA final-state features using train proteins only, and a second ridge predicted the base residual.

The base `R2=.37010` rises to `.41431`, an incremental `+.04421`; base nMSE `.62990` falls to `.58569`, and the correction explains `7.02%` of the base prediction's residual squared error. Centered cosine rises from `.61854` to `.65584`. Protein-paired MSE improves by `-.006624` (CI `[-.007139,-.006155]`, `p<.0001`) and decoder JS by `-.000259` (`[-.000409,-.000113]`, `p=.00055`). Decoder top-1 changes by `-.00171` (`[-.00754,+.00403]`, `p=.595`), so the unique information is robust in hidden-state error and small in distribution space, but does not improve discrete decoder decisions.

**Corrected full-final-state conditioning:** the base and residualizer now receive all 1,024 standardized coordinates of final `h_t`; no PCA is applied to the conditioning state. Both use ridge regression (`alpha=1`), fitted only on train proteins in float64 for numerical stability. Layer 12 alone is reduced to 256 train-PCA coordinates, ridge-residualized against the full state, and used to predict the full 1,024-D base residual. The test set remains untouched and homology-separated.

The stronger full-state base reaches `R2=.47956`, nMSE `.52044`, centered cosine `.69373`, decoder JS `.051314`, KL `.254913`, and top-1 `.48777`. Adding unique layer-12 information reaches `R2=.49124`, nMSE `.50876`, cosine `.70267`, JS `.051420`, KL `.255438`, and top-1 `.49085`. The incremental full-state `R2` is `+.01168`, or a 2.244% residual-error reduction. Protein-paired MSE changes by `-.001750` (CI `[-.001899,-.001608]`, `p<.0001`). However, JS changes adversely by `+.000105` (CI `[-.000003,+.000216]`, `p=.0534`), KL by `+.000523` (`[-.000218,+.001244]`, `p=.156`), and top-1 by `+.00304` (`[-.00197,+.00797]`, `p=.230`).

The prior `+.0442 R2` claim was inflated by conditioning on compressed final features. The corrected result supports only a small, reliable **latent** residual beyond full final `h_t`; it provides no evidence that this extra information improves decoder distributions or decisions. Accordingly, the report no longer claims broad decoder-functional state insufficiency. It retains the narrower statement that final `h_t` does not exhaust linearly recoverable future-state detail.

### 6.3 Biological stratification

The original structural stratification used the preserved SmoothL1-only predictor: validation SmoothL1 fell from `0.17174` to `0.03819` (77.8%) with RITA frozen. The amended full-objective predictor is analyzed separately below; the old structural findings are not silently replaced.

For the latest behavioral replay, the same faithful architecture was retrained from scratch using up to 64 evenly distributed transitions per protein: 130,682 train and 30,143 validation examples, versus 24,504/5,652 previously. RITA remained frozen and the objective remained unit SmoothL1 plus teacher-to-student KL. Validation latent/KL/total changed `.17558/.57619/.75177 -> .04705/.19853/.24558` by epoch 12. Against the earlier sparse full-objective endpoint `.05135/.19666/.24801`, dense fitting improves latent loss 8.37% and total 0.98%, while KL is 0.95% worse. Epoch-12 train total is `.06870`, so substantial generalization error remains despite 5.33-fold denser sampling; this limitation is reported rather than treated as solved.

On TAPE annotations, mean error is coil `.1117`, helix `.1180`, and beta `.1273`. Within-protein helix-minus-coil error is `+0.00514` (CI `[+0.00278,+0.00763]`, BH `q<.001`), beta-minus-coil is `+0.00742` (`[+0.00488,+0.00996]`, `q<.001`), and transition-minus-stable is `+0.00149` (`[+0.00028,+0.00273]`, `q=.0186`). High-minus-low long-range-contact error is `+0.00625` (`[+0.00412,+0.00851]`, `q<.001`).

Protein transitions are predictably harder in structured and contact-dense contexts, especially beta and long-range-contact positions. This is evidence that the modality does not eliminate the need for broader contextual computation.

### 6.4 Behavior coupling

**Original result (preserved):** the SmoothL1-only predictor's raw error had an adverse ordering on all three views: UniRef error versus correct probability `rho=+.8076`, ProteinGym error versus DMS `macro rho=+.2102`, and better-minus-worse generated-sequence error `+.02360`. That result could not distinguish predictability from trajectory scale.

The amendment uses the full-objective warmed predictor and seven nonredundant error-oriented quantities; lower always means more predictable. Centering uses the mean future state from all homology-separated UniRef train transitions. Raw MSE is mean squared coordinate error, normalized state MSE divides summed error by squared distance from the train mean, transition-relative MSE divides by the actual `||h_(t+1)-h_t||^2`, centered-cosine error is `1-cos`, decoder JS compares teacher and predicted next-state distributions, and faithful total is exactly per-transition SmoothL1 plus teacher-to-student KL.

On 487 homology-separated UniRef test proteins:

| Error metric | Spearman vs correct probability (95% CI) | Spearman vs NTP loss (95% CI) |
|---|---:|---:|
| raw SmoothL1 | +.833 `[+.791,+.869]` | -.782 `[-.826,-.730]` |
| raw MSE | +.832 `[+.789,+.868]` | -.780 `[-.825,-.728]` |
| normalized state MSE | -.420 `[-.496,-.338]` | +.439 `[+.359,+.513]` |
| transition-relative MSE | -.0926 `[-.182,-.0010]` | +.0938 `[+.0020,+.179]` |
| centered-cosine error | -.303 `[-.390,-.213]` | +.324 `[+.234,+.408]` |
| decoder JS | +.898 `[+.869,+.921]` | -.831 `[-.864,-.790]` |
| faithful total | +.889 `[+.856,+.914]` | -.821 `[-.858,-.777]` |

All permutation tests are `p=.00050` except transition-relative MSE (`p=.0370/.0325`); all remain below `.05` after separate seven-test BH correction. The scale-free latent metrics therefore favor good NTP behavior, reversing the raw result. In contrast, decoder divergence and the complete loss retain an even stronger adverse ordering. This is neither Outcome A nor pure Outcome B: raw metric scaling was a real confound, but faithful decoder-functional predictability remains misaligned.

Across the unchanged 14-assay ProteinGym panel, macro Spearman versus DMS is: raw SmoothL1 `+.204` (CI `[+.141,+.257]`), raw MSE `+.206` (`[+.144,+.260]`), normalized MSE `+.041` (`[-.044,+.123]`), transition-relative MSE `-.066` (`[-.164,+.046]`), centered-cosine error `+.104` (`[+.023,+.174]`), decoder JS `+.226` (`[+.160,+.285]`), and faithful total `+.235` (`[+.172,+.289]`). Normalized and transition-relative results are null (`p=.366/.249`); the other five remain significant after seven-test BH correction (`q<=.0304`). Because larger error means less predictability, decoder JS and faithful total again order fitter variants adversely. Within-assay coupling to official RITA fitness is also adverse for decoder JS (`macro rho=+.602`, CI `[+.490,+.702]`) and faithful total (`+.580`, `[+.473,+.676]`), while normalized MSE is null (`-.009`, `[-.163,+.123]`).

The fixed Native generation groups give the same faithful-objective answer. Better-minus-worse is raw SmoothL1 `+.0261` (CI `[+.00625,+.0463]`), raw MSE `+.0580` (`[+.0145,+.1018]`), normalized MSE `+.152` (`[+.0952,+.214]`), transition-relative MSE `-1.075` (`[-2.316,-.214]`), centered-cosine error `+.0812` (`[+.0511,+.1135]`), decoder JS `+.0450` (`[+.0153,+.0764]`), and faithful total `+.276` (`[+.0988,+.464]`). All label-permutation `p<=.0139` and seven-test BH `q<=.0139`. Transition-relative error alone says better generations are more predictable; every other measure, including both scale-free normalized/cosine measures and the decoder-functional/full losses, says they are less predictable.

Accordingly, the original universal statement that "more predictable transitions select worse behavior" was too broad and is retracted. The first amendment established a narrower adverse relationship for the implemented faithful objective, but still used the sparsely fitted predictor and absolute decoder error.

**Dense-predictor and decoder-transition-normalized replay:** all locked data and behavioral labels were reused. To avoid unstable division by individual near-zero transitions, relative decoder JS is computed per sequence as

`sum_t JS(p_(t+1),p_pred) / (sum_t JS(p_(t+1),p_t) + epsilon)`.

It is an error fraction: lower means that NextLat leaves less of the actual decoder transition unexplained. The dense-checkpoint results are:

| Error metric | UniRef vs correct probability | UniRef vs NTP loss | ProteinGym vs DMS | Generation better-minus-worse |
|---|---:|---:|---:|---:|
| raw SmoothL1 | +.837 | -.784 | +.205 | +.02493 |
| raw MSE | +.836 | -.782 | +.207 | +.05492 |
| normalized state MSE | -.291 | +.321 | +.110 | +.1361 |
| transition-relative MSE | -.090 | +.090 | +.039 | -5.082 |
| centered-cosine error | -.196 | +.228 | +.142 | +.07376 |
| absolute decoder JS | +.892 | -.823 | +.226 | +.04462 |
| decoder-transition-relative JS | +.528 | -.464 | +.170 | -.05647 |
| faithful total | +.885 | -.816 | +.237 | +.2706 |

For the new relative decoder metric, UniRef CIs are `[+.453,+.592]` versus correct probability and `[-.535,-.385]` versus NTP loss (both permutation `p=.00050`, saved BH `q=.00057/.00067`). ProteinGym macro `rho=+.170` has assay-bootstrap CI `[+.108,+.230]`, sign-flip `p=.00030`, BH `q=.00048`; relative error versus official RITA fitness is `+.322` (`[+.194,+.433]`, `q=.00120`). Because larger residual fraction accompanies better behavior, these are adverse even after controlling for decoder-transition magnitude. Generation is null and slightly favorable in direction (`-.0565`, CI `[-.325,+.105]`, permutation `p=.979`, BH `q=.979`).

Absolute decoder JS and faithful total remain adverse on all three views after dense fitting. On ProteinGym their macro correlations are `+.226` (`[+.156,+.287]`) and `+.237` (`[+.171,+.292]`), both BH `q=.00048`. Better generations retain larger absolute JS (`+.04462`, `[+.01479,+.07568]`, `q=.0128`) and faithful total (`+.2706`, `[+.0925,+.4573]`, `q=.0128`). The dense fit therefore does not explain the prior result as sparse-predictor underfitting. The relative metric also rules out decoder-transition magnitude on UniRef and ProteinGym, although its generation result is explicitly null. These remain associations and do not prove that a gradient step would causally harm a protein, but they provide no positive behavioral rationale for optimization.

### 6.5 Learned-auxiliary gradient safety and NextLat decision

**Original audit (preserved, methodologically mismatched):** a SmoothL1-only warmed predictor was evaluated with the full SmoothL1+KL loss. On six prefixes its all-backbone norm ratio was `.379x` (CI `[.219,.635]`), cosine `+.138` (`[-.034,+.298]`), and retention `1.010` (`[.927,1.072]`). The mismatch means this is not the definitive faithful audit.

**Corrected warmup:** a fresh predictor, saved separately, optimized the implemented unit `SmoothL1 + teacher-to-student KL`; target future states and the teacher head path stayed detached, while RITA parameter versions were checked before and after. On the unchanged validation split, latent loss fell `.17174 -> .05135`, KL `.55248 -> .19666`, and total `.72422 -> .24801` (65.8% total reduction; selected epoch 11 of 12). Only predictor parameters entered the optimizer.

Across 12 fixed 128-residue test prefixes, this fully warmed loss requests a full-backbone auxiliary/NTP norm ratio of `.3649` (cluster-bootstrap CI `[.2770,.4678]`). Global cosine is `+.1646` (`[+.0204,+.3039]`, sign-flip `p=.0546`) and NTP-direction retention is `1.0429` (`[.9891,1.0968]`). Ratios/cosines/retention by broad depth are early `.3468/-.0851/.9596`, middle `.3931/+.0192/1.0041`, and late `.4260/+.4752/1.1588`; early and middle cosine intervals cross zero, while late cosine CI `[+.290,+.642]` is aligned. Individual global ratios range `.164-.803`, and three of 12 global cosines are negative.

The corrected audit does not qualitatively rescue or worsen gradient safety: ChemFM's systematically opposing `2.8--4.6x` LoRA-subspace failure is absent, and protein full-backbone gradients are moderate and on average aligned. A value near `.0822` would produce 3% pressure **only in this measured full-backbone space**. It is not a rank-32 LoRA coefficient: projection into a particular adapter parameterization can materially change norms and angles.

The full-backbone audit is insufficient to characterize a future adapter parameterization. The explicitly requested gradient-only rank-32 audit below therefore supersedes it for adapter-space safety. No full-backbone number is used to recommend or calibrate LoRA training.

### 6.6 Rank-32 LoRA-subspace gradient audit

The live pinned architecture was inspected before adapter construction. RITA-M uses separate linear attention projections named `transformer.layers.{0..23}.self_attention.{query,key,value,proj}`; these are not Llama module names. Standard attention-only PEFT LoRA 0.13.2 was attached to exactly those 96 modules with rank 32, alpha 32 (scale 1), zero dropout, no trained bias, and verified standard no-op initialization (nonzero A, zero B). This creates 6,291,456 trainable values in 192 LoRA tensors. At this initialization A gradients are zero and B gradients define the initial trainable tangent. Automated assertions verified that every matched module is under `self_attention`, every non-LoRA RITA parameter and every dense-predictor parameter is frozen, no optimizer exists, and all parameter version counters are unchanged after the audit.

The sample is 32 deterministic length-stratified test proteins from 32 distinct homology clusters, each using the same 128-residue prefix. Protein-cluster bootstrap results are:

| LoRA parameter space | auxiliary/NTP norm ratio (95% CI) | cosine (95% CI) | NTP-direction retention (95% CI) | negative cosine |
|---|---:|---:|---:|---:|
| all attention adapters | .500 `[.439,.564]` | +.213 `[+.096,+.329]` | 1.080 `[1.021,1.138]` | 8/32 = 25.0% `[9.4%,40.6%]` |
| early layers 0--7 | .363 `[.316,.410]` | -.045 `[-.149,+.062]` | .985 `[.946,1.022]` | 18/32 = 56.3% `[40.6%,71.9%]` |
| middle layers 8--15 | .400 `[.357,.443]` | -.072 `[-.142,+.002]` | .960 `[.930,.990]` | 19/32 = 59.4% `[43.8%,75.0%]` |
| late layers 16--23 | .661 `[.565,.761]` | +.385 `[+.246,+.517]` | 1.184 `[1.088,1.270]` | 6/32 = 18.8% `[6.3%,34.4%]` |

Relative to the existing RITA full-backbone audit, projection into the actual adapters raises the mean ratio from about `.365` to `.500` and cosine from `+.165` to `+.213`. It exposes mild localized opposition in early/middle layers, especially middle retention below one, but the global update remains aligned and moderate. This is qualitatively unlike ChemFM's trained-LoRA ratios of roughly `2.8--4.6x` with substantial middle/late conflict. **ChemFM-style gradient conflict does not reappear in RITA's initialized rank-32 attention-LoRA subspace.**

This answers gradient compatibility only. It does not overturn the behavioral stop: dense faithful and decoder-transition-relative quantities still provide no evidence that reducing the objective improves Native NTP, ProteinGym fitness, or generation quality. Therefore no Native or NextLat adapter training is recommended or performed.

### 6.7 Direct answers after amendment

1. **Does the actual amino acid add information?** Yes: C exceeds both A and C-shuffle on full-state and decoder metrics.
2. **How large is the gain?** C-minus-A is `+.0169 R2`, `-.000318` decoder JS, and `+.00647` decoder top-1; C-minus-shuffle is `+.02385 R2` and `+.00776` top-1.
3. **Does scale-free predictability couple to Native NTP?** Purely latent normalized/relative/cosine metrics are favorable, confirming raw scale confounding. Dense-predictor decoder-transition-relative JS is adverse (`rho=+.528` versus correct probability and `-.464` versus NTP loss).
4. **Does ProteinGym agree?** Yes for decoder-normalized behavior: relative JS is significantly adverse versus DMS (`macro rho=+.170`, CI `[+.108,+.230]`).
5. **Does generation agree?** Absolute decoder JS and full loss remain adverse, but relative JS is null (`p=.979`). The report does not claim three-way replication for the normalized decoder metric.
6. **Does an earlier layer add residual information beyond full final `h_t`?** A small amount: `+.01168 R2` and 2.24% residual reduction. Decoder JS/KL do not improve, and top-1 change is unresolved.
7. **Did dense fitting change the result?** No materially. It uses 130,682 transitions and improves validation latent loss 8.37% over the sparse full-objective checkpoint, but faithful behavior correlations remain adverse.
8. **What is the gradient evidence?** The actual attention-LoRA ratio is `.500`, cosine `+.213`, and retention `1.080`; global geometry is compatible, with mild early/middle conflict.
9. **Is the ChemFM mechanism reproduced?** Partially: faithful-objective behavioral decoupling survives state-, predictor-, and decoder-transition-scale controls; ChemFM-style LoRA gradient domination is absent.
10. **Is rank-32 LoRA training justified?** No. The gradient audit removes incompatibility as a blocker but cannot supply the missing positive behavioral rationale.

## 7. Cross-modality conclusion

### Evidence for chemistry/SMILES specificity

- RITA's faithful NextLat gradient is `.365x` NTP in the full backbone and `.500x` in the initialized rank-32 attention-LoRA subspace, both with positive global mean cosine. ChemFM reported `2.8--4.6x` in its trained LoRA subspace. Severe gradient domination is therefore not modality-general.
- The realized next amino acid adds a small reproducible amount of future-state information after conditioning on `h_t`; it is not a useless input.
- Several state-scale-free latent metrics reverse the originally adverse raw-error ordering on UniRef. Raw hidden-state magnitude, rather than latent predictability alone, explains an important part of the first protein result.

### Evidence for a general autoregressive-Transformer problem

- Both ChemFM and RITA have negative adjacent tangent correlation and decoder-functional chord-orthogonal components, undermining literal straight/geodesic assumptions across token modalities.
- In both modalities, a transition objective can become substantially more predictable without establishing behavioral usefulness.
- RITA reproduces the narrower surrogate-decoupling concern in the actual faithful loss: lower decoder JS and lower SmoothL1+KL consistently order Native NTP, DMS fitness, and free-running quality in the undesired direction, even after substantially denser predictor fitting.
- Decoder-transition-relative JS remains adverse on UniRef and ProteinGym, so absolute transition magnitude does not explain those two functional results.
- Full-state conditioning leaves a small amount of linearly recoverable latent information in layer 12, but no decoder-functional gain. This is weak evidence for distributed future-state detail, not proof of a functionally inadequate final recurrent state.
- Additional hidden-state smoothness is not inherently functional; relevant decoder information occupies components that simple geometric regularizers would suppress.

### Ambiguous results

- RITA local tangent persistence has a modest favorable association with native token probability, and secondary-structure transitions are slightly more curved. These show sensitivity to protein organization but do not establish causality or a trainable STP mechanism.
- The full-backbone audit uses 12 prefixes and the LoRA-subspace audit uses 32 distinct test clusters. The adapter result is an initialization-time tangent-space diagnostic, not evidence about gradients after LoRA parameters have trained away from their standard no-op initialization.
- Scale-free results are not uniform: normalized/cosine latent predictability is favorable on UniRef, while decoder-transition-relative JS is adverse on UniRef and ProteinGym but null in generated sequences. The adverse conclusion applies to the decoder-functional/full faithful objective, not to an intrinsic universal value of latent predictability.
- Dense fitting reduced validation latent error but left a large train/validation gap. An even denser or differently regularized predictor might generalize better, although the requested 100k-plus control did not alter the ordering.
- TAPE and ProteinGym targets cannot be certified absent from RITA's original UniRef100 pretraining. This limits absolute generalization claims, though it does not invalidate within-checkpoint surrogate comparisons.
- The generation quality composite is deliberately simple and independent, not a substitute for a broad protein-design benchmark or structure prediction.

## 8. Recommendation

Do not launch Native/STP/NextLat rank-32 LoRA training from this result. STP remains closed. Faithful NextLat learns the intended transition mapping and its actual initialized attention-LoRA gradients are globally moderate and compatible, but its decoder-functional/full objective remains behaviorally reversed after dense fitting, and decoder-transition normalization preserves that reversal on UniRef and ProteinGym. Full-state conditioning weakens the distributed-state objection to a small latent-only effect, so neither state sufficiency nor gradient conflict is now the decisive blocker. The ChemFM mechanism is therefore **partially reproduced**: behavioral surrogate decoupling survives, while severe LoRA gradient domination does not. The chemistry-specific contribution is raw latent-scale confounding; the modality-general concern is optimizing a tractable transition surrogate without evidence that it selects useful behavior.

**Final question:** after controlling for full final-state conditioning, a 130k-transition predictor fit, actual decoder-transition magnitude, and the actual initialized rank-32 attention-LoRA trainable subspace, there is still strong associational evidence that faithful NextLat is behaviorally misaligned enough to stop before training. The LoRA audit is reassuring on optimization safety: it does not reproduce ChemFM-style domination. That removes a negative argument but cannot supply the missing positive relationship between the objective and useful protein-model behavior.

Reconsider training only after a new frozen diagnostic identifies a transition quantity with preregistered positive coupling to Native NTP, ProteinGym fitness, and free-running quality. The rank-32 attention target set and initialization-time gradient geometry are now established; any future pilot should still remain one A6000, matched Native and auxiliary arms, a fixed modest residue budget, predictor learning unscaled, and adapter pressure monitored during training. That is a gate for future work, not a recommendation to run it now.

## 9. Reproducibility and artifact map

Random seed is `20260914`. Local extraction used PyTorch 2.3.0+cu121, Transformers 4.45.2, NumPy 1.26.4, SciPy 1.14.1, scikit-learn 1.5.2, and pandas 2.2.3 on an NVIDIA RTX 4050 Laptop GPU. Frozen-state caching required 84.5 s for UniRef, 23.5 s for TAPE, and 6.3 s for reversals, with measured peak allocation about 747 MB. A requested Thunder A6000 was unavailable; an L40/6-vCPU/100-GB instance was used only to run MMseqs2, then deleted. No diagnostic inference result came from a substituted model.

The amendment reused the same local environment, checkpoint, cached states, manifests, ProteinGym variants and fitness scores, and generated sequences/quality labels. It did not regenerate or relabel data. The new predictor checkpoint is separate from the old one. The LoRA decision is machine-readable and records that adapters were instantiated for gradients only, with neither optimization nor parameter updates.

The latest coupling artifact computes and stores Benjamini-Hochberg q-values directly for the UniRef outcome families, ProteinGym DMS/RITA-fitness families, and generation family. The report no longer relies on prose-only multiplicity calculations.

Primary artifacts:

- `protein/runs/audit/audit.json` and `fitness_parity.json`: model/tokenizer/fitness audit.
- `protein/data/diagnostic_pool.jsonl`: locked homology-separated pool.
- `protein/runs/uniref/geometry.json`, `behavior_coupling.json`, and `stp_gradient.json`: STP evidence.
- `protein/runs/tape/stratified_geometry.json` and `orthogonal.json`: biological and functional STP tests.
- `protein/runs/uniref/nextlat_probes.json`, `faithful_nextlat_predictor.json`, and `nextlat_gradient.json`: transition and safety evidence.
- `protein/runs/tape/nextlat_stratified.json`: structural NextLat evidence.
- `protein/runs/proteingym/scores.json`: locked functional panel.
- `protein/runs/generation/native_replay.json`: frozen generation replay.
- `protein/runs/amendment/faithful_full_predictor.pt` and `.json`: separately saved complete-objective warmup and audit trail.
- `protein/runs/amendment/full_target_and_residual.json`: full-1024D A/B/C/shuffle probes, PCA ceiling, and conditional layer-12 test.
- `protein/runs/amendment/scale_free_coupling.json`: seven-metric UniRef, locked ProteinGym, and unchanged-generation replay.
- `protein/runs/amendment/nextlat_full_gradient.json`: corrected 12-protein full-backbone gradient audit with broad-depth intervals.
- `protein/runs/amendment/full_conditioning_residual.json`: layer-12 residual test conditioned on all 1,024 final-state coordinates.
- `protein/runs/amendment/faithful_dense_predictor.pt` and `.json`: 130,682-transition faithful predictor and frozen-backbone audit trail.
- `protein/runs/amendment/scale_free_coupling_dense.json`: dense-predictor coupling, stable decoder-transition-relative JS, and saved BH q-values.
- `protein/runs/amendment/nextlat_lora_r32_gradient.json`: 32-cluster rank-32 attention-LoRA gradient geometry and no-update audit.
- `protein/runs/amendment/decision.json`: prespecified LoRA-gate outcome and explicit non-execution record.
- `protein/runs/summary.json`: compact estimates and output hashes.
- `protein/runs/reproducibility.json`: package, hardware, command, revision, manifest, and archive provenance.

Focused validation is under `protein/tests/`. Cached tensors and downloaded bulk archives are intentionally ignored; compact manifests, exact IDs, hashes, outputs, and regeneration scripts remain under `protein/`.
