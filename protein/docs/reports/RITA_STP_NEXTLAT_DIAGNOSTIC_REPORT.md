# RITA-M STP and NextLat diagnostic report

Date: 2026-09-14  
Checkpoint: `lightonai/RITA_m` at `d819157a3a96d278500232b2cbb2a02d9646bcf6`  
Status: frozen diagnostic complete; no RITA parameters were optimized

## Executive decision

Neither released STP nor faithful NextLat merits a rank-32 LoRA experiment on the present evidence.

RITA-M does not exhibit the straight, locally persistent native trajectory assumed by a literal tube/geodesic account of STP: mean adjacent tangent correlation is `-0.4045` (protein bootstrap 95% CI `[-0.4055,-0.4034]`). Structural differences exist but are small and internally inconsistent, the primary geometry survives sequence reversal, and removing only 10% of the chord-orthogonal component harms the correct-residue margin. Some local straightness correlates with native prediction quality, but longer-scale tube radius and path efficiency relate to ProteinGym fitness and free-running quality in the opposite direction from the STP premise. Released STP also requests a large and variable gradient (`2.99x` NTP, 95% CI `[1.74,4.51]`). These results do not overturn ChemFM's small, inconclusive positive behavioral signal; they say that RITA supplies no stronger mechanistic reason to repeat it.

NextLat's proximal premise is present only weakly. A ridge probe predicts the next state from the current state with `R2=0.3527`; adding the realized next residue raises this to `0.3697`, while a matched shuffled residue falls to `0.3483`. The gain is real but small and improves decoder top-1 agreement by only `0.75` percentage points. A capacity-controlled additional-layer input makes prediction worse, not better. More importantly, transition error has the wrong behavioral ordering: larger error associates with better native next-token behavior, higher ProteinGym fitness (`macro rho=+0.210`, CI `[+0.146,+0.265]`), and better preregistered Native generations. Predictor warmup greatly improves validation error, and its gradient is much safer than in ChemFM (`0.379x` NTP rather than `2.8--4.6x` in the earlier LoRA audit), but faithful unit weighting still exceeds modest auxiliary pressure and the optimized surrogate remains behaviorally misaligned.

A pressure-controlled NextLat coefficient of approximately `0.079` would yield 3% backbone pressure in this audit while leaving predictor optimization unscaled. That is a clearly distinct possible method, not faithful NextLat. It is not recommended for training yet because pressure control addresses interference, not the observed reversed relationship between predictability and useful behavior.

## 1. Scope and preregistered exclusions

This study characterized one frozen 300M-parameter autoregressive protein LM. It performed no STP fine-tuning, NextLat backbone fine-tuning, matched Native LoRA training, objective sweep, rank sweep, full continued pretraining, RITA-L confirmation, broad geometry suite, or large generation/structure campaign. Predictor-only fitting was used for faithful NextLat exactly so that its learned auxiliary gradient could be measured without changing RITA.

The independent resampling unit is a protein or MMseqs cluster, not a residue. Reported intervals use whole-cluster/protein bootstrap resampling. Paired biological contrasts use proteins present in both strata and sign-flip tests. ProteinGym uncertainty is across assays. Families of related structural and generation comparisons were interpreted with Benjamini-Hochberg correction.

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

Probes used 49,008 train, 11,304 validation, and 11,688 test transitions from 2,042/471/487 homology-separated proteins, capped at 24 stable positions per protein. Inputs and targets were each capacity-controlled to 256 PCA dimensions fitted on train only; target PCA retained 61.6% of train variance. Ridge was primary and a two-layer width-256 MLP was the sole nonlinear check.

| Input | Ridge R2 | nMSE | centered cosine | decoder JS | decoder top-1 | gold probability | gold rank | gold margin |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A: current `h_t` | .3527 | .6473 | .6021 | .05185 | .4961 | .07416 | 7.336 | -.7800 |
| B: realized residue only | .0274 | .9726 | .2042 | .07525 | .2931 | .06174 | 7.906 | -.6255 |
| C: `[h_t,e(x_(t+1))]` | .3697 | .6303 | .6314 | .05155 | .5035 | .07416 | 7.329 | -.7834 |
| C-shuffle | .3483 | .6518 | .5989 | .05204 | .4946 | .07397 | 7.344 | -.7810 |

C reduces protein-level raw state MSE relative to A by `-0.00255` (CI `[-0.00265,-0.00245]`, paired `p<.0001`) and relative to C-shuffle by `-0.00321` (CI `[-0.00330,-0.00313]`, `p<.0001`). C also reduces decoder probability-distribution disagreement relative to shuffle, but not detectably relative to A. The MLP reproduces the ordering: A/C/shuffle `R2=.3488/.3650/.3393` and decoder top-1 `.5027/.5151/.4802`.

The realized residue therefore adds genuine information, not just dimensions. However, the incremental `R2` is only `+0.017`, and decoder-functional improvements over the current state are tiny or absent. B alone has little explanatory power.

### 6.2 State sufficiency

Under the same 256-dimensional input bottleneck and probe capacity, final state plus layer 12 performs worse than final state alone: ridge `R2=.3435` versus `.3527`, decoder top-1 `.4818` versus `.4961`, protein MSE difference `+0.00138` (CI `[+0.00111,+0.00166]`, `p<.0001`), and decoder disagreement difference `+0.0000617` (CI `[+0.0000479,+0.0000773]`). The nonlinear probe agrees (`R2=.3428` versus `.3488`).

This test does not prove that computation is never distributed across layers; it only rejects the specified practical hypothesis that one additional informative layer supplies substantial capacity-controlled information missing from final `h_t`.

### 6.3 Biological stratification

The faithful predictor was trained on cached states only, with RITA frozen. Validation SmoothL1 fell from `0.17174` to `0.03819` (77.8%) before early stopping.

On TAPE annotations, mean error is coil `.1117`, helix `.1180`, and beta `.1273`. Within-protein helix-minus-coil error is `+0.00514` (CI `[+0.00278,+0.00763]`, BH `q<.001`), beta-minus-coil is `+0.00742` (`[+0.00488,+0.00996]`, `q<.001`), and transition-minus-stable is `+0.00149` (`[+0.00028,+0.00273]`, `q=.0186`). High-minus-low long-range-contact error is `+0.00625` (`[+0.00412,+0.00851]`, `q<.001`).

Protein transitions are predictably harder in structured and contact-dense contexts, especially beta and long-range-contact positions. This is evidence that the modality does not eliminate the need for broader contextual computation.

### 6.4 Behavior coupling

On the homology-separated UniRef test proteins, higher transition error correlates with *higher* correct-next-residue probability (`rho=+0.8076`, CI `[+0.7680,+0.8402]`) and *lower* NTP loss (`rho=-0.7066`, CI `[-0.7534,-0.6518]`; both permutation `p<.001`). Equivalently, the most NextLat-predictable proteins are those on which Native RITA predicts residues worse. This strong relation may partly reflect hidden-state scale or sequence-family covariates and is correlational, but its direction is adverse to the proposed surrogate.

On ProteinGym, official forward-plus-reverse RITA fitness has macro Spearman `+0.3158` (assay-bootstrap CI `[+0.2201,+0.4021]`), validating that the panel contains meaningful RITA signal. NextLat error itself has macro correlation `+0.2102` with DMS fitness (`[+0.1457,+0.2655]`, sign-flip `p=.0003`): fitter variants have less predictable, not more predictable, transitions. Error correlates strongly with RITA fitness within assays (`macro rho=+0.5478`, `[+0.4423,+0.6377]`).

The Native generation replay agrees. The better-quality half has NextLat error higher by `+0.02360` (CI `[+0.00632,+0.04183]`, permutation `p=.0122`, family BH `q=.0305`). This is not a null result: all three behavior views order the surrogate in the undesired direction.

### 6.5 Learned-auxiliary gradient safety and NextLat decision

After predictor-only warmup, faithful SmoothL1 plus teacher-to-student KL requests an all-backbone gradient norm `0.379x` NTP (CI `[0.219,0.635]`). Mean cosine is `+0.138` (CI `[-0.034,+0.298]`); mean NTP-direction retention is `1.010` (CI `[0.927,1.072]`). Early and middle depth groups remain mildly opposing on average (cosines `-0.075` and `-0.085`), while late layers are aligned (`+0.385`). One of six global batches is opposing with ratio `1.00`, cosine `-0.186`, and retention `0.813`.

This is important cross-modality evidence: the exact ChemFM failure signature of universally dominant `2.8--4.6x` auxiliary pressure did **not** reproduce. Protein NextLat is safer in magnitude after learning. Faithful unit weighting nevertheless applies about 38% rather than low-single-digit pressure, and safety does not repair adverse surrogate/behavior coupling. Scaling the backbone contribution by `0.0791` would target 3% pressure while predictor learning remains unscaled.

**NextLat decision: no faithful rank-32 LoRA experiment.** The realized amino acid adds only modest decoder-functional information, the additional-layer control provides no rescue, difficult biological contexts are less predictable, and increased predictability consistently selects worse behavior. A pressure-controlled modification is the only technically defensible form if independent future evidence motivates training, but it is a new labeled method and is not warranted by this diagnostic alone.

## 7. Cross-modality conclusion

### Evidence for chemistry/SMILES specificity

- RITA's warmed faithful NextLat gradient is `0.379x` NTP, far below ChemFM's `2.8--4.6x` LoRA ratios, and its global mean cosine is not negative. The severity of gradient domination is therefore architecture/modality/training-regime dependent rather than universal.
- The realized next amino acid adds a small reproducible amount of future-state information after conditioning on `h_t`; it is not a useless input.

### Evidence for a general autoregressive-Transformer problem

- Both ChemFM and RITA have negative adjacent tangent correlation and decoder-functional chord-orthogonal components, undermining literal straight/geodesic assumptions across token modalities.
- In both modalities, a transition objective can become substantially more predictable without establishing behavioral usefulness.
- RITA strengthens the surrogate-decoupling concern: lower transition error is consistently associated with worse native NTP behavior, worse DMS fitness ordering, and worse free-running quality.
- Additional hidden-state smoothness is not inherently functional; relevant decoder information occupies components that simple geometric regularizers would suppress.

### Ambiguous results

- RITA local tangent persistence has a modest favorable association with native token probability, and secondary-structure transitions are slightly more curved. These show sensitivity to protein organization but do not establish causality or a trainable STP mechanism.
- The NextLat gradient audit uses six representative prefixes; its interval is informative but not a high-powered map of every batch or LoRA subspace.
- TAPE and ProteinGym targets cannot be certified absent from RITA's original UniRef100 pretraining. This limits absolute generalization claims, though it does not invalidate within-checkpoint surrogate comparisons.
- The generation quality composite is deliberately simple and independent, not a substitute for a broad protein-design benchmark or structure prediction.

## 8. Recommendation

Do not launch Native/STP/NextLat rank-32 LoRA training from this result. STP lacks its proposed native geometry and poses functional/gradient risk. Faithful NextLat learns the intended transition mapping and is safer than in ChemFM, but the mapping's behavioral ordering is reversed; optimizing it would repeat the central conceptual error even if gradient conflict were controlled.

Reconsider training only after a new frozen diagnostic identifies a transition or geometry quantity with preregistered positive coupling to Native NTP, ProteinGym fitness, and free-running quality. If that occurs, the next experiment should remain one A6000, LoRA rank 32, matched Native and auxiliary arms, a fixed modest residue budget, predictor learning unscaled, and auxiliary backbone pressure measured and capped before training. That is a gate for future work, not a recommendation to run it now.

## 9. Reproducibility and artifact map

Random seed is `20260914`. Local extraction used PyTorch 2.3.0+cu121, Transformers 4.45.2, NumPy 1.26.4, SciPy 1.14.1, scikit-learn 1.5.2, and pandas 2.2.3 on an NVIDIA RTX 4050 Laptop GPU. Frozen-state caching required 84.5 s for UniRef, 23.5 s for TAPE, and 6.3 s for reversals, with measured peak allocation about 747 MB. A requested Thunder A6000 was unavailable; an L40/6-vCPU/100-GB instance was used only to run MMseqs2, then deleted. No diagnostic inference result came from a substituted model.

Primary artifacts:

- `protein/runs/audit/audit.json` and `fitness_parity.json`: model/tokenizer/fitness audit.
- `protein/data/diagnostic_pool.jsonl`: locked homology-separated pool.
- `protein/runs/uniref/geometry.json`, `behavior_coupling.json`, and `stp_gradient.json`: STP evidence.
- `protein/runs/tape/stratified_geometry.json` and `orthogonal.json`: biological and functional STP tests.
- `protein/runs/uniref/nextlat_probes.json`, `faithful_nextlat_predictor.json`, and `nextlat_gradient.json`: transition and safety evidence.
- `protein/runs/tape/nextlat_stratified.json`: structural NextLat evidence.
- `protein/runs/proteingym/scores.json`: locked functional panel.
- `protein/runs/generation/native_replay.json`: frozen generation replay.
- `protein/runs/summary.json`: compact estimates and output hashes.
- `protein/runs/reproducibility.json`: package, hardware, command, revision, manifest, and archive provenance.

Focused validation is under `protein/tests/`. Cached tensors and downloaded bulk archives are intentionally ignored; compact manifests, exact IDs, hashes, outputs, and regeneration scripts remain under `protein/`.
