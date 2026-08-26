# cLM-JEPA continuation context

Work only inside `C:\Users\arnav.DHEERAJACER\CLM-JEPA`. Read `AGENTS.md` and `docs/CLM_JEPA_Plan.md` before changing experiments. The plan is authoritative; this file records current state and paths.

## Operating constraints

1. Preserve official LLM-JEPA tensor semantics and ChemFM tokenization, serialization, LoRA, generation, and scoring unless a report explicitly documents a controlled deviation.
2. Exact generated top-1 is the primary forward-reaction metric. CE and representation metrics are mechanism diagnostics.
3. Do not call the observed geometry “collapse”: variance contraction and common-direction concentration are extreme, but effective rank and residual retrieval remain nontrivial.
4. Do not tune or select checkpoints from diagnostic panels.
5. Credentials belong only in environment variables. Do not commit tokens, keys, or passwords.
6. Use the canonical trainer in `src/train.py` on both RTX 4050 and A6000. Hardware-specific files are execution wrappers or diagnostics, not alternative scientific trainers.
7. Preserve reusable code and compact decision evidence; remove caches, transient logs, and completed one-off profiling/debug runners.

## Provenance and hardware

- GitHub: `https://github.com/ArnavSharma938/CLM-JEPA.git`
- LLM-JEPA reference commit: `ea0017c654ad917066ff32afc88276bea8ca5f7e`
- ChemFM reference commit: `ee35b23d03de1a8e97b8e04dcdfb1d579de70f02`
- ChemFM-1B revision: `f99dc2e89726539bb9cf31b2e2b4360650bac6a8`
- Local GPU: RTX 4050 Laptop, 6,141 MiB
- Cloud configuration used: one A6000, six vCPUs, 200 GB storage, Thunder base/prototyping image

## Code layout

See `docs/CODE_LAYOUT.md` for the full map. Key entrypoints:

- `src/train.py`: canonical ChemFM training, validation, checkpoint/resume, and W&B.
- `src/chemfm.py`: tokenizer, collation, LoRA loading, and generation.
- `src/jepa.py`: original endpoint cLM-JEPA readouts, predictor-token objective, MSE, and exact SIGReg.
- `src/vjepa2_1.py`: dense causal suffix/context prediction, four-depth fusion,
  train-only predictor, and functional EMA ChemFM target.
- `src/representation_eval.py`: standard frozen representation evaluation.
- `src/eval_uspto_mit_five_view_a6000.py`: official five-view beam-10 endpoint evaluation.
- `scripts/`: report-specific diagnostics, setup utilities, A6000 execution wrappers, and the GSM8K upstream reference.
- `references/`: pinned upstream source only.

The trainer exposes three families: native ChemFM; original endpoint cLM-JEPA
conditions implemented by `src/jepa.py`; and the separate dense causal
V-JEPA-2.1-style condition implemented by `src/vjepa2_1.py`. They share NTP,
ChemFM loading, data order, optimizer/scheduler, checkpoint dispatch, and
generation. Native and endpoint conditions retain their historical extended
vocabulary for checkpoint/control parity. Dense uses the base vocabulary
because its predictor is latent-only. PCSF and projection conditions are
historical and are not active trainer branches.

## Gate and experiment status

### Gates 0–3

Passed. Gate 3 evaluated 7,168 frozen examples across seven datasets and retained k=0 and k=1. The protocol, corrections, and aggregate evidence are retained in `docs/reports/00_METHOD_AND_PROTOCOL_FIDELITY.md`. Gate-3 sample CSVs and the obsolete gate runner were removed; a rerun would require regenerating the samples from the ignored full datasets.

### Gate 4

Passed for the reduced USPTO-MIT pilot with fixed reliable settings: LR `1e-4`, seed 533 primary, seed 917 directional replication, four epochs, native NTP retained, effective auxiliary weight one, and 50% auxiliary-loss dropout. This was not broad HPO.

### Original Gate 5 and geometry

The selected epoch-3 native and symmetric cosine cLM-JEPA checkpoints tied at 2/32 exact top-1, so the strict gate failed. On 512 frozen identities native/cLM-JEPA top-1 was 24/17; exact McNemar p=0.143. Symmetric cLM-JEPA target variance was approximately 495× below native and mean-direction energy was 0.999617. Centering/top-PC removal recovered margin 0.285808 and retrieval 0.859375. Pair strength did not predict CE or generated-rank improvement.

### Target stop-gradient

On 512 identities native/stop-gradient top-1 was 24/26; difference +0.39 pp, 95% CI [-1.37,+2.15], McNemar p=0.8318. Variance increased 12.03× versus symmetric cLM-JEPA but remained 46.3× below native. Target CE was 7.69% worse than native.

### SIGReg studies

- Batch-2 k ablation: native/k0/k1 top-1 was 7/2/3 of 256; neither readout restored native-scale geometry.
- Exact batch 128: geometry moved near native scale, but updates fell 16× and generation degraded to 0/256 top-1 with CE 1.080978; this comparison is cadence-confounded.
- RTX 4050 batch-16 preflight: streamed/direct values and parameter gradients matched exactly; exact N=16 halves update cadence.
- Frozen gradient assay: SIGReg endpoint norm remained 0.0376–0.0427 across the observed contraction trajectory.
- Cadence-matched A6000 batch-16 run: epoch-4 native/SIGReg top-1 was 6/3 of 256; source/target variance remained 16.9×/14.8× below native.

### MSE+SIGReg and official endpoint

MSE alone did not restore native-scale geometry. MSE+SIGReg restored epoch-4 source/target variance to 90.3%/55.2% of native, but exact top-1 tied 6/256 and target CE was 3.36% worse.

The official five-view endpoint then compared this selected epoch-4 MSE+SIGReg checkpoint with cadence-matched native on a frozen sequential sample. At 1,280 unique reactions:

- native top-1: 3.906% (50/1,280);
- cLM-JEPA top-1: 3.125% (40/1,280);
- paired difference: -0.781 pp;
- 95% bootstrap CI: [-1.719,+0.156] pp;
- native-only/cLM-only: 24/14;
- exact McNemar p=0.1433.

The prespecified 99% futility upper bound was +0.458 pp, below the +1 pp effect of interest, so evaluation stopped without extending to 3,300. Top-5 also favored native after Holm correction: 26.797% versus 23.516%, adjusted p=0.00108.

### GSM8K LLM-JEPA reference

The reduced two-epoch DeepSeek-1.5B run was not a successful behavioral control: NTP/LLM-JEPA accuracy was 36/28 of 300, difference -2.67 pp, 95% CI [-6.33,+1.00], p=0.229. LLM-JEPA target variance was only 1.45× below NTP, versus ChemFM's approximately 495× contraction. This does not support LoRA alone as a sufficient explanation for the ChemFM geometry.

### Frozen mechanistic audit

The selected MSE+SIGReg endpoint's active auxiliary gradient was `0.212x` the LoRA NTP-gradient norm, cosine `-0.042`, and 99.82% orthogonal. The conflict was localized and SIGReg-dominated: layers 17-21 had active-auxiliary/early-NTP cosine `-0.107`. Swapping those cLM layers into native worsened CE by `+0.015607`; restoring native layers 17-21 in cLM removed 55.1% of the full CE gap. cLM layers 12-16 instead improved the native background by `-0.003447`.

True and shuffled active auxiliary gradients had cosine `0.902`; the pair-specific residual was `0.096x` the NTP norm and nearly orthogonal to NTP. Source/target MSE decomposition measurements are reported in `docs/reports/02_ENDPOINT_MECHANISM_AUDITS.md`.

### Frozen SIGReg pair-specificity audit

Four disjoint batches of 16 reactions were evaluated at MSE+SIGReg epochs 1, 2, and 4 with four fresh SIGReg slice draws per batch. SIGReg increased correct-versus-deranged discrimination, cosine margin, center separation, and variance in all 48 cLM measurements. Applied SIGReg grew from `1.51x` to `4.06x` the applied MSE norm from epochs 1 to 4, while the pair-dependent fraction of the full auxiliary gradient fell from `0.468` to `0.398`. SIGReg/NTP cosine remained near `-0.05`. Full details are in Part II of `docs/reports/02_ENDPOINT_MECHANISM_AUDITS.md`.

### Pair-Center Spread Floor experiment

PCSF tested a reference-relative, one-sided standard-deviation floor on positive-pair centers with no covariance, rank, Gaussian, or whitening constraint. Frozen calibration fixed `rho=0.80` and `beta=4.2`; no coefficient sweep was run. The four-epoch condition produced pair-center sigma `0.535x` the matched-native reference. On the fixed 256 panel it scored 4/256 top-1 versus native 6/256 and had 2.49% higher target CE. Full details are in Part I of `docs/reports/03_ENDPOINT_INTERVENTION_EXPERIMENTS.md`.

### Projection-space MSE+SIGReg (historical condition)

Both MSE and exact SIGReg were moved from raw ChemFM EOS states into one shared `2048->2048->2048->64` hidden-BN/ReLU projection head. BatchNorm saw the full 32-row source-plus-target logical JEPA batch. During this experiment PCSF and direct raw MSE+SIGReg were removed from its production path. The projector condition was later removed, and endpoint MSE+SIGReg was restored as a maintained endpoint condition.

The four-epoch seed-533 run completed 320 updates in 32.33 minutes. Projected z effective rank was about three. The matched 256-reaction target-token CE was `0.256497`, versus direct MSE+SIGReg `0.248779` and native `0.240683`. On the first 512 reactions of the frozen official manifest, native/direct/projected exact top-1 was `18/15/4`; projected versus direct was `-2.148` pp, 95% CI `[-3.516,-0.781]`, McNemar `p=0.00342`. The 512 panel was budget-bounded during execution and is descriptive, not the original confirmatory 1,280 endpoint. Full details are in Part II of `docs/reports/03_ENDPOINT_INTERVENTION_EXPERIMENTS.md`.

### Dense causal V-JEPA 2.1

The primary-source-locked translation uses causal product-suffix masks, dense
`1/sqrt(distance)` context L1, depths 6/11/17/22, a 24-block width-384
noncausal latent predictor, and a fixed-0.99925 EMA target. The exact
JEPA-disabled path is ordinary ChemFM NTP and generation remains unchanged.
The test suite passed 102 tests with one intentional skip.

The seed-533 A6000 pilot completed 320 updates on the same 1,280/256 protocol.
Exact one-view top-1 tied native/direct at 6/256, but dense top-10 was 39/256
versus 52/256 native. On 2,748 matched causal target tokens, dense CE was
`.253547` versus `.243832` native and `.253330` direct MSE+SIGReg. Dense states
were closer to native at every supervised depth, yet global k=0 retrieval was
only 33.59% versus 45.31% native and 83.98% direct. Exact component VJPs showed
the latent predictor receiving substantially larger dense-loss gradients than
ChemFM. The selected epoch-4 checkpoint and all JSONs are preserved locally;
the A6000 instance was deleted after SHA-256 verification. Full details are in
`docs/reports/04_DENSE_CAUSAL_VJEPA2_1_EXPERIMENT.md`.

## Current measured state

For the fixed USPTO-MIT pilot endpoints, endpoint MSE+SIGReg, PCSF,
projection-space MSE+SIGReg, and dense causal V-JEPA 2.1-style conditions all
recorded exact top-1 and/or CE values no higher than the corresponding native
comparison. Dense supervision had higher CKA and lower RMS displacement from
native than direct endpoint MSE+SIGReg at every reported token depth; its
global raw retrieval was `33.59%`, target-token CE `.253547`, top-1 `6/256`,
and top-10 `39/256`. These measurements do not cover MetaTrans,
retrosynthesis, additional seeds, or larger training exposure.

The consolidated report index and artifact paths are in `docs/reports/README.md`. The dense causal implementation and measurements are in report 04.
