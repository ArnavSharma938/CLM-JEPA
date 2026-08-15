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
- `src/jepa.py`: readouts, cosine/MSE objectives, stop-gradient option, and exact streamed SIGReg.
- `src/representation_eval.py`: standard frozen representation evaluation.
- `src/eval_uspto_mit_five_view_a6000.py`: official five-view beam-10 endpoint evaluation.
- `scripts/`: report-specific diagnostics, setup utilities, A6000 execution wrappers, and the GSM8K upstream reference.
- `references/`: pinned upstream source only.

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

True and shuffled active auxiliary gradients had cosine `0.902`; the pair-specific residual was only `0.096x` the NTP norm and nearly orthogonal to NTP. Source/target MSE decomposition did not support target-branch interference as the main mechanism. Full details and the one controlled next experiment are in `docs/reports/04_MECHANISTIC_GRADIENT_AND_BLOCK_SWAP_AUDIT.md`.

## Current conclusion

For the fixed USPTO-MIT pilot endpoints, repairing global JEPA geometry was not sufficient to improve official reaction generation. The selected cLM-JEPA endpoint did not meet the +1 pp exact-top-1 effect of interest. The negative CE effect is most specifically associated with SIGReg pressure and realized parameter changes in layers 17-21, not uniform model-wide gradient opposition. This conclusion does not cover MetaTrans or retrosynthesis training, additional seeds, larger training exposure, or an objective that couples the auxiliary relationship differently to autoregressive decoding.

The consolidated report index and artifact paths are in `docs/reports/README.md`. Reports 01-07 from the former chronology were merged into `01_COSINE_TO_MSE_SIGREG_DIAGNOSIS.md`; the decisive MSE+SIGReg, official endpoint, and mechanistic audit results are reports 02-04.
