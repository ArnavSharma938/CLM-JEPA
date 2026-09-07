# Decoder-visible token-conditioned training experiment

Status: implementation, matched treatment training, development evaluation, and
the contingent full-state baseline are complete. Native controls and the official
five-view endpoint were reused; no Native retraining, representation audit, or
new panel generation was performed.

## Sources inspected before finalizing implementation

Inspected on 2026-09-06:

- [NextLat paper v4](https://arxiv.org/html/2511.05963v4), section 3.2 and
  appendix C (also checked v1). Official source pinned at
  `3770be6009cea2b3c455a9ce7f2ca88b504bb955`:
  [model_nextlat.py](https://github.com/JaydenTeoh/NextLat/blob/3770be6009cea2b3c455a9ce7f2ca88b504bb955/models/model_nextlat.py)
  and `config/fineweb/1B/nextlat_finewebedu_1b_100b_horizon1.yaml`.
  `NextLatDynamicsModel` implements normalized concatenation, three linear layers,
  GELU and a residual update. `_nextlat_loss_function` uses detached future states
  with SmoothL1, a detached current decoder for predicted logits, and detached
  teacher logits. `_categorical_kl_loss` uses `kl_div(log_p, log_q,
  log_target=True, reduction="none")`, sums vocabulary dimensions and averages
  valid tokens. `compute_loss` uses Fabric autocast; temporary detached input
  tensors receive gradients that are subsequently propagated into the original
  hidden states **and token embeddings**. Those temporary detaches do not freeze
  the current-state transformer path.
- [EAGLE paper](https://arxiv.org/html/2401.15077), section 3.1. Official v1
  branch pinned at `4a9cf3a1f6cd4a294e6d30a4e7c77cba246d7ca5`:
  [training](https://github.com/SafeAILab/EAGLE/blob/4a9cf3a1f6cd4a294e6d30a4e7c77cba246d7ca5/eagle/train/main.py)
  shifts both input token IDs and future feature targets one step; the
  [model](https://github.com/SafeAILab/EAGLE/blob/4a9cf3a1f6cd4a294e6d30a4e7c77cba246d7ca5/eagle/model/cnets.py)
  embeds under `no_grad` and concatenates embeddings with current features.
  This supplies the advanced-token precedent, not ChemFM's layer selection or
  training objective. EAGLE's draft network and frozen target-model training are
  not transplanted into this experiment.
- [EAGLE-3 paper](https://arxiv.org/html/2503.01840), sections 2–3; official main
  pinned at `cb7e0841fe0c206c6ed74a197ad5e2a1f13f5a2b`:
  [traineagle3/cnets.py](https://github.com/SafeAILab/EAGLE/blob/cb7e0841fe0c206c6ed74a197ad5e2a1f13f5a2b/eagle/traineagle3/cnets.py).
  It removes feature reconstruction constraints and trains against token
  distributions; the code converts logits to FP32 and uses detached target
  probabilities. This supports distinguishing feature reconstruction from token
  function, not an exact reproduction of EAGLE-3.
- [ChemFM paper](https://arxiv.org/abs/2410.21422), reaction-prediction training
  discussion; [official repository](https://github.com/TheLuoFengLab/ChemFM),
  master `ee35b23d03de1a8e97b8e04dcdfb1d579de70f02`. Read the local pinned
  `finetuning/reaction_prediction/{main.py,utils.py}` and synthesis config.
  Current upstream Python source is text-identical to the local files after
  newline normalization. The collator serializes source and product separately,
  concatenates them, masks source labels, and retains the product suffix.
  Preserve local `src/chemfm.py` BF16, LoRA, tokenizer-resize behavior and 402-row
  checkpoint/generation vocabulary. Restrict only the auxiliary decoder to 392.
- Local authoritative paths: `src/train.py`, `src/chemfm.py`, `src/stp.py`,
  `scripts/run_stp_matrix.py`, `scripts/run_stp_completion.py`, and
  `src/eval_uspto_mit_five_view_a6000.py`. Report 03's ridge C-to-D
  reconstruction/function dissociation motivates decoder-visible projection;
  it does not establish generation improvement from this training method.

Fetched reference code and the ChemFM PDF are cached under
`runs/decoder_projected/reference_sources/`; they are not runtime dependencies.
Installed versions: torch 2.3.0+cu121, transformers 4.45.2, peft 0.13.2.
No dependency changes were made.

## Locked experiment and intentional differences

`src/decoder_projected.py` uses final post-RMSNorm states captured by the existing
STP context manager. `W0` is the active initial Native/STP model head, sliced to
392 rows, not the inactive PEFT original module. Treatments start with the same
seeded `load_lora_model` initialization used by the archived controls. Starting
from an epoch-4 Native checkpoint and training another four epochs would not be
a matched-duration comparison; no such continuation is performed.

Center W0 in FP32; perform one compact CPU SVD with fixed relative tolerance
`1e-6`, absolute tolerance zero. Retain every singular value above
`1e-6 * largest_singular_value`, and record all singular values and reconstruction
error. Store U and Sigma-V-transpose as FP32 buffers; resume restores them without
another SVD. The real-model smoke measured rank **352** at this fixed tolerance,
with relative centered-decoder reconstruction error `6.96e-6`; rank is measured,
not forced to 391. This result pertains to the currently implemented starting
initialization. The starting-checkpoint interpretation has been raised for user
clarification before any treatment optimization.

The predictor is LayerNorm(4096), Linear(4096,256), GELU, Linear(256,256), GELU,
Linear(256,r). Its final weight and bias start at zero. Input order is `[h,e]`
as specified here (upstream code uses `[e,h]`). Predictor width, biases, zero
final initialization, fixed initial decoder snapshot, projection, detached token
embedding, enforced FP32 auxiliary operations, and ChemFM-only gradient scaling
are explicit experimental changes. Upstream uses a detached live head, rather
than this immutable initial snapshot. No speculative decoding is enabled.

Every eligible target is an actual product token: include the first token after
`<prostart>`, exclude product `<eos>`, source framing and padding. Reject empty
or truncated targets. SmoothL1 averages dimensions, then eligible transitions
within each row, then rows. KL is teacher-to-prediction over 392 logits at T=1,
with the same row-first averaging. This differs from upstream's global valid-token
averaging. Both objectives have coefficient one.

One calibration uses the first shuffled physical training batch and all trainable
ChemFM parameters (LoRA, saved embedding and head), excluding the predictor.
Measure un-clipped norms at alpha=1, set `min(1, .05*g_N/g_A)`, and lock it.
Only the current-state auxiliary path, including its residual, is scaled.
The target and direct advanced-token embedding path are detached. Predictor
gradients are not multiplied by alpha. ChemFM retains the existing norm-1
gradient clipping; predictor gradients are recorded and passed to AdamW without
that ChemFM clipping operation.

Predictor initialization and calibration preserve the training RNG stream;
calibration uses its own identically seeded DataLoader generator. Training data
order, collation, NTP suffix labels, AdamW, LR `1e-4`, weight decay `.01`, betas
`.9/.999`, epsilon `1e-8`, cosine-with-min-LR `1e-5`, 5% warmup, batch 4,
accumulation 4, rank/alpha 8, dropout .1, SDPA and no gradient checkpointing match
the established runner. Four epochs give 320 optimizer steps. Save epoch recovery
checkpoints to avoid losing completed epochs; use epoch 4 for evaluation. Keep
the trainer's existing two-row final validation path for compatibility.

The auxiliary module is separate from ChemFM. Adapter serialization saves only
ordinary ChemFM parameters; `auxiliary_training_state.pt` contains predictor,
coordinates, alpha and metadata separately. The ordinary official evaluator
never loads this file. Tests cover saved/reloaded ordinary Llama generation;
epoch-4 treatment generation was evaluated through the ordinary official
evaluator without loading predictor state.

## Reuse, evaluation and decision

Preflight verified 256 unique reactions, 5 existing views each, 1,280 rows,
and **55,746** eligible transitions per epoch. Input hashes, Native prediction
hashes, model config, resource estimates and paths are in
`runs/decoder_projected/preflight.json`. The frozen 512-reaction panel SHA-256 is
`a2e6202a4abaf9a70f4700e04299a09964d38c10fce004022dc43e759aa6057d`.

Reuse `runs/stp_matrix/a6000/stage_a/r8_l0.02/native/seed_{seed}/evaluation/`.
Native top-1 is 2.5390625% (533), 2.1484375% (917), and 3.90625% (1301).
Treatment scores and the contingent full-state baseline are complete for both
primary seeds.

| Native seed | Top-1 % | Top-3 % | Top-5 % | Top-10 % | View-candidate validity % |
|---|---:|---:|---:|---:|---:|
| 533 | 2.5391 | 15.2344 | 22.8516 | 35.3516 | 98.9883 |
| 917 | 2.1484 | 14.6484 | 21.8750 | 30.2734 | 98.6992 |
| 1301 (contingent control) | 3.9063 | 14.2578 | 22.4609 | 33.0078 | 98.6445 |

| Treatment | Seed | Native top-1 % | Treatment top-1 % | Effect pp | Treatment-only / Native-only |
|---|---:|---:|---:|---:|---:|
| decoder-projected | 533 | 2.5391 | 3.7109 | +1.1719 | 9 / 3 |
| decoder-projected | 917 | 2.1484 | 3.7109 | +1.5625 | 11 / 3 |
| decoder-projected mean | — | 2.3438 | 3.7109 | **+1.3672** | — |
| full-state baseline | 533 | 2.5391 | 3.5156 | +0.9766 | 8 / 3 |
| full-state baseline | 917 | 2.1484 | 3.5156 | +1.3672 | 10 / 3 |
| full-state baseline mean | — | 2.3438 | 3.5156 | **+1.1719** | — |

The runner invokes the existing official five-view evaluator and its `summarize`
command: beam 10, 10 candidates per view, reciprocal-rank aggregation and
canonical identity scoring. One worker keeps compute sequential. Summaries
include paired top-1, top-3/5/10 and both official-view and aggregate validity.
The first two effects both positive means proceed; both nonpositive means stop;
otherwise run only seed 1301. For the requested three-seed direction/mean rule,
the prespecified conservative interpretation is proceed only with at least two
positive effects **and** a positive three-seed mean. No weight sweep.

The positive development gate triggered the full-state baseline. Both baseline
seeds completed. Mean top-1 effects are +1.3672 pp for decoder-projected and
+1.1719 pp for full-state; decoder-projected is selected descriptively for later
confirmation by the specified generation criterion.

## Execution and checks

Commands from repository root, using `.venv/Scripts/python.exe` on Windows:

```text
python scripts/run_decoder_projected.py preflight
python scripts/run_decoder_projected.py smoke
python scripts/run_decoder_projected.py run
```

The smoke used one shortest valid 42-token row, no optimizer updates, and peaked
at 2,181,778,432 allocated bytes. It checked finite forward/backward losses,
nonzero predictor gradient, exact ordinary logits, and hook removal. Its alpha
and gradient norms are **smoke-only**, not the experiment calibration. Evidence:
`runs/decoder_projected/smoke.json` and `smoke.log`.

The matched runs were executed on Thunder RTX A6000 (48 GB), six vCPUs and 100 GB
disk, using four evaluator workers with one CPU thread each. Projected seeds 533
and 917 took 417.9 s and 410.0 s (peak 6.65 GB); full-state seeds 533 and 917
completed on the same instance. The predictor state is training-only and ordinary
generation used the ChemFM checkpoint without predictor/SVD components.

The existing serialization validator referenced a removed
`extract_source_and_target` helper. The necessary fix derives source/target
regions from unchanged collator labels and attention masks. No serialization
or NTP labels changed. Existing user deletions under `references/llm-jepa/`
were left untouched.

Verification: 38 focused tests passed (37 in the objective/trainer/STP/official
evaluator/decision suite, plus the final frozen-snapshot/resume test). Coverage
includes gradient isolation/scaling, row averaging, KL direction, SVD softmax
equivalence, transition boundaries, calibration, ordinary saved-model generation,
zero residual initialization, snapshot independence, SVD reuse on resume, and
the decision gate. Python compilation and tracked-diff whitespace checks passed.
