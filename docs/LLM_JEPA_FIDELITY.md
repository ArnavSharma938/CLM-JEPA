# LLM-JEPA fidelity audit

## Verdict

The cLM-JEPA core in `src/jepa.py` is a faithful executable
transcription of the default, non-additive, cosine-loss path in official
LLM-JEPA `finetune.py`. It is not a textual copy of that file and the repository
as a whole is not yet functionally identical to the complete upstream training
system. Chat-specific preprocessing was deliberately replaced by official
ChemFM reaction preprocessing, while several pending Gate 4/5 training,
generation, and checkpoint details still require correction before fine-tuning.

This distinction is important: the JEPA representation/loss computation is
verified; the pending Gate 4/5 runner is not yet certified as an end-to-end
faithful ChemFM training backend.

## Audited sources

- LLM-JEPA commit: `ea0017c654ad917066ff32afc88276bea8ca5f7e`
- ChemFM commit: `ee35b23d03de1a8e97b8e04dcdfb1d579de70f02`
- `references/llm-jepa/finetune.py` SHA-256:
  `52c5ee3048b36f34ebea9dc1305bd5a353ab79b23416c008d5290ef8f4984392`
- `references/llm-jepa/run.sh` SHA-256:
  `c50ce29b7baa0eadcad223280b05030f7c876de00ec95719cc01a910769ce4b4`
- `references/llm-jepa/evaluate.py` SHA-256:
  `9aee3b521b7532cda4fb9d54d02ddb3859eac59e6f97b9481a704b4fd8a20ab9`
- Audited local core: `src/jepa.py` SHA-256:
  `8eecc2cb992790b77b7649ba684b1b3b4921666e252dd0a21f955dd9b9eb3b21`

The hashes prove which snapshots were inspected; differing hashes also make
clear that the files are not claimed to be byte-identical.

## `finetune.py` coverage

Nearly all LLM-JEPA behavior is indeed concentrated in `finetune.py`. The
audit covered the whole file by responsibility, not only the loss lines.

| Official `finetune.py` area | Local disposition | Fidelity result |
|---|---|---|
| 30â€“59: chat message selection | Replaced by ChemFM task direction in `ReactionCollator` | Intentional backend substitution. |
| 62â€“220: full/user/assistant tokenization and predictor insertion | ChemFM collator constructs native, source-only, and target-only views; `CLMJEPA` appends predictor IDs | Semantically equivalent views, different serialization. |
| 222â€“419: chat label creation and model-specific templates | Replaced by ChemFM target-only labels and reaction markers | Intentional; chat behavior must not leak into ChemFM. |
| 420â€“516: tokenizer/model/LoRA setup | Split across `src/chemfm.py` and `src/jepa.py` | Predictor strings and LoRA target modules match; initialization follows ChemFM rather than LLM-JEPA. |
| 519â€“750: `RepresentationTrainer` | Ported to `CLMJEPA` | Default non-additive cosine path verified exactly. |
| 753â€“778: profiler callback | W&B compute counters instead | Not algorithmically required; FLOP estimates are not yet equivalent. |
| 780â€“929: CLI and dataset selection | Gate-specific scripts and chemistry manifests | Intentional experiment-layer replacement. |
| 937â€“1008: Trainer scheduling/configuration | Direct loop in `src/train.py` | Not yet fully faithful to ChemFM; see blockers. |
| 1012â€“1045: regular versus representation trainer | `condition` selects native/monitor/cLM-JEPA/shuffled | Same experimental distinction plus required controls. |
| 1063â€“1102: training and final save | Direct training loop writes JSON only | Material checkpointing gap before Gate 4. |

## Exact JEPA-core mapping

### Predictor tokens and ordering

`finetune.py:134â€“140` appends predictor strings in descending order from the
requested count, and `finetune.py:439â€“445` defines ten predictor tokens.
`src/jepa.py:13â€“25` defines the same strings; `src/jepa.py:132â€“135` appends
`predictor_k, ..., predictor_1`. This is exact for `k >= 0`.

`k=-1` is a project-requested condition, not an upstream predictor count. It
selects the second-to-last active source token and does not append a token.

### Row construction and isolation

The default official path at `finetune.py:603â€“621` concatenates rows in this
order:

1. native language-model rows;
2. user/source-predictor rows;
3. assistant/target-only rows.

It performs one model call with `output_hidden_states=True` at line 648 and
splits last-layer states into the same three blocks at lines 663â€“665.
`src/jepa.py:140â€“159` uses the identical row order, all-ignored labels for the
two auxiliary blocks, one model call, and the last hidden layer. Dynamic
padding replaces upstream fixed-length chat padding without changing any
active token or row boundary.

Source and target views are separate rows, so neither can attend to the other.
The native row remains the only row with non-ignored language-model labels.

### Final-state extraction

`finetune.py:538â€“564` computes an index from the number of active tokens plus
the configured negative offset. Lines 683â€“685 compute source and target
indices; lines 710â€“711 select the representations.

`src/jepa.py:161â€“167` computes the corresponding active lengths and selects:

- final predictor for `k > 0`;
- source `<eos>` for `k=0`;
- second-to-last source token for project condition `k=-1`;
- target `<eos>` for every target view.

ChemFM's explicit `<eos>` requirement follows the plan and is more specific
than the model-dependent chat EOT offset used upstream.

### Loss and gradients

`finetune.py:714` computes cosine similarity, line 735 computes
`1 - mean(cosine)`, and line 739 computes
`gamma * lm_loss + lambda * jepa_loss`. `src/jepa.py:168â€“170` implements the
same operations. Both source and target branches remain attached to the shared
backbone, so gradients flow through both, as upstream.

The upstream alternatives at lines 720â€“730 (L2, MSE, InfoNCE) are deliberately
not part of the primary port. The plan excludes InfoNCE and allows MSE only as
a rescue after verified cosine failure; Gate 3 did not trigger that rescue.

### JEPA loss dropout

`finetune.py:572â€“578` skips JEPA when
`torch.rand(1).item() > jepa_ratio`. `src/jepa.py:102â€“105` uses the same RNG,
strict comparison, and native-only fallback. The Gate 4 configuration resolves
`jepa_ratio = 1 - alpha` and `lambda = lambda_eff / (1 - alpha)` before the
call, matching the plan's interpretation of upstream loss dropout.

### Optional additive mask

The special additive-mask path at `finetune.py:566â€“601` is not ported. The plan
says to use it only if numerically equivalent and measurably more efficient.
The local implementation uses the upstream default three-row path instead, so
this is not a fidelity defect for the selected method.

## ChemFM preservation audit

The chemistry backend was compared separately because copying LLM-JEPA chat
code would itself violate the plan.

- ChemFM `utils.py:120â€“144` mean-initializes newly added embeddings. Local
  `src/chemfm.py:95â€“100` and `src/jepa.py:16â€“25` preserve that policy.
- ChemFM `utils.py:147â€“223` constructs reaction markers, independently
  tokenizes source and target, masks all source labels, right-pads, and returns
  generation prompts. Local `ReactionCollator` preserves these operations for
  forward, metabolism, and retrosynthesis directions.
- ChemFM `main.py:61â€“123` loads Llama, applies attention dropout, disables the
  cache for training, and targets all seven attention/MLP projections with
  LoRA. Local loading preserves those choices and also preserves ChemFM's
  `embed_tokens`/`lm_head` modules-to-save behavior.
- ChemFM `score.py` canonicalizes with RDKit, removes invalid strings from
  valid predictions, and computes ranked exact recovery. Local task scoring
  preserves canonical exact matching and counts invalid generations as
  failures, while adding the plan's task-specific metrics.

## Evidence beyond unit tests

### Static control-flow comparison

Every branch of `RepresentationTrainer` was classified above. The selected
default path was traced from CLI arguments through data view construction,
row packing, one model call, state slicing, loss construction, and trainer
selection. Optional branches not used by the plan are explicitly listed
rather than silently treated as equivalent.

### Real ChemFM-1B execution

Gate 3 exercised the actual pinned ChemFM-1B checkpoint, reaction tokenizer,
predictor-token extension, source/target extraction, final hidden-state
selection, and cosine diagnostics on 7,168 chemical examples across seven
datasets. All five positions produced finite, non-collapsed target states and
pair-specific signal on every dataset. Peak allocated CUDA memory recorded in
the retained result ranged from 2,396,944,384 to 3,494,853,632 bytes.

This validates the frozen representation path with the real model. Gate 3 did
not exercise backward, optimizer, scheduler, generation, or checkpoint save.

### Executable golden reference

`tests/test_llm_jepa_parity.py` independently reconstructs the official default
packing and loss instead of calling the local helper twice. On a deterministic
tiny Llama it compares packed IDs, labels, indices, source states, target
states, native loss, JEPA loss, combined loss, and every parameter gradient.
The strongest comparison uses `rtol=0` and `atol=0`.

Additional tests establish exact native behavior at lambda zero, one model
call, causal isolation, source sensitivity, target isolation, explicit target
EOS extraction, shared-backbone JEPA gradients, zero monitor-only JEPA
gradient, and reproducible unequal-target shuffling.

### Causal and row-isolation invariants

The one-call construction remains safe because transformer attention is
row-local. Predictor suffixes occur after the source and therefore cannot
alter preceding source states. Auxiliary target rows cannot alter source or
native rows. These invariants are verified both by tensor inspection and by
mutation tests.

## Material gaps before Gate 4

The audit found the following end-to-end gaps. Gate 4 must not be launched
until they are corrected and smoke-tested on one real batch.

1. `src/train.py` has no model/adapter checkpoint save or resume path,
   whereas ChemFM saves every epoch and the plan requires frozen validation
   checkpoint selection.
2. The local scheduler is `CosineAnnealingLR` without ChemFM's 5% warmup.
   ChemFM uses `cosine_with_min_lr`; optimizer betas/epsilon also need to be
   resolved from the official configuration rather than implicit defaults.
3. Local beam generation omits ChemFM's `early_stopping='never'` and
   `length_penalty=0.0` and uses `max_new_tokens` rather than the official
   combined maximum-length rule. These may change rankings.
4. Gradient accumulation is not implemented even though the plan requires
   matching effective batch/optimizer steps across controls.
5. The direct loop has not been exercised end to end with actual W&B logging,
   validation generation, diagnostics, and artifact writing.
6. The official ChemFM loop contains a debug limit at `main.py:132â€“134`
   (`if it > 10: break`). Local code correctly does not copy it, but this
   intentional correction must remain documented.

These gaps concern the pending training backend, not the already completed
frozen Gate 3 relationship assay or the verified JEPA loss core.

## What can and cannot be claimed

Supported claim: the local default cosine JEPA computation is functionally
equivalent to the corresponding default `RepresentationTrainer` computation
in pinned LLM-JEPA `finetune.py`, after the explicit ChemFM serialization and
EOS adaptations required by the plan.

Unsupported claims:

- that local and upstream files are textually identical;
- that optional additive-mask/L2/MSE/InfoNCE paths are implemented;
- that the current Gate 4/5 runner preserves the complete official ChemFM
  training, generation, evaluation, and checkpoint path;
- that unit tests alone establish paper-level reproduction;
- that CUDA results will be bit-identical across hardware or library builds.
