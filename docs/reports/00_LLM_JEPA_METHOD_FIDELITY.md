# LLM-JEPA method fidelity audit

## Verdict

The default non-additive cosine JEPA core in `src/jepa.py` is functionally equivalent to the corresponding path in official LLM-JEPA `finetune.py`, after the required ChemFM serialization and EOS substitutions. The implementation is not textually identical and does not claim support for every optional upstream objective. The ChemFM training backend now includes checkpoint save/resume, matched gradient accumulation, the official optimizer/scheduler settings, beam-generation settings, W&B integration, and frozen validation selection.

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
  `98c2287b31456575619047a86fc24a05f2a1f9788616160e4106593cfc48a2f7`

The hashes prove which snapshots were inspected; differing hashes also make
clear that the files are not claimed to be byte-identical.

## `finetune.py` coverage

Nearly all LLM-JEPA behavior is indeed concentrated in `finetune.py`. The
audit covered the whole file by responsibility, not only the loss lines.

| Official `finetune.py` area | Local disposition | Fidelity result |
|---|---|---|
| 30–59: chat message selection | Replaced by ChemFM task direction in `ReactionCollator` | Intentional backend substitution. |
| 62–220: full/user/assistant tokenization and predictor insertion | ChemFM collator constructs native, source-only, and target-only views; `CLMJEPA` appends predictor IDs | Semantically equivalent views, different serialization. |
| 222–419: chat label creation and model-specific templates | Replaced by ChemFM target-only labels and reaction markers | Intentional; chat behavior must not leak into ChemFM. |
| 420–516: tokenizer/model/LoRA setup | Split across `src/chemfm.py` and `src/jepa.py` | Predictor strings and LoRA target modules match; initialization follows ChemFM rather than LLM-JEPA. |
| 519–750: `RepresentationTrainer` | Ported to `CLMJEPA` | Default non-additive cosine path verified exactly. |
| 753–778: profiler callback | W&B compute counters instead | Not algorithmically required; FLOP estimates are not yet equivalent. |
| 780–929: CLI and dataset selection | Gate-specific scripts and chemistry manifests | Intentional experiment-layer replacement. |
| 937–1008: Trainer scheduling/configuration | Direct loop in `src/train.py` | ChemFM optimizer, scheduler, accumulation, and checkpoint semantics implemented. |
| 1012–1045: regular versus representation trainer | `condition` selects native/monitor/cLM-JEPA/shuffled | Same experimental distinction plus required controls. |
| 1063–1102: training and final save | `src/train.py` checkpoints adapters and resumable state every epoch | Required ChemFM-compatible lifecycle implemented. |

## Exact JEPA-core mapping

### Predictor tokens and ordering

`finetune.py:134–140` appends predictor strings in descending order from the
requested count, and `finetune.py:439–445` defines ten predictor tokens.
`src/jepa.py:13–25` defines the same strings; `src/jepa.py:132–135` appends
`predictor_k, ..., predictor_1`. This is exact for `k >= 0`.

`k=-1` is a project-requested condition, not an upstream predictor count. It
selects the second-to-last active source token and does not append a token.

### Row construction and isolation

The default official path at `finetune.py:603–621` concatenates rows in this
order:

1. native language-model rows;
2. user/source-predictor rows;
3. assistant/target-only rows.

It performs one model call with `output_hidden_states=True` at line 648 and
splits last-layer states into the same three blocks at lines 663–665.
`src/jepa.py:140–159` uses the identical row order, all-ignored labels for the
two auxiliary blocks, one model call, and the last hidden layer. Dynamic
padding replaces upstream fixed-length chat padding without changing any
active token or row boundary.

Source and target views are separate rows, so neither can attend to the other.
The native row remains the only row with non-ignored language-model labels.

### Final-state extraction

`finetune.py:538–564` computes an index from the number of active tokens plus
the configured negative offset. Lines 683–685 compute source and target
indices; lines 710–711 select the representations.

`src/jepa.py:161–167` computes the corresponding active lengths and selects:

- final predictor for `k > 0`;
- source `<eos>` for `k=0`;
- second-to-last source token for project condition `k=-1`;
- target `<eos>` for every target view.

ChemFM's explicit `<eos>` requirement follows the plan and is more specific
than the model-dependent chat EOT offset used upstream.

### Loss and gradients

`finetune.py:714` computes cosine similarity, line 735 computes
`1 - mean(cosine)`, and line 739 computes
`gamma * lm_loss + lambda * jepa_loss`. `src/jepa.py:168–170` implements the
same operations. Both source and target branches remain attached to the shared
backbone, so gradients flow through both, as upstream.

The upstream alternatives at lines 720–730 (L2, MSE, InfoNCE) are deliberately
not part of the primary port. The plan excludes InfoNCE and allows MSE only as
a rescue after verified cosine failure; Gate 3 did not trigger that rescue.

### JEPA loss dropout

`finetune.py:572–578` skips JEPA when
`torch.rand(1).item() > jepa_ratio`. `src/jepa.py:102–105` uses the same RNG,
strict comparison, and native-only fallback. The Gate 4 configuration resolves
`jepa_ratio = 1 - alpha` and `lambda = lambda_eff / (1 - alpha)` before the
call, matching the plan's interpretation of upstream loss dropout.

### Optional additive mask

The special additive-mask path at `finetune.py:566–601` is not ported. The plan
says to use it only if numerically equivalent and measurably more efficient.
The local implementation uses the upstream default three-row path instead, so
this is not a fidelity defect for the selected method.

## ChemFM preservation audit

The chemistry backend was compared separately because copying LLM-JEPA chat
code would itself violate the plan.

- ChemFM `utils.py:120–144` mean-initializes newly added embeddings. Local
  `src/chemfm.py:95–100` and `src/jepa.py:16–25` preserve that policy.
- ChemFM `utils.py:147–223` constructs reaction markers, independently
  tokenizes source and target, masks all source labels, right-pads, and returns
  generation prompts. Local `ReactionCollator` preserves these operations for
  forward, metabolism, and retrosynthesis directions.
- ChemFM `main.py:61–123` loads Llama, applies attention dropout, disables the
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

## Historical issues resolved before Gate 4

The original audit found six backend issues. The current implementation resolves them as follows:

| Audit finding | Current implementation |
|---|---|
| No adapter checkpoint save/resume | Epoch checkpoints include adapter weights, optimizer/scheduler state, RNG state, and resume metadata. |
| Scheduler/optimizer mismatch | ChemFM AdamW betas/epsilon, 5% warmup, cosine decay, and minimum LR are explicit in `src/train.py`. |
| Beam-generation mismatch | ChemFM beam width, length penalty, early-stopping, maximum length, and canonical scoring are preserved. |
| No gradient accumulation | Physical and effective batch semantics are explicit and checkpoint-resumable. |
| Training path not exercised end to end | Gate 4/5 and later A6000 experiments exercised training, W&B logs, checkpoint selection, generation, diagnostics, and artifact writing. |
| Upstream ChemFM debug break | The local loop intentionally omits the reference's eleven-step debug limit. |

## What can and cannot be claimed

Supported claim: the local default cosine JEPA computation is functionally
equivalent to the corresponding default `RepresentationTrainer` computation
in pinned LLM-JEPA `finetune.py`, after the explicit ChemFM serialization and
EOS adaptations required by the plan.

Unsupported claims:

- that local and upstream files are textually identical;
- that optional additive-mask/L2/MSE/InfoNCE paths are implemented;
- that unit tests alone establish paper-level reproduction;
- that CUDA results will be bit-identical across hardware or library builds.
