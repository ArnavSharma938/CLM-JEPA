# Paper-STP completion experiment preregistration

This protocol is frozen before any new training or new-checkpoint generation
evaluation. It starts from Report 08 final commit
`841f6e95635f88706b7fcdec14866e3c31932629`; the eventual execution commit is
recorded in the completed report. The official STP source remains
`galilai-group/llm-jepa@ea0017c654ad917066ff32afc88276bea8ca5f7e`.

The existing 512-reaction manifest is a development/selection panel, not an
untouched confirmation panel. It is
`data/clm_jepa_uspto_mit_official_endpoint/prespecified_stage1_512.jsonl`,
SHA-256
`a2e6202a4abaf9a70f4700e04299a09964d38c10fce004022dc43e759aa6057d`.
All comparisons use its official five R-SMILES views, beam 10, ten returned
candidates per view, canonicalization, and reciprocal-rank aggregation.

## Completed, reusable, and missing matrix

| Formulation | Rank / alpha | Lambda | Seeds | Status and reuse |
|---|---:|---:|---|---|
| Native NTP | 8 / 8 | 0 | 533, 917, 1301 | Complete; reuse checkpoints and 512-view predictions |
| Released STP | 8 / 8 | 0.02 | 533, 917, 1301 | Complete; reuse checkpoints, predictions, and diagnostics |
| Native NTP | 128 / 128 | 0 | 533, 917 | Complete; reuse Report 08 controls |
| Released STP | 128 / 128 | 0.02 | 533, 917 | Complete; reuse; this is the only completed rank-128 STP evidence |
| Paper STP | 8 / 8 | 0.02 | 533, 917 | Complete; reuse Report 08 checkpoints and predictions |
| Released STP | 8 / 8 | 0.005, 0.08 | 533, 917 | Complete; reuse only as released-formulation lambda evidence |
| Paper STP | 128 / 128 | 0.02 | 533, 917 | **Required new** formulation-by-capacity cell |
| Paper STP | 8 / 8 | 0.08, 0.12 | 533, 917 | **Required new** paper-specific lambda screen |
| Paper STP | 8 / 8 | 0.16 | 533, 917 | Conditional edge point only |
| Paper STP | 128 / 128 | selected lambda if not 0.02 | 533, 917 | Conditional selected-lambda rank completion only |

Thus the unconditional new budget is exactly six trajectories. No Native,
released-STP, or paper-rank-8/lambda-0.02 trajectory is rerun.

Report 08 establishes that released STP at rank 128/lambda 0.02 did not improve
the treatment effect over released STP at rank 8/lambda 0.02 on seeds 533 and
917. It does not establish a formulation-independent rank result. Likewise,
its released-STP lambda screen does not select a paper-STP lambda. The frozen
batch diagnostic measured mean lambda-weighted STP/NTP norm ratios of 0.0450
for released STP and 0.00776 for paper STP at lambda 0.02, approximately a
5.8-fold difference; equal lambda is therefore not treated as equal auxiliary
pressure.

## Frozen existing-beam diagnostics

Before training, the complete archived ordered beams are analyzed without
changing inference. Gold presence and rank are recomputed from all 50
canonical candidates, using the exact stable official reciprocal-rank rule.
The diagnostic records:

- per-view gold rank and aggregate gold score/rank;
- top-1/3/5/10, invalid candidates, candidate diversity, and cross-view
  agreement;
- Native-only and treatment-only top-1 transitions;
- candidates promoted above the gold product and their view ranks;
- mutually exclusive Native-only failure classes: `beam_entry_absent` if gold
  is absent from every treatment beam, `cross_view_aggregation` if gold is
  top-1 in at least one treatment view but not after aggregation, and
  `within_beam_ranking` if gold remains in a treatment beam but is never a
  per-view top-1.

For teacher-forced token localization, material events are fixed as the first
target position where Native and treatment differ in top-1 correctness, cross
a correct-token rank threshold of 3, 5, or 10, differ in correct-token rank by
at least five places, or differ in correct-token margin by at least 0.5 logits.
These diagnostics are explanatory and cannot change the official endpoint.

## Training constants

Every new trajectory uses seeds 533 and 917, the frozen 1,280-row training
manifest, four epochs/320 optimizer updates, physical batch size 4, gradient
accumulation 4, effective batch size 16, learning rate `1e-4`, fused AdamW,
the existing cosine schedule, BF16/FP32 behavior, SDPA, one paper-STP sample
per example, final post-RMSNorm state, framing-excluded semantic path, FP32
cosine, symmetric gradients, and final-checkpoint-only writing. LoRA uses
alpha/rank 1, dropout 0.1, target modules `q_proj`, `k_proj`, `v_proj`,
`o_proj`, `gate_proj`, `up_proj`, and `down_proj`, with `embed_tokens` and
`lm_head` in `modules_to_save`.

## Required stages

### A: paper formulation by capacity at lambda 0.02

Train paper STP rank/alpha 128/128 at lambda 0.02 for both seeds. Reuse the
rank-128 Native and released-STP checkpoints. The primary quantities are the
same-rank paper-minus-Native treatment effects and the seed-paired interaction

`(Paper128 - Native128) - (Paper8 - Native8)`.

Absolute rank-128 accuracy is not evidence for a capacity interaction.

### B: paper-specific rank-8 lambda screen

Train paper STP rank 8 at lambda 0.08 and 0.12 for both seeds and reuse lambda
0.02. Selection is based first on mean paired official exact top-1 treatment
effect versus the same Native seed. Top-k, teacher-forced metrics, STP loss,
gradient norms/cosines, and geometry are secondary.

Lambda 0.12 triggers lambda 0.16 only if its mean treatment effect exceeds
both 0.02 and 0.08 by at least 0.01 absolute accuracy and both lambda-0.12 seed
effects are nonnegative. This is the frozen meaning of “materially better” for
the edge condition.

The default selected lambda is 0.02. An alternative is eligible only if its
mean treatment effect exceeds 0.02 by at least 0.005 and both seed effects are
nonnegative. Among eligible alternatives, choose the largest mean effect; if
two means differ by less than 0.005, choose the lower lambda. If lambda 0.16 is
run, it must itself exceed the otherwise selected candidate by at least 0.005
with nonnegative effects to replace it.

### C: selected-lambda paper capacity completion

If the selected paper lambda is 0.02, Stage A is the complete rank comparison.
Otherwise train exactly two paper-STP rank-128 trajectories at the selected
lambda and reuse the rank-128 Native controls.

Rank 128 is selected as the best-supported paper capacity only if the mean
rank interaction is at least +0.005 and both seed-specific interaction
contrasts are nonnegative. Otherwise rank 8 is retained. This rule is frozen
before inspecting any selected-lambda rank-128 result.

### D: fairly characterized formulation comparison

The released reference remains rank 8/lambda 0.02. Compare it with the selected
paper rank/lambda through same-rank treatment effects relative to Native.
Direct model-versus-model paired differences are reported, but if selected
ranks differ they are descriptive rather than a pure formulation effect.

A formulation advantage is called supported only if the mean treatment-effect
contrast is at least 0.005 in magnitude, both seed contrasts have the same
sign, and the two-way seed/reaction bootstrap 95% interval excludes zero.
Otherwise the formulation difference is unresolved. No claim is made from a
numerically larger mean alone.

## Reported endpoints and stopping

For every new checkpoint report exact top-1/3/5/10, all five individual views,
teacher-forced CE/correct-token rate/margin, paper-STP loss, raw and
lambda-weighted gradient ratios, NTP-gradient cosine, gold beam survival/rank,
aggregation behavior, training time, and peak VRAM. Preserve all negative
runs and per-reaction outputs.

This experiment stops after the required and conditionally triggered cells.
It does not launch new confirmation seeds or a new endpoint. The final report
will specify one later confirmation using new paired training seeds and a
newly prespecified untouched 512-reaction panel from the remaining official
test set, with no further model selection.
