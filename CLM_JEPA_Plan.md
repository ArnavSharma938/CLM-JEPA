# cLM-JEPA with ChemFM-1B
## Research and implementation plan for JEPA fine-tuning of a causal chemical language model

**Scope.** This study adapts **LLM-JEPA** to **ChemFM-1B**. For every task, ChemFM retains its ordinary target-token generation loss while an auxiliary JEPA loss predicts the hidden representation of the **same target molecule or precursor set** from the source. The JEPA branch is training-only; inference remains ordinary ChemFM generation.

The study uses three tasks:

1. **forward reaction-product prediction** — primary methodological task;
2. **one-step human metabolite generation** — difficult low-data biochemical task;
3. **single-step retrosynthesis** — secondary reverse-direction task.

The paper is limited to ChemFM-1B fine-tuning. Do not claim cross-model generality or pretraining efficacy.

---

## 1. Global execution rules

### 1.1 Project root and implementation base

Keep all work under:

`C:\Users\arnav.DHEERAJACER\CLM-JEPA`

Use the official **LLM-JEPA** repository as the method reference and the official **ChemFM** repository as the model/training backend. Preserve ChemFM's tokenizer, LoRA setup, reaction collator, generation, and scoring code. Add cLM-JEPA through minimal wrappers or trainer overrides; do not rewrite ChemFM into a new framework.

### 1.2 Compute

Use local hardware for development and any run that fits. If local execution reaches CUDA OOM after ordinary memory measures—mixed precision, gradient checkpointing, smaller physical batch, and gradient accumulation—stop and request permission to use Thunder Compute. The request must state the exact run, expected runtime/cost, launch command, artifacts copied back, and abort conditions. Use exactly **1× NVIDIA A6000, 6 vCPU, no template, and 200 GB storage** unless explicitly changed.

### 1.3 Stop-work rules

Stop before expensive runs if verified debugging cannot resolve any of the following:

- ChemFM-1B/tokenizer inference cannot be reproduced;
- native fine-tuning cannot nearly memorize a fixed tiny set;
- source and target JEPA views are not isolated;
- the target representation is collapsed or dominated by serialization shortcuts;
- correctly paired source-target examples are not more predictable than matched shuffles;
- JEPA cannot decrease without materially damaging generation;
- gains disappear after correcting preprocessing, leakage, decoding, or compute matching.

Do not answer a failed gate by increasing model size, changing the task, adding an external teacher, or spending more compute.

### 1.4 Efficient Codex execution

Use tokens and compute efficiently. Inspect only the files and symbols needed for the current gate, make targeted edits, keep reports concise until the final report, reuse official code, and avoid unrelated refactors or broad literature summaries. Do not run a full hyperparameter grid when staged or multi-fidelity selection can answer the question.

---

## 2. Research questions and claims

### 2.1 Research questions

1. Does cLM-JEPA improve ChemFM-1B generation over matched native fine-tuning?
2. Are improvements caused by predicting the correctly paired target representation rather than extra computation or generic regularization?
3. Are gains reproducible across forward synthesis, human metabolism, and retrosynthesis?
4. Are gains larger or more stable under low-data and chemically out-of-distribution evaluation?

### 2.2 Permitted claims

A defensible positive result requires:

1. a learnable, leakage-free native baseline;
2. identical data, trainable parameters, optimization exposure, checkpointing, and decoding between matched conditions;
3. cLM-JEPA outperforming native fine-tuning and matched-shuffled JEPA;
4. no material loss of validity or native generative quality;
5. the same direction on a second pilot seed and five-seed final evaluation;
6. noncollapsed, pair-specific representation behavior.

Claim boundaries:

- **ChemFM-1B fine-tuning claim:** requires improvement on at least two tasks.
- **reliable ChemFM-1B method claim:** requires consistent gains across seeds, tasks, and at least one OOD or low-data regime.
- **low-data claim:** requires a larger or more consistent benefit in prespecified reduced-data conditions than at full data.
- **general cLM claim:** not supported by this study.
- **pretraining claim:** not supported by this study.

---

## 3. Tasks, exact inputs, and datasets

### 3.0 Shared input/output contract

Follow ChemFM's reaction collator rather than inventing a new prompting scheme:

```text
training sequence = <SOURCE_MARKER> source <eos> <TARGET_MARKER> target <eos>
generation prompt = <SOURCE_MARKER> source <eos> <TARGET_MARKER>
labels            = -100 on the complete source span; token IDs on the target span
```

Use `<rstart>`, `<prostart>`, and `<eos>` from ChemFM's `string_template.json`. Join disconnected molecular components with `.`. The native loss and JEPA target branch must receive the **same target serialization for each example**. Never supply atom maps, reaction classes, metabolizing enzymes, template labels, or split metadata to the model unless a task below explicitly says so.

### 3.1 Task A — forward reaction-product prediction

**Scientific mapping**

```text
context/source = all reported non-product reaction components
                 (reactants + reagents/agents + solvents), mixed with `.`
target         = one reported major product

<rstart>{mixed_source}<eos><prostart>{product}<eos>
```

Use the same mixed-input definition for native and cLM-JEPA runs. Randomize component order only as a training augmentation; validation and test use a frozen enumeration policy.

| Role | Dataset | Use and validity boundary |
|---|---|---|
| Pilot/compatibility | **USPTO-MIT synthesis**, official ChemFM split and R-SMILES files | Reproduce ChemFM's native pipeline and select the initial method. This is not headline evidence: the random split is near saturation and R-SMILES deliberately reduces source-target string discrepancy. |
| Established difficult ID benchmark | **USPTO-STEREO**, published mixed-input split | Measures exact connectivity and stereochemistry under a recognized in-distribution benchmark. Do not describe it as OOD. |
| Main open benchmark | **ORDerly-forward**, keeping only records with exactly one product while preserving their original split assignment | Input is the union of ORDerly reactants, solvents, and agents, mixed with `.`; target is the single product. Report the retained counts and do not compare the filtered score directly with the published unfiltered ORDerly number. |
| External test | **non-USPTO-forward**, filtered with the same rule | Report separately as cross-source generalization. Because non-USPTO records have different component completeness/role quality, audit missing fields and do not treat this set as a definitive prospective benchmark. |
| Secondary OOD test | one frozen **local-reaction-template-held-out** split derived from ORDerly-forward | Use templates only to partition data, never as model input. Keep all examples of a template in one partition, pre-register extraction and frequency rules, and report ID and OOD results separately. |

Serialization analysis:

1. Use official R-SMILES only for direct ChemFM compatibility.
2. Use non-root-aligned canonical or independently randomized SMILES for the principal scientific result.
3. Report a gain that occurs only with R-SMILES as serialization-dependent, not evidence of improved chemical abstraction.

### 3.2 Task B — one-step human metabolite generation

**Scientific mapping**

```text
context/source = one parent molecule
target         = one one-step human metabolic product

<rstart>{parent}<eos><prostart>{metabolite}<eos>
```

One parent-metabolite pair is one training example. Keep every pair sharing a parent in the same split.

**Dataset: MetaTrans.** Use the released data under two clearly distinguished conditions:

1. **Full MetaTrans training collection:** contains curated database pairs **and** SyGMa/generic-rule-derived pairs; do not call every training target experimentally verified.
2. **Curated-source subset:** include only directly database-sourced pairs when the released provenance can be reconstructed exactly. If provenance is absent or ambiguous, omit this subset rather than guessing.
3. **Primary test:** the published held-out drug test set, kept unchanged and evaluated at the parent level.
4. **Secondary OOD validation:** a parent-grouped scaffold split constructed from the remaining training data, with no parent or scaffold shared across partitions.

The published test contains only known positive metabolites and is incomplete. Therefore:

- it validly measures recovery/ranking of recorded metabolites;
- precision is only a **lower-bound precision** because unrecorded predictions may be real metabolites;
- it cannot support claims about recognizing nonmetabolized compounds or clinical false-positive rates.

Use exactly the same molecule standardization in every split and audit salts, invalid structures, stereochemistry, duplicate parents, and alternate representations. Do not add USPTO reaction pairs to this task.

### 3.3 Task C — single-step retrosynthesis

**Scientific mapping**

```text
context/source = one product molecule
target         = the recorded precursor reactant set

<prostart>{product}<eos><rstart>{precursor_1.precursor_2...}<eos>
```

Canonicalize each precursor independently and sort components deterministically for non-R-SMILES experiments. The target is an unordered set even though ChemFM generates a sequence.

| Role | Dataset | Use and validity boundary |
|---|---|---|
| Pilot/recognized ID benchmark | **USPTO-50K**, published split, reaction class withheld | Establish native learning, multi-component decoding, and cLM-JEPA feasibility. R-SMILES is a compatibility condition only. |
| Main difficult benchmark | released **local-reaction-template-held-out USPTO-480K** split from *Retrosynthesis Extrapolation* | Use the published split unchanged and non-root-aligned target serialization for the headline OOD result. The benchmark is intentionally severe and exact match is expected to be very low. |
| External test | **non-USPTO-retro** | Report separately after preprocessing is frozen; differences in reaction recording and component assignment make it supportive rather than definitive evidence. |

Do not supply reaction classes or local templates to the model. Exact-match accuracy is necessary but insufficient because multiple precursor sets may be valid; Section 10 therefore includes validity, novelty, round-trip, and limited manual review.

---

## 4. Exact method

Preserve ChemFM's collation, target-only causal loss, LoRA loading, generation, and canonical scoring in `ChemFM/finetuning/reaction_prediction/{utils.py,main.py,score.py,string_template.json}`. Port only the LLM-JEPA behavior needed from `llm-jepa/finetune.py`: appended predictor tokens, final non-padding-state extraction, isolated source/target computation, cosine JEPA loss, and `jepa_ratio` loss dropout. Follow the LLM-JEPA paper and `run.sh` for the meaning and tuning order of `k`, `lambda`, and loss dropout. Do not add a separate predictor MLP.

For source `x` and target `y`:

\[
\mathcal L_{\mathrm{cLM-JEPA}}
=
\mathcal L_{\mathrm{native}}(x,y)
+
\lambda\left[1-\cos(z_x,z_y)\right].
\]

- `z_x` is the final-layer state of the last predictor token after the complete source.
- `k=0` uses the source `<eos>` state directly, matching LLM-JEPA's identity predictor.
- `k>0` appends the same predictor-token pattern used by LLM-JEPA and uses the final predictor state.
- `z_y` is the final-layer state of the target `<eos>` from an independently encoded target-only sequence.
- Source and target views cannot attend to each other.
- The primary formulation allows gradients through both shared-backbone branches, matching LLM-JEPA.
- Use cosine distance. InfoNCE is excluded; MSE is one rescue ablation only after verified cosine failure.

Compute the native row, source-predictor row, and target-only row in one concatenated batch when practical, as in LLM-JEPA. Use custom additive masking only if it is numerically equivalent and measurably more efficient.

---

## 5. Controls

Every selected experiment compares:

1. **Native fine-tuning:** ordinary target-token cross-entropy, `λ=0`, without JEPA view computation.
2. **Monitor-only JEPA:** execute the same source and target JEPA branches and log the JEPA loss, but exclude it from backpropagation. This controls for extra forward computation and tests whether native fine-tuning implicitly improves JEPA alignment.
3. **cLM-JEPA:** ordinary native loss plus the correctly paired JEPA loss; this is the proposed method.
4. **Matched-shuffled JEPA:** replace each target representation with a different target matched as closely as practical on token length and molecular size; native token targets remain correct. This tests whether pair-specific representation prediction matters.
5. **JEPA-only failure diagnostic:** on a tiny fixed subset only, set the native-loss coefficient to zero and optimize only JEPA. This is expected to damage or fail to preserve generation and is never a candidate configuration.

---

## 6. Minimal data and leakage checks

Download each dataset only from the paper-linked official release. Record the source URL/version, file hash, and measured split counts. Create one compact manifest per dataset containing example ID, split, source, target, canonical identities, and token lengths. Verify only what changes scientific validity:

- preserve published splits for benchmark comparability; report any cross-split exact duplicates instead of silently rewriting the benchmark;
- for newly constructed splits, prevent canonical reaction-pair overlap and keep all examples from one metabolism parent or one held-out reaction template together;
- verify that truncation never removes target endings;
- keep all preprocessing, hyperparameter, checkpoint, and decoding decisions independent of test data;
- count invalid generations as failures.

If an official split has substantial duplicate leakage, report the official result and a separately labeled leakage-clean sensitivity result. Do not build additional data infrastructure beyond these checks.

---

## 7. ChemFM-1B pilot program

Use one immutable reduced USPTO-MIT train/validation manifest.

### Gate 0 — official reproduction

1. Load ChemFM-1B and its reaction tokenizer.
2. Reproduce official reaction collation and generation prompts.
3. Verify target-only labels, beam generation, canonicalization, and hidden-state extraction.
4. Check the official training loop for accidental debug limits before using it unchanged.

### Gate 1 — native tiny overfit

Use fixed 32- and 128-example subsets. Require near-memorization on 32 examples, strongly decreasing target loss, valid decoded products, and no source-label leakage.

### Gate 2 — implementation soundness

Test:

- exact equivalence between native and cLM-JEPA code with `λ=0`;
- predictor tokens cannot alter preceding source states or native logits;
- target changes cannot affect source states or native logits;
- source changes affect predictor states;
- correct target `<eos>` extraction;
- JEPA gradients reach the intended shared parameters;
- monitor-only produces no JEPA gradient;
- shuffled targets are reproducible and never equal the correct target;
- saved predictions reproduce all reported metrics offline.

### Gate 3 — frozen relationship assay

Compare `k ∈ {0,1,2,3,4}` without fine-tuning. For each `k`, measure:

- correct-pair versus random and matched-shuffle cosine margin;
- target variance, effective rank, and mean-direction energy;
- held-out linear or ridge source-to-target explained variance;
- retrieval against length- and heavy-atom-matched negatives.

These are **method-selection diagnostics**, not paper-level task outcomes. Use them to eliminate clearly unusable `k` values. Permit one train-set mean-centering and normalization rescue; if no `k` retains pair-specific signal, stop.

### Gate 4 — compute-efficient hyperparameter selection

Follow the original LLM-JEPA ordering:

1. **Tune the native learning rate first**, with JEPA disabled, and freeze it.
2. Keep all other native ChemFM/LoRA settings fixed unless the native baseline fails.
3. Use Gate 3 to retain at most two `k` values.
4. Jointly select JEPA weight and JEPA-loss dropout using:
   - `λ_eff ∈ {0.5,1,2,4}`;
   - dropout `α ∈ {0,0.5,0.75}`;
   - actual `λ = λ_eff/(1-α)`, following the LLM-JEPA observation that `λ(1-α)` should remain approximately constant;
   - `jepa_ratio = 1-α` in the ported LLM-JEPA implementation.

Avoid a full grid. Use a small TPE-guided Hyperband/ASHA study—or an equivalent scripted successive-halving procedure if that is simpler—with epoch budgets of 1, 2, and 4 and at most 12 JEPA trials. Rank trials by the prespecified validation generative metric, with validity as a hard guardrail. Replicate the top configuration on a second seed before freezing it.

Do not tune on retrieval, JEPA loss, or test performance.

### Gate 5 — control confirmation

With the frozen configuration, run native, monitor-only, cLM-JEPA, and matched-shuffled JEPA on two seeds. Run the JEPA-only diagnostic only on the tiny subset. Proceed only if cLM-JEPA beats native and shuffled directionally on both seeds without worse validity or native validation loss.

---

## 8. Task expansion and data efficiency

After the USPTO-MIT pilot passes:

1. run forward prediction on USPTO-STEREO and ORDerly-forward;
2. run human metabolite generation;
3. run USPTO-50K retrosynthesis;
4. run the strict OOD retrosynthesis benchmark only if USPTO-50K succeeds.

Use nested data subsets:

- forward: 1%, 5%, 10%, and 100%;
- metabolism: 10%, 25%, 50%, and 100%;
- retrosynthesis: 10% and 100%.

Run reduced fractions only after the native full-data pipeline is valid. Native and cLM-JEPA must use the same examples, order, and number of optimizer steps.

---

## 9. Fairness and checkpointing

Native, monitor-only, shuffled, and cLM-JEPA runs must share:

- checkpoint, tokenizer, LoRA modules/rank, and trainable parameter count;
- dataset manifest, serialization, and example order;
- optimizer, scheduler, batch, accumulation, and maximum epochs;
- validation frequency, early-stopping rule, decoding, and evaluator.

Report both:

1. **exposure-matched results:** equal examples and optimizer steps;
2. **compute-aware results:** model calls, effective tokens, JEPA-active batches, wall time, throughput, peak VRAM, and estimated FLOPs.

Monitor-only must perform the same JEPA computation as cLM-JEPA. Checkpoints are selected only by the frozen native-task validation metric.

---

## 10. Generative evaluation

Representation retrieval and mapping metrics are diagnostics only. The paper's evidence must come from generated chemical outputs.

### 10.1 Forward reaction prediction

Use ChemFM's beam generation and ranking logic in `main.py::evaluate` and `score.py`:

- beam size 10;
- top-1, top-3, top-5, and top-10 exact product accuracy after one frozen RDKit standardization/canonicalization pipeline;
- invalid-SMILES rate and unique valid candidates;
- stereochemistry-aware exact accuracy;
- connectivity-correct but stereochemistry-wrong rate;
- separate results on USPTO-STEREO, filtered ORDerly-forward, non-USPTO-forward, and template-held-out reactions.

For the USPTO-MIT compatibility condition, reproduce official R-SMILES augmentation/ranking. Do not use R-SMILES ranking for the principal nonaligned result unless it can be generated without changing the declared input policy.

### 10.2 Human metabolite generation

Evaluate at fixed top-5, top-10, and top-20 output windows, grouped by parent:

- fraction of parents with at least one, at least half, and all recorded metabolites recovered;
- metabolite recall@k and total recorded metabolites recovered;
- lower-bound precision@k and average output size;
- valid and unique prediction rates.

Primary identity uses exact standardized molecular structure with stereochemistry retained. Also report the MetaTrans historical criterion—fingerprint Tanimoto similarity exactly 1—as a secondary comparability metric, because it can ignore stereochemical or charge distinctions. Never describe measured precision as the true biochemical false-positive rate.

### 10.3 Retrosynthesis

After canonicalizing each precursor and comparing unordered precursor sets, report:

- top-1, top-3, top-5, and top-10 exact precursor-set accuracy;
- invalid and duplicate prediction rates;
- train-template, novel-template, and invalid-template fractions using the released OOD benchmark procedure;
- round-trip top-k recovery with one frozen forward model, clearly labeled as surrogate-dependent;
- a fixed, blinded manual chemistry review of a small prespecified OOD sample if the paper makes chemical-feasibility or novel-reaction claims.

Exact match remains the primary benchmark metric, but it cannot by itself establish that unmatched predictions are chemically wrong.

### 10.4 Statistics

- Pilot decisions: two fixed seeds.
- Final approved comparisons: five fixed seeds, matching LLM-JEPA's stability protocol.
- Report every seed, mean, standard deviation, paired effect size, and paired bootstrap confidence interval over test examples.
- Use two-sided paired tests for prespecified primary comparisons.

---

## 11. Representation and mechanism diagnostics

Compare pretrained, native, monitor-only, shuffled, and cLM-JEPA checkpoints using:

- native and JEPA loss curves;
- correct-pair versus matched-shuffle margins;
- target variance, effective rank, and mean-direction energy;
- held-out linear/ridge mapping;
- SVD of source-target differences;
- alternate-SMILES invariance;
- predictor sensitivity to removing or replacing a necessary reaction component.

These analyses explain results; they must not replace generative evaluation or determine checkpoints after the protocol is frozen.

---

## 12. Full-run protocol

### 12.1 Epoch calibration

Use fixed maximums from the relevant official ChemFM configurations as safety caps:

- forward: 20 epochs;
- retrosynthesis: 10 epochs;
- metabolism: 20 epochs;
- pilot/HPO: at most 4 epochs, with 1/2/4-epoch multi-fidelity budgets.

During the reduced pilot, evaluate validation generation after every epoch and use patience 3 after epoch 4 to identify a reasonable epoch count for each task/data fraction. Freeze that epoch count before full comparisons. Full native, monitor-only, shuffled, and cLM-JEPA runs then use the **same fixed number of epochs and optimizer steps**; do not stop conditions independently. Select each run's checkpoint using only its frozen validation metric.

Validation selectors:

- forward: top-1 exact product accuracy;
- metabolism: recall@5, with lower-bound precision@5 as tie-breaker;
- retrosynthesis: top-1 exact precursor-set accuracy.

Test evaluation occurs once on the selected checkpoint. Never copy ChemFM's current test-per-epoch behavior.

### 12.2 Required full comparisons

For each task/data fraction that passes its reduced pilot, run:

1. native;
2. monitor-only;
3. matched-shuffled JEPA;
4. cLM-JEPA.

Use five fixed seeds. Freeze learning rate, `k`, `lambda`, loss-dropout rate, LoRA configuration, batch/accumulation, scheduler, decoding, epoch count, and checkpoint rule before launch. Do not launch a full cell merely because it appears in the plan.

---

## 13. Logging and report

Use `WANDB_PROJECT=clm-jepa`; keep credentials only in environment variables.

Log native/JEPA/total loss, task metrics, validity, gradient norms, learning rate, JEPA-active batches, tokens, wall time, throughput, VRAM, seed, data fraction, and resolved hyperparameters. Avoid parameter histograms and frequent embedding dumps.

Generate a reproducible final Markdown/LaTeX report and PDF under `reports/` containing the exact protocol, every pilot and failure, all controls, generated-output metrics, low-data/OOD results, representation diagnostics, compute/cost, commands, and conclusions restricted to the evidence.

Interpretation:

- **Strong evidence:** cLM-JEPA improves at least two tasks, beats matched shuffling, preserves generation, repeats across seeds, and helps in low-data or OOD evaluation.
- **Narrow evidence:** only one task or one regime improves, or only representation geometry changes.
- **Negative evidence:** no leakage-free, noncollapsed cLM-JEPA condition improves generated outputs over native and shuffled controls.

---

## 14. Required reading

1. Huang, LeCun, Balestriero. **LLM-JEPA: Large Language Models Meet Joint Embedding Predictive Architectures.**
2. Official `galilai-group/llm-jepa` repository: `finetune.py`, `run.sh`, and `evaluate.py`.
3. Cai et al. **ChemFM as a scaling law guided foundation model pre-trained on informative chemicals.**
4. Official `TheLuoFengLab/ChemFM` reaction-prediction code: `main.py`, `utils.py`, `score.py`, `string_template.json`, and task configs.
5. Schwaller et al. **Molecular Transformer: A Model for Uncertainty-Calibrated Chemical Reaction Prediction.**
6. Zhong et al. **Root-aligned SMILES: A Tight Representation for Chemical Reaction Prediction.**
7. Litsa, Das, Kavraki. **Prediction of drug metabolites using neural machine translation.**
8. Toniato et al. **ORDerly: Data Sets and Benchmarks for Chemical Reaction Data.**
9. Bradshaw et al. **Challenging Reaction Prediction Models to Generalize to Novel Chemistry.**
10. Choe, Chen, Jung. **Assessing the extrapolation capability of template-free retrosynthesis models.**
11. Falkner, Klein, Hutter. **BOHB: Robust and Efficient Hyperparameter Optimization at Scale.**
