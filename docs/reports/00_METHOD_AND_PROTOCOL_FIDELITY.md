# Method and protocol fidelity

## Scope

The ChemFM implementation preserves the default LLM-JEPA training computation while substituting reaction-specific serialization and endpoint readouts. The fidelity claim applies to the symmetric, unnormalized JEPA path used in the original cosine experiments and to the separately verified raw-MSE branch. It does not claim byte-for-byte identity with upstream code or cover unused upstream additive-mask, InfoNCE, normalized-MSE, or multi-token-predictor branches.

## Pinned references

| Component | Source | Commit/revision |
|---|---|---|
| LLM-JEPA | `references/llm-jepa/` | `ea0017c654ad917066ff32afc88276bea8ca5f7e` |
| ChemFM | `references/chemfm/` | `ee35b23d03de1a8e97b8e04dcdfb1d579de70f02` |
| ChemFM-1B | local model snapshot | `f99dc2e89726539bb9cf31b2e2b4360650bac6a8` |
| LeJEPA/SIGReg | official repository | `c293d291ca87cd4fddee9d3fffe4e914c7272052` |

## Preserved computation

- Native ChemFM next-token prediction remains active and generation is unchanged.
- Source and target sequences are evaluated in one concatenated model call.
- The source representation is either the appended predictor-token state (`k=1`) or the existing final source EOS state (`k=0`). The target is the final target EOS state.
- Symmetric conditions backpropagate the JEPA loss through both branches. The stop-gradient experiment detached only the target argument of JEPA; native NTP still updated the shared target pathway.
- The established auxiliary dropout is sampled with probability 0.5. Its outer coefficient is 2, so the expected coefficient of the primary JEPA prediction loss is 1.
- ChemFM tokenization, reaction serialization, LoRA modules, target masking, optimizer, scheduler, and canonical generation/scoring are retained unless a report states a controlled execution-only change.

The original cosine loss is `mean(1 - cos(z_source, z_target))`. The MSE ablation uses the official LLM-JEPA branch exactly: `mean((z_source - z_target)^2)`, without endpoint normalization.

## Verification

The executable parity suite covers predictor-token insertion, k=0/k=1 readouts, target masking, concatenated-row construction, symmetric and detached gradient paths, auxiliary dropout, cosine/MSE values, and native-loss preservation. The original golden comparison matched representation tensors, scalar JEPA loss, and relevant gradients. The MSE/SIGReg extension added direct tests for raw MSE, the two-view coefficient conversion, and streamed-versus-materialized SIGReg loss and gradients; the focused final suite reported `23 passed`.

Exact streamed SIGReg uses a no-graph sufficient-statistics pass followed by RNG-replayed recomputation and endpoint VJPs. It does not use a queue, stale embeddings, or an average of smaller-batch SIGReg losses. On the batch-16 equivalence case:

| Quantity | Result |
|---|---:|
| Direct / streamed SIGReg | 1.6730396748 / 1.6730396748 |
| Loss absolute error | 0 |
| Representation-gradient relative L2 error | `1.75e-8` |
| Parameter-gradient relative L2 error | 0 |
| Parameter-gradient cosine | 1.0000000000 |

## Evaluation protocols

Three evaluation paths serve different purposes:

1. Training-time validation in `src/train.py` is a small checkpoint-selection/completion check.
2. One-view frozen panels measure mechanism diagnostics and do not represent the official ChemFM benchmark.
3. The official endpoint evaluator uses one canonical reaction identity, all five official R-SMILES views, beam width 10 with ten candidates per view, ChemFM canonical product handling, and official reciprocal-rank aggregation. Metrics are computed per unique reaction identity.

Exact generated top-1 is the primary forward-reaction endpoint. Teacher-forced CE and representation metrics are diagnostics and cannot replace it.

## Scope limits

Gate 3 evaluated 7,168 frozen examples across seven datasets and retained both k=0 and k=1 for later testing. Completed training experiments remain limited to the reduced USPTO-MIT pilot plus a reduced GSM8K reference. The evidence does not cover full-scale USPTO-MIT training, MetaTrans, retrosynthesis, multiple training seeds for the selected endpoint, or objectives not explicitly tested.

## Implementation paths

- Trainer: `src/train.py`
- ChemFM integration and generation: `src/chemfm.py`
- JEPA/MSE/SIGReg: `src/jepa.py`
- Representation metrics: `src/representation_eval.py`, `src/metrics.py`
- Official five-view endpoint: `src/eval_uspto_mit_five_view_a6000.py`
- Pinned upstream code: `references/chemfm/`, `references/llm-jepa/`
