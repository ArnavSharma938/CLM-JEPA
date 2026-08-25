# V-JEPA 2.1 to causal ChemFM mapping

## Reference lock

- Paper: V-JEPA 2.1, arXiv:2603.14482v3 (37-page PDF, revised 2026-06-11).
- Code: `facebookresearch/vjepa2` commit
  `204698b45b3712590f06245fbfba32d3be539812`, specifically
  `app/vjepa_2_1/`.
- Maintained implementation: `src/vjepa2_1.py`; trainer condition:
  `clm_jepa_vjepa2_1`.

## Preserved mechanics and causal translation

| V-JEPA 2.1 mechanism | ChemFM translation |
|---|---|
| Masked spatiotemporal blocks | A causal product boundary whose suffix is replaced by train-only mask latents |
| Eight local masks at scale 0.15 and two global masks at scale 0.70 | Ten resumable suffix-mask identities: eight short 15% horizons and two long 70% horizons |
| Visible patch context | All source tokens, target marker, and teacher-forced product prefix before the boundary |
| Dense masked prediction | Per-token predictions of EMA ChemFM states throughout the true product suffix |
| Dense context prediction | Per-token predictions of EMA states throughout the visible causal prefix |
| Context proximity weight `1/sqrt(d_min)` | `1/sqrt(boundary - absolute_token_position)`, continuous across source and product-prefix tokens |
| Four equally spaced encoder levels | ChemFM block outputs 6, 11, 17, and 22 (HF hidden-state depths; block indices 5, 10, 16, 21) |
| Learned norm per level, concatenate/fuse | Four learned LayerNorm outputs concatenated, then `4D -> D -> 384` with GELU |
| 24-block, width-384, 12-head ViT predictor | Train-only noncausal 1D rotary transformer with the same depth, width, heads, MLP ratio, initialization, and residual scaling |
| Separate masked/context output heads | Separate linear projections from 384 to four concatenated ChemFM target levels |
| EMA target encoder | Fixed-`0.99925` Polyak EMA of trainable ChemFM encoder state and target level norms; frozen ChemFM weights are shared |
| Stop-gradient target | EMA forward runs under `no_grad`; no target parameter enters the optimizer |
| Dense L1 objective | Equal mean over four depth slices, tokens, and latent elements |
| Progressive context objective | Zero, then linear ramp, then coefficient 0.5; the paper's 15k--30k interval is scaled to 1/9--2/9 of the planned ChemFM update budget |

The reference implementation computes the context term as the mean of the
weighted elementwise error over visible tokens/elements. It does **not**
renormalize by the sum of proximity weights; the causal implementation follows
that executable reference behavior.

## Causality and generation invariants

ChemFM performs one ordinary full teacher-forced causal forward. Causal
attention makes every state before the sampled boundary exactly equal to the
same state from the physically truncated prefix. The latent predictor receives
those visible states and replaces every future student feature with the sampled
zero-initialized mask embedding before noncausal predictor attention. It can
therefore integrate all of `C_t`, but cannot read any representation computed
from the true `M_t` tokens.

The EMA target performs a second causal forward over the complete true
reaction. These full-sequence states are targets only. They are detached before
the dense L1 objective.

The historical LLM-JEPA vocabulary predictor tokens are not added in this
condition. ChemFM's tokenizer size, LM head, native target-token CE, decoding,
and generation code are unchanged. Setting dense JEPA weight to zero takes the
ordinary one-forward NTP path without sampling a mask.

## Memory and state

Duplicating all frozen ChemFM-1B parameters would waste approximately another
full model copy. The target therefore stores EMA copies only for trainable
backbone parameters (LoRA and trainable saved embedding state) and performs a
functional target call through the shared frozen backbone. The checkpoint
contains the student adapter through PEFT plus predictor, online/target norms,
EMA tensors, EMA update count, progressive schedule budget, and exact mask
sampler call index.

## Intentional sequence-specific adaptations

1. The 2D/3D mask geometry becomes multiscale causal suffix prediction; this is
   the only mask family that guarantees a valid autoregressive prefix.
2. RoPE uses absolute 1D token coordinates instead of video space-time
   coordinates.
3. The context schedule retains the reference shape and endpoint but is scaled
   proportionally to the much shorter controlled ChemFM runs. A literal 15k
   delay would make the defining context objective inactive for the entire
   pilot.
4. The target `<prostart>` marker remains visible, matching the real generation
   prompt. Molecular product tokens and product EOS form the eligible future.

No endpoint cosine/MSE/SIGReg, gradient surgery, stop-gradient student branch,
LLM-JEPA predictor tokens, or generation-time module is active in this
condition.
