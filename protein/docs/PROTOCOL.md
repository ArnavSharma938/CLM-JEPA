# RITA-M frozen diagnostic protocol

## Frozen scope

RITA-M is never updated. The only learned modules are frozen-state probes and
the allowed predictor-only NextLat warmup. STP/NextLat backbone fine-tuning,
Native LoRA, sweeps, and larger RITA confirmations are out of scope.

## Token contract

The pinned tokenizer emits one token for every canonical residue and appends
one `<EOS>` token. It emits no BOS. Its artifact contains `<PAD>` id 1 and
`<EOS>` id 2 but publishes an empty special-token map, so padding is constructed
explicitly and neither generic special-token attribute is trusted.

For a residue sequence `x[0:L]`, hidden state `h[t]` occupies the same token
position as `x[t]`. Eligible residue transitions are `h[t] -> x[t+1]` and
`h[t] -> h[t+1]` for `0 <= t < L-1`. The first residue is not scored because
there is no BOS state. EOS and PAD are excluded from STP and NextLat positions.

## Independent units

Primary intervals and paired tests resample proteins or complete 30%-identity
clusters, never individual residue positions. Structural stratum statistics
are first reduced within protein. ProteinGym statistics use assays as the
independent unit.

## Locked minimal analyses

- final hidden state, plus decoder block 12 only for the state-sufficiency test;
- curvature, adjacent tangent correlation, three span bins, normalized tube
  radius, and path efficiency;
- helix, beta, coil, secondary-structure-transition, and low/high long-range
  contact strata;
- forward/reverse comparison on a deterministically selected subset;
- 10% chord-orthogonal attenuation with hidden-norm restoration;
- ridge and one width-controlled nonlinear future-state predictor;
- final-state A/B/C/C-shuffle decomposition and one block-12 state control;
- released-STP and predictor-warmed faithful-NextLat gradient audits;
- 100 base-model generations under one sampling configuration.
