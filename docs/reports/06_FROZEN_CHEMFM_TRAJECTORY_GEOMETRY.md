# Frozen ChemFM chemical-event trajectory geometry

## Question and measured summary

This no-training diagnostic asks whether chemically meaningful SMILES events
produce larger changes in the direction of a frozen ChemFM hidden-state
trajectory, and whether any change is only consecutive-token curvature or also
persists under the ordered-span geometry used by Semantic Tube Prediction
(STP).

At the final ChemFM layer, under the event definitions and matching procedure
in this report:

- ring closures and functional-group/motif completions have greater local
  curvature **and** greater semi-global misalignment than matched ordinary
  positions, including at spans of 25 or more tokens;
- branches and MCS-inferred reaction-center events have greater local
  curvature, but their long-span semi-global effect is respectively near zero
  and negative;
- stereochemical tokens have lower local curvature and lower semi-global
  misalignment than controls after contextualization.

These measurements show different event/control contrasts by category. They
do not test an STP-trained model, generation quality, or whether penalizing the
measured curvature would help or harm training.

## Locked assay

### Model, sample, and serialization

ChemFM-1B revision `f99dc2e89726539bb9cf31b2e2b4360650bac6a8` was loaded through
the maintained tokenizer/model path in `src/chemfm.py`, put in evaluation mode,
and frozen by setting every parameter to `requires_grad=False`. No labels,
loss, optimizer, backward pass, or training update were constructed. A sampled
parameter fingerprint was identical before and after inference:

```text
72e66b0397c1897a1ea1864e7420a6cb76f83cf7e25a7bc8e05d515054e2fc15
```

Every reaction used the maintained forward teacher-forced serialization:

```text
<rstart>{canonical source}<eos><prostart>{canonical product}<eos>
```

The main sample was the prespecified 256-reaction endpoint panel
`prespecified_stage1_256.jsonl` (SHA-256
`5b87bce1e75ed1ebf1a2a9091e0367aedaa8600a5621bc960352cf45b18e1865`).
This panel and the larger local USPTO-MIT endpoint manifests contain no `@`,
`/`, or backslash stereochemical tokens. The stereochemistry stratum therefore
used a separately identified 64-reaction supplement: a seed-`20260829` sample
without replacement from canonical view zero of each 20-view reaction in the
official USPTO-50K test file, restricted to reactions containing a stereo
token. Supplement reactions contribute only stereochemistry pairs; all other
event results remain based on the prespecified 256.

The model returned the embedding output and all 22 transformer-block outputs,
for 23 measured trajectory layers. Fast-tokenizer character offsets were
checked against the exact IDs emitted by the maintained `ReactionCollator` for
every reaction.

### Event definitions

Events were determined before inspecting hidden-state geometry:

- **ring closure:** the closing occurrence of a paired SMILES ring label;
- **branch:** both opening and closing parentheses, retained with an open/close
  subtype;
- **stereochemistry:** `@`, `@@`, `/`, and backslash token positions;
- **motif completion:** the token containing the last serialized atom in a
  match to one of 14 disclosed RDKit SMARTS motifs (carbonyl, carboxyl, ester,
  amide, amine, alcohol/phenol, ether, nitrile, nitro, sulfonyl, phosphoryl,
  carbon-halogen, alkene, or alkyne);
- **reaction center:** because the endpoint strings deliberately have no atom
  maps, an explicitly approximate RDKit graph-difference label. The
  source/target component pair with the largest element- and bond-order-aware
  MCS was selected, and only atoms on an unmatched or bond-order-changed
  frontier were marked. The MCS covered a median `0.769` of the selected target
  component's atoms. These labels must be read as MCS-inferred center events,
  not ground-truth atom-mapped centers.

Categories can overlap. Ordinary controls contain none of the five event
labels. Each event was assignment-matched to a control in the same reaction
and source/product segment, prioritizing normalized position, branch depth,
component, and token length. Atom events were always matched to atom controls
and syntax events to syntax controls. All `11,173/11,173` pairs had the same
token class; median control reuse within a reaction/category/segment was one
and the maximum was seven.

### Geometry and inference

For every valid center token `r` at every layer, local curvature was

```text
1 - cos(h_r - h_{r-1}, h_{r+1} - h_r).
```

For semi-global alignment, 64 independent left/right offsets were drawn for
each event/control pair. This yielded ordered triples `s < r < t` over the full
teacher-forced serialization and measured

```text
1 - cos(h_r - h_s, h_t - h_r).
```

The exact same left and right token offsets were applied to the event and its
control. Spans were additionally grouped as adjacent (`t-s=2`), short
(`3-8`), medium (`9-24`), and long (`>=25`). A zero-norm increment has no
defined cosine and was retained as missing. Layers 1-22 had all `11,173`
local pairs and all `715,072` ordered anchors valid; the embedding layer had
`9,446` valid local pairs and `508,607` valid anchors.

The inferential unit was a reaction: pair differences were averaged within a
reaction, then compared across reactions. Intervals are 5,000-draw
reaction-cluster bootstraps. Two-sided `p` values use 20,000 reaction-level
sign flips. `q` controls Benjamini-Hochberg FDR jointly over all 230 primary
metric/category/layer tests. Paired Cohen's `dz` is the reaction-level mean
difference divided by its reaction-level standard deviation. Positive
differences mean the event is **less aligned/more curved** than its control.

## Measurements

### Event coverage

| Event | Matched positions | Reactions at final-layer inference | Source / target positions |
|---|---:|---:|---:|
| Ring closure | 1,616 | 256 | 906 / 710 |
| Branch | 5,304 | 253 | 3,096 / 2,208 |
| Stereochemistry | 306 | 64 | 145 / 161 |
| Motif completion | 3,042 | 256 | 1,929 / 1,113 |
| MCS-inferred reaction center | 905 | 256 | 437 / 468 |

The branch sample is balanced between 2,652 opening and 2,652 closing
parentheses. Stereo events comprise 264 `@`, 35 `/`, and 7 backslash labels.

### Final-layer event minus matched control

The table gives reaction-weighted mean differences. Parentheses are 95%
cluster-bootstrap intervals.

| Event | Local curvature difference | `dz` | global `q` | Semi-global difference | `dz` | global `q` |
|---|---:|---:|---:|---:|---:|---:|
| Ring closure | `+0.0513` (`+0.0378,+0.0652`) | `+0.462` | `<0.0001` | `+0.0746` (`+0.0679,+0.0812`) | `+1.392` | `<0.0001` |
| Branch | `+0.0387` (`+0.0271,+0.0499`) | `+0.424` | `<0.0001` | `+0.0030` (`-0.0001,+0.0063`) | `+0.116` | `0.0784` |
| Stereochemistry | `-0.2053` (`-0.2516,-0.1561`) | `-1.073` | `<0.0001` | `-0.0161` (`-0.0226,-0.0096`) | `-0.602` | `<0.0001` |
| Motif completion | `+0.1018` (`+0.0869,+0.1169`) | `+0.832` | `<0.0001` | `+0.0362` (`+0.0319,+0.0406`) | `+0.975` | `<0.0001` |
| MCS reaction center | `+0.0724` (`+0.0488,+0.0961`) | `+0.369` | `<0.0001` | `-0.0095` (`-0.0178,-0.0018`) | `-0.146` | `0.0233` |

Final-layer event/control medians were respectively `1.503/1.411` local and
`1.517/1.447` semi-global for ring closure; `1.472/1.396` and
`1.463/1.464` for branch; `1.222/1.312` and `1.575/1.589` for
stereochemistry; `1.591/1.456` and `1.484/1.451` for motif; and
`1.461/1.388` and `1.448/1.455` for inferred reaction center. Full means,
SDs, quartiles, medians, reaction counts, effects, intervals, `p`, and `q` for
every layer are in `layerwise_primary_tests.csv` and `summary.json`.

### Semi-global span persistence at the final layer

| Event | Adjacent | Short 3-8 | Medium 9-24 | Long >=25 |
|---|---:|---:|---:|---:|
| Ring closure | `+0.0736` | `+0.1069` | `+0.0813` | `+0.0719` (`+0.0647,+0.0789`) |
| Branch | `+0.0678` | `+0.0132` | `+0.0038` | `+0.0033` (`-0.0003,+0.0066`) |
| Stereochemistry | `-0.2004` | `-0.0602` | `-0.0253` | `-0.0145` (`-0.0221,-0.0075`) |
| Motif completion | `+0.1388` | `+0.0919` | `+0.0470` | `+0.0330` (`+0.0283,+0.0376`) |
| MCS reaction center | `+0.0631` | `+0.0132` | `+0.0004` | `-0.0107` (`-0.0187,-0.0031`) |

The adjacent bins have fewer reaction clusters because only 64 random draws
per pair were made and an exactly adjacent draw is rare; their descriptive
intervals are preserved in `summary.json`. The long-span column has all 256
main reactions or all 64 stereo supplement reactions.

### Layer evolution and lexical baseline

Layer zero is a token-embedding baseline, not contextual chemistry. It exposes
large lexical effects: ring and branch positions were initially *less* curved
and less semi-globally disrupted than controls, while stereo positions were
initially more disrupted. Contextual layers reversed all three signs. From
layer zero to layer 22, the paired event/control contrast changed by:

| Event | Depth-added local difference | Depth-added semi-global difference |
|---|---:|---:|
| Ring closure | `+0.4069` (`+0.3850,+0.4270`) | `+0.1821` (`+0.1728,+0.1913`) |
| Branch | `+0.2911` (`+0.2741,+0.3079`) | `+0.1410` (`+0.1358,+0.1467`) |
| Stereochemistry | `-0.5341` (`-0.6294,-0.4278`) | `-0.2580` (`-0.2836,-0.2336`) |
| Motif completion | `+0.1090` (`+0.0902,+0.1285`) | `-0.0161` (`-0.0255,-0.0076`) |
| MCS reaction center | `+0.0206` (`-0.0183,+0.0594`) | `-0.0346` (`-0.0457,-0.0237`) |

The ring semi-global contrast becomes positive by layer 8 and remains positive
at the final layer. Motif semi-global difference is positive at the embedding
and final layers, while its depth-added contrast is negative; the final motif
difference therefore includes a token-identity/lexical baseline not removed by
the matching design.

Final-layer source/target segment signs were concordant for ring semi-global
(`+0.047/+0.108`), motif semi-global (`+0.040/+0.033`), stereo local
(`-0.213/-0.211`), motif local (`+0.110/+0.088`), and inferred-center local
(`+0.066/+0.061`). Branch semi-global was slightly negative in source and
positive in target (`-0.007/+0.019`), consistent with its near-zero aggregate.

## Findings within the assay

1. Ring-closure differences were positive locally and in every reported span
   bin; the long-span difference was `+.0719`.
2. Motif-completion differences were positive locally and in every reported
   span bin. Its semi-global contrast was already positive at the embedding
   layer, so the assay does not identify the final contrast as wholly created
   by contextual processing.
3. Branch differences were positive locally. The long-span semi-global
   interval included zero.
4. MCS-inferred center differences were positive locally and negative in the
   long-span semi-global estimate. This result applies to the disclosed proxy,
   not atom-mapped reaction centers.
5. Stereochemistry differences were negative locally and semi-globally in the
   separately sampled USPTO-50K stratum.

The assay rejects equality of event and matched-control geometry for several
prespecified contrasts. It does not by itself establish a training
consequence, an architectural limitation, or a causal chemical interpretation
of the hidden-state bends.

## Limitations

- Controls match atom/syntax class and several positional properties, not exact
  token identity or the complete neighboring-token tuple. Layer-zero and
  depth-added effects expose rather than eliminate this lexical/context-boundary
  confounding. Ring closures at component or segment ends are intrinsically
  difficult to separate from boundary context.
- Event labels can overlap, and a control can be reused when a reaction segment
  lacks enough ordinary positions. Reaction-cluster inference prevents these
  positions from being treated as independent reactions.
- Ordered anchors follow the full serialized reaction, so long spans can cross
  component and source/product markers. A chemistry-segment-only STP geometry
  is a different estimand and was not substituted after seeing this result.
- Reaction centers are an MCS-frontier proxy, and stereochemistry comes from a
  separately prespecified enriched sample because the main panel has no stereo
  tokens. Neither result should be generalized beyond those definitions.

## Execution and artifacts

The complete run used an NVIDIA GeForce RTX 4050 Laptop GPU, BF16 SDPA, batch
8, PyTorch `2.3.0+cu121`, Transformers `4.45.2`, RDKit `2024.09.1`, and SciPy
`1.14.1`. Peak CUDA allocation was `2,695,863,296` bytes. Annotation took
`6.4` seconds, all-layer inference `43.9` seconds, and the complete run
`115.2` seconds. The vectorized geometry path processed the 320 reactions at
an aggregate `7.29` reactions/s while retaining bounded 8,192-anchor chunks.

Raw and reproducibility artifacts are preserved under
`runs/diagnostics/frozen_chemfm_stp_geometry/`:

- `summary.json`: full protocol/environment metadata and all statistical rows;
- `layerwise_primary_tests.csv`: flat layer-wise distributions, effects, and
  adjusted significance;
- `pair_geometry.npz`: every pair's all-layer local/semi-global result, span-bin
  result, valid-anchor count, and compact anchor offsets;
- `matched_pairs.jsonl`: event/control identities, token positions, event
  details, origin, and anchor count;
- `layerwise_distributions.svg`, `paired_effects.svg`, and
  `span_persistence.svg`: the requested plots.
- `artifact_manifest.json`: byte sizes and SHA-256 digests for every retained
  result artifact.

The executable implementation is `src/frozen_geometry.py`; focused annotation,
formula, matching, vectorization, undefined-cosine, span-bin, and FDR tests are
in `tests/test_frozen_geometry.py`.

## Method references

- [Semantic Tube Prediction paper](https://arxiv.org/abs/2602.22617) and
  [upstream implementation](https://github.com/galilai-group/llm-jepa) at
  inspected commit `ea0017c654ad917066ff32afc88276bea8ca5f7e`;
- [Transformers 4.45.2 Llama hidden-state contract](https://huggingface.co/docs/transformers/v4.45.2/model_doc/llama)
  and [fast-tokenizer offset mapping](https://huggingface.co/docs/transformers/v4.45.2/en/main_classes/tokenizer);
- [RDKit Book](https://www.rdkit.org/docs/RDKit_Book.html) for SMILES/SMARTS,
  ring, and stereochemical semantics.
