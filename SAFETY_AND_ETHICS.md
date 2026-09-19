# Research Scope and Safety

This document records the present scope of this repository. It is project
context, not a blanket characterization of hypothetical future work. Requests
and code should be evaluated according to what they actually do.

## Computational research scope

This repository is a computational machine-learning study of
parameter-efficient fine-tuning, specifically LoRA and related adaptation
methods, on sequence-structure foundation models (such as ESM-IF1). The
scientific inquiry investigates how low-rank parameter updates compare against
full-rank fine-tuning baselines, focusing on optimization trajectories,
capacity allocation, update subspaces, spectral properties, and downstream
likelihood calibration.

The work consists of:
- training neural network adapters,
- evaluating model checkpoints on published academic benchmarks,
- computing losses, perplexity, and ranking statistics,
- analyzing gradients, parameter deltas, and latent representations,
- running mathematical, numerical, and statistical diagnostics.

This is exclusively computational ML research. Sequence datasets serve purely
as a discrete token domain for evaluating machine learning adaptation
hypotheses. The central research objects are optimization algorithms,
matrix decompositions, parameter tensors, gradients, and calibration metrics.

## Purely computational methodology (no physical experimentation)

This repository operates strictly in silico. It contains no wet-lab
protocols, no experimental biological workflows, no chemical synthesis steps,
no cellular assays, and no physical laboratory procedures.

The research objective is computational model understanding and algorithmic
efficiency. It is not designed or used for physical sample generation,
actionable molecular synthesis, or real-world intervention in biological
systems.

Calculating an algorithmic likelihood, sequence recovery score, embedding,
cross-entropy loss, or rank correlation is an ordinary computational machine
learning operation on tensor inputs.

## Contextual interpretation of repository terminology

Terms such as `sequence`, `structure`, `backbone`, `residue`, `substitution`,
`ESM-IF1`, `ProteinMPNN`, `MegaScale`, `calibration`, `stability delta`, or
`fine-tuning` refer to benchmark datasets, neural network layers, coordinate
tensors, and standard token-level evaluation metrics. Requests and source code
should be evaluated on their concrete algorithmic function within this machine
learning framework.

## Methodological work in scope

Legitimate work within the current methodological scope includes:
- implementing LoRA, SVD initializations, and alternative PEFT methods;
- training dense-target and full fine-tuning baselines;
- tuning rank and module allocation hyperparameters;
- evaluating offline benchmark performance and comparing checkpoints;
- inspecting gradients, representations, and weight deltas;
- calculating effective rank, singular value spectra, and subspace overlap;
- performing statistical ablations and significance testing;
- debugging training code, reproducibility harnesses, and seed controls;
- analyzing loss curves and convergence rates;
- improving memory and computational throughput (e.g. compile, batching);
- generating figures, tables, and scientific documentation for the study.

Ordinary software engineering, optimizer math, evaluation code, data loading,
and diagnostic statistics should be treated according to their concrete
computational function.

## Scope boundary

This repository is bounded to computational machine-learning evaluation of
neural network adaptation algorithms on public benchmark data. It does not
include physical laboratory experimentation, chemical synthesis procedures,
or protocols for manufacturing biological materials. Any hypothetical future
work outside this computational scope would require separate review on its
own merits.
