#!/usr/bin/env python
"""Execute a tiny real-checkpoint parity check against ProteinGym's RITA source."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.constants import CANONICAL_AA, RITA_FITNESS_SOURCE
from src.fitness import official_rita_fitness
from src.modeling import load_rita


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("official_source", type=Path)
    parser.add_argument("output", type=Path); args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("upstream_rita_fitness", args.official_source)
    upstream = importlib.util.module_from_spec(spec); spec.loader.exec_module(upstream)
    loaded = load_rita(); sequence = CANONICAL_AA
    observed = official_rita_fitness(loaded.model, loaded.tokenizer, sequence, "cuda")
    expected = float(upstream.calc_fitness(loaded.model, np.asarray([sequence]), loaded.tokenizer)[0])
    difference = abs(observed - expected)
    # Concurrent FP16 CUDA reductions are not guaranteed bitwise identity; this
    # tolerance is <0.1% of the two-direction score and catches alignment errors.
    tolerance = 5e-3
    if difference > tolerance: raise AssertionError(f"official RITA parity failed: {observed} vs {expected}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"sequence": sequence, "local_score": observed,
        "upstream_score": expected, "absolute_difference": difference, "absolute_tolerance": tolerance,
        "upstream_source": RITA_FITNESS_SOURCE, "passed": True}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__": main()
