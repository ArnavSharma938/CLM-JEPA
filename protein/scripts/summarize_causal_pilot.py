#!/usr/bin/env python
"""Aggregate the eight paired replicates using the locked inference plan."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.pilot import load_protocol, sha256_file
from src.pilot_stats import (
    bh_adjust, exact_spearman_permutation_p, hierarchical_paired_ci,
    paired_summary, spearman_bootstrap_ci,
)


def load(path):
    return json.loads(path.read_text())


def result_path(root, replicate, arm, fraction, stage):
    return root / "evaluation" / f"replicate_{replicate}" / arm / (
        f"checkpoint_{fraction:03d}_{stage}.json"
    )


def paired_metric(payloads, getter):
    return paired_summary(
        [getter(row["native"]) for row in payloads],
        [getter(row["nextlat"]) for row in payloads],
    )


def per_sequence_generation_value(row, metric):
    """Return the record-level analogue of a generation summary metric."""
    if metric == "termination_rate":
        return float(row["terminated"])
    return row[metric]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/causal_pilot.json")
    parser.add_argument("--esmfold", type=Path)
    args = parser.parse_args()
    protocol = load_protocol(args.protocol)
    training_integrity = []
    for replicate in range(8):
        arms = {
            arm: load(args.root / "training" / f"replicate_{replicate}" / arm /
                      "training.json")
            for arm in ("native", "nextlat")
        }
        keys = ("seed", "protocol_sha256", "manifest_sha256", "initial_lora_sha256",
                "total_primary_residues", "total_directional_residues",
                "total_optimizer_steps")
        matched = {key: arms["native"][key] == arms["nextlat"][key] for key in keys}
        if not all(matched.values()):
            raise AssertionError({"replicate": replicate, "paired_mismatch": matched})
        training_integrity.append({
            "replicate": replicate, "paired_fields_match": matched,
            "native_final_lora_sha256": arms["native"]["final_lora_sha256"],
            "nextlat_final_lora_sha256": arms["nextlat"]["final_lora_sha256"],
            "primary_residues": arms["native"]["total_primary_residues"],
            "directional_residues_per_arm": arms["native"]["total_directional_residues"],
            "optimizer_steps": arms["native"]["total_optimizer_steps"],
        })
    fractions = (25, 50, 100)
    primary_by_fraction = {}
    paired_final = None
    for fraction in fractions:
        paired = []
        for replicate in range(8):
            paired.append({
                arm: load(result_path(args.root, replicate, arm, fraction, "primary"))
                for arm in ("native", "nextlat")
            })
        primary_by_fraction[str(fraction)] = {
            "bidirectional_ntp_loss": paired_metric(
                paired, lambda row: row["lm"]["bidirectional_ntp_loss"]
            ),
            "proteingym_macro_spearman": paired_metric(
                paired, lambda row: row["proteingym"]["macro_spearman"]
            ),
            "correct_probability": paired_metric(
                paired, lambda row: row["lm"]["correct_probability"]
            ),
            "top1": paired_metric(paired, lambda row: row["lm"]["top1"]),
        }
        if fraction == 100:
            paired_final = paired
    primary = {
        key: primary_by_fraction["100"][key]
        for key in (
            "bidirectional_ntp_loss", "proteingym_macro_spearman",
            "correct_probability", "top1",
        )
    }
    primary["bidirectional_perplexity"] = paired_metric(
        paired_final, lambda row: float(np.exp(row["lm"]["bidirectional_ntp_loss"]))
    )
    primary_test_keys = ("bidirectional_ntp_loss", "proteingym_macro_spearman")
    qvalues = bh_adjust([
        primary[key]["exact_sign_flip_p"] for key in primary_test_keys
    ])
    for key, qvalue in zip(primary_test_keys, qvalues):
        primary[key]["bh_q"] = qvalue
    primary["proteingym_macro_spearman"]["hierarchical_ci95"] = hierarchical_paired_ci(
        [[assay["spearman"] for assay in pair["native"]["proteingym"]["assays"]]
         for pair in paired_final],
        [[assay["spearman"] for assay in pair["nextlat"]["proteingym"]["assays"]]
         for pair in paired_final],
    )
    robustness = {
        "lm_orientation": {
            key: paired_metric(paired_final, lambda row, metric=key: row["lm"][metric])
            for key in ("forward_ntp_loss", "reverse_ntp_loss")
        },
        "lm_length_bins": {
            name: paired_metric(
                paired_final,
                lambda row, bin_name=name: row["lm"]["length_bins"][bin_name][
                    "bidirectional_ntp_loss"
                ],
            )
            for name in ("64_159", "160_319", "320_512")
        },
    }
    robustness["lm_orientation"].update({
        f"{orientation}_perplexity": paired_metric(
            paired_final,
            lambda row, metric=f"{orientation}_ntp_loss": float(
                np.exp(row["lm"][metric])
            ),
        )
        for orientation in ("forward", "reverse")
    })
    categories = sorted(set.intersection(*[
        set(pair["native"]["proteingym"]["categories"])
        & set(pair["nextlat"]["proteingym"]["categories"])
        for pair in paired_final
    ]))
    robustness["proteingym_categories"] = {
        category: paired_metric(
            paired_final,
            lambda row, name=category: row["proteingym"]["categories"][name],
        ) for category in categories
    }

    representation_pairs = [
        {
            arm: load(result_path(args.root, replicate, arm, 100, "representations"))
            for arm in ("native", "nextlat")
        } for replicate in range(8)
    ]
    representations = {
        "swissprot_ec_accuracy": paired_metric(
            representation_pairs,
            lambda row: row["representations"]["swissprot_ec"]["forward_test_accuracy"],
        ),
        "swissprot_ec_top5": paired_metric(
            representation_pairs,
            lambda row: row["representations"]["swissprot_ec"]["forward_test_top5"],
        ),
        "swissprot_ec_orientation_delta": paired_metric(
            representation_pairs,
            lambda row: row["representations"]["swissprot_ec"][
                "test_orientation_accuracy_delta_reverse_minus_forward"
            ],
        ),
        "secondary_macro_accuracy": paired_metric(
            representation_pairs,
            lambda row: row["representations"]["secondary_structure"]["macro_accuracy"],
        ),
        "contact_macro_average_precision": paired_metric(
            representation_pairs,
            lambda row: row["representations"]["long_range_contact"][
                "protein_macro_average_precision"
            ],
        ),
    }

    generation_pairs = [
        {
            arm: load(result_path(args.root, replicate, arm, 100, "generation"))
            for arm in ("native", "nextlat")
        } for replicate in range(8)
    ]
    generation_keys = (
        "termination_rate", "length", "canonical_fraction", "entropy",
        "max_homopolymer", "unique_bigram_fraction", "composition_l1",
        "rita_bidirectional_fitness",
    )
    generation = {
        key: paired_metric(
            generation_pairs, lambda row, metric=key: row["generation"]["summary"][metric]
        ) for key in generation_keys
    }

    for key in generation_keys:
        native_nested, nextlat_nested = [], []
        for pair in generation_pairs:
            native_by_id = {
                row["id"]: per_sequence_generation_value(row, key)
                for row in pair["native"]["generation"]["records"]
            }
            nextlat_by_id = {
                row["id"]: per_sequence_generation_value(row, key)
                for row in pair["nextlat"]["generation"]["records"]
            }
            identifiers = sorted(set(native_by_id) & set(nextlat_by_id))
            identifiers = [identifier for identifier in identifiers
                           if native_by_id[identifier] is not None
                           and nextlat_by_id[identifier] is not None]
            native_nested.append([native_by_id[identifier] for identifier in identifiers])
            nextlat_nested.append([nextlat_by_id[identifier] for identifier in identifiers])
        generation[key]["hierarchical_ci95"] = hierarchical_paired_ci(
            native_nested, nextlat_nested, seed=20260915 + generation_keys.index(key),
        )

    mechanism = {}
    coupling_rows = []
    for fraction in (0, 25, 50, 100):
        rows = [
            load(args.root / "mechanism" / f"replicate_{replicate}" /
                 f"checkpoint_{fraction:03d}.json")
            for replicate in range(8)
        ]
        transition_metrics = {
            "transition_smooth_l1": "smooth_l1",
            "transition_kl": "kl",
            "transition_faithful_total": "faithful_total",
            "transition_relative_js": "decoder_transition_relative_js",
            "transition_decoder_top1_agreement": "decoder_top1_agreement",
        }
        mechanism[str(fraction)] = {
            **{
                output_name: paired_summary(
                    [row["summary"]["native"][input_name] for row in rows],
                    [row["summary"]["nextlat"][input_name] for row in rows],
                )
                for output_name, input_name in transition_metrics.items()
            },
            "gradient": {
                part: {
                    metric: paired_summary(
                        [0.0] * len(rows),
                        [row["gradient_summary"][part][metric]["mean"] for row in rows],
                    )
                    for metric in ("norm_ratio", "cosine",
                                   "ntp_direction_retention", "negative_cosine_fraction")
                }
                for part in ("global", "early", "middle", "late")
            },
            "hidden_state_drift": {
                part: paired_summary(
                    [float(np.mean([record["arms"]["native"]["drift"][part]
                                    for record in row["records"]])) for row in rows],
                    [float(np.mean([record["arms"]["nextlat"]["drift"][part]
                                    for record in row["records"]])) for row in rows],
                ) for part in ("early", "middle", "late")
            },
        }
        if fraction:
            for replicate, row in enumerate(rows):
                pair = {
                    arm: load(result_path(args.root, replicate, arm, fraction, "primary"))
                    for arm in ("native", "nextlat")
                }
                coupling_rows.append({
                    "replicate": replicate, "fraction": fraction,
                    "predictability_gain": (
                        row["summary"]["native"]["faithful_total"]
                        - row["summary"]["nextlat"]["faithful_total"]
                    ),
                    "ntp_gain": (
                        pair["native"]["lm"]["bidirectional_ntp_loss"]
                        - pair["nextlat"]["lm"]["bidirectional_ntp_loss"]
                    ),
                    "proteingym_gain": (
                        pair["nextlat"]["proteingym"]["macro_spearman"]
                        - pair["native"]["proteingym"]["macro_spearman"]
                    ),
                })
    coupling = {
        endpoint: {
            "spearman_all_replicate_checkpoints": float(spearmanr(
                [row["predictability_gain"] for row in coupling_rows],
                [row[endpoint] for row in coupling_rows],
            ).statistic),
            "n": len(coupling_rows),
        }
        for endpoint in ("ntp_gain", "proteingym_gain")
    }
    final_predictability = [row["predictability_gain"] for row in coupling_rows
                            if row["fraction"] == 100]
    final_endpoints = {
        "ntp_gain": [row["ntp_gain"] for row in coupling_rows if row["fraction"] == 100],
        "proteingym_gain": [row["proteingym_gain"] for row in coupling_rows
                            if row["fraction"] == 100],
        "swissprot_ec_accuracy_gain": [
            pair["nextlat"]["representations"]["swissprot_ec"]["forward_test_accuracy"]
            - pair["native"]["representations"]["swissprot_ec"]["forward_test_accuracy"]
            for pair in representation_pairs
        ],
        "secondary_accuracy_gain": [
            pair["nextlat"]["representations"]["secondary_structure"]["macro_accuracy"]
            - pair["native"]["representations"]["secondary_structure"]["macro_accuracy"]
            for pair in representation_pairs
        ],
        "contact_ap_gain": [
            pair["nextlat"]["representations"]["long_range_contact"]["protein_macro_average_precision"]
            - pair["native"]["representations"]["long_range_contact"]["protein_macro_average_precision"]
            for pair in representation_pairs
        ],
        "generation_fitness_gain": [
            pair["nextlat"]["generation"]["summary"]["rita_bidirectional_fitness"]
            - pair["native"]["generation"]["summary"]["rita_bidirectional_fitness"]
            for pair in generation_pairs
        ],
    }
    for metric in ("termination_rate", "entropy", "max_homopolymer",
                   "unique_bigram_fraction", "composition_l1"):
        final_endpoints[f"generation_{metric}_nextlat_minus_native"] = [
            pair["nextlat"]["generation"]["summary"][metric]
            - pair["native"]["generation"]["summary"][metric]
            for pair in generation_pairs
        ]
    coupling["final_checkpoint_across_replicates"] = {
        endpoint: {
            "spearman": float(spearmanr(final_predictability, values).statistic),
            "ci95": spearman_bootstrap_ci(
                final_predictability, values, 20260915 + index
            ),
            "exact_permutation_p": exact_spearman_permutation_p(
                final_predictability, values
            ),
            "n": len(values),
        } for index, (endpoint, values) in enumerate(final_endpoints.items())
    }

    efficiency = []
    for replicate in range(8):
        native_payload = load(result_path(
            args.root, replicate, "native", 100, "primary"
        ))
        native_final = native_payload["lm"]["bidirectional_ntp_loss"]
        native_pg_final = native_payload["proteingym"]["macro_spearman"]
        reached, reached_pg = None, None
        for fraction in fractions:
            payload = load(result_path(
                args.root, replicate, "nextlat", fraction, "primary"
            ))
            value = payload["lm"]["bidirectional_ntp_loss"]
            if value <= native_final:
                reached = reached if reached is not None else fraction / 100
            if payload["proteingym"]["macro_spearman"] >= native_pg_final:
                reached_pg = reached_pg if reached_pg is not None else fraction / 100
        efficiency.append({"replicate": replicate,
                           "nextlat_fraction_reaching_native_final_lm": reached,
                           "nextlat_fraction_reaching_native_final_proteingym": reached_pg})
    esmfold = None
    if args.esmfold and args.esmfold.exists():
        raw_esmfold = load(args.esmfold)["results"]
        esm_pairs = []
        for replicate in range(8):
            matched = {}
            for arm in ("native", "nextlat"):
                suffix = str(result_path(
                    args.root, replicate, arm, 100, "generation"
                )).replace("\\", "/")
                candidates = [value for key, value in raw_esmfold.items()
                              if key.replace("\\", "/").endswith(suffix)]
                if len(candidates) != 1:
                    raise ValueError(f"ESMFold result mismatch: {suffix}")
                matched[arm] = candidates[0]
            esm_pairs.append(matched)
        esmfold = {
            metric: paired_metric(
                esm_pairs, lambda row, name=metric: row["summary"][name]
            )
            for metric in ("mean_plddt", "fraction_plddt_ge_70",
                           "nonlocal_ca_clash_fraction")
        }
    output = {
        "protocol_sha256": sha256_file(args.protocol),
        "summarizer_sha256": sha256_file(Path(__file__)),
        "training_integrity": training_integrity,
        "primary_final": primary,
        "robustness": robustness,
        "learning_curves": primary_by_fraction,
        "data_efficiency": efficiency,
        "representations": representations,
        "generation": generation,
        "esmfold": esmfold,
        "mechanism": mechanism,
        "predictability_behavior_coupling": coupling,
        "replicate_count": 8,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
