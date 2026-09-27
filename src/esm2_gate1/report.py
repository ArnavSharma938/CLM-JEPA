from __future__ import annotations

import argparse
import json
from pathlib import Path


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _trajectory(history: list[dict], tolerance: float = 1e-4) -> str:
    values = [float(item["validation_nll"]) for item in history[-4:]]
    if len(values) < 4:
        return "insufficient"
    differences = [right - left for left, right in zip(values, values[1:])]
    if all(value < -tolerance for value in differences):
        return "improving"
    if all(value > tolerance for value in differences):
        return "deteriorating"
    if all(abs(value) <= tolerance for value in differences):
        return "plateaued"
    return "mixed"


def _fmt(value: float | None) -> str:
    return "not run" if value is None else f"{value:.6f}"


def build_report(manifests: Path, runs: Path, output: Path) -> None:
    corpus = _json(manifests / "corpus_audit.json")
    distance = _json(manifests / "distance_audit.json")
    preflight = _json(runs / "preflight.json")
    parity = _json(runs / "fair_esm_parity.json")
    backend = _json(runs / "backend_benchmark.json")
    execution_profile = _json(runs / "execution_profile.json")
    gate = _json(runs / "gate1_summary.json")
    lines = [
        "# ESM-2-650M BackboneRef Gate-1 results",
        "",
        "## Provenance and corpus",
        "",
        f"- Dayhoff revision: `{corpus['dataset_revision']}`",
        f"- ESM-2 revision: `{preflight['source']['resolved_model_revision']}`",
        f"- ESM-2 weight SHA-256: `{preflight['source']['model_weight_sha256']}`",
        f"- Repository commit / Gate-1 source hash: `{preflight['repository_commit']}` / "
        f"`{preflight['esm2_gate1_source_sha256']}`",
        f"- BRn / BRq / intersection backbones: {corpus['brn_unique_backbones']:,} / "
        f"{corpus['brq_unique_backbones']:,} / {corpus['brn_brq_intersection_backbones']:,}",
        f"- Malformed / invalid-length / exact-duplicate / low-complexity removals: "
        f"{corpus['malformed_records_removed']:,} / {corpus.get('invalid_length_records_removed', 0):,} / "
        f"{corpus['exact_duplicates_across_backbones_removed']:,} / "
        f"{corpus['low_complexity_backbones_removed']:,}",
        f"- Low-complexity removals from the preferred BRn intersection: "
        f"{corpus['low_complexity_preferred_backbones_removed']:,}",
        f"- BRn-only fallback backbones: {corpus['brn_only_fallback_backbones_added']:,}",
        f"- Structure ZIP SHA-256: `{distance['structure_zip_sha256']}`",
        f"- Structure accessions / missing intersection accessions: "
        f"{distance['released_structure_accessions']:,} / {distance['intersection_structure_coverage']['missing']:,}",
        f"- Accession grouping check (BRq rows/accession min-max): "
        f"{corpus['brq_rows_per_accession_min']:.0f}-{corpus['brq_rows_per_accession_max']:.0f}; "
        f"BRn rows/accession distribution: `{json.dumps(corpus['brn_scan']['rows_per_accession'], sort_keys=True)}`",
        f"- Foldseek / MMseqs versions: `{distance['tool_versions']['foldseek']}` / "
        f"`{distance['tool_versions']['mmseqs']}`",
        "",
        "The representative sequence is the minimum stable SHA-256 over accession, generation description, and sequence. "
        "Backbone accession, not sequence row, is the independent unit.",
        "",
        "| Manifest | Backbones | Sequences | Residues | SHA-256 |",
        "|---|---:|---:|---:|---|",
    ]
    for name, metadata in {**corpus["manifests"], **distance["outputs"]}.items():
        stats = metadata["statistics"]
        lines.append(
            f"| {name} | {stats['backbones']:,} | {stats['sequences']:,} | {stats['residues']:,} | "
            f"`{metadata['manifest_sha256']}` |"
        )

    lines += [
        "",
        "Length and per-sequence entropy summaries (minimum / median / mean / maximum):",
        "",
        "| Manifest | Length | Entropy (bits) |",
        "|---|---|---|",
    ]
    for name, metadata in {**corpus["manifests"], **distance["outputs"]}.items():
        stats = metadata["statistics"]
        length = stats["length"]
        entropy = stats["entropy_bits"]
        lines.append(
            f"| {name} | {length['minimum']:.0f} / {length['median']:.0f} / "
            f"{length['mean']:.1f} / {length['maximum']:.0f} | "
            f"{entropy['minimum']:.3f} / {entropy['median']:.3f} / "
            f"{entropy['mean']:.3f} / {entropy['maximum']:.3f} |"
        )
    train_composition = corpus["manifests"]["train_50k"]["statistics"]["amino_acid_frequencies"]
    lines += [
        "",
        "The 50k amino-acid frequencies are: "
        + ", ".join(f"{aa}={frequency:.4f}" for aa, frequency in train_composition.items())
        + ".",
        "The preregistered natural-reference low-complexity rule and thresholds were: "
        f"`{json.dumps(corpus['low_complexity_thresholds'], sort_keys=True)}`.",
    ]
    lines += [
        "",
        "### Natural-reference composition and complexity comparison",
        "",
        "| Population | Sequences | Entropy median | Max-residue q99 | Homopolymer q99 | Unique-residue median |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, stats in corpus["composition_complexity_comparison"].items():
        lines.append(
            f"| {name} | {stats['sequences']:,} | {stats['entropy_bits']['median']:.3f} | "
            f"{stats['maximum_residue_fraction']['q99']:.3f} | "
            f"{stats['maximum_homopolymer_fraction']['q99']:.3f} | "
            f"{stats['unique_residue_fraction']['median']:.3f} |"
        )
    natural_aa = corpus["composition_complexity_comparison"][
        "natural_length_matched_uniref50"
    ]["amino_acid_frequencies"]
    synthetic_aa = corpus["composition_complexity_comparison"][
        "synthetic_before_low_complexity_filter"
    ]["amino_acid_frequencies"]
    lines += [
        "",
        "Natural versus prefilter-synthetic amino-acid frequencies: "
        + ", ".join(
            f"{aa}={natural_aa[aa]:.4f}/{synthetic_aa[aa]:.4f}" for aa in natural_aa
        )
        + " (natural/synthetic). No composition matching was applied.",
    ]

    lines += [
        "",
        "## Model and execution validation",
        "",
        f"- Architecture: {preflight['layers']} layers, hidden size {preflight['hidden_dimension']}, "
        f"{preflight['attention_heads']} heads, FFN size {preflight['intermediate_dimension']}",
        f"- Runtime versions: Transformers `{preflight['environment']['transformers']}`, "
        f"PyTorch `{preflight['environment']['torch']}`",
        f"- Loaded parameters: {preflight['total_pretrained_parameters']:,}",
        f"- Dense target matrices/scalars: {preflight['target_matrix_count']:,} / {preflight['target_scalar_count']:,}",
        f"- FT / DTFT / LoRA-r8 / LoRA-r64 trainables: {preflight['ft_trainable_count']:,} / "
        f"{preflight['dtft_trainable_count']:,} / {preflight['lora_r8_trainable_count']:,} / "
        f"{preflight['lora_r64_trainable_count']:,}",
        f"- DTFT/FT scalar coverage: {preflight['dtft_ft_coverage_fraction']:.6f}",
        f"- Fair-ESM parity max-logit/NLL differences: "
        f"{parity['masked_logits_max_abs_difference']:.3g} / {parity['nll_absolute_difference']:.3g}",
        f"- Selected attention backend after end-to-end profiling: `{execution_profile['backend']}`",
        f"- torch.compile selected for training: "
        f"`{execution_profile['selected_training']['compile_model']}`",
        "",
        "| Dense target family (per layer) | Weight shape | Matrices |",
        "|---|---|---:|",
    ]
    target_families = {}
    for target in preflight["target_modules"]:
        family = target["name"].split(".layer.", 1)[1].split(".", 1)[1]
        entry = target_families.setdefault(family, {"shape": target["shape"], "count": 0})
        entry["count"] += 1
    for family, values in sorted(target_families.items()):
        lines.append(
            f"| {family} | {values['shape'][0]} x {values['shape'][1]} | {values['count']} |"
        )
    lines += [
        "",
        "| Backend | Passed | tokens/s | Peak VRAM (GB) |",
        "|---|:---:|---:|---:|",
    ]
    for item in backend["benchmarks"]:
        lines.append(
            f"| {item['backend']} | {'yes' if item.get('passed') else 'no'} | "
            f"{item.get('tokens_per_second', float('nan')):.1f} | "
            f"{item.get('peak_vram_bytes', 0) / 1e9:.2f} |"
        )

    selected_train = execution_profile["selected_training"]
    selected_eval = execution_profile["selected_evaluation"]
    lines += [
        "",
        "### End-to-end code optimization",
        "",
        "The full data-loader → transfer → model → loss path was profiled on the A6000. Training additionally "
        "included backward, gradient clipping, and fused AdamW; FT was used as the worst-memory condition. "
        "All candidate measurements are preserved in `execution_profile.json`.",
        "",
        "| Path | Token budget | Workers | Gradient checkpointing | torch.compile | Residues/s | Peak VRAM (GB) |",
        "|---|---:|---:|:---:|:---:|---:|---:|",
        f"| training | {selected_train['token_budget']:,} | {selected_train['workers']} | "
        f"{selected_train['gradient_checkpointing']} | {selected_train['compile_model']} | "
        f"{selected_train['residues_per_second']:.1f} | "
        f"{selected_train['peak_vram_bytes'] / 1e9:.2f} |",
        f"| evaluation | {selected_eval['token_budget']:,} | {selected_eval['workers']} | n/a | "
        f"{selected_eval['compile_model']} | "
        f"{selected_eval['residues_per_second']:.1f} | {selected_eval['peak_vram_bytes'] / 1e9:.2f} |",
        "",
        "Evaluation reuses one loaded model/checkpoint across standard, structural-OOD, and natural sets; "
        "identical best/last checkpoint states use hard links to avoid duplicate serialization and storage.",
        "",
        "Training comparison maxima (residues/s; peak VRAM belongs to the maximizing row):",
        "",
        "| Factor | Setting | Best residues/s | Peak VRAM (GB) |",
        "|---|---|---:|---:|",
    ]
    training_rows = execution_profile["training"]
    comparison_groups = []
    for field, label in (
        ("backend", "attention backend"),
        ("gradient_checkpointing", "gradient checkpointing"),
        ("token_budget", "train token budget"),
        ("workers", "workers"),
        ("compile_model", "torch.compile"),
    ):
        for setting in sorted({row[field] for row in training_rows}, key=str):
            candidates = [row for row in training_rows if row[field] == setting and row["passed"]]
            if candidates:
                comparison_groups.append((label, setting, max(candidates, key=lambda row: row["residues_per_second"])))
    for label, setting, row in comparison_groups:
        lines.append(
            f"| {label} | {setting} | {row['residues_per_second']:.1f} | "
            f"{row['peak_vram_bytes'] / 1e9:.2f} |"
        )
    lines += [
        "",
        "Evaluation token-budget/worker/compile candidates are retained in `execution_profile.json`; "
        "the selected compiled 16,384-token, four-worker path is shown above.",
    ]

    lines += [
        "", "## Learning-rate screen", "",
        "| Method | LR | Minimum validation NLL | Terminal validation NLL | Selected |",
        "|---|---:|---:|---:|:---:|",
    ]
    for mode, learning_rate in gate["selected_learning_rates"].items():
        selection = _json(runs / "hpo" / "10k" / mode / "selection.json")
        for candidate in selection["candidates"]:
            lines.append(
                f"| {mode} | {candidate['learning_rate']:.3g} | "
                f"{candidate['minimum_validation_nll']:.6f} | "
                f"{candidate['terminal_validation_nll']:.6f} | "
                f"{'yes' if candidate['learning_rate'] == learning_rate else 'no'} |"
            )
    lines += [
        "",
        "Candidate trajectories are retained under `hpo/10k/<method>/selection.json`; the registered NLL tie "
        "tolerance is 1e-4 and ties prefer the lower LR.",
    ]

    base_values = {
        dataset: _json(runs / "evaluation" / "10k" / "base" / "fixed" / dataset / "summary.json")[
            "masked_token_nll"
        ]
        for dataset in ("standard_test", "remote_ood_test", "natural_test")
    }
    lines += [
        "",
        "## Base distribution-pressure comparison",
        "",
        f"Base masked-token NLL was {base_values['standard_test']:.6f} on standard synthetic, "
        f"{base_values['remote_ood_test']:.6f} on structural-OOD synthetic, and "
        f"{base_values['natural_test']:.6f} on length-matched UniRef50.",
        "",
        "## Raw seed-11 NLL",
        "",
        "| Load | Set | Base | FT | DTFT | LoRA-r8 | LoRA-r64 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for load in ("10k", "50k"):
        for dataset in ("standard_test", "remote_ood_test"):
            values = {}
            for mode in ("base", "ft", "dtft", "lora8", "lora64"):
                leaf = "fixed" if mode == "base" else "seed_11"
                path = runs / "evaluation" / load / mode / leaf / dataset / "summary.json"
                values[mode] = _json(path)["masked_token_nll"] if path.exists() else None
            lines.append(
                f"| {load} | {dataset} | {_fmt(values['base'])} | {_fmt(values['ft'])} | "
                f"{_fmt(values['dtft'])} | {_fmt(values['lora8'])} | {_fmt(values['lora64'])} |"
            )

    lines += [
        "",
        "## Gate quantities (seed 11)",
        "",
        "| Load | Rank | Set | A_DTFT | G_LoRA | F_gap | 95% CI A / G / F | Candidate | Replicate |",
        "|---|---|---|---:|---:|---:|---|:---:|:---:|",
    ]
    for load, ranks in gate["seed11_gates"].items():
        for rank, datasets in ranks.items():
            for dataset, result in datasets.items():
                fraction = "NA" if result["F_gap"] is None else f"{result['F_gap']:.3f}"
                intervals = result["paired_backbone_bootstrap"]

                def interval(name: str) -> str:
                    value = intervals[name]
                    if value["lower_95"] is None:
                        return "NA"
                    return f"[{value['lower_95']:.4f},{value['upper_95']:.4f}]"

                lines.append(
                    f"| {load} | {rank} | {dataset} | {result['A_DTFT']:.6f} | "
                    f"{result['G_LoRA']:.6f} | {fraction} | "
                    f"{interval('A_DTFT')} / {interval('G_LoRA')} / {interval('F_gap')} | "
                    f"{'yes' if result['candidate_gap'] else 'no'} | "
                    f"{'yes' if result['replication_trigger'] else 'no'} |"
                )
    lines += [
        "",
        "### Continuous structural-distance analysis",
        "",
        "| Load | Rank | Set | Linear slope G(distance) | Spearman rho | p-value |",
        "|---|---|---|---:|---:|---:|",
    ]
    for load, ranks in gate["seed11_gates"].items():
        for rank, datasets in ranks.items():
            for dataset, result in datasets.items():
                trend = result["distance_trend"]
                lines.append(
                    f"| {load} | {rank} | {dataset} | {trend['linear_slope']:.6f} | "
                    f"{trend['spearman_rho']:.4f} | {trend['spearman_pvalue']:.3g} |"
                )
    for load, ranks in gate["seed11_combined_distance_trends"].items():
        for rank, trend in ranks.items():
            lines.append(
                f"| {load} | {rank} | standard+OOD | {trend['linear_slope']:.6f} | "
                f"{trend['spearman_rho']:.4f} | {trend['spearman_pvalue']:.3g} |"
            )
    lines += [
        "",
        "Distance quintiles and all bootstrap intervals are retained in `gate1_summary.json`; "
        "the resampling unit is backbone/protein.",
        "Structural distance is `1 - maximum query-normalized Foldseek TM-score` to the named training load. "
        "Sequence proximity is the best-bit MMseqs2 hit with identity and minimum query/target coverage retained.",
        "",
        "### Registered screen-pattern flags",
        "",
        "Load-dependence deltas (`F_gap(50k) - F_gap(10k)`): "
        + ", ".join(
            f"{item['rank']}/{item['dataset']}="
            + (
                "NA" if item["delta_50k_minus_10k"] is None
                else "{:.3f}".format(item["delta_50k_minus_10k"])
            )
            for item in gate["screen_patterns"]["load_dependence"]
        )
        + ".",
        "Rank-sensitive candidate gaps (r8 >=10%, r64 <10%): "
        + (
            json.dumps(gate["screen_patterns"]["rank_sensitivity"], sort_keys=True)
            if gate["screen_patterns"]["rank_sensitivity"] else "none"
        )
        + ".",
        "Material r64 candidate gaps: "
        + (
            json.dumps(gate["screen_patterns"]["r64_material_gaps"], sort_keys=True)
            if gate["screen_patterns"]["r64_material_gaps"] else "none"
        )
        + ".",
        "",
        "### FT versus DTFT target-coverage check (10k)",
        "",
        "| Set | Base NLL | FT NLL | DTFT NLL | (DTFT-FT)/(Base-FT) | Equivalent within 10% |",
        "|---|---:|---:|---:|---:|:---:|",
    ]
    for dataset, values in gate["coverage_10k"].items():
        fraction = values["fraction_of_base_to_ft_gain"]
        lines.append(
            f"| {dataset} | {values['base_nll']:.6f} | {values['ft_nll']:.6f} | "
            f"{values['dtft_nll']:.6f} | {'NA' if fraction is None else f'{fraction:.3f}'} | "
            f"{'yes' if values['effectively_equivalent_at_10_percent'] else 'no'} |"
        )

    if gate["seed23_replication"]:
        lines += [
            "",
            "### Triggered seed-23 replication",
            "",
            "| Load/rank | Set | A_DTFT | G_LoRA | F_gap | 95% CI A / G / F | Candidate gap replicated |",
            "|---|---|---:|---:|---:|---|:---:|",
        ]
        for pair, datasets in gate["seed23_replication"].items():
            for dataset, result in datasets.items():
                fraction = "NA" if result["F_gap"] is None else f"{result['F_gap']:.3f}"
                intervals = result["paired_backbone_bootstrap"]

                def replicate_interval(name: str) -> str:
                    value = intervals[name]
                    if value["lower_95"] is None:
                        return "NA"
                    return f"[{value['lower_95']:.4f},{value['upper_95']:.4f}]"

                lines.append(
                    f"| {pair} | {dataset} | {result['A_DTFT']:.6f} | "
                    f"{result['G_LoRA']:.6f} | {fraction} | "
                    f"{replicate_interval('A_DTFT')} / {replicate_interval('G_LoRA')} / "
                    f"{replicate_interval('F_gap')} | "
                    f"{'yes' if result['candidate_gap'] else 'no'} |"
                )

    lines += [
        "",
        "## Natural-protein retention",
        "",
        "| Load | Method | Base natural NLL | Method natural NLL | Delta retention |",
        "|---|---|---:|---:|---:|",
    ]
    for load, methods in gate["natural_retention"].items():
        for mode, values in methods.items():
            lines.append(
                f"| {load} | {mode} | {values['reference_nll']:.6f} | "
                f"{values['method_nll']:.6f} | {values['delta']:.6f} |"
            )

    lines += [
        "",
        "## Training trajectories and resources",
        "",
        "| Load | Method | Seed | Status | Best validation NLL | Steps | Sequences | Residues | Effective residues/step | Residues/s | Padding | Runtime (h) | Peak VRAM (GB) | Best checkpoint |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for execution_path in sorted((runs / "evidence").glob("*/*/seed_*/execution.json")):
        execution = _json(execution_path)
        history_path = execution_path.with_name("history.json")
        history = _json(history_path) if history_path.exists() else []
        seed = execution_path.parent.name.removeprefix("seed_")
        mode = execution_path.parent.parent.name
        load = execution_path.parent.parent.parent.name
        lines.append(
            f"| {load} | {mode} | {seed} | "
            f"{execution.get('trajectory_classification', _trajectory(history))} | "
            f"{execution['best_validation_nll']:.6f} | {execution['optimizer_steps']:,} | "
            f"{execution['sequences']:,} | {execution['residues']:,} | "
            f"{execution['effective_residues_per_step']['mean']:.1f} | "
            f"{execution['residues_per_second']:.1f} | {execution['padding_fraction']:.3f} | "
            f"{execution['wall_seconds'] / 3600:.2f} | {execution['peak_vram_bytes'] / 1e9:.2f} | "
            f"`{execution_path.parent / 'best.pt'}` |"
        )

    lines += [
        "",
        "Best and resumable terminal checkpoints are retained beside each evidence run. Execution JSON records "
        "sequences, residues, effective residue budgets, padding, throughput, runtime, and VRAM.",
        "",
        "## Gate-1 decision",
        "",
        f"**{gate['gate1_decision']}**",
        "",
        "## Limitations",
        "",
        "- Foldseek uses query-normalized TM-score with exact refinement. Nested-set pruning gives exact 50k maxima for possible OOD backbones; already-standard backbones retain their exact 10k maximum as a certified 50k lower bound because that is sufficient for the fixed 0.5 split.",
        "- One deterministic ProteinMPNN generation represents each backbone in primary manifests.",
        "- Bootstrap intervals quantify fixed-test protein sampling uncertainty, not training-seed uncertainty.",
        "- This screen does not test or interpret H1-H4.",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-root", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    build_report(args.manifest_root, args.run_root, args.output)


if __name__ == "__main__":
    main()
