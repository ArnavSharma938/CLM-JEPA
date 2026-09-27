from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import norm


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _metric_rows(base_dir: Path, dtft_dir: Path) -> tuple[list[dict], dict]:
    base_true = {row["pair_id"]: row for row in read_jsonl(base_dir / "test_true.jsonl")}
    dtft_true = {row["pair_id"]: row for row in read_jsonl(dtft_dir / "test_true.jsonl")}
    base_decoys: dict[str, list[dict]] = defaultdict(list)
    dtft_decoys: dict[str, list[dict]] = defaultdict(list)
    for row in read_jsonl(base_dir / "test_decoy.jsonl"):
        base_decoys[row["pair_id"]].append(row)
    for row in read_jsonl(dtft_dir / "test_decoy.jsonl"):
        dtft_decoys[row["pair_id"]].append(row)
    if set(base_true) != set(dtft_true) or set(base_true) != set(base_decoys) or set(base_true) != set(dtft_decoys):
        raise RuntimeError("Base and DTFT test true/decoy results do not cover the same pairs")
    result = []
    for pair_id, btrue in base_true.items():
        dtrue = dtft_true[pair_id]
        bdecoys, ddecoys = base_decoys[pair_id], dtft_decoys[pair_id]
        if len(bdecoys) != len(ddecoys) or len(bdecoys) != 4:
            raise RuntimeError(f"pair {pair_id} does not have exactly four fixed decoys per model")
        if btrue["supervised_tokens"] != dtrue["supervised_tokens"]:
            raise RuntimeError(f"true-pair token counts differ for {pair_id}")
        if any(row["supervised_tokens"] != btrue["supervised_tokens"] for row in bdecoys + ddecoys):
            raise RuntimeError(f"decoy peptide target token count differs for {pair_id}")
        base_true_nll = btrue["loss_sum"] / btrue["supervised_tokens"]
        dtft_true_nll = dtrue["loss_sum"] / dtrue["supervised_tokens"]
        base_decoy_nll = sum(row["loss_sum"] for row in bdecoys) / sum(row["supervised_tokens"] for row in bdecoys)
        dtft_decoy_nll = sum(row["loss_sum"] for row in ddecoys) / sum(row["supervised_tokens"] for row in ddecoys)
        result.append({
            "pair_id": pair_id,
            "cluster_id": btrue["receptor_cluster_id"],
            "tokens": int(btrue["supervised_tokens"]),
            "base_true_nll": base_true_nll,
            "dtft_true_nll": dtft_true_nll,
            "base_decoy_nll": base_decoy_nll,
            "dtft_decoy_nll": dtft_decoy_nll,
            "adaptation_gain": base_true_nll - dtft_true_nll,
            "base_specificity": base_decoy_nll - base_true_nll,
            "dtft_specificity": dtft_decoy_nll - dtft_true_nll,
            "acquired_receptor_dependence": (
                dtft_decoy_nll - dtft_true_nll - base_decoy_nll + base_true_nll
            ),
        })
    metadata = {
        "test_pairs": len(result),
        "fixed_decoys_per_pair": 4,
        "heldout_receptor_clusters": len({row["cluster_id"] for row in result}),
        "base_true_peptide_nll": sum(row["base_true_nll"] * row["tokens"] for row in result) / sum(row["tokens"] for row in result),
        "dtft_true_peptide_nll": sum(row["dtft_true_nll"] * row["tokens"] for row in result) / sum(row["tokens"] for row in result),
        "base_decoy_peptide_nll": sum(row["base_decoy_nll"] * row["tokens"] for row in result) / sum(row["tokens"] for row in result),
        "dtft_decoy_peptide_nll": sum(row["dtft_decoy_nll"] * row["tokens"] for row in result) / sum(row["tokens"] for row in result),
        "A_DTFT_token_pooled": sum(row["adaptation_gain"] * row["tokens"] for row in result) / sum(row["tokens"] for row in result),
    }
    return result, metadata


def _cluster_means(rows: list[dict], field: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[row["cluster_id"]].append(float(row[field]))
    return {cluster: float(np.mean(values)) for cluster, values in grouped.items()}


def _bootstrap(values: dict[str, float], *, replicates: int, seed: int) -> dict:
    clusters = sorted(values)
    array = np.asarray([values[key] for key in clusters], dtype=np.float64)
    rng = np.random.default_rng(seed)
    indexes = rng.integers(0, len(array), size=(replicates, len(array)))
    means = array[indexes].mean(axis=1)
    return {
        "cluster_equal_weight_mean": float(array.mean()),
        "cluster_sd": float(array.std(ddof=1)),
        "ci95_cluster_bootstrap": [float(value) for value in np.quantile(means, [0.025, 0.975])],
        "bootstrap_probability_nonpositive": float(np.mean(means <= 0)),
        "bootstrap_probability_nonnegative": float(np.mean(means >= 0)),
        "independent_clusters": len(array),
        "bootstrap_replicates": replicates,
    }


def planning_proxy_detectability(values: dict[str, float], *, material_effect: float, seed: int = 9381) -> dict:
    clusters = sorted(values)
    array = np.asarray([values[key] for key in clusters], dtype=np.float64)
    n = len(array)
    sd = float(array.std(ddof=1))
    mde = float((norm.ppf(0.975) + norm.ppf(0.80)) * sd / np.sqrt(n))
    rng = np.random.default_rng(seed)
    centered = array - array.mean()
    replicates = 10_000
    sampled = centered[rng.integers(0, n, size=(replicates, n))] + material_effect
    means = sampled.mean(axis=1)
    standard_errors = sampled.std(axis=1, ddof=1) / np.sqrt(n)
    power = float(np.mean(np.abs(means / np.maximum(standard_errors, 1e-15)) > norm.ppf(0.975)))
    return {
        "variance_proxy_source": "Base-to-DTFT adaptation-gain variability across receptor clusters; not DTFT-versus-LoRA paired-difference variance",
        "proxy_cluster_sd_base_to_dtft_adaptation_gain": sd,
        "proxy_mde_alpha_0.05_power_0.80": mde,
        "delta_material_0.10_A_DTFT": material_effect,
        "proxy_power_at_delta_material": power,
        "independent_clusters": n,
        "simulation_replicates": replicates,
        "actual_dtft_vs_lora_paired_variance": None,
        "informational_only_not_a_gate0_stopping_criterion": True,
    }


def analyze(
    base_dir: Path,
    dtft_dir: Path,
    *,
    bootstrap_replicates: int = 20_000,
    primary_a_dtft: float | None = None,
) -> dict:
    rows, metadata = _metric_rows(base_dir, dtft_dir)
    by_cluster = {
        "A_DTFT": _cluster_means(rows, "adaptation_gain"),
        "S_Base": _cluster_means(rows, "base_specificity"),
        "S_DTFT": _cluster_means(rows, "dtft_specificity"),
        "C_DTFT": _cluster_means(rows, "acquired_receptor_dependence"),
    }
    uncertainty = {
        name: _bootstrap(values, replicates=bootstrap_replicates, seed=7100 + index)
        for index, (name, values) in enumerate(by_cluster.items())
    }
    a = uncertainty["A_DTFT"]["cluster_equal_weight_mean"]
    c = uncertainty["C_DTFT"]["cluster_equal_weight_mean"]
    ci_a = uncertainty["A_DTFT"]["ci95_cluster_bootstrap"]
    ci_c = uncertainty["C_DTFT"]["ci95_cluster_bootstrap"]
    # The Gate-1 material threshold stays anchored to the predeclared Stage-4
    # equal-cluster estimand even when Stage 6 adds unused clusters for precision.
    primary_a = a if primary_a_dtft is None else float(primary_a_dtft)
    # Use Base-to-DTFT cluster-gain variability only as an informational proxy;
    # it is not the future paired DTFT-versus-LoRA variance.
    power = planning_proxy_detectability(
        by_cluster["A_DTFT"], material_effect=0.10 * max(primary_a, 0.0)
    )
    passed = bool(a > 0 and ci_a[0] > 0 and c > 0 and ci_c[0] > 0 and c >= 0.10 * a)
    return {
        **metadata,
        "uncertainty_by_receptor_cluster": uncertainty,
        "A_DTFT": a,
        "C_DTFT": c,
        "C_over_A": c / a if a != 0 else None,
        "primary_estimand": "equal-receptor-cluster-weighted mean peptide-NLL gain",
        "primary_A_DTFT_for_material_threshold": primary_a,
        "meaningful_target_dependence_rule": "C_DTFT > 0 with cluster-bootstrap 95% lower bound > 0 and C_DTFT/A_DTFT >= 0.10",
        "stage4_passed": passed,
        "stage6_planning_proxy": power,
        "test_pair_scores": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Cluster-paired adaptation and conditionality analysis")
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--dtft-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--primary-a-dtft", type=float,
        help="Stage-4 equal-cluster A_DTFT used for the Gate-1 material threshold when analyzing an expanded Stage-6 panel",
    )
    args = parser.parse_args()
    result = analyze(args.base_dir, args.dtft_dir, primary_a_dtft=args.primary_a_dtft)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {key: value for key, value in result.items() if key not in {"test_pair_scores", "uncertainty_by_receptor_cluster"}}
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
