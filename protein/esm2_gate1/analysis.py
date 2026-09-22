from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _rows(path: Path) -> dict[str, dict]:
    return {
        row["backbone_id"]: row
        for row in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line)
    }


def _pooled_nll(rows: list[dict]) -> float:
    return sum(row["loss_sum"] for row in rows) / sum(row["masked_tokens"] for row in rows)


def paired_bootstrap(
    base: dict[str, dict], dtft: dict[str, dict], lora: dict[str, dict], *, seed: int = 1701, draws: int = 10_000
) -> dict:
    ids = sorted(set(base) & set(dtft) & set(lora))
    if set(ids) != set(base) or set(ids) != set(dtft) or set(ids) != set(lora):
        raise RuntimeError("paired bootstrap inputs do not contain identical backbones")
    rng = np.random.default_rng(seed)
    values = np.empty((draws, 3), dtype=np.float64)
    arrays = {}
    for label, rows in (("base", base), ("dtft", dtft), ("lora", lora)):
        arrays[label] = (
            np.asarray([rows[item]["loss_sum"] for item in ids], dtype=np.float64),
            np.asarray([rows[item]["masked_tokens"] for item in ids], dtype=np.float64),
        )
    for draw in range(draws):
        sampled = rng.integers(0, len(ids), len(ids))
        b, d, l = (
            arrays[label][0][sampled].sum() / arrays[label][1][sampled].sum()
            for label in ("base", "dtft", "lora")
        )
        adaptation = b - d
        gap = l - d
        values[draw] = (adaptation, gap, gap / adaptation if adaptation > 0 else np.nan)
    result = {}
    for index, name in enumerate(("A_DTFT", "G_LoRA", "F_gap")):
        finite = values[:, index][np.isfinite(values[:, index])]
        result[name] = {
            "lower_95": float(np.quantile(finite, 0.025)) if finite.size else None,
            "upper_95": float(np.quantile(finite, 0.975)) if finite.size else None,
        }
    return result


def gate_quantities(base_path: Path, dtft_path: Path, lora_path: Path) -> dict:
    base, dtft, lora = _rows(base_path), _rows(dtft_path), _rows(lora_path)
    base_nll = _pooled_nll(list(base.values()))
    dtft_nll = _pooled_nll(list(dtft.values()))
    lora_nll = _pooled_nll(list(lora.values()))
    adaptation = base_nll - dtft_nll
    gap = lora_nll - dtft_nll
    fraction = gap / adaptation if adaptation > 0 else None
    candidate = fraction is not None and fraction >= 0.10
    borderline = fraction is not None and 0.08 <= fraction < 0.10
    return {
        "raw_nll": {"base": base_nll, "dtft": dtft_nll, "lora": lora_nll},
        "A_DTFT": adaptation,
        "G_LoRA": gap,
        "F_gap": fraction,
        "candidate_gap": candidate,
        "borderline_gap": borderline,
        "replication_trigger": candidate or borderline,
        "registered_gap_threshold": 0.10,
        "registered_borderline_lower": 0.08,
        "paired_backbone_bootstrap": paired_bootstrap(base, dtft, lora),
    }


def coverage_check(base_path: Path, ft_path: Path, dtft_path: Path) -> dict:
    base, ft, dtft = _rows(base_path), _rows(ft_path), _rows(dtft_path)
    b, f, d = (_pooled_nll(list(rows.values())) for rows in (base, ft, dtft))
    gain = b - f
    difference = d - f
    fraction = difference / gain if gain > 0 else None
    return {
        "base_nll": b, "ft_nll": f, "dtft_nll": d,
        "ft_minus_dtft_nll": difference,
        "fraction_of_base_to_ft_gain": fraction,
        "effectively_equivalent_at_10_percent": fraction is not None and abs(fraction) <= 0.10,
    }


def nll_delta(reference_path: Path, method_path: Path) -> dict:
    reference, method = _rows(reference_path), _rows(method_path)
    if set(reference) != set(method):
        raise RuntimeError("NLL delta requires identical backbone sets")
    reference_nll = _pooled_nll(list(reference.values()))
    method_nll = _pooled_nll(list(method.values()))
    return {
        "reference_nll": reference_nll,
        "method_nll": method_nll,
        "delta": method_nll - reference_nll,
    }


def _distance_trend_rows(dtft: dict[str, dict], lora: dict[str, dict], load: str) -> dict:
    from scipy.stats import spearmanr

    ids = sorted(set(dtft) & set(lora))
    if set(ids) != set(dtft) or set(ids) != set(lora):
        raise RuntimeError("distance trend inputs do not contain identical backbones")
    similarity_key = f"structural_similarity_{load}"
    distances = np.asarray([1.0 - float(dtft[item][similarity_key]) for item in ids])
    gaps = np.asarray([float(lora[item]["nll"]) - float(dtft[item]["nll"]) for item in ids])
    if np.ptp(distances) == 0:
        slope = intercept = rho = pvalue = float("nan")
    else:
        slope, intercept = np.polyfit(distances, gaps, 1)
        rho, pvalue = spearmanr(distances, gaps)
    bins = []
    edges = np.quantile(distances, np.linspace(0, 1, 6))
    for index in range(5):
        selected = (distances >= edges[index]) & (
            distances <= edges[index + 1] if index == 4 else distances < edges[index + 1]
        )
        bins.append({
            "distance_min": float(edges[index]), "distance_max": float(edges[index + 1]),
            "proteins": int(selected.sum()),
            "mean_G_LoRA": float(gaps[selected].mean()) if selected.any() else None,
        })
    return {
        "distance_definition": f"1 - {similarity_key}",
        "linear_slope": float(slope), "linear_intercept": float(intercept),
        "spearman_rho": float(rho), "spearman_pvalue": float(pvalue), "quintiles": bins,
    }


def distance_trend(dtft_path: Path, lora_path: Path, load: str) -> dict:
    return _distance_trend_rows(_rows(dtft_path), _rows(lora_path), load)


def combined_distance_trend(
    dtft_paths: list[Path], lora_paths: list[Path], load: str
) -> dict:
    if len(dtft_paths) != len(lora_paths) or not dtft_paths:
        raise ValueError("combined distance trends require paired nonempty path lists")
    dtft: dict[str, dict] = {}
    lora: dict[str, dict] = {}
    for dtft_path, lora_path in zip(dtft_paths, lora_paths):
        for destination, source in ((dtft, _rows(dtft_path)), (lora, _rows(lora_path))):
            overlap = set(destination) & set(source)
            if overlap:
                raise RuntimeError(f"combined distance sets overlap on {len(overlap)} backbones")
            destination.update(source)
    return _distance_trend_rows(dtft, lora, load)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--dtft", required=True, type=Path)
    parser.add_argument("--lora", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = gate_quantities(args.base, args.dtft, args.lora)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
