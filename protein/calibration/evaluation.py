from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from .data import SequenceRecord, VariantRecord
from .modeling import autocast_context
from .optimized_forward import model_logits, prefetch_batches


def teacher_forced_scores(
    model: torch.nn.Module,
    batch: dict[str, Any],
    alphabet: Any,
    device: torch.device,
    *,
    reuse_backbones: bool = True,
) -> dict[str, Any]:
    tokens = batch["tokens"].to(device, non_blocking=True)
    target = tokens[:, 1:]
    with autocast_context(device):
        logits = model_logits(model, batch, device, reuse_backbones=reuse_backbones, tokens=tokens)
        token_nll = F.cross_entropy(logits.float(), target, reduction="none", ignore_index=alphabet.padding_idx)
    valid = target.ne(alphabet.padding_idx)
    sequence_logp = -(token_nll * valid).sum(dim=1)
    token_count = valid.sum(dim=1)
    predicted = logits.argmax(dim=1)
    correct = ((predicted == target) & valid).sum(dim=1)
    # One bulk device-to-host transfer avoids a synchronization per sequence.
    host_nll, host_valid = token_nll.detach().cpu(), valid.detach().cpu()
    return {
        "sequence_logp": sequence_logp.detach().cpu(),
        "token_count": token_count.detach().cpu(),
        "correct": correct.detach().cpu(),
        "token_nll": [x[m].tolist() for x, m in zip(host_nll, host_valid)],
    }


def _sequence_rows(model: torch.nn.Module, loader: Iterable[dict[str, Any]], alphabet: Any, device: torch.device) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    model.eval()
    with torch.inference_mode():
        for batch in prefetch_batches(loader, device):
            scores = teacher_forced_scores(model, batch, alphabet, device)
            for i, record in enumerate(batch["records"]):
                assert isinstance(record, SequenceRecord)
                count = int(scores["token_count"][i])
                logp = float(scores["sequence_logp"][i])
                nll_per_token = -logp / count
                recovery = float(scores["correct"][i]) / count
                rows.append(
                    {
                        "domain": record.domain_id,
                        "backbone_key": f"{record.backbone_path}::{record.chain_id}",
                        "sequence_id": record.sequence_id,
                        "ddg": record.ddg,
                        "target_score": -record.ddg,
                        "fitness": -record.ddg,
                        "substitution_count": record.substitution_count,
                        "mutation_count": record.mutation_count,
                        "sequence_length": record.sequence_length,
                        "foldseek_qtm_max_to_train": record.foldseek_qtm_max_to_train,
                        "d_struct": record.d_struct,
                        "target_logp": logp,
                        "variant_logp": logp,
                        "token_count": count,
                        "target_nll_per_token": nll_per_token,
                        "variant_nll_per_token": nll_per_token,
                        "target_recovery": recovery,
                        "variant_recovery": recovery,
                        "token_losses": scores["token_nll"][i],
                    }
                )
    return rows


_variant_rows = _sequence_rows


def evaluate_sequences(
    model: torch.nn.Module,
    sequence_loader: Iterable[dict[str, Any]],
    native_loader: Iterable[dict[str, Any]],
    alphabet: Any,
    device: torch.device,
    output_dir: Path,
) -> dict[str, Any]:
    target_rows = _sequence_rows(model, sequence_loader, alphabet, device)
    native_rows = _sequence_rows(model, native_loader, alphabet, device)
    native_by_backbone = {row["backbone_key"]: row for row in native_rows}
    for row in target_rows:
        native = native_by_backbone[row["backbone_key"]]
        row["native_logp"] = native["target_logp"]
        row["native_recovery"] = native["target_recovery"]
        row["delta_logp"] = row["target_logp"] - row["native_logp"]
    frame = pd.DataFrame(target_rows)
    domain_rows = []
    for domain, group in frame.groupby("domain", sort=True):
        rho = spearmanr(group["delta_logp"], group["target_score"]).statistic if len(group) >= 2 else np.nan
        labels = (group["ddg"] < 0).astype(int)
        auc = roc_auc_score(labels, group["delta_logp"]) if labels.nunique() == 2 else np.nan
        domain_rows.append({"domain": domain, "spearman": rho, "auroc": auc, "n": len(group)})
    domains = pd.DataFrame(domain_rows)
    frame["stability_bin"] = pd.qcut(frame["ddg"], q=min(10, frame["ddg"].nunique()), duplicates="drop")
    calibration = (
        frame.groupby("stability_bin", observed=True)
        .agg(n=("ddg", "size"), mean_ddg=("ddg", "mean"), mean_delta_logp=("delta_logp", "mean"))
        .reset_index()
    )
    domain_distance = frame.groupby("domain")["d_struct"].first().sort_values()
    domain_tier = {}
    for label, chunk in zip(("near", "middle", "far"), np.array_split(domain_distance.index.to_numpy(), 3)):
        domain_tier.update({domain: label for domain in chunk})
    frame["ood_tier"] = frame["domain"].map(domain_tier)
    ood = {}
    for tier, group in frame.groupby("ood_tier", sort=False):
        domain_subset = domains[domains["domain"].isin(group["domain"].unique())]
        ood[tier] = {
            "n_domains": int(group["domain"].nunique()),
            "heldout_nll": float(np.average(group["target_nll_per_token"], weights=group["token_count"])),
            "domain_spearman_mean": float(domain_subset["spearman"].mean()),
            "domain_auroc_mean": float(domain_subset["auroc"].mean()),
        }
    metrics = {
        "heldout_nll": float(np.average(frame["target_nll_per_token"], weights=frame["token_count"])),
        "sequence_recovery": float(frame.groupby("domain")["native_recovery"].first().mean()),
        "domain_spearman_mean": float(domains["spearman"].mean()),
        "domain_spearman_median": float(domains["spearman"].median()),
        "domain_auroc_mean": float(domains["auroc"].mean()),
        "n_sequences": int(len(frame)),
        "n_variants": int(len(frame)),
        "n_domains": int(frame["domain"].nunique()),
        "ood_tertiles": ood,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    frame["stability_bin"] = frame["stability_bin"].astype(str)
    frame.to_parquet(output_dir / "sequence_metrics.parquet", index=False)
    frame.to_parquet(output_dir / "variant_metrics.parquet", index=False)
    domains.to_parquet(output_dir / "domain_metrics.parquet", index=False)
    calibration["stability_bin"] = calibration["stability_bin"].astype(str)
    calibration.to_parquet(output_dir / "stability_calibration.parquet", index=False)
    (output_dir / "summary.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    return metrics


evaluate_variants = evaluate_sequences
