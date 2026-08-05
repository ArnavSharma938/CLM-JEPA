from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class TrackingContext:
    task: str
    dataset: str
    condition: str
    seed: int
    data_fraction: float
    resolved_hyperparameters: Mapping[str, Any]


def _safe_config(context: TrackingContext) -> dict[str, Any]:
    config = asdict(context)
    flattened = str(config).lower()
    if any(marker in flattened for marker in ("api_key", "password", "secret", "token=")):
        raise ValueError("tracking configuration must not contain credentials")
    return config


class WandbTracker:
    """Section 13 logging with credentials supplied only through the environment."""

    def __init__(
        self,
        context: TrackingContext,
        *,
        run_name: str,
        enabled: bool = True,
        wandb_module=None,
    ) -> None:
        self.enabled = enabled
        self.started_at = time.perf_counter()
        self.total_tokens = 0
        self.jepa_active_batches = 0
        self.run = None
        if not enabled:
            return
        if not os.environ.get("WANDB_API_KEY"):
            raise EnvironmentError("WANDB_API_KEY must be set in the environment")
        project = os.environ.get("WANDB_PROJECT", "clm-jepa")
        if project != "clm-jepa":
            raise ValueError("WANDB_PROJECT must be clm-jepa")
        if wandb_module is None:
            import wandb as wandb_module
        kwargs = {
            "project": project,
            "name": run_name,
            "config": _safe_config(context),
            "job_type": "fine-tuning",
            "reinit": True,
        }
        entity = os.environ.get("WANDB_ENTITY")
        if entity:
            kwargs["entity"] = entity
        self.run = wandb_module.init(**kwargs)

    def log_training_step(
        self,
        *,
        step: int,
        native_loss: float,
        jepa_loss: float | None,
        total_loss: float,
        gradient_norm: float,
        learning_rate: float,
        jepa_active: bool,
        batch_tokens: int,
        model_calls: int,
        effective_tokens: int,
        peak_vram_bytes: int,
        estimated_flops: float | None = None,
    ) -> None:
        self.total_tokens += int(batch_tokens)
        self.jepa_active_batches += int(jepa_active)
        elapsed = max(time.perf_counter() - self.started_at, 1e-12)
        payload = {
            "train/native_loss": native_loss,
            "train/jepa_loss": jepa_loss,
            "train/total_loss": total_loss,
            "train/gradient_norm": gradient_norm,
            "train/learning_rate": learning_rate,
            "compute/jepa_active_batch": int(jepa_active),
            "compute/jepa_active_batches": self.jepa_active_batches,
            "compute/batch_tokens": int(batch_tokens),
            "compute/total_tokens": self.total_tokens,
            "compute/model_calls": int(model_calls),
            "compute/effective_tokens": int(effective_tokens),
            "compute/wall_time_seconds": elapsed,
            "compute/tokens_per_second": self.total_tokens / elapsed,
            "compute/peak_vram_bytes": int(peak_vram_bytes),
        }
        if estimated_flops is not None:
            payload["compute/estimated_flops"] = estimated_flops
        if self.run is not None:
            self.run.log(payload, step=step)

    def log_evaluation(
        self,
        *,
        step: int,
        split: str,
        task_metrics: Mapping[str, float],
        validity: float,
        native_loss: float,
    ) -> None:
        payload = {
            f"{split}/native_loss": native_loss,
            f"{split}/validity": validity,
        }
        payload.update({f"{split}/{name}": value for name, value in task_metrics.items()})
        if self.run is not None:
            self.run.log(payload, step=step)

    def finish(self, summary: Mapping[str, Any] | None = None) -> None:
        if self.run is None:
            return
        if summary:
            self.run.summary.update(dict(summary))
        self.run.finish()
