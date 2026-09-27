from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal


Mode = Literal["base", "ft", "dtft", "lora8", "lora64"]


@dataclass(frozen=True)
class TrainConfig:
    mode: Mode
    seed: int
    lr: float
    manifest: Path
    output_dir: Path
    diagnostic_panel: Path | None = None
    epochs: int = 30
    batch_size: int = 32
    coordinate_noise_angstrom: float = 0.1
    num_workers: int = 6
    validation_fraction: float = 0.5
    precision: str = "bf16"
    adapter_weight_decay: float = 0.0
    compile_training: bool = False
    compile_mode: str = "max-autotune"
    instrument_checkpoints: bool = True
    save_checkpoints: bool = True
    early_stopping_patience_checks: int | None = None
    minimum_epochs: float = 0.0

    def validate(self) -> None:
        if self.batch_size != 32:
            raise ValueError("The preregistered effective batch size is exactly 32")
        if self.coordinate_noise_angstrom != 0.1:
            raise ValueError("The preregistered coordinate noise is exactly 0.1 A")
        if self.epochs > 30:
            raise ValueError("Thirty epochs is a hard cap")
        if self.precision != "bf16":
            raise ValueError("The calibration protocol requires bf16")
        if self.mode.startswith("lora") and self.adapter_weight_decay != 0:
            raise ValueError("LoRA factor weight decay must be zero")
        if self.compile_mode not in {"default", "reduce-overhead", "max-autotune"}:
            raise ValueError("Unsupported torch.compile mode")
        if self.seed == 7 and "evidence" in self.output_dir.parts:
            raise ValueError("HPO seed 7 cannot be used for uncertainty estimates")
        if self.early_stopping_patience_checks is not None and self.early_stopping_patience_checks < 1:
            raise ValueError("Early-stopping patience must be at least one validation check")
        if not 0 <= self.minimum_epochs <= self.epochs:
            raise ValueError("minimum_epochs must be between zero and the epoch cap")

    def serializable(self) -> dict:
        payload = asdict(self)
        payload["manifest"] = str(self.manifest)
        payload["output_dir"] = str(self.output_dir)
        payload["diagnostic_panel"] = str(self.diagnostic_panel) if self.diagnostic_panel else None
        return payload


HPO_LRS = {
    "ft": (3e-8, 1e-7, 3e-7),
    "dtft": (3e-8, 1e-7, 3e-7),
    "lora8": (1e-6, 3e-6, 1e-5, 3e-5, 1e-4),
    "lora64": (1e-6, 3e-6, 1e-5, 3e-5, 1e-4),
}
EVIDENCE_SEEDS = (11, 23, 37)
CONFIRMATION_SEEDS = (53, 71)
