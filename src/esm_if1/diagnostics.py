from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable

import torch
import torch.nn as nn

from .lora import LoRALinear
from .training import sequence_nll


@dataclass(frozen=True)
class SpectrumSummary:
    singular_values: torch.Tensor
    frobenius_squared: float
    captured_energy: float
    stable_rank: float
    stable_rank_is_exact: bool
    effective_rank: float | None
    spectral_entropy: float | None
    r90: int | str
    r95: int | str
    r99: int | str
    full_spectrum: bool


def _threshold_rank(values: torch.Tensor, fro2: float, threshold: float, full: bool) -> int | str:
    reached = torch.nonzero(torch.cumsum(values.square(), 0) / fro2 >= threshold)
    if len(reached):
        rank = int(reached[0]) + 1
        return rank if full else f"<= {rank} (uncertified randomized bound)"
    return f"> {len(values)}"


def adaptive_spectrum(matrix: torch.Tensor, start_k: int = 128) -> SpectrumSummary:
    """Measure a gradient spectrum without materializing unnecessary full SVDs."""
    if matrix.ndim != 2:
        raise ValueError("Spectrum input must be a matrix")
    matrix = matrix.detach().float()
    limit = min(matrix.shape)
    fro2 = float(torch.sum(matrix.square(), dtype=torch.float64))
    if fro2 == 0:
        zeros = torch.zeros(limit)
        return SpectrumSummary(zeros, 0.0, 1.0, 0.0, True, 0.0, 0.0, 0, 0, 0, True)
    k = min(start_k, limit)
    full = False
    while True:
        if k == limit:
            values = torch.linalg.svdvals(matrix)
            full = True
            break
        _, values, _ = torch.svd_lowrank(matrix, q=k, niter=4)
        captured = float(values.square().sum(dtype=torch.float64)) / fro2
        if captured >= 0.99:
            break
        next_k = min(limit, {128: 256, 256: 512}.get(k, limit))
        if next_k == k:
            break
        k = next_k
    captured = float(values.square().sum(dtype=torch.float64)) / fro2
    stable_rank = fro2 / float(values[0].square())
    entropy = None
    effective = None
    if full:
        probabilities = values.square().double() / fro2
        probabilities = probabilities[probabilities > 0]
        entropy = float(-(probabilities * probabilities.log()).sum())
        effective = math.exp(entropy)
    return SpectrumSummary(
        singular_values=values.cpu(),
        frobenius_squared=fro2,
        captured_energy=captured,
        stable_rank=stable_rank,
        stable_rank_is_exact=full,
        effective_rank=effective,
        spectral_entropy=entropy,
        r90=_threshold_rank(values, fro2, 0.90, full),
        r95=_threshold_rank(values, fro2, 0.95, full),
        r99=_threshold_rank(values, fro2, 0.99, full),
        full_spectrum=full,
    )


class StreamingPCs:
    """Bounded-memory incremental rank-64 activation sketch."""

    def __init__(self, dimension: int, rank: int = 64) -> None:
        self.dimension = dimension
        self.rank = min(rank, dimension)
        self.rows = 0
        self.mean = torch.zeros(dimension, dtype=torch.float64)
        self.basis = torch.empty(0, dimension)
        self.energy = torch.empty(0)

    def update(self, values: torch.Tensor) -> None:
        values = values.detach().reshape(-1, self.dimension).float().cpu()
        if not len(values):
            return
        # Deterministic bounded row coverage keeps the secondary activation
        # diagnostic small relative to the supervised gradient pass.
        if len(values) > 64:
            indices = torch.linspace(0, len(values) - 1, 64).round().long()
            values = values[indices]
        self.rows += len(values)
        batch_mean = values.double().mean(0)
        self.mean += (batch_mean - self.mean) * (len(values) / self.rows)
        centered = values - self.mean.float()
        if len(self.energy):
            synthetic = self.energy.sqrt().unsqueeze(1) * self.basis
            centered = torch.cat((synthetic, centered), dim=0)
        q = min(self.rank, centered.shape[0], centered.shape[1])
        if q:
            _, singular, vectors = torch.pca_lowrank(centered, q=q, center=False, niter=2)
            self.basis = vectors[:, :q].T
            self.energy = singular[:q].square()

    def state(self) -> dict[str, Any]:
        return {
            "rows": self.rows,
            "mean": self.mean.float(),
            "components": self.basis.to(torch.bfloat16),
            "captured_energy": self.energy,
            "rank": self.rank,
            "method": "deterministic-64-row-per-microbatch incremental randomized SVD sketch",
        }


def _dense_targets(model: nn.Module) -> dict[str, nn.Linear]:
    wrapped = {
        name: module.base for name, module in model.named_modules() if isinstance(module, LoRALinear)
    }
    if wrapped:
        return wrapped
    from .targets import target_matrices

    return {item.name: item.module for item in target_matrices(model)}


def checkpoint_diagnostics(
    model: nn.Module,
    loader: Iterable[dict[str, Any]],
    alphabet: Any,
    device: torch.device,
) -> dict[str, Any]:
    targets = _dense_targets(model)
    prior_training = model.training
    requires_grad = {name: module.weight.requires_grad for name, module in targets.items()}
    for module in targets.values():
        module.weight.requires_grad_(True)
    gradient_sum = {name: torch.zeros_like(module.weight, device="cpu", dtype=torch.float32) for name, module in targets.items()}
    norm_values: dict[str, list[float]] = {name: [] for name in targets}
    pc_sketches = {name: StreamingPCs(module.in_features, rank=64) for name, module in targets.items()}
    active_masks: dict[str, torch.Tensor] = {}

    def hook_for(name: str):
        def hook(_module, inputs, _output):
            values = inputs[0].detach()
            if values.ndim != 3:
                return
            mask = None
            for candidate in active_masks.values():
                if tuple(values.shape[:2]) == tuple(candidate.shape):
                    mask = candidate
                    break
                if tuple(values.shape[:2]) == tuple(candidate.T.shape):
                    mask = candidate.T
                    break
            selected = values[mask] if mask is not None else values.reshape(-1, values.shape[-1])
            pc_sketches[name].update(selected)

        return hook

    handles = [module.register_forward_hook(hook_for(name)) for name, module in targets.items()]
    batches = 0
    total_examples = 0
    model.eval()
    try:
        for batch in loader:
            model.zero_grad(set_to_none=True)
            coord_valid = ~batch["padding_mask"]
            token_valid = batch["tokens"][:, 1:].ne(alphabet.padding_idx)
            active_masks = {"encoder": coord_valid, "decoder": token_valid}
            loss_sum, _ = sequence_nll(model, batch, alphabet, device)
            batch_examples = len(batch["records"])
            (loss_sum / batch_examples).backward()
            for name, module in targets.items():
                if module.weight.grad is None:
                    raise RuntimeError(f"Missing diagnostic gradient for {name}")
                gradient = module.weight.grad.detach().float().cpu()
                gradient_sum[name] += gradient * batch_examples
                norm_values[name].append(float(torch.linalg.norm(gradient)))
            batches += 1
            total_examples += batch_examples
    finally:
        for handle in handles:
            handle.remove()
        model.zero_grad(set_to_none=True)
        for name, module in targets.items():
            module.weight.requires_grad_(requires_grad[name])
        model.train(prior_training)
    if batches == 0:
        raise ValueError("Diagnostic loader is empty")
    modules = {}
    for name in targets:
        mean_gradient = gradient_sum[name] / total_examples
        spectrum = adaptive_spectrum(mean_gradient)
        norms = torch.tensor(norm_values[name])
        spectrum_payload = asdict(spectrum)
        spectrum_payload["singular_values"] = spectrum.singular_values
        modules[name] = {
            "mean_gradient": mean_gradient.to(torch.bfloat16),
            "frobenius_norm": float(torch.linalg.norm(mean_gradient)),
            "spectral_norm": float(spectrum.singular_values[0]) if len(spectrum.singular_values) else 0.0,
            "microbatch_frobenius_mean": float(norms.mean()),
            "microbatch_frobenius_sd": float(norms.std(unbiased=True)) if len(norms) > 1 else 0.0,
            "spectrum": spectrum_payload,
            "activation_pcs": pc_sketches[name].state(),
        }
    return {"microbatches": batches, "examples": total_examples, "modules": modules}
