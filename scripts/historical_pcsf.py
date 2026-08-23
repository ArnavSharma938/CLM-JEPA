"""Archived PCSF mathematics for reproducing historical diagnostics.

PCSF is intentionally absent from ``src/`` and the active training path.  The
definitions remain here only so prior reports and read-only audit scripts can
still be inspected or reproduced against their frozen checkpoints.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


def pair_centers(source_states: torch.Tensor, target_states: torch.Tensor) -> torch.Tensor:
    if source_states.ndim != 2 or target_states.shape != source_states.shape:
        raise ValueError("PCSF expects matched 2D source/target states")
    return (source_states.float() + target_states.float()) * 0.5


def pair_center_variance(
    centers: torch.Tensor, *, unbiased: bool = True,
) -> torch.Tensor:
    if centers.ndim != 2 or centers.size(0) < 2:
        raise ValueError("pair-center spread expects at least two 2D samples")
    values = centers.float()
    centered = values - values.mean(dim=0, keepdim=True)
    denominator = (values.size(0) - int(unbiased)) * values.size(1)
    return centered.square().sum() / denominator


def pair_center_standard_deviation(
    centers: torch.Tensor, *, epsilon: float = 1e-8, unbiased: bool = True,
) -> torch.Tensor:
    if epsilon <= 0.0:
        raise ValueError("PCSF epsilon must be positive")
    return torch.sqrt(pair_center_variance(centers, unbiased=unbiased) + epsilon)


def pcsf_loss(
    source_states: torch.Tensor,
    target_states: torch.Tensor,
    reference_centers: torch.Tensor,
    *,
    rho: float,
    epsilon: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not 0.0 < rho <= 1.0:
        raise ValueError("PCSF rho must be in (0, 1]")
    centers = pair_centers(source_states, target_states)
    if reference_centers.shape != centers.shape:
        raise ValueError("reference centers must match the active logical batch")
    current_sigma = pair_center_standard_deviation(centers, epsilon=epsilon)
    reference_sigma = pair_center_standard_deviation(
        reference_centers.detach(), epsilon=epsilon,
    )
    return (
        torch.relu(rho * reference_sigma - current_sigma).square(),
        current_sigma,
        reference_sigma,
    )


@dataclass
class PreparedPCSF:
    loss: torch.Tensor
    mean: torch.Tensor
    sigma: torch.Tensor
    reference_sigma: torch.Tensor
    threshold: torch.Tensor
    expected_samples: int
    dimensions: int

    @property
    def above_floor(self) -> bool:
        return bool((self.threshold > self.sigma).item())

    def representation_gradients(
        self, source_states: torch.Tensor, target_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        centers = pair_centers(source_states, target_states).detach()
        if not self.above_floor:
            zeros = torch.zeros_like(centers)
            return zeros, zeros
        center_gradient = (
            -2.0
            * (self.threshold - self.sigma)
            * (centers - self.mean)
            / ((self.expected_samples - 1) * self.dimensions * self.sigma)
        )
        endpoint_gradient = center_gradient * 0.5
        return endpoint_gradient, endpoint_gradient

    def surrogate(
        self, source_states: torch.Tensor, target_states: torch.Tensor,
    ) -> torch.Tensor:
        source_gradient, target_gradient = self.representation_gradients(
            source_states, target_states,
        )
        return (
            source_states * source_gradient.to(source_states.dtype)
        ).sum() + (
            target_states * target_gradient.to(target_states.dtype)
        ).sum()


@dataclass
class StreamingPCSF:
    expected_samples: int
    dimensions: int
    rho: float
    epsilon: float
    current_chunks: list[torch.Tensor]
    reference_chunks: list[torch.Tensor]
    samples: int = 0

    def update(
        self,
        source_states: torch.Tensor,
        target_states: torch.Tensor,
        reference_centers: torch.Tensor,
    ) -> None:
        centers = pair_centers(source_states, target_states).detach()
        references = reference_centers.detach().float()
        if centers.size(1) != self.dimensions or references.shape != centers.shape:
            raise ValueError("PCSF current/reference chunk shape mismatch")
        self.current_chunks.append(centers)
        self.reference_chunks.append(references)
        self.samples += centers.size(0)

    def finalize(self) -> PreparedPCSF:
        if self.samples != self.expected_samples:
            raise ValueError("incomplete historical PCSF logical batch")
        current = torch.cat(self.current_chunks, dim=0)
        reference = torch.cat(self.reference_chunks, dim=0)
        loss, sigma, reference_sigma = pcsf_loss(
            current, current, reference, rho=self.rho, epsilon=self.epsilon,
        )
        return PreparedPCSF(
            loss=loss,
            mean=current.mean(dim=0, keepdim=True),
            sigma=sigma,
            reference_sigma=reference_sigma,
            threshold=self.rho * reference_sigma,
            expected_samples=self.expected_samples,
            dimensions=self.dimensions,
        )


class PairCenterSpreadFloor:
    def __init__(self, *, rho: float, epsilon: float = 1e-8):
        if not 0.0 < rho <= 1.0 or epsilon <= 0.0:
            raise ValueError("invalid historical PCSF configuration")
        self.rho = rho
        self.epsilon = epsilon

    def __call__(self, source_states, target_states, reference_centers):
        return pcsf_loss(
            source_states, target_states, reference_centers,
            rho=self.rho, epsilon=self.epsilon,
        )

    def start_streaming(self, *, expected_samples: int, dimensions: int) -> StreamingPCSF:
        return StreamingPCSF(
            expected_samples=expected_samples,
            dimensions=dimensions,
            rho=self.rho,
            epsilon=self.epsilon,
            current_chunks=[],
            reference_chunks=[],
        )
