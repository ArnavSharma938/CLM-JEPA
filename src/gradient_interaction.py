"""Published two-objective gradient interaction rules for cLM-JEPA.

The functions operate on Gram statistics rather than flattened parameter
vectors.  This is equivalent to the published vector formulas and avoids a
second multi-million-element allocation for the LoRA gradient vector.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np
import torch
from scipy.optimize import minimize_scalar


EPSILON = 1e-30
CAGRAD_EPSILON = 1e-4
CAGRAD_C = 0.5


@dataclass(frozen=True)
class GradientCombination:
    method: str
    main_coefficient: float
    auxiliary_coefficient: float
    cosine: float
    auxiliary_to_main_norm_ratio: float
    modification_norm: float
    modification_relative_to_raw_sum: float
    conflict: bool
    auxiliary_gate: float | None = None
    cagrad_weight_main: float | None = None
    cagrad_weight_auxiliary: float | None = None
    cagrad_lambda: float | None = None

    def as_dict(self) -> dict[str, float | str | bool | None]:
        return asdict(self)


def _gram(
    main: Sequence[torch.Tensor], auxiliary: Sequence[torch.Tensor],
) -> tuple[float, float, float]:
    if len(main) != len(auxiliary) or not main:
        raise ValueError("main and auxiliary gradients must be nonempty and aligned")
    device = main[0].device
    # A single FP32 device reduction avoids per-tensor host synchronizations and
    # slow consumer-GPU FP64 kernels.  The resulting scalar Gram matrix is then
    # promoted to Python/NumPy double for the two-variable algebra.
    terms = torch.zeros(3, device=device, dtype=torch.float32)
    for left, right in zip(main, auxiliary):
        if left.shape != right.shape:
            raise ValueError("gradient tensors must have matching shapes")
        left32 = left.detach().float()
        right32 = right.detach().float()
        terms += torch.stack((
            left32.square().sum(), right32.square().sum(),
            (left32 * right32).sum(),
        ))
    return tuple(float(value) for value in terms.cpu())


def _linear_combination_norm(
    main_coefficient: float, auxiliary_coefficient: float,
    main_squared: float, auxiliary_squared: float, dot: float,
) -> float:
    squared = (
        main_coefficient * main_coefficient * main_squared
        + auxiliary_coefficient * auxiliary_coefficient * auxiliary_squared
        + 2.0 * main_coefficient * auxiliary_coefficient * dot
    )
    return float(np.sqrt(max(0.0, squared)))


def combine_gradients(
    method: str,
    main: Sequence[torch.Tensor],
    auxiliary: Sequence[torch.Tensor],
    *,
    cagrad_c: float = CAGRAD_C,
) -> GradientCombination:
    """Return coefficients for a published two-gradient combination.

    ``auxiliary`` must already include the experiment's active-step JEPA
    coefficient.  ``pcgrad`` follows the requested asymmetric form: only the
    conflicting component of the auxiliary gradient is projected away.
    ``aux_similarity`` is the weighted rule in Proposition 1 / Algorithm 2 of
    Du et al. (2018).  ``cagrad`` follows the official two-task solver and its
    original ``1 / (1 + c)`` rescaling.
    """
    if method not in {"weighted_sum", "pcgrad", "cagrad", "aux_similarity"}:
        raise ValueError(f"unknown gradient interaction method: {method}")
    main_squared, auxiliary_squared, dot = _gram(main, auxiliary)
    main_norm = np.sqrt(max(0.0, main_squared))
    auxiliary_norm = np.sqrt(max(0.0, auxiliary_squared))
    cosine = dot / max(main_norm * auxiliary_norm, EPSILON)
    cosine = float(np.clip(cosine, -1.0, 1.0))
    ratio = auxiliary_norm / max(main_norm, EPSILON)
    conflict = dot < 0.0
    main_coefficient = 1.0
    auxiliary_coefficient = 1.0
    gate = None
    cagrad_weight_main = None
    cagrad_weight_auxiliary = None
    cagrad_lambda = None

    if method == "pcgrad" and conflict and main_squared > 0.0:
        main_coefficient = 1.0 - dot / main_squared
    elif method == "aux_similarity":
        gate = max(0.0, cosine)
        auxiliary_coefficient = gate
    elif method == "cagrad":
        if cagrad_c < 0.0:
            raise ValueError("CAGrad c must be nonnegative")
        g0_norm = 0.5 * np.sqrt(
            main_squared + auxiliary_squared + 2.0 * dot + CAGRAD_EPSILON
        )
        coefficient = cagrad_c * g0_norm

        def objective(weight_main: float) -> float:
            weight_auxiliary = 1.0 - weight_main
            gw_squared = (
                weight_main * weight_main * main_squared
                + weight_auxiliary * weight_auxiliary * auxiliary_squared
                + 2.0 * weight_main * weight_auxiliary * dot
            )
            gw_dot_g0 = 0.5 * (
                weight_main * (main_squared + dot)
                + weight_auxiliary * (auxiliary_squared + dot)
            )
            return gw_dot_g0 + coefficient * np.sqrt(
                max(0.0, gw_squared) + CAGRAD_EPSILON
            )

        solution = minimize_scalar(
            objective, bounds=(0.0, 1.0), method="bounded",
            options={"xatol": 1e-8},
        )
        if not solution.success:
            raise RuntimeError(f"CAGrad dual solve failed: {solution.message}")
        cagrad_weight_main = float(solution.x)
        cagrad_weight_auxiliary = 1.0 - cagrad_weight_main
        gw_norm_without_epsilon = _linear_combination_norm(
            cagrad_weight_main, cagrad_weight_auxiliary,
            main_squared, auxiliary_squared, dot,
        )
        gw_norm = np.sqrt(
            gw_norm_without_epsilon * gw_norm_without_epsilon + CAGRAD_EPSILON
        )
        cagrad_lambda = coefficient / (gw_norm + CAGRAD_EPSILON)
        scale = 1.0 / (1.0 + cagrad_c)
        main_coefficient = scale * (
            0.5 + cagrad_lambda * cagrad_weight_main
        )
        auxiliary_coefficient = scale * (
            0.5 + cagrad_lambda * cagrad_weight_auxiliary
        )

    delta_main = main_coefficient - 1.0
    delta_auxiliary = auxiliary_coefficient - 1.0
    modification_norm = _linear_combination_norm(
        delta_main, delta_auxiliary, main_squared, auxiliary_squared, dot,
    )
    raw_sum_norm = _linear_combination_norm(
        1.0, 1.0, main_squared, auxiliary_squared, dot,
    )
    return GradientCombination(
        method=method,
        main_coefficient=float(main_coefficient),
        auxiliary_coefficient=float(auxiliary_coefficient),
        cosine=cosine,
        auxiliary_to_main_norm_ratio=float(ratio),
        modification_norm=modification_norm,
        modification_relative_to_raw_sum=(
            modification_norm / max(raw_sum_norm, EPSILON)
        ),
        conflict=conflict,
        auxiliary_gate=gate,
        cagrad_weight_main=cagrad_weight_main,
        cagrad_weight_auxiliary=cagrad_weight_auxiliary,
        cagrad_lambda=cagrad_lambda,
    )


def apply_combination(
    parameters: Sequence[torch.nn.Parameter],
    main: Sequence[torch.Tensor],
    auxiliary: Sequence[torch.Tensor],
    combination: GradientCombination,
) -> None:
    if not (len(parameters) == len(main) == len(auxiliary)):
        raise ValueError("parameters and gradient lists must be aligned")
    # LoRA exposes hundreds of small tensors.  Foreach turns the elementwise
    # scale/add into multi-tensor CUDA launches instead of two launches per
    # parameter while preserving the published linear combination.
    combined = torch._foreach_mul(list(main), combination.main_coefficient)
    torch._foreach_add_(
        combined, list(auxiliary), alpha=combination.auxiliary_coefficient,
    )
    for parameter, gradient in zip(parameters, combined):
        parameter.grad = gradient.to(dtype=parameter.dtype)
