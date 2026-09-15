"""Capacity-controlled conditional residual probe used by the amendment."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge


def fit_conditional_residual_ridge(
    train_final: np.ndarray,
    train_extra: np.ndarray,
    train_target: np.ndarray,
    test_final: np.ndarray,
    test_extra: np.ndarray,
    *,
    alpha: float = 1.0,
    residualizer_alpha: float = 1e-4,
):
    """Predict target from final features, then only from extra-feature residuals.

    The extra representation is first linearly residualized against the final
    representation on train data. Consequently, an exact duplicate of the
    final representation has no conditional feature with which to improve the
    test prediction.
    """
    base = Ridge(alpha=alpha, fit_intercept=True).fit(train_final, train_target)
    base_train = base.predict(train_final)
    base_test = base.predict(test_final)
    residualizer = Ridge(alpha=residualizer_alpha, fit_intercept=True).fit(train_final, train_extra)
    unique_train = train_extra - residualizer.predict(train_final)
    unique_test = test_extra - residualizer.predict(test_final)
    correction = Ridge(alpha=alpha, fit_intercept=True).fit(
        unique_train, train_target - base_train
    )
    correction_test = correction.predict(unique_test)
    return {
        "base_train": base_train.astype(np.float32),
        "base_test": base_test.astype(np.float32),
        "correction_test": correction_test.astype(np.float32),
        "combined_test": (base_test + correction_test).astype(np.float32),
        "unique_train_rms": float(np.sqrt(np.mean(unique_train ** 2))),
        "unique_test_rms": float(np.sqrt(np.mean(unique_test ** 2))),
    }
