import numpy as np

from src.residual_probe import fit_conditional_residual_ridge


def test_duplicated_final_features_have_no_conditional_information():
    rng = np.random.default_rng(14)
    train_final = rng.normal(size=(120, 12))
    test_final = rng.normal(size=(40, 12))
    weights = rng.normal(size=(12, 7))
    train_target = train_final @ weights + rng.normal(scale=.1, size=(120, 7))
    result = fit_conditional_residual_ridge(
        train_final, train_final.copy(), train_target,
        test_final, test_final.copy(), alpha=1.0,
    )
    assert result["unique_train_rms"] < 1e-10
    assert result["unique_test_rms"] < 1e-10
    np.testing.assert_allclose(result["combined_test"], result["base_test"], atol=1e-6, rtol=0)
