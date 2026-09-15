from src.stats import benjamini_hochberg


def test_benjamini_hochberg_preserves_order_and_monotonicity():
    adjusted = benjamini_hochberg([.04, .001, .02, .8])
    assert adjusted == [.05333333333333334, .004, .04, .8]
