import numpy as np
import pandas as pd
import pytest
from usq.project2.supplement import paired_bootstrap
from test_project2_ledger import market


def test_paired_bootstrap_reproducible_and_identity():
    a = np.random.default_rng(8).normal(0, .01, 80)
    result = paired_bootstrap(a, a, reps=100)
    assert result["cagr_difference"] == 0
    assert result["cagr_diff_ci_low"] == 0
    assert result["cagr_diff_ci_high"] == 0
    b = a+.001
    first = paired_bootstrap(a, b, reps=100)
    assert first == paired_bootstrap(a, b, reps=100)
    assert first["mean_diff_ci_low"] == pytest.approx(.252)


def test_bootstrap_requires_aligned_complete_observations():
    with pytest.raises(ValueError):
        paired_bootstrap(np.zeros(40), np.zeros(39))
    with pytest.raises(ValueError):
        paired_bootstrap(np.zeros(40), np.full(40, np.nan))


def test_explicit_equal_weight_pool_is_not_constant_prediction():
    m = market()
    score = pd.DataFrame(1., index=m.index, columns=m.columns)
    daily, w, _, _ = m.backtest(score, "20230101", "20230107", all_eligible=True, cost_bps=0, weights=True)
    np.testing.assert_allclose(w.iloc[2], [.5, .5])
    assert daily.iloc[2].gross_return == pytest.approx(.05)
