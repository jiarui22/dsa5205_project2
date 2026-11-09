import numpy as np
import pandas as pd

from usq.project2.factors import corr, rank, residual, slog, tsrank, wq, surprise_events
from usq.project2.screen import hac_t, information_series


def prices(seed=3):
    rng = np.random.default_rng(seed)
    c = pd.DataFrame(np.exp(rng.normal(0, .02, (300, 40)).cumsum(axis=0))*30)
    return {"close": c, "open": c*.99, "high": c*1.02, "low": c*.97,
            "volume": pd.DataFrame(rng.uniform(100, 1000, c.shape)), "returns": c.pct_change()}


def test_rank_ties_and_full_time_windows():
    x = pd.DataFrame([[1., 1., 3.], [2., 1., 1.], [1., 3., 2.]])
    np.testing.assert_allclose(rank(x).iloc[0], [.5, .5, 1.])
    assert tsrank(x, 3).iloc[:2].isna().all().all()
    assert tsrank(x, 3).iloc[2, 0] == .5


def test_cross_sectional_ols_has_intercept_and_is_orthogonal():
    p = prices()
    x, y = slog(p["volume"]*1e6), slog(p["close"]*1e6)
    u = pd.DataFrame(1, index=x.index, columns=x.columns)
    r = residual(y, x, u)
    np.testing.assert_allclose(r.mean(axis=1), 0, atol=1e-12)
    np.testing.assert_allclose((r*x).sum(axis=1), 0, atol=1e-10)


def test_every_candidate_is_causal():
    from usq.project2.factors import WQ_IDS
    p = prices()
    changed = {k: x.copy() for k, x in p.items()}
    for value in changed.values():
        value.iloc[260:] *= 3
    for number in WQ_IDS:
        a, b = wq(number, p), wq(number, changed)
        pd.testing.assert_frame_equal(a.iloc[:260], b.iloc[:260], obj=f"WQ{number}")


def test_wq101_formula_and_missing_propagation():
    p = prices()
    result = wq(101, p)
    np.testing.assert_allclose(result, (p["close"]-p["open"])/(p["high"]-p["low"]+.001))
    p["high"].iloc[200, 0] = np.nan
    assert np.isnan(wq(101, p).iloc[200, 0])


def test_information_pairs_and_hac():
    x = prices()["close"]
    u = x.notna().astype("uint8")
    s = information_series(x, x, u)
    np.testing.assert_allclose(s.rankic, 1)
    assert np.isfinite(hac_t(pd.Series(np.random.default_rng(2).normal(size=100))))


def test_constant_float32_prediction_cannot_have_rankic():
    x = prices()["close"].astype("float32")
    constant = x*0+np.float32(.012345)
    u = x.notna().astype("uint8")
    s = information_series(constant, x, u)
    assert s.ic.isna().all()
    assert s.rankic.isna().all()


def test_surprise_uses_prior_announced_quarters_and_pre_release_cap():
    periods = pd.date_range("2015-03-31", periods=13, freq="QE")
    release = periods+pd.Timedelta(days=30)
    q = pd.DataFrame({"sid": "a", "period_end": periods,
                      "fyearq": periods.year, "fqtr": periods.quarter,
                      "release_date": release, "available_ts": release+pd.Timedelta(days=1),
                      "earnings_q_available_ts": release+pd.Timedelta(days=1),
                      "earnings_q": np.r_[np.arange(12.), 100.]})
    cal = pd.bdate_range("2018-01-01", "2018-06-01").strftime("%Y%m%d")
    cap = pd.DataFrame(1000., index=cal, columns=["a"])
    cap.loc["20180430", "a"] = 2000.
    e = surprise_events(q, cap)
    assert len(e) == 1
    assert e.iloc[0].expected_ni == 12.
    assert e.iloc[0].cap_date == "20180427"
    assert abs(e.iloc[0].value-.088) < 1e-12
    q.loc[11, "earnings_q_available_ts"] = pd.Timestamp("2018-05-02")
    # All required historical constituents must precede the target announcement.
    empty = surprise_events(q, cap)
    assert empty.empty
