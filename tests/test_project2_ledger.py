import numpy as np
import pandas as pd
from usq.project2.ledger import Market


def market():
    m = Market.__new__(Market)
    m.index = pd.Index([f"2023010{i}" for i in range(1, 8)])
    m.columns = pd.Index(["a", "b"])
    m.close = np.full((7, 2), 100.)
    m.universe = np.ones((7, 2), dtype=bool)
    m.tradable = np.ones((7, 2), dtype=bool)
    m.ret = np.array([[0., 0.], [.9, 0.], [.1, 0.], [0., 0.], [0., 0.], [0., 0.], [0., 0.]])
    m.terminal = np.zeros((7, 2), dtype=bool)
    m.source_return = np.zeros((7, 2))
    m.spy = np.zeros(7)
    return m


def test_delayed_close_execution_and_cost_reconcile():
    m = market()
    s = pd.DataFrame([[2., 1.]]*7, index=m.index, columns=m.columns)
    daily, weights, events, summary = m.backtest(s, "20230101", "20230107", top_n=1, cost_bps=10, rebalance_every=1, weights=True)
    assert weights.iloc[1].sum() == 0
    assert weights.iloc[2, 0] == 1
    assert daily.iloc[1].gross_return == 0  # cannot earn execution-day .9
    assert abs(daily.iloc[2].gross_return-.1) < 1e-12
    assert abs(daily.iloc[1].cost-.001/1.001) < 1e-12
    assert daily.iloc[-1].cost > 0
    np.testing.assert_allclose(daily.net_return, daily.gross_return-daily.cost)


def test_constant_scores_are_cash_not_sid_order_strategy():
    m = market()
    s = pd.DataFrame(3., index=m.index, columns=m.columns)
    daily, weights, _, _ = m.backtest(s, "20230101", "20230107", top_n=1, weights=True)
    assert (weights == 0).all().all()
    assert (daily.net_return == 0).all()


def test_unfillable_buy_stays_cash_without_future_reselection():
    m = market()
    m.tradable[1, 0] = False
    s = pd.DataFrame([[2., 1.]]*7, index=m.index, columns=m.columns)
    daily, weights, _, _ = m.backtest(s, "20230101", "20230107", top_n=1, rebalance_every=5, weights=True)
    assert (weights.iloc[2:6] == 0).all().all()


def test_terminal_missing_close_uses_event_then_cash():
    m = market()
    m.ret[2, 0] = np.nan
    m.terminal[2, 0] = True
    m.source_return[2, 0] = -.25
    m.tradable[2:, 0] = False
    s = pd.DataFrame([[2., 1.]]*7, index=m.index, columns=m.columns)
    daily, weights, events, _ = m.backtest(s, "20230101", "20230107", top_n=1, cost_bps=0, rebalance_every=5, weights=True)
    assert daily.iloc[2].gross_return == -.25
    assert weights.iloc[3].sum() == 0
    assert events.iloc[0].policy == "terminal_source_return_cash_settlement"
