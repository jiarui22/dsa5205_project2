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


def test_terminal_source_return_overrides_finite_stale_mark(tmp_path):
    # Exercise the constructor: ffill makes the missing terminal close produce
    # a finite zero return while the vendor records an actual terminal loss.
    p = tmp_path/'panels'; p.mkdir()
    dates = pd.Index(['20230103', '20230104', '20230105', '20230106'])
    close = pd.DataFrame(100., index=dates, columns=['a', 'b'])
    close.loc['20230105', 'a'] = np.nan
    fields = {'close_adj': close,
              'universe': pd.DataFrame(1, index=dates, columns=close.columns),
              'trade_available': close.notna().astype(int),
              'adjustment_segment': pd.DataFrame(1., index=dates, columns=close.columns),
              'daily_return': pd.DataFrame(0., index=dates, columns=close.columns),
              'delisting_event': pd.DataFrame(0, index=dates, columns=close.columns),
              'indices_total_return_close': pd.DataFrame({'SPY': 100.}, index=dates)}
    fields['daily_return'].loc['20230105', 'a'] = -.5
    fields['delisting_event'].loc['20230105', 'a'] = 1
    for name, frame in fields.items():
        frame.to_pickle(p/f'{name}.pkl')
    m = Market(tmp_path)
    assert m.ret[2, 0] == 0  # finite stale mark previously masked the loss
    score = pd.DataFrame([[2., 1.]]*4, index=dates, columns=close.columns)
    daily, _, events, _ = m.backtest(score, dates[0], dates[-1], top_n=1,
                                    cost_bps=0, rebalance_every=5)
    assert daily.loc['20230105', 'gross_return'] == -.5
    assert daily.loc['20230105', 'cash_weight'] == 1
    assert daily.net_nav.iloc[-1] == .5
    assert events.iloc[0].policy == 'terminal_source_return_cash_settlement'


def test_terminal_last_mark_assumption_is_logged_even_if_mark_is_finite():
    m = market()
    m.terminal[2, 0] = True
    m.source_return[2, 0] = np.nan
    m.ret[2, 0] = 0.
    score = pd.DataFrame([[2., 1.]]*7, index=m.index, columns=m.columns)
    _, _, events, _ = m.backtest(score, m.index[0], m.index[-1], top_n=1,
                                cost_bps=0, rebalance_every=5)
    assert events.iloc[0].policy == 'terminal_last_mark_cash_assumption'


def test_top100_boundary_ties_do_not_depend_on_panel_column_order():
    names = pd.Index([f'S{i:03d}' for i in range(250)])
    dates = pd.Index(['20230103', '20230104', '20230105', '20230106'])
    values = pd.Series(np.r_[np.ones(200)*2, np.ones(50)], index=names)
    def held(columns):
        m = Market.__new__(Market)
        m.index, m.columns = dates, columns
        shape = (len(dates), len(columns))
        m.close = np.full(shape, 100.)
        m.universe = m.tradable = np.ones(shape, dtype=bool)
        m.ret = m.source_return = np.zeros(shape)
        m.terminal = np.zeros(shape, dtype=bool)
        m.spy = np.zeros(len(dates))
        score = pd.DataFrame([values.reindex(columns).to_numpy()]*len(dates), index=dates, columns=columns)
        _, w, _, _ = m.backtest(score, dates[0], dates[-1], top_n=100,
                                 cost_bps=0, rebalance_every=5, weights=True)
        return set(w.columns[w.iloc[2]>0])
    chosen = held(names)
    assert len(chosen) == 100
    assert chosen <= set(names[:200])  # score priority is preserved
    assert chosen != set(names[:100])  # no inherited SID-order portfolio
    assert chosen == held(names[::-1])
    assert chosen == held(names)
