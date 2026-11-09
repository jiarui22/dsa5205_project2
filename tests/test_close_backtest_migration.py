"""Close-only timing, costs, risk alignment, and no-VWAP integration checks."""
import json
import numpy as np
import pandas as pd
import pytest

from usq.data import PanelStore
from usq.backtest.engine import run_backtest, group_backtests
from usq.barra.model import build_model
from usq.barra.attribution import analyze
from usq import workflow


def small():
    dates = pd.bdate_range('2023-01-02', periods=8).strftime('%Y%m%d')
    price = pd.DataFrame({'A': [100, 110, 121, 133.1, 146.41, 161.051, 177.1561, 194.87171],
                          'B': [100.]*8}, index=dates)
    score = price*0+1; score.A = 2
    return score, price, price.notna()


def test_entry_is_next_close_not_same_close_and_initial_cost_has_own_day():
    score, price, universe = small()
    bt = run_backtest(score, price, universe, top_n=1, cost_bps=10)
    first = bt.daily.iloc[0]
    assert bt.daily.index[0] == price.index[1]
    assert first.gross_return == 0
    assert first.net_return == pytest.approx(-.001/1.001)
    assert first.turnover == pytest.approx(.5/1.001)
    assert bt.daily.iloc[1].gross_return == pytest.approx(.1)
    assert bt.weights.iloc[0].eq(0).all()
    assert bt.post_close_weights.iloc[0].A == 1
    assert bt.post_close_weights.iloc[-1].eq(0).all()
    assert bt.daily.iloc[-1].cost == pytest.approx(.001*1.1)


def test_same_day_signal_changes_cannot_change_that_days_holdings_or_trades():
    score, price, universe = small()
    a = run_backtest(score, price, universe, top_n=1, cost_bps=0)
    changed = score.copy(); changed.iloc[3:, 0] = 0
    b = run_backtest(changed, price, universe, top_n=1, cost_bps=0)
    pd.testing.assert_frame_equal(a.daily.iloc[:3], b.daily.iloc[:3])
    pd.testing.assert_frame_equal(a.post_close_weights.iloc[:3], b.post_close_weights.iloc[:3])


def test_bad_execution_quote_does_not_choose_the_runner_up():
    score, price, universe = small()
    price.iloc[1, 0] = np.nan
    with pytest.raises(ValueError, match='execution close'):
        run_backtest(score, price, universe, top_n=1)


def test_future_untradability_does_not_silently_reselect():
    score, price, universe = small()
    tradable = universe.copy(); tradable.iloc[1, 0] = False
    with pytest.raises(ValueError, match='no future-based reselection'):
        run_backtest(score, price, universe, top_n=1, trade_available=tradable)


def test_unknown_adjustment_link_is_fatal_when_held():
    score, price, universe = small()
    segment = price*0+1; segment.iloc[2:, 0] = 2
    with pytest.raises(ValueError, match='missing held return'):
        run_backtest(score, price, universe, top_n=1, adjustment_segment=segment)


def test_low_frequency_drift_no_intermediate_trades_and_self_financing_cost():
    score, price, universe = small()
    bt = run_backtest(score, price, universe, top_n=2, cost_bps=10, rebalance_every=3)
    assert bt.daily.iloc[1:3].traded_notional.eq(0).all()
    assert bt.post_close_weights.iloc[1].A == pytest.approx(.55/1.05)
    assert np.allclose(bt.daily.cost, bt.daily.traded_notional*.001, atol=1e-12)
    expected = (bt.weights*(price/price.shift(1)-1).reindex(bt.weights.index).fillna(0)).sum(axis=1)
    assert np.allclose(bt.daily.net_return, expected-bt.daily.cost-bt.daily.borrow_cost)


def test_group_returns_include_entry_cost_even_with_zero_beginning_exposure():
    score, price, universe = small()
    groups = group_backtests(score, price, universe, groups=2, cost_bps=10)
    assert groups.iloc[0].notna().all() and groups.iloc[0].lt(0).all()


def make_store(tmp_path):
    rng = np.random.default_rng(723)
    dates = pd.bdate_range('2021-01-01', periods=620).strftime('%Y%m%d')
    ids = pd.Index([f'S{i:02d}' for i in range(60)])
    close = pd.DataFrame(50*np.exp(np.cumsum(rng.normal(.0002, .01, (620, 60)), axis=0)), index=dates, columns=ids)
    store = PanelStore(tmp_path/'panels')
    cap = close*pd.Series(rng.lognormal(16, 1, 60), index=ids)
    for name, frame in {'close_adj': close, 'adjustment_segment': close*0+1,
        'market_cap': cap, 'volume': pd.DataFrame(rng.lognormal(14, .4, close.shape), index=dates, columns=ids),
        'crsp_shares_out': cap/close,
        'universe': close*0+1, 'trade_available': close*0+1,
        'book_equity': cap*rng.lognormal(-.5, .4, close.shape),
        'earnings_ttm': cap*rng.normal(.05, .02, close.shape),
        'operating_cash_flow_ttm': cap*rng.normal(.06, .02, close.shape),
        'total_assets': cap*rng.lognormal(.7, .4, close.shape),
        'total_liabilities': cap*rng.lognormal(-.5, .4, close.shape)}.items():
        store.save(name, frame)
    store.save('industry', pd.DataFrame(np.tile(np.where(np.arange(60)%2, 'Tech', 'Other'), (620, 1)), index=dates, columns=ids), numeric=False)
    # Deliberately different columns: a benchmark is not aligned to stock SIDs.
    store.save('indices_total_return_close', pd.DataFrame({'SPY': np.arange(620)+100.}, index=dates))
    score = close*0+np.nan
    score.iloc[-10:] = rng.normal(size=(10, 60))
    score.to_pickle(tmp_path/'scores.pkl')
    return store, dates, score


def test_close_workflow_barra_and_spy_need_no_vwap_and_ignore_nan_vwap(tmp_path):
    store, dates, score = make_store(tmp_path)
    cfg = {'data_dir': str(store.root), 'run_dir': str(tmp_path/'run'),
        'backtest': {'top_n': 10, 'groups': 3, 'cost_bps': 10, 'borrow_bps_annual': 300,
                     'benchmark': 'spy'},
        'barra': {'model_dir': str(tmp_path/'risk'), 'start': dates[-9], 'end': dates[-1]}}
    with pytest.warns(UserWarning):
        workflow.backtest(cfg, tmp_path/'scores.pkl', with_barra=True)
    a = pd.read_csv(tmp_path/'run/backtest/daily.csv', dtype={'date': str})
    attr = pd.read_csv(tmp_path/'run/barra/daily_returns.csv')
    assert np.allclose(a.net_return, attr.portfolio)
    assert a.benchmark_return.iloc[0] == 0
    assert a.benchmark_return.iloc[1] == pytest.approx(1/(100+611))
    metadata = json.loads((tmp_path/'run/barra/model_metadata.json').read_text())
    assert metadata['price_field'] == 'close_adj'
    assert metadata['cash_flow_field'] == 'operating_cash_flow_ttm'
    assert 'crsp_shares_out' in metadata['turnover_input']
    assert '非 SPY' in metadata['benchmark_name']
    model = workflow.barra_model(cfg)
    assert model.alignment.iloc[0].signal_date == dates[-10]
    assert model.alignment.iloc[0].entry_date == dates[-10]
    pd.testing.assert_frame_equal(model.returns, (store.load('close_adj')/store.load('close_adj').shift(1)-1))
    for name in ['vwap', 'vwap_adj', 'amount']:
        store.save(name, store.load('close_adj')*np.nan)
    # Adding/changing an UNUSED VWAP must neither invalidate risk cache nor P&L.
    workflow.barra_model(cfg)
    cfg['run_dir'] = str(tmp_path/'run_nan_vwap')
    workflow.backtest(cfg, tmp_path/'scores.pkl', with_barra=True)
    b = pd.read_csv(tmp_path/'run_nan_vwap/backtest/daily.csv', dtype={'date': str})
    pd.testing.assert_frame_equal(a, b)
    assert 'VWAP' not in (tmp_path/'run_nan_vwap/backtest/report.html').read_text()
    assert 'VWAP' not in (tmp_path/'run_nan_vwap/barra/report.html').read_text()


def test_missing_spy_does_not_fall_back_to_cap_weighted_reference(tmp_path):
    score, price, universe = small()
    bt = run_backtest(score, price, universe, top_n=1)
    with pytest.raises(ValueError, match='no silent'):
        workflow.attach_performance_benchmark(bt, PanelStore(tmp_path), price, universe, {})
