import pandas as pd
from usq.database.universe_reconstruction import monthly_liquidity_universe


def test_monthly_selection_and_daily_price_gate():
    dates = pd.bdate_range('2020-01-28','2020-02-10')
    close = pd.DataFrame(10.,index=dates,columns=['A','B'])
    dollars = pd.DataFrame({'A':100.,'B':10.},index=dates)
    close.loc['2020-02-04','A'] = 4.
    dollars.loc['2020-02-04':,'B'] = 1000.
    u = monthly_liquidity_universe(close,dollars,close.notna(),keep=.5,lookback=2,min_history=3)
    assert not u.loc[:'2020-01-31'].any().any()
    assert u.loc['2020-02-03','A'] and not u.loc['2020-02-03','B']
    assert not u.loc['2020-02-04','A']
    assert u.loc['2020-02-05','A']
    assert not u.loc['2020-02-04':,'B'].any()


def test_current_signal_close_is_included_and_ties_follow_column_order():
    dates = pd.bdate_range('2020-01-28','2020-02-03')
    close = pd.DataFrame(10.,index=dates,columns=['Z','A','B'])
    dollars = close*10
    u = monthly_liquidity_universe(close,dollars,close.notna(),keep=.5,lookback=2,min_history=3)
    assert u.iloc[-1].tolist() == [True,True,False]
    dollars.loc['2020-02-03','B'] = 10000.
    changed = monthly_liquidity_universe(close,dollars,close.notna(),keep=.5,lookback=2,min_history=3)
    assert changed.iloc[-1].tolist() == [True,False,True]


def test_history_requires_complete_recent_window_and_is_not_ipo_age():
    dates = pd.bdate_range('2020-01-27','2020-02-04')
    close = pd.DataFrame(10.,index=dates,columns=['A'])
    close.loc['2020-01-30','A'] = float('nan')
    u = monthly_liquidity_universe(close,close*100,close.notna(),keep=1.,lookback=2,min_history=3)
    assert not u.loc['2020-02-03':].any().any()


def test_future_rows_do_not_change_earlier_membership():
    dates = pd.bdate_range('2020-01-27','2020-03-10')
    close = pd.DataFrame(10.,index=dates,columns=['A','B'])
    dollars = close.copy()
    dollars.loc['2020-02-20':,'B'] = 10000.
    args = dict(keep=.5,lookback=2,min_history=3)
    full = monthly_liquidity_universe(close,dollars,close.notna(),**args)
    past = monthly_liquidity_universe(close.loc[:'2020-02-10'],dollars.loc[:'2020-02-10'],close.loc[:'2020-02-10'].notna(),**args)
    pd.testing.assert_frame_equal(past,full.loc[:'2020-02-10'])
