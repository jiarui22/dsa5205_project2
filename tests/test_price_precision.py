import numpy as np
import pandas as pd
import pytest

from usq.data.adjustments import canonical_close, adjusted_bars, observed_returns
from usq.data.labels import close_forward_returns
from usq.database.common import save_panel
from usq.project2.factors import price_inputs, wq, rank


def inputs():
    rng = np.random.default_rng(5205)
    dates = pd.Index(pd.bdate_range('2017-01-02', periods=320).strftime('%Y%m%d'))
    columns = pd.Index(list('ABCDEFGH'))
    r = pd.DataFrame(rng.normal(0, .012, (320, 8)), index=dates, columns=columns)
    r.iloc[30:45, :3] = 0
    close = (1+r).cumprod().mul(np.arange(8)+10)
    # A split changes the raw price basis without changing total return.
    close.iloc[160:, :4] /= 3
    source_returns=r.copy()
    source_returns.iloc[60::20] += .001  # Cash distributions enter total returns.
    panels = {name: close*scale for name,scale in [('close',1),('open',.999),('high',1.02),('low',.98)]}
    panels.update(daily_return=source_returns, adjustment_segment=close*0+1,
                  volume=close*0+100, returns=source_returns)
    panels['adj_factor']=(canonical_close(close,source_returns,panels['adjustment_segment'])/close).astype('float32')
    return panels, close*0+1


def test_canonical_price_matches_source_returns_and_stays_flat():
    panels, _ = inputs()
    c = canonical_close(panels['close'], panels['daily_return'], panels['adjustment_segment'])
    np.testing.assert_allclose(c.iloc[1:].to_numpy()/c.shift().iloc[1:].to_numpy()-1,
                               panels['daily_return'].iloc[1:], rtol=1e-12, atol=3e-16)
    assert c.iloc[30:45, :3].eq(c.iloc[29, :3]).all().all()
    assert c.dtypes.eq('float64').all()


def test_wq_features_ignore_arbitrary_stored_normalization_exactly():
    panels, u = inputs()
    first = price_inputs(panels.__getitem__, u)
    for j,name in enumerate(panels['adj_factor']):
        panels['adj_factor'][name] = (panels['adj_factor'][name].astype('float64')*10.**(j-4)).astype('float32')
    second = price_inputs(panels.__getitem__, u)
    for number in [4, 19, 26, 35, 38, 52]:
        pd.testing.assert_frame_equal(rank(wq(number, first)), rank(wq(number, second)), check_exact=True)


def test_future_changes_do_not_change_prices_features_or_mature_labels():
    panels, u = inputs()
    first = price_inputs(panels.__getitem__, u)
    a = canonical_close(panels['close'],panels['daily_return'],panels['adjustment_segment'])
    y, ends = close_forward_returns(a, segment=panels['adjustment_segment'])
    for name in ['open','high','low','close','daily_return']:
        panels[name].iloc[280:] *= 1.1
    second = price_inputs(panels.__getitem__, u)
    b = canonical_close(panels['close'],panels['daily_return'],panels['adjustment_segment'])
    z, _ = close_forward_returns(b, segment=panels['adjustment_segment'])
    pd.testing.assert_frame_equal(a.iloc[:280],b.iloc[:280],check_exact=True)
    pd.testing.assert_frame_equal(y.loc[ends.lt(u.index[280])],z.loc[ends.lt(u.index[280])],check_exact=True)
    for number in [4,19,26,35,38,52]:
        pd.testing.assert_frame_equal(wq(number,first).iloc[:280],wq(number,second).iloc[:280],check_exact=True)


def test_quarantine_carries_known_return_but_identity_break_resets():
    p = pd.DataFrame({'A':[100.,np.nan,50.,25.,30.]})
    r = pd.DataFrame({'A':[np.nan,.1,.2,.5,.2]})
    seg = pd.DataFrame({'A':[1,1,1,2,2]})
    c = canonical_close(p,r,seg)
    np.testing.assert_allclose(c.A,[100.,np.nan,132.,25.,30.],equal_nan=True)
    daily = observed_returns(c,r,seg)
    assert daily.iloc[:4].isna().all().all()
    assert daily.iloc[4,0] == .2
    with pytest.raises(ValueError,match='invalid linking return'):
        canonical_close(p,r.mask(r.eq(.2)),seg)


def test_float64_panel_serialization_does_not_quantize(tmp_path):
    p = pd.DataFrame({'A':[1.,1.+1e-10]})
    info = save_panel(p,tmp_path/'price.pkl',p.index,p.columns,numeric_dtype='float64')
    assert info['dtype']=='float64'
    pd.testing.assert_frame_equal(pd.read_pickle(tmp_path/'price.pkl'),p,check_exact=True)


def test_next_close_labels_equal_ten_source_return_intervals():
    panels, _ = inputs()
    c=canonical_close(panels['close'],panels['daily_return'],panels['adjustment_segment'])
    target,_=close_forward_returns(c,10,panels['adjustment_segment'])
    compounded=pd.DataFrame(1.,index=c.index,columns=c.columns)
    for offset in range(2,12):
        compounded*=1+panels['daily_return'].shift(-offset)
    np.testing.assert_allclose(target.iloc[:-11],(compounded-1).iloc[:-11],rtol=5e-13,atol=5e-13)
