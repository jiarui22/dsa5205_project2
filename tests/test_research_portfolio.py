import numpy as np
import pandas as pd
from usq.project2.ledger import stable_tie_ranks,top_positions
from usq.project2.research_portfolio import buffered_positions,normalized_weights,additional_features,ResearchMarket


def test_buffer_retains_eligible_existing_positions_and_fills_from_best():
    scores=np.arange(200.,0,-1);eligible=np.ones(200,bool);held=np.zeros(200,bool);held[[0,119,149,150]]=True
    keys=stable_tie_ranks(pd.Index([str(i) for i in range(200)]))
    selected=buffered_positions(scores,eligible,keys,held)
    assert len(selected)==len(set(selected))==100
    assert {0,119,149}.issubset(set(selected)) and 150 not in selected
    assert set(range(98)).issubset(set(selected))
    eligible[119]=False;selected=buffered_positions(scores,eligible,keys,held)
    assert 119 not in selected
    nohold=buffered_positions(scores,eligible,keys,np.zeros(200,bool))
    np.testing.assert_array_equal(nohold,top_positions(scores,eligible,keys))


def test_buffer_ties_column_invariance_and_constant_cash():
    names=pd.Index([f's{i}' for i in range(200)]);scores=np.repeat([1.,0.],100);held=np.arange(200)%3==0;ok=np.ones(200,bool)
    a=buffered_positions(scores,ok,stable_tie_ranks(names),held)
    b=buffered_positions(scores[::-1],ok,stable_tie_ranks(names[::-1]),held[::-1])
    assert set(names[a])==set(names[::-1][b])
    assert not len(buffered_positions(np.ones(200),ok,stable_tie_ranks(names),held))


def test_capped_weights_conserve_capital_and_do_not_use_input_order():
    raw=np.r_[1000.,np.arange(1.,100.)];w=normalized_weights(raw,.02)
    np.testing.assert_allclose(w.sum(),1);assert w.max()<=.02+1e-12
    np.testing.assert_allclose(w,normalized_weights(raw[::-1],.02)[::-1])
    np.testing.assert_allclose(normalized_weights(np.ones(20),.02),.02)


def test_extra_features_are_causal_and_do_not_bridge_segments():
    rng=np.random.default_rng(3);c=pd.DataFrame(np.exp(rng.normal(0,.02,(400,5)).cumsum(0))*20);seg=c*0+1;cap=c*1e7;volume=c*0+1000
    original=additional_features(c,seg,cap,c,volume)
    changed=c.copy();changed.iloc[350:]*=20
    future=additional_features(changed,seg,cap,c,volume)
    for name in original:pd.testing.assert_frame_equal(original[name].iloc[:350],future[name].iloc[:350])
    seg.iloc[300:]=2;x=additional_features(c,seg,cap,c,volume)
    assert x['momentum_12_1'].iloc[310].isna().all()
    assert x['momentum_12_1'].iloc[330].isna().all()
    assert x['low_volatility_63'].iloc[320].isna().all()


def test_terminal_execution_day_does_not_create_future_based_buffer_replacement():
    market=ResearchMarket.__new__(ResearchMarket);market.index=pd.bdate_range('2020-01-01',periods=8).strftime('%Y%m%d');market.columns=pd.Index([str(i) for i in range(160)])
    shape=(8,160);market.close=np.ones(shape)*100;market.universe=np.ones(shape,bool);market.tradable=np.ones(shape,bool)
    market.ret=np.zeros(shape);market.terminal=np.zeros(shape,bool);market.source_return=np.full(shape,np.nan);market.spy=np.zeros(8);market.cap=np.ones(shape)*1e8
    score=pd.DataFrame(np.tile(np.arange(160.,0,-1),(8,1)),index=market.index,columns=market.columns)
    score.iloc[3:,0]=40.5  # incumbent falls to rank 120, inside retention band
    market.terminal[4,0]=True;market.source_return[4,0]=-1;market.tradable[4,0]=False
    d,w,_,_=market.backtest(score,market.index[0],market.index[-1],rebalance_every=3,buffer_n=150,weights=True)
    assert w.iloc[5,100]==0  # no new substitute bought using T+1 terminal knowledge
    assert d.cash_weight.iloc[4]>.009
