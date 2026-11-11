"""Research-only buffered selection and capitalization weights on the same ledger."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .ledger import top_positions
from .extensions import CachedMarket


def buffered_positions(values,eligible,tie_ranks,held,top_n=100,buffer_n=150):
    candidates=np.flatnonzero(eligible&np.isfinite(values))
    if not len(candidates) or np.unique(values[candidates]).size<2:return candidates[:0]
    if buffer_n<top_n:raise ValueError('Buffer must be at least entry count')
    ordered=candidates[np.lexsort((tie_ranks[candidates],-values[candidates]))]
    retained=ordered[:buffer_n][np.asarray(held,dtype=bool)[ordered[:buffer_n]]][:top_n]
    entrants=ordered[~np.isin(ordered,retained)][:max(top_n-len(retained),0)]
    return np.r_[retained,entrants]


def normalized_weights(raw,max_weight=None):
    """Water-filling with a per-name target cap; infeasible residual stays cash."""
    raw=np.asarray(raw,dtype=float)
    if (~np.isfinite(raw)).any() or (raw<=0).any():raise ValueError('Positive finite weights required')
    if not len(raw):return raw
    if max_weight is None:return raw/raw.sum()
    if not 0<max_weight<=1:raise ValueError('Invalid weight cap')
    if len(raw)*max_weight<=1:return np.full(len(raw),max_weight)
    remaining=np.ones(len(raw),bool);out=np.zeros(len(raw));budget=1.
    while remaining.any():
        ids=np.flatnonzero(remaining);proposal=raw[ids]/raw[ids].sum()*budget;hit=proposal>max_weight
        if not hit.any():out[ids]=proposal;break
        fixed=ids[hit];out[fixed]=max_weight;remaining[fixed]=False;budget=1-out.sum()
    return out


class ResearchMarket(CachedMarket):
    def __init__(self,database):
        super().__init__(database)
        from pathlib import Path
        cap=pd.read_pickle(Path(database)/'panels/market_cap.pkl')
        if not cap.index.equals(self.index) or not cap.columns.equals(self.columns):raise ValueError('Cap axes differ')
        self.cap=cap.to_numpy();self.buffer_n=None;self.weighting='equal';self.max_weight=None;self.decisions=[]

    def backtest(self,*args,buffer_n=None,weighting='equal',max_weight=None,**kwargs):
        if weighting not in ['equal','sqrt_cap','cap']:raise ValueError('Unknown weighting')
        self.buffer_n=buffer_n;self.weighting=weighting;self.max_weight=max_weight;self.decisions=[]
        result=super().backtest(*args,**kwargs)
        result[3].update(buffer_n=buffer_n,weighting=weighting,max_target_weight=max_weight)
        return result

    def _select_positions(self,values,eligible,tie_ranks,top_n,held,signal_index):
        if self.buffer_n is None:order=top_positions(values,eligible,tie_ranks,top_n)
        else:order=buffered_positions(values,eligible,tie_ranks,held,top_n,self.buffer_n)
        self.decisions.append({'signal_date':self.index[signal_index],'candidate_count':int((eligible&np.isfinite(values)).sum()),'selected':len(order),'retained':int(np.asarray(held)[order].sum()),'new_entries':int((~np.asarray(held)[order]).sum())})
        return order

    def _allocation_weights(self,order,signal_index):
        if self.weighting=='equal' and self.max_weight is None:return super()._allocation_weights(order,signal_index)
        raw=np.ones(len(order)) if self.weighting=='equal' else np.asarray(self.cap[signal_index,order],dtype=float)
        if self.weighting=='sqrt_cap':raw=np.sqrt(raw)
        return normalized_weights(raw,self.max_weight)


def additional_features(close,segments,cap,raw_close,volume):
    """Causal, fixed definitions; raw dollar volume is not adjusted-price volume."""
    for x in [segments,cap,raw_close,volume]:
        if not x.index.equals(close.index) or not x.columns.equals(close.columns):raise ValueError('Feature axes differ')
    close=close.where(close>0)
    same=segments.gt(0)&segments.eq(segments.shift())
    returns=(close/close.shift()-1).where(same)
    momentum=(close.shift(21)/close.shift(252)-1).where(segments.eq(segments.shift(21))&segments.eq(segments.shift(252))&segments.gt(0))
    lowvol=-returns.rolling(63,min_periods=63).std(ddof=1)
    size=np.log(cap.where(cap>0))
    dollars=(raw_close.where(raw_close>0)*volume.where(volume>0))
    liquidity=np.log(dollars.rolling(20,min_periods=20).mean()).where(segments.eq(segments.shift(19))&segments.gt(0))
    return {'momentum_12_1':momentum,'low_volatility_63':lowvol,'log_cap':size,'log_dollar_volume_20':liquidity}
