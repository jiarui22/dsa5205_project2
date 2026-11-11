"""Causal total-return prices with one deterministic normalization per segment."""
import numpy as np
import pandas as pd

PRICE_PRECISION_POLICY = 'canonical_total_return_float64_v1'


def canonical_close(close, returns, segments):
    """Anchor at the first observed raw close; compound past source returns only.

    No stored adjustment factor or future normalization enters this calculation.
    A quarantined bar stays missing, but a known source return can carry the
    internal chain through it. Unknown identities always break the chain.
    """
    for other in (returns, segments):
        if not close.index.equals(other.index) or not close.columns.equals(other.columns):
            raise ValueError('Canonical price inputs must have identical axes')
    p = close.to_numpy(dtype='float64')
    r = returns.to_numpy(dtype='float64')
    seg = segments.to_numpy(dtype='float64')
    out = np.full(p.shape, np.nan, dtype='float64')
    state = np.full(p.shape[1], np.nan, dtype='float64')
    for i in range(len(p)):
        known = np.isfinite(seg[i]) & (seg[i] > 0)
        observed = np.isfinite(p[i]) & (p[i] > 0) & known
        same = np.zeros(len(state), dtype=bool) if i == 0 else known & (seg[i] == seg[i-1])
        linked = same & np.isfinite(state) & np.isfinite(r[i]) & (r[i] > -1)
        if np.any(same & np.isfinite(state) & observed & ~linked):
            raise ValueError('Known adjustment segment contains an invalid linking return')
        state = np.where(linked, state * (1 + r[i]), np.where(observed, p[i], np.nan))
        if np.any(observed & (~np.isfinite(state) | (state <= 0))):
            raise ValueError('Nonpositive or overflowing canonical price chain')
        out[i] = np.where(observed, state, np.nan)
    return pd.DataFrame(out, index=close.index, columns=close.columns)


def adjusted_bars(bars, returns, segments):
    """Float64 OHLC on the same return chain; preserve exact raw quote ratios."""
    close = bars['close'].astype('float64')
    adjusted = {'close': canonical_close(close, returns, segments)}
    for field in ('open', 'high', 'low'):
        adjusted[field] = bars[field].astype('float64').div(close).mul(adjusted['close'])
    return adjusted


def observed_returns(close, returns, segments):
    """Source returns on adjacent observed marks, without ratio-rounding noise."""
    valid = close.notna() & close.shift().notna() & segments.gt(0) & segments.eq(segments.shift())
    return returns.astype('float64').where(valid)
