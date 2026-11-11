from __future__ import annotations

import numpy as np
import pandas as pd
from .panels import PanelStore
from usq.utils import zscore


NEXT_CLOSE_LABEL_POLICY = {
    'label_version': 'next_close_10d_v2',
    'mode': 'close_compound', 'horizon': 10, 'horizon_sessions': 10,
    'entry_offset': 1, 'exit_offset': 11,
    'formula': 'close_adj[t+11]/close_adj[t+1]-1',
    'standardize': False, 'neutralization': None, 'vwap_required': False,
}


def read_label_end(store, reference):
    """Read date columns BEFORE setting the index; CSV inference can make ints."""
    frame = pd.read_csv(store.root/'label_end.csv', dtype=str)
    if list(frame.columns) != ['date', 'label_end']:
        raise ValueError('label_end.csv must have date,label_end columns')
    result = frame.set_index('date').label_end
    if not result.index.equals(reference.index) or result.index.has_duplicates:
        raise ValueError('Full label-end calendar differs from labels')
    if not result.dropna().str.fullmatch(r'\d{8}').all() or not result.dropna().gt(result.dropna().index).all():
        raise ValueError('Invalid future label end dates')
    return result


def close_forward_returns(price, horizon=10, segment=None, *, delay=1):
    """Close[t+delay+h]/close[t+delay]-1, indexed by signal date t.

    The default enters at the next close and holds for ten return intervals.
    Explicit delay=0 reproduces the historical signal-close label only.
    With dated adjustment segments, require the entire interval to stay in one
    positive known segment, including the signal-to-entry identity link.
    Intermediate absent quotes alone need not invalidate endpoint returns.
    Without segment evidence, conservatively require all holding-period marks.
    Never ffill prices or sum arithmetic daily returns.
    """
    if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 1:
        raise ValueError('horizon must be a positive integer trading-session count')
    if not isinstance(delay, int) or isinstance(delay, bool) or delay < 0:
        raise ValueError('delay must be a nonnegative integer trading-session count')
    if price.index.has_duplicates or price.columns.has_duplicates or not price.index.is_monotonic_increasing:
        raise ValueError('Invalid complete price axes')
    p = price.astype('float64')
    valid = np.isfinite(p) & p.gt(0)
    exit_offset = delay + horizon
    good = valid.shift(-delay, fill_value=False) & valid.shift(-exit_offset, fill_value=False)
    if segment is not None:
        if not segment.index.equals(price.index) or not segment.columns.equals(price.columns):
            raise ValueError('Adjustment segments must have identical full axes')
        known = np.isfinite(segment) & segment.gt(0)
        for offset in range(exit_offset + 1):
            good &= known.shift(-offset, fill_value=False) & segment.shift(-offset).eq(segment)
    else:
        for offset in range(delay + 1, exit_offset):
            good &= valid.shift(-offset, fill_value=False)
    raw = (p.shift(-exit_offset) / p.shift(-delay) - 1).where(good)
    end = pd.Series(price.index, index=price.index, name='label_end').shift(-exit_offset)
    return raw, end


def daily_close_returns(price, segment=None):
    """Backward-looking close[t]/close[t-1]-1, without fill or identity bridging."""
    p = price.astype('float64')
    valid = np.isfinite(p) & p.gt(0)
    good = valid & valid.shift(1, fill_value=False)
    if segment is not None:
        if not segment.index.equals(price.index) or not segment.columns.equals(price.columns):
            raise ValueError('Adjustment segment axes differ')
        good &= segment.gt(0) & np.isfinite(segment) & segment.eq(segment.shift(1))
    return (p / p.shift(1) - 1).where(good)


def forward_returns(price, horizon=10, delay=1):
    """Signal t close -> enter t+delay VWAP -> exit t+delay+horizon VWAP.

    Require every mark in the window, and retain label_end for exact split purging.
    Prices must use a consistently adjusted basis; no ffill/pct_change defaults.
    """
    if horizon < 1 or delay < 1:
        raise ValueError("horizon and delay must be positive")
    raw = price.shift(-(delay + horizon)) / price.shift(-delay) - 1
    good = pd.DataFrame(True, index=price.index, columns=price.columns)
    for offset in range(delay, delay + horizon + 1):
        good &= price.shift(-offset).gt(0) & price.shift(-offset).notna()
    end = pd.Series(price.index, index=price.index, name="label_end").shift(-(delay + horizon))
    return raw.where(good), end


def neutralize(target, cap, industry, mask, standardize=True, min_obs=20):
    """Daily WLS residual versus log cap + industries, weights=sqrt(cap).

    Controls use SIGNAL-date information. Full industry dummies span the intercept.
    Residuals are not portfolio returns. Missing controls remain missing.
    """
    out = target * np.nan
    for d in target.index:
        valid = mask.loc[d].eq(1) & target.loc[d].notna() & cap.loc[d].gt(0) & industry.loc[d].notna()
        ids = target.columns[valid]
        if len(ids) < min_obs:
            continue
        c = cap.loc[d, ids].to_numpy(float)
        logc = np.log(c)
        X = np.column_stack([logc - logc.mean(), pd.get_dummies(industry.loc[d, ids], dtype=float).to_numpy()])
        if len(ids) <= np.linalg.matrix_rank(X) + 2:
            continue
        y = target.loc[d, ids].to_numpy(float)
        sw = (c / np.median(c)) ** .25
        beta = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)[0]
        out.loc[d, ids] = y - X @ beta
    return zscore(out) if standardize else out


def prepare(store: PanelStore, horizon=10, standardize=None, mode="close_compound"):
    if horizon != 10:
        raise ValueError("this project fixes the label horizon at ten trading days")
    if mode in ('close_compound', 'signal_close_compound'):
        if standardize not in (None, False):
            raise ValueError('Direct close_compound ylabel is not standardized/neutralized')
        if store.path('close_adj').exists():
            cp = store.load('close_adj')
        else:
            close = store.load('close')
            adj = store.load('adj_factor', close)
            if (adj.dropna() <= 0).any().any():
                raise ValueError('adj_factor must be positive')
            cp = close * adj
            store.save('close_adj', cp)
        segment = store.load('adjustment_segment', cp) if store.path('adjustment_segment').exists() else None
        delay = 1 if mode == 'close_compound' else 0
        raw, end = close_forward_returns(cp, horizon, segment, delay=delay)
        for field in ['forward_10d_raw', 'ylabel', 'label_10']:
            store.save(field, raw)
        store.save('returns', daily_close_returns(cp, segment))
        store.save('label_available', raw.notna().astype('float32'))
        end.to_csv(store.root / 'label_end.csv', index_label='date')
        from usq.utils import json_dump
        policy = dict(NEXT_CLOSE_LABEL_POLICY)
        if delay == 0:
            policy.update(label_version='signal_close_10d_v1', mode=mode,
                          formula='close_adj[t+10]/close_adj[t]-1', entry_offset=0, exit_offset=10)
        json_dump({**policy,
            'segment_check': segment is not None, 'vwap_required': False,
            'label_is_executable_return': False,
            'timing': 'Signal after t close; next-close target spans t+1 to t+11; legacy mode is explicit.'},
            store.root / 'label_metadata.json')
        return store.manifest()
    if standardize is None:
        standardize = True  # Explicit legacy conventions retain their prior behavior.
    close = store.load("close")
    volume = store.load("volume", close)
    if store.path("vwap").exists():
        vwap = store.load("vwap", close)
    else:
        amount = store.load("amount", close)
        vwap = (amount / volume.where(volume > 0)).where(amount > 0)
        store.save("vwap", vwap)
    adj = store.load("adj_factor", close)
    if (adj.dropna() <= 0).any().any():
        raise ValueError("adj_factor must be positive; formula is adjusted_price=raw_price*factor")
    cap = store.load("market_cap", close)
    ind = store.load("industry", close, numeric=False)
    mask = store.load("universe", close).eq(1)
    for field in ("open", "high", "low"):
        raw_field = store.load(field, close)
        store.save(f"{field}_adj", raw_field * adj)
    adjusted = (vwap * adj).where(vwap > 0)
    if mode == "vwap_compound":
        raw, end = forward_returns(adjusted, horizon)
    elif mode == "close_sum":
        # Alternative target convention: sum future daily close returns.
        # Still require all future marks; never reproduce pandas sum(skipna=True).
        cp = close * adj
        daily = cp / cp.shift(1) - 1
        raw = sum(daily.shift(-k) for k in range(1, horizon + 1))
        end = pd.Series(close.index, index=close.index, name="label_end").shift(-horizon)
    else:
        raise ValueError("label mode must be close_compound, signal_close_compound, vwap_compound or legacy close_sum")
    label = neutralize(raw, cap, ind, mask, standardize)
    store.save("vwap_adj", adjusted)
    store.save("close_adj", close * adj)
    store.save("forward_10d_raw", raw)
    store.save("label_10", label)
    end.to_csv(store.root / "label_end.csv", index_label="date")
    from usq.utils import json_dump
    json_dump({"mode": mode, "horizon": horizon, "standardize": standardize,
               "neutralization": "signal-date log cap + full industry dummies; WLS weights=sqrt(cap)",
               "price_adjustment": "raw_price * adj_factor"}, store.root / "label_metadata.json")
    return store.manifest()
