"""Source-linked factors. All operators are backward-looking, with full windows.

WQ source: Kakushadze (2015), https://arxiv.org/abs/1601.00991, Appendix A.
No formula here requires VWAP or ADV (the latter means DOLLAR volume).
Financial quantities use USD millions before the signed-log transformation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def rank(x):
    return x.rank(axis=1, pct=True, method="average")


def tsrank(x, n):
    return x.rolling(n, min_periods=n).rank(pct=True, method="average")


def corr(x, y, n):
    return x.rolling(n, min_periods=n).corr(y).clip(-1, 1)


def std(x, n):
    return x.rolling(n, min_periods=n).std(ddof=1)


def mean(x, n):
    return x.rolling(n, min_periods=n).mean()


def total(x, n):
    return x.rolling(n, min_periods=n).sum()


def minimum(x, n):
    return x.rolling(n, min_periods=n).min()


def maximum(x, n):
    return x.rolling(n, min_periods=n).max()


def scale(x):
    return x.div(x.abs().sum(axis=1).replace(0, np.nan), axis=0)


def slog(x):
    return np.sign(x) * np.log1p(np.abs(x) / 1e6)


def residual(y, x, universe, min_count=30):
    """Daily OLS with intercept; paired observations only, no future estimates."""
    ok = universe.eq(1) & np.isfinite(y) & np.isfinite(x)
    xx, yy = x.where(ok), y.where(ok)
    xc, yc = xx.sub(xx.mean(axis=1), axis=0), yy.sub(yy.mean(axis=1), axis=0)
    slope = (xc * yc).sum(axis=1) / (xc * xc).sum(axis=1).replace(0, np.nan)
    return (yc - xc.mul(slope, axis=0)).where(ok).where(ok.sum(axis=1).ge(min_count), axis=0)


WQ_IDS = (2, 3, 4, 6, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20,
          22, 23, 24, 26, 30, 33, 34, 35, 37, 38, 40, 44, 45, 46, 49, 51, 52, 53, 54, 55, 101)


def wq(number, p):
    o, h, l, c, v, r = (p[k] for k in ("open", "high", "low", "close", "volume", "returns"))
    d = c.diff()
    if number == 2:
        return -corr(rank(np.log(v.where(v > 0)).diff(2)), rank((c-o)/o), 6)
    if number == 3:
        return -corr(rank(o), rank(v), 10)
    if number == 4:
        return -tsrank(rank(l), 9)
    if number == 6:
        return -corr(o, v, 10)
    if number == 8:
        x = total(o, 5) * total(r, 5)
        return -rank(x-x.shift(10))
    if number in (9, 10):
        n = 5 if number == 9 else 4
        x = d.where((minimum(d, n) > 0) | (maximum(d, n) < 0), -d)
        x = x.where(minimum(d, n).notna())
        return x if number == 9 else rank(x)
    if number == 12:
        return np.sign(v.diff()) * -d
    if number in (13, 16):
        x = c if number == 13 else h
        return -rank(rank(x).rolling(5, min_periods=5).cov(rank(v)))
    if number == 14:
        return -rank(r.diff(3)) * corr(o, v, 10)
    if number == 15:
        return -total(rank(corr(rank(h), rank(v), 3)), 3)
    if number == 18:
        return -rank(std((c-o).abs(), 5) + c-o + corr(c, o, 10))
    if number == 19:
        return -np.sign(c-c.shift(7)+c.diff(7)) * (1+rank(1+total(r, 250)))
    if number == 20:
        return -rank(o-h.shift()) * rank(o-c.shift()) * rank(o-l.shift())
    if number == 22:
        return -corr(h, v, 5).diff(5) * rank(std(c, 20))
    if number == 23:
        return (-h.diff(2)).where(mean(h, 20) < h, 0).where(mean(h, 20).notna())
    if number == 24:
        return (-(c-minimum(c, 100))).where(mean(c, 100).diff(100)/c.shift(100) <= .05,
                                             -c.diff(3)).where(mean(c, 100).diff(100).notna())
    if number == 26:
        return -maximum(corr(tsrank(v, 5), tsrank(h, 5), 5), 3)
    if number == 30:
        x = np.sign(d) + np.sign(d.shift()) + np.sign(d.shift(2))
        return (1-rank(x)) * total(v, 5)/total(v, 20).replace(0, np.nan)
    if number == 33:
        return rank(o/c-1)
    if number == 34:
        return rank(1-rank(std(r, 2)/std(r, 5).replace(0, np.nan))+1-rank(d))
    if number == 35:
        return tsrank(v, 32)*(1-tsrank(c+h-l, 16))*(1-tsrank(r, 32))
    if number == 37:
        return rank(corr((o-c).shift(), c, 200)) + rank(o-c)
    if number == 38:
        return -rank(tsrank(c, 10))*rank(c/o)
    if number == 40:
        return -rank(std(h, 10))*corr(h, v, 10)
    if number == 44:
        return -corr(h, rank(v), 5)
    if number == 45:
        return -rank(mean(c.shift(5), 20))*corr(c, v, 2)*rank(corr(total(c, 5), total(c, 20), 2))
    if number in (46, 49, 51):
        a = (c.shift(20)-c.shift(10))/10 - (c.shift(10)-c)/10
        if number == 46:
            return (-d).where(a >= 0, 1).where(a <= .25, -1).where(a.notna())
        return (-d).where(a >= (-.1 if number == 49 else -.05), 1).where(a.notna())
    if number == 52:
        return (minimum(l, 5).shift(5)-minimum(l, 5))*rank((total(r, 240)-total(r, 20))/220)*tsrank(v, 5)
    if number == 53:
        return -(((c-l)-(h-c))/(c-l).replace(0, np.nan)).diff(9)
    if number == 54:
        return -((l-c)/(l-h).replace(0, np.nan))*(o/c)**5
    if number == 55:
        x = (c-minimum(l, 12))/(maximum(h, 12)-minimum(l, 12)).replace(0, np.nan)
        return -corr(rank(x), rank(v), 6)
    if number == 101:
        return (c-o)/(h-l+.001)
    raise ValueError(f"Unimplemented WQ number {number}")


def price_inputs(load, universe):
    """Forward-anchor each known segment; future back-adjustment constants cancel.

    Rank inputs use only the contemporaneously eligible cross-section. A missing
    boundary bar prevents rolling operators from spanning a segment discontinuity.
    """
    seg, adj = load("adjustment_segment"), load("adj_factor")
    reset = seg.ne(seg.shift()) & seg.gt(0)
    anchor = adj.where(reset).ffill().where(seg.gt(0))
    factor = adj/anchor
    boundary = reset & seg.shift().notna()
    eligible = universe.eq(1)
    p = {}
    for field in ("open", "high", "low", "close"):
        p[field] = (load(field)*factor).where(~boundary & seg.gt(0) & eligible).astype("float32")
    p["volume"] = load("volume").where(~boundary & seg.gt(0) & eligible)
    p["returns"] = load("returns").where(eligible)
    return p


def surprise_events(q, cap):
    """CMB simple expectation: NI[q-4]+mean(last 8 YoY NI dollar changes).

    Each constituent must have been released strictly before this release. The
    denominator is cap on the last session BEFORE the release, not availability.
    """
    dates = pd.to_datetime(cap.index, format="%Y%m%d")
    sidpos = {s: i for i, s in enumerate(cap.columns)}
    values = cap.to_numpy()
    events = []
    for sid, group in q.groupby("sid", sort=False):
        if sid not in sidpos:
            continue
        group = group.copy()
        group["ordinal"] = group.fyearq*4+group.fqtr
        group = group.loc[~group.ordinal.duplicated(keep=False)]
        rows = {int(row.ordinal): row for row in group.itertuples() if pd.notna(row.earnings_q)}
        for k, row in rows.items():
            if pd.isna(row.release_date) or pd.isna(row.available_ts):
                continue
            required = set(range(k-12, k))
            if not required.issubset(rows):
                continue
            if any(pd.isna(rows[j].earnings_q_available_ts) or
                   rows[j].earnings_q_available_ts >= row.release_date for j in required):
                continue
            changes = [rows[k-j].earnings_q-rows[k-j-4].earnings_q for j in range(1, 9)]
            expected = rows[k-4].earnings_q + np.mean(changes)
            pos = dates.searchsorted(row.release_date, side="left")-1
            if pos < 0 or pos >= len(dates):
                continue
            denominator = values[pos, sidpos[sid]]
            if not np.isfinite(denominator) or denominator <= 0:
                continue
            available = max(row.available_ts, row.earnings_q_available_ts)
            if pd.isna(available):
                continue
            events.append({"sid": sid, "period_end": row.period_end,
                           "available_date": available, "release_date": row.release_date,
                           "cap_date": cap.index[pos], "actual_ni": row.earnings_q,
                           "expected_ni": expected, "cap": float(denominator),
                           "value": (row.earnings_q-expected)/denominator})
    columns = ["sid", "period_end", "available_date", "release_date", "cap_date", "actual_ni", "expected_ni", "cap", "value"]
    out = pd.DataFrame(events, columns=columns).sort_values(["sid", "available_date", "period_end"])
    if out.empty:
        return out
    out = out.drop_duplicates(["sid", "available_date"], keep="last")
    return out.loc[out.period_end.eq(out.groupby("sid").period_end.cummax())].reset_index(drop=True)
