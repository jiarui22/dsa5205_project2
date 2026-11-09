"""Long-only classroom close ledger with explicit corporate/stale-mark treatment.

This does not relax the repository's strict production engine. Signals never
reselect names using tomorrow's quotes. Unfillable buys stay cash; unsellable
positions remain held. Corporate settlement exceptions are saved as event rows.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from usq.backtest.metrics import performance


class Market:
    def __init__(self, database):
        from pathlib import Path
        p = Path(database)/"panels"
        def load(name):
            return pd.read_pickle(p/f"{name}.pkl")
        c = load("close_adj")
        self.index, self.columns = c.index, c.columns
        self.close = c.to_numpy()
        self.universe = load("universe").to_numpy() == 1
        self.tradable = (load("trade_available").to_numpy() == 1) & np.isfinite(self.close) & (self.close > 0)
        seg = load("adjustment_segment")
        # A stale mark is an accounting convention, NOT an assertion of a known
        # economic zero return. A later same-segment quote catches up the mark.
        marks = c.ffill()
        prior_seg = seg.where(seg > 0).ffill().shift()
        valid = seg.eq(prior_seg) & seg.gt(0)
        self.ret = (marks/marks.shift()-1).where(valid).to_numpy()
        self.source_return = load("daily_return").to_numpy()
        self.terminal = load("delisting_event").to_numpy() == 1
        spy = load("indices_total_return_close")["SPY"]
        self.spy = spy.pct_change(fill_method=None).to_numpy()

    def backtest(self, scores, start, end, *, cost_bps=10, rebalance_every=5,
                 smooth=1, top_n=100, group=None, groups=5, weights=False,
                 all_eligible=False):
        if not scores.columns.equals(self.columns):
            raise ValueError("score sid order differs")
        x = scores.reindex(self.index).where(self.universe)
        # Equal predictions convey no ordering: go to cash, not first 100 SIDs.
        if not all_eligible:
            varying = x.nunique(axis=1, dropna=True).ge(2)
            x = x.where(varying, axis=0)
        pct = x.rank(axis=1, pct=True)
        if smooth > 1:
            pct = pct.rolling(smooth, min_periods=1).mean().where(x.notna())
        signal = pct.to_numpy()
        lo, hi = self.index.searchsorted(start), self.index.searchsorted(end, side="right")
        current = np.zeros(len(self.columns), dtype=float)
        records, holdings, events = [], [], []
        rate, first_trade = cost_bps/10000., None
        for i in range(lo, hi):
            held = current.copy()
            r = self.ret[i].astype(float).copy()
            terminal = self.terminal[i] & (held > 0)
            missing = (held > 0) & ~np.isfinite(r)
            source_ok = missing & terminal & np.isfinite(self.source_return[i]) & (self.source_return[i] >= -1)
            r[source_ok] = self.source_return[i, source_ok]
            for j in np.flatnonzero(source_ok):
                events.append({"date": self.index[i], "sid": self.columns[j], "policy": "terminal_source_return_cash_settlement", "weight": held[j], "return": float(r[j])})
            unresolved = (held > 0) & ~np.isfinite(r)
            for j in np.flatnonzero(unresolved):
                events.append({"date": self.index[i], "sid": self.columns[j], "policy": "terminal_last_mark_cash_assumption" if terminal[j] else "stale_last_mark_pending_quote", "weight": held[j], "return": 0.})
            r[~np.isfinite(r)] = 0.  # explicit last-mark accounting, exceptions above
            if ((held > 0) & (r < -1)).any():
                raise ValueError("invalid held simple return")
            gross = float(held@r)
            pre_nav = 1+gross
            if pre_nav <= 0:
                raise ValueError("NAV exhausted")
            assets = held*(1+r)
            # Mandatory distributions are cash, not synthetic stock sales.
            assets[terminal] = 0.
            scheduled = i > 0 and (i-lo) % rebalance_every == 1 % rebalance_every
            rebalance = scheduled and self.index[i-1] in scores.index
            final = i == hi-1
            fee = traded = 0.
            if rebalance or final:
                desired = np.zeros_like(current)
                if not final:
                    eligible = self.universe[i-1] & np.isfinite(signal[i-1])
                    positions = np.flatnonzero(eligible)
                    if all_eligible:
                        if group is not None:
                            raise ValueError("all_eligible is an allocation baseline, not a score group")
                        order = positions
                    elif group is None:
                        order = positions[np.argsort(-signal[i-1, positions], kind="stable")[:top_n]]
                    else:
                        bins = np.minimum(np.floor((np.nan_to_num(signal[i-1], nan=-1)-.5/max(len(positions), 1))*groups).astype(int)+1, groups)
                        order = positions[bins[positions] == group]
                    if len(order):
                        desired[order] = 1/len(order)
                executable = self.tradable[i] & ~self.terminal[i]
                desired[~executable] = 0.  # no replacement using future quotes
                locked = ~executable & (assets > 0)
                locked_value = float(assets[locked].sum())
                for _ in range(100):
                    nav = pre_nav-fee
                    target = desired*nav
                    free_budget = max(nav-locked_value, 0.)
                    if target.sum() > free_budget and target.sum() > 0:
                        target *= free_budget/target.sum()
                    target[locked] = assets[locked]
                    new_fee = rate*np.abs(target-assets).sum()
                    if abs(new_fee-fee) < 1e-12:
                        fee = float(new_fee)
                        break
                    fee = float(new_fee)
                traded = float(np.abs(target-assets).sum())
                current = target/(pre_nav-fee)
                if traded > 0 and first_trade is None:
                    first_trade = i
            else:
                current = assets/pre_nav
            if current.sum() > 1+1e-8 or (current < -1e-10).any():
                raise ValueError("cash/leverage constraint violated")
            # One common SPY investment window, even if a model chooses cash.
            benchmark = 0. if i <= lo+1 else float(self.spy[i])
            if not np.isfinite(benchmark):
                raise ValueError("SPY missing")
            records.append({"date": self.index[i], "gross_return": gross, "net_return": gross-fee,
                            "cost": fee, "turnover": traded/(2*pre_nav), "traded_notional": traded,
                            "benchmark_return": benchmark, "cash_weight": 1-current.sum()})
            if weights:
                holdings.append(held.astype("float32"))
        daily = pd.DataFrame(records).set_index("date")
        daily["net_nav"] = (1+daily.net_return).cumprod()
        daily["benchmark_nav"] = (1+daily.benchmark_return).cumprod()
        daily["relative_nav"] = daily.net_nav/daily.benchmark_nav
        daily["drawdown"] = daily.net_nav/daily.net_nav.cummax().clip(lower=1)-1
        summary = performance(daily.net_return, daily.turnover)
        bench = performance(daily.benchmark_return)
        relative = performance(daily.relative_nav.pct_change(fill_method=None).fillna(daily.relative_nav.iloc[0]-1))
        summary.update(spy_annual_return=bench["annual_return"],
                       annual_return_minus_spy=summary["annual_return"]-bench["annual_return"],
                       relative_nav_annual_return=relative["annual_return"],
                       relative_nav_max_drawdown=relative["max_drawdown"],
                       exception_rows=len(events), cost_bps=cost_bps, smooth=smooth,
                       rebalance_every=rebalance_every, timing="T signal -> T+1 close trade -> T+2 first held return")
        return daily, pd.DataFrame(holdings, index=daily.index, columns=self.columns) if weights else None, pd.DataFrame(events), summary
