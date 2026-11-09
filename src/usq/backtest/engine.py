"""Adjusted-close research ledger: signal T -> trade T+1 CLOSE -> earn at T+2.

Rows are daily close marks. Costs occur on trade dates, including initial entry
and optional final liquidation. Selection never uses future price availability.
Adjusted prices are return proxies, not literal fills or a full corporate ledger.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from .metrics import performance
from usq.data.labels import daily_close_returns


@dataclass
class BacktestResult:
    daily: pd.DataFrame
    weights: pd.DataFrame  # beginning of interval, indexed by exit date
    summary: dict
    post_close_weights: pd.DataFrame | None = None


def score_groups(score, universe, groups=50):
    """Tie-aware percentile bins. Equal signals stay together; empty bins allowed."""
    if groups < 2:
        raise ValueError("groups >= 2 required")
    x = score.where(universe.eq(1))
    rank = x.rank(axis=1, method="average")
    count = x.count(axis=1)
    pct = rank.sub(.5).div(count.replace(0, np.nan), axis=0)
    return (np.floor(pct * groups) + 1).clip(1, groups).where(x.notna())


def target_weights(score, universe, top_n=100, mode="long", group=None, groups=50):
    x = score.where(universe.eq(1)).replace([np.inf, -np.inf], np.nan)
    if group is not None:
        chosen = score_groups(x, universe, groups).eq(group)
        return chosen.div(chosen.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    if top_n < 1:
        raise ValueError("top_n >= 1 required")
    if mode == "long":
        # Stable sid order for ties, recorded in the input schema.
        chosen = x.rank(axis=1, ascending=False, method="first").le(top_n) & x.notna()
        return chosen.div(chosen.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    if mode == "long_short":
        r = x.rank(axis=1, pct=True)
        r = r.sub(r.mean(axis=1), axis=0)
        return r.div(r.abs().sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    raise ValueError("mode must be long or long_short")


def run_backtest(score, close_adj, universe, top_n=100, mode="long", cost_bps=5.,
                 borrow_bps_annual=300., group=None, groups=50, benchmark_returns=None,
                 start=None, end=None, adjustment_segment=None, trade_available=None,
                 rebalance_every=1, liquidate_end=True):
    if not np.isfinite(cost_bps) or not np.isfinite(borrow_bps_annual) or not (0 <= cost_bps < 10000) or borrow_bps_annual < 0:
        raise ValueError("finite cost_bps in [0,10000) and nonnegative borrow cost required")
    cal = close_adj.index
    if cal.has_duplicates or close_adj.columns.has_duplicates or not cal.is_monotonic_increasing:
        raise ValueError('invalid complete close calendar/security axes')
    if not score.columns.equals(close_adj.columns) or not universe.index.equals(cal) or not universe.columns.equals(close_adj.columns):
        raise ValueError("align ordered sids and full calendar before backtesting")
    if score.index.has_duplicates or not score.index.is_monotonic_increasing or not score.index.isin(cal).all():
        raise ValueError("score contains dates outside the price calendar")
    if not universe.isin([0, 1]).all().all():
        raise ValueError('universe must be explicit 0/1')
    if not isinstance(rebalance_every, int) or isinstance(rebalance_every, bool) or rebalance_every < 1:
        raise ValueError('rebalance_every must be a positive integer')
    if trade_available is not None:
        if not trade_available.index.equals(cal) or not trade_available.columns.equals(close_adj.columns):
            raise ValueError('full tradability axes differ')
        if not trade_available.isin([0, 1]).all().all():
            raise ValueError('tradability must be explicit 0/1')
    x = score.reindex(cal)
    active_signal = x.notna().any(axis=1)
    if active_signal.any():
        observed = np.flatnonzero(active_signal.to_numpy())
        if not active_signal.iloc[observed[0]:observed[-1] + 1].all():
            raise ValueError("internal missing score day; explicit cash/hold policy is required")
    if not active_signal.any():
        raise ValueError("no score intervals in requested window")
    observed = np.flatnonzero(active_signal.to_numpy())
    first = int(observed[0])+1
    last = min(int(observed[-1])+2, len(cal)-1, int(cal.searchsorted(end or cal[-1], side='right'))-1)
    if first > last or not (cal[first:last+1] >= (start or cal[0])).any():
        raise ValueError('no executable score intervals in requested window')
    target = target_weights(x, universe, top_n, mode, group, groups).to_numpy(float)
    interval = daily_close_returns(close_adj, adjustment_segment).to_numpy(float)
    prices = close_adj.to_numpy(float)
    tradable = trade_available.eq(1).to_numpy() if trade_available is not None else np.ones(prices.shape, bool)
    current = np.zeros(len(close_adj.columns), dtype=float)
    rows, before, after = [], [], []
    rate = cost_bps/1e4
    for i in range(first, last+1):
        held = current.copy()
        r = interval[i]
        bad = (held != 0) & ~np.isfinite(r)
        if bad.any():
            sid = close_adj.columns[np.flatnonzero(bad)[0]]
            raise ValueError(f'missing held return: {cal[i]} {sid}; supply corporate-action/delisting marks')
        safe_r = np.where(np.isfinite(r), r, 0.)  # only unheld missing marks are irrelevant
        gross = float(held @ safe_r)
        borrow = float(np.abs(np.minimum(held, 0)).sum()*borrow_bps_annual/1e4/252)
        pre_nav = 1+gross-borrow
        assets = held*(1+safe_r)
        if pre_nav <= 0:
            raise ValueError('portfolio NAV exhausted')
        rebalance = active_signal.iloc[i-1] and ((i-first) % rebalance_every == 0)
        if liquidate_end and i == last:
            desired = np.zeros_like(current)
        elif rebalance:
            desired = target[i-1]
        else:
            desired = None
        fee = traded = 0.
        if desired is not None:
            # Desired weights are fractions of POST-cost NAV. Solve the small
            # self-financing fixed point; charging on pre-fee targets overinvests.
            fee = rate*np.abs(desired*pre_nav-assets).sum()
            for _ in range(200):
                updated = rate*np.abs(desired*(pre_nav-fee)-assets).sum()
                if abs(updated-fee) <= 1e-13*max(1., pre_nav):
                    fee = float(updated)
                    break
                fee = float(updated)
            else:
                raise ValueError('transaction cost fixed point did not converge')
            if pre_nav-fee <= 0:
                raise ValueError('portfolio NAV exhausted by costs')
            changes = desired*(pre_nav-fee)-assets
            attempted = np.abs(changes) > 1e-12
            executable = np.isfinite(prices[i]) & (prices[i] > 0) & tradable[i]
            if (attempted & ~executable).any():
                sid = close_adj.columns[np.flatnonzero(attempted & ~executable)[0]]
                raise ValueError(f'missing execution close/tradability: {cal[i]} {sid}; no future-based reselection')
            traded = float(np.abs(changes).sum())
            current = desired.copy()
        else:
            current = assets/pre_nav
        before.append(held); after.append(current.copy())
        rows.append({'gross_return': gross, 'cost': fee, 'borrow_cost': borrow,
            'net_return': gross-fee-borrow, 'turnover': traded/(2*pre_nav),
            'traded_notional': traded, 'gross_exposure': float(np.abs(held).sum()),
            'post_close_gross_exposure': float(np.abs(current).sum())})
    index = cal[first:last+1]
    daily = pd.DataFrame(rows, index=index)
    weights = pd.DataFrame(before, index=index, columns=close_adj.columns)
    post = pd.DataFrame(after, index=index, columns=close_adj.columns)
    keep = daily.index >= (start or daily.index[0])
    daily, weights, post = daily.loc[keep], weights.loc[keep], post.loc[keep]
    for name in ("gross", "net"):
        daily[f"{name}_nav"] = (1 + daily[f"{name}_return"]).cumprod()
    daily["drawdown"] = daily.net_nav / daily.net_nav.cummax().clip(lower=1) - 1
    if benchmark_returns is not None:
        b = benchmark_returns.reindex(daily.index)
        if b.isna().any():
            raise ValueError("benchmark missing on a return date")
        daily["benchmark_return"] = b
        daily["benchmark_nav"] = (1 + b).cumprod()
        daily["relative_nav"] = daily.net_nav / daily.benchmark_nav
    summary = {"net": performance(daily.net_return, daily.turnover), "gross": performance(daily.gross_return, daily.turnover),
               "timing": "T after-close signal; T+1 CLOSE execution/cost; first held close-to-close return at T+2; rows=daily close",
               "mode": mode, "cost_bps_per_traded_dollar": cost_bps,
               "borrow_bps_annual": borrow_bps_annual,
               "rebalance_every": rebalance_every, "liquidate_end": liquidate_end,
               "price_field": "close_adj", "execution_delay_sessions": 1,
               "return_basis": "adjusted-close return proxy; dividends only if embedded in adj_factor; not literal adjusted-price auction fills",
               "cost_basis": "actual close-date traded notional / start-of-day NAV; target fractions of post-cost NAV",
               "turnover_basis": "half traded notional / pretrade NAV",
               "held_return_policy": "missing/unknown segment raises; no zero imputation or future-based reselection"}
    return BacktestResult(daily, weights, summary, post)


def group_backtests(score, close_adj, universe, groups=50, **kwargs):
    series = {}
    counts = score_groups(score.reindex(close_adj.index), universe, groups)
    for g in range(1, groups + 1):
        if not counts.eq(g).any().any():
            series[f"G{g:02d}"] = pd.Series(dtype=float)
            continue
        result = run_backtest(score, close_adj, universe, group=g, groups=groups, **kwargs)
        # A zero-stock group is undefined rather than a synthetic zero-return group.
        exposed = (result.daily.gross_exposure > 0) | (result.daily.post_close_gross_exposure > 0) | result.daily.cost.gt(0)
        series[f"G{g:02d}"] = result.daily.net_return.where(exposed)
    return pd.DataFrame(series)
