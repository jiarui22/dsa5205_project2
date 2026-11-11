"""Stage one: source-linked reproduction and temporally sealed factor selection."""
from __future__ import annotations
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from usq.database.fundamentals import expand_events
from usq.data.labels import NEXT_CLOSE_LABEL_POLICY
from .factors import WQ_IDS, price_inputs, rank, residual, slog, surprise_events, wq
from .protocol import bind_inputs


POLICY = {
    'label_policy': NEXT_CLOSE_LABEL_POLICY,
    "seed": 5205, "train_start": "20150101", "direction_end": "20181231",
    "selection_end": "20211231", "validation_year": 2022,
    "test_start": "20230101", "test_end": "20251231",
    "direction_rule": "sign of mean 2015-2018 daily RankIC (labels matured by 2018 end)",
    "min_stability_rankic": .01, "min_coverage": .5, "min_daily_pairs": 30,
    "min_positive_stability_years": 2, "max_abs_factor_corr": .8,
    "target_wq_count": 10, "candidate_ids": list(WQ_IDS),
    "selection_order": "2019-2021 directional RankIC descending; then average absolute daily rank correlation",
    "financial_units_before_slog": "USD millions",
    "inputs": "as-of daily universe; segment-forward-anchored adjusted OHLC; original shares volume; no VWAP formulas",
    "temporal_caveat": "latest downloaded financial statements, not strict historical PIT; revised experiment on previously inspected 2023-2025 outcomes, not a fresh blind holdout",
    "label": "close_adj[T+11]/close_adj[T+1]-1; development labels must end by relevant phase end",
    "coverage_denominator": "daily universe=1 with a mature finite target",
}


def write_json(value, path):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def information_series(x, y, universe, min_pairs=30):
    ok = universe.eq(1) & np.isfinite(x) & np.isfinite(y)
    a, b = x.where(ok).astype("float64"), y.where(ok).astype("float64")
    count = ok.sum(axis=1)
    ar, br = rank(a), rank(b)
    varies = ar.std(axis=1).gt(1e-12) & br.std(axis=1).gt(1e-12)
    return pd.DataFrame({"ic": a.corrwith(b, axis=1).where((count >= min_pairs) & varies),
                         "rankic": ar.corrwith(br, axis=1).where((count >= min_pairs) & varies),
                         "pairs": count,
                         "coverage": count/(universe.eq(1) & np.isfinite(y)).sum(axis=1).replace(0, np.nan)})


def hac_t(series, lag=10):
    x = series.dropna().to_numpy(float)
    if len(x) <= lag+1:
        return float("nan")
    z = x-x.mean()
    variance = z@z/len(x)
    for k in range(1, lag+1):
        variance += 2*(1-k/(lag+1))*(z[k:]@z[:-k])/len(x)
    return x.mean()/np.sqrt(max(variance, 1e-20)/len(x))


def phase_summary(s):
    x = s.rankic.dropna()
    return {"rankic": float(x.mean()), "ic": float(s.ic.mean()),
            "rankicir_daily": float(x.mean()/x.std()), "hac_t_lag10": float(hac_t(x)),
            "coverage": float(s.coverage.mean()), "days": len(x)}


def run(database, output, reuse_factors=None):
    started = time.time()
    root, out = Path(database), Path(output)
    for sub in ("factors", "evaluation", "events"):
        (out/sub).mkdir(parents=True, exist_ok=True)
    policy_path = out/"research_protocol.json"
    if policy_path.exists() and json.loads(policy_path.read_text()) != POLICY:
        raise ValueError("Frozen protocol differs; use a new experiment output directory")
    write_json(POLICY, policy_path)  # freeze before reading any factor outcome
    if reuse_factors:
        import shutil
        source = Path(reuse_factors)
        for sub in ("factors", "events"):
            for path in sorted((source/sub).glob("*")):
                dest = out/sub/path.name
                if not dest.exists():
                    shutil.copy2(path, dest)
    panels = root/"panels"
    universe = pd.read_pickle(panels/"universe.pkl")
    label_end = bind_inputs(root, out, universe, create=True)
    def load(name):
        x = pd.read_pickle(panels/f"{name}.pkl")
        if not x.index.equals(universe.index) or not x.columns.equals(universe.columns):
            raise ValueError(f"Full panel axes differ: {name}")
        return x
    target = load("ylabel")
    dev = (universe.index >= "20150101") & (universe.index <= "20211231")
    ydev = target.loc[dev].where(label_end.loc[dev].le("20211231"), axis=0)
    udev = universe.loc[dev]
    summaries, annuals, catalog = [], [], {}

    def evaluate(name, x, source, logic):
        path = out/"factors"/f"{name}.pkl"
        if x is not None:
            x = x.replace([np.inf, -np.inf], np.nan).astype("float32")
            x.to_pickle(path)
        else:
            x = pd.read_pickle(path)
        s = information_series(x.loc[dev], ydev, udev)
        # Direction split purges the final ten labels rather than reading 2019.
        s.loc[(s.index <= "20181231") & label_end.reindex(s.index).gt("20181231"), ["ic", "rankic"]] = np.nan
        s.to_parquet(out/"evaluation"/f"{name}_daily.parquet")
        first, stable = s.loc[:"20181231"], s.loc["20190101":]
        direction = 1 if first.rankic.mean() >= 0 else -1
        a, b = phase_summary(first), phase_summary(stable)
        years = s.groupby(s.index.str[:4]).agg({"ic": "mean", "rankic": "mean", "coverage": "mean", "pairs": "mean"})
        for year, row in years.iterrows():
            annuals.append({"factor": name, "year": year, **row.to_dict(), "signed_rankic": direction*row["rankic"]})
        positive = int((years.loc["2019":"2021", "rankic"]*direction > 0).sum())
        passed = (b["rankic"]*direction >= POLICY["min_stability_rankic"] and
                  b["coverage"] >= POLICY["min_coverage"] and positive >= 2)
        result = {"factor": name, "family": "WQ" if name.startswith("WQ") else "fundamental" if name.startswith("F") else "control",
                  "direction": direction, "early_rankic": a["rankic"],
                  "stability_rankic_signed": b["rankic"]*direction,
                  "stability_rankicir_signed": b["rankicir_daily"]*direction,
                  "stability_hac_t_signed": b["hac_t_lag10"]*direction,
                  "stability_coverage": b["coverage"], "positive_stability_years": positive,
                  "passes_predeclared_rules": bool(passed)}
        summaries.append(result)
        catalog[name] = {"source": source, "logic": logic, "direction": direction,
                         "shape": list(x.shape), "storage": "float32", "no_future_fill": True}
        print(json.dumps({"stage": "factor", **result, "elapsed_seconds": round(time.time()-started)}, ensure_ascii=False), flush=True)

    cap = load("market_cap")
    fund_specs = [("F01_EP_reconstructed", "earnings_ttm", "market_cap", "东方证券2024，第8、23页；signed-log OLS净利润对市值残差"),
                  ("F02_BP_reconstructed", "book_equity", "market_cap", "东方证券2024，第8、29页；signed-log OLS净资产对市值残差"),
                  ("F03_SPQ_reconstructed", "revenue_q", "market_cap", "东方证券2024，第8、25页；signed-log OLS单季收入对市值残差"),
                  ("F04_ROEQ_reconstructed", "earnings_q", "book_equity", "东方证券2024，第8、37页；signed-log OLS单季利润对净资产残差")]
    for name, numerator, denominator, source in fund_specs:
        if (out/"factors"/f"{name}.pkl").exists():
            evaluate(name, None, source, "高残差表示在相同解释变量下更强的基本面；方向只用开发期确定")
            continue
        y, x = load(numerator), cap if denominator == "market_cap" else load(denominator)
        evaluate(name, residual(slog(y), slog(x), universe), source, "金额单位为百万美元；截面含截距OLS残差")
    name = "F05_FCF_yield"
    if not (out/"factors"/f"{name}.pkl").exists():
        ev = cap+load("total_liabilities")-load("cash")
        f = (load("operating_cash_flow_ttm")-load("capital_expenditures_ttm"))/ev.where(ev > 0)
        industry = load("industry")
        financial = industry.apply(lambda s: pd.to_numeric(s.astype(str).str.replace("SIC2:", "", regex=False), errors="coerce")).ge(60) & industry.apply(lambda s: pd.to_numeric(s.astype(str).str.replace("SIC2:", "", regex=False), errors="coerce")).le(69)
        f = f.where(~financial & universe.eq(1))
    else:
        f = None
    evaluate(name, f, "东北证券2026，第5页；美国Compustat口径迁移", "FCF/(市值+总负债-现金)，仅因子层面排除SIC2 60–69，不复制原报告完整策略")
    name = "F06_earnings_surprise"
    if not (out/"factors"/f"{name}.pkl").exists():
        q = pd.read_parquet(root/"metadata/fundamental_quarterly.parquet")
        events = surprise_events(q, cap)
        events.to_parquet(out/"events/earnings_surprise.parquet", index=False)
        f = expand_events(events, universe.index, universe.columns).where(universe.eq(1))
        del q, events
    else:
        f = None
    evaluate(name, f, "招商证券2026，第12–13、27页；简单预期法", "利润同比变化额8季均值预测；公告前市值归一化；公告后可用并ffill")
    for name, numerator, denominator, source in fund_specs:
        control = name.replace("reconstructed", "ratio").replace("F0", "C0")
        x = cap if denominator == "market_cap" else load(denominator)
        f = load(numerator)/x.where(x > 0)
        evaluate(control, f.where(universe.eq(1)), source, "传统比值对照")
    del cap, f
    need_prices = any(not (out/"factors"/f"WQ{n:03d}.pkl").exists() for n in WQ_IDS)
    prices = price_inputs(load, universe) if need_prices else None
    for number in WQ_IDS:
        name = f"WQ{number:03d}"
        f = None if (out/"factors"/f"{name}.pkl").exists() else wq(number, prices)
        evaluate(name, f, "Kakushadze (2015), 101 Formulaic Alphas, Appendix A", f"原文Alpha#{number}；不依赖VWAP/ADV；算子语义见factors.py")
    del prices, f
    summary = pd.DataFrame(summaries)
    summary.to_csv(out/"evaluation/factor_screen.csv", index=False)
    pd.DataFrame(annuals).to_csv(out/"evaluation/factor_annual.csv", index=False)
    candidates = summary.loc[summary.family.eq("WQ") & summary.passes_predeclared_rules].sort_values("stability_rankic_signed", ascending=False)
    selected, chosen_ranks, decisions = [], [], []
    sample = udev.loc["20190101":"20211231"].iloc[::5]
    for row in candidates.itertuples():
        x = pd.read_pickle(out/"factors"/f"{row.factor}.pkl").reindex(sample.index).where(sample.eq(1))
        xr = rank(x)*row.direction
        correlations = {name: float(xr.corrwith(r, axis=1).abs().mean()) for name, r in chosen_ranks}
        maximum_corr = max(correlations.values(), default=0.)
        accepted = maximum_corr < POLICY["max_abs_factor_corr"] and len(selected) < POLICY["target_wq_count"]
        decisions.append({"factor": row.factor, "accepted": accepted, "max_abs_daily_rank_corr": maximum_corr, "correlations": correlations})
        if accepted:
            selected.append(row.factor)
            chosen_ranks.append((row.factor, xr))
    selection = {"wq": selected, "fundamental": summary.loc[summary.family.eq("fundamental"), "factor"].tolist(),
                 "directions": dict(zip(summary.factor, summary.direction.astype(int))),
                 "decisions": decisions, "selected_without_2022_2025_outcomes": True,
                 "protocol_sha256": hashlib.sha256(policy_path.read_bytes()).hexdigest()}
    write_json(selection, out/"factor_selection.json")
    write_json(catalog, out/"factor_catalog.json")
    print(json.dumps({"stage": "screen_complete", "selected": selected, "elapsed_seconds": round(time.time()-started)}, ensure_ascii=False), flush=True)
    return selection


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--reuse-factors")
    args = parser.parse_args()
    run(args.database, args.output, args.reuse_factors)
