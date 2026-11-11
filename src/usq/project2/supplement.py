"""Descriptive extensions, never used to reselect the sealed main experiment."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from usq.backtest.metrics import performance
from usq.data.labels import NEXT_CLOSE_LABEL_POLICY, close_forward_returns
from usq.barra.exposures import build_exposures
from .factors import rank
from .ledger import Market, stable_tie_ranks, top_positions
from .risk import ResearchStore
from .screen import information_series, write_json
from .protocol import bind_inputs


POLICY = {
    'label_policy': NEXT_CLOSE_LABEL_POLICY,
    'tie_break': 'Top100 boundary ties: SHA256(seed=5205,SID), same rule in ledger and monthly target snapshots',
    "status": "predeclared v2 descriptive diagnostics on previously inspected dates; no winner reselection and no fresh holdout claim",
    "seed": 5205, "bootstrap_reps": 2000, "bootstrap_block": 20,
    "bootstrap": "paired circular moving blocks of trading dates; percentile 95% intervals, not multiple-test adjusted",
    "horizons": [1, 5, 10, 20],
    "factor_phases": [["2015-2018", "20150101", "20181231"], ["2019-2021", "20190101", "20211231"], ["2022", "20220101", "20221231"]],
    "single_factor_portfolios": "all actually selected factors, frozen directions, main execution and costs; no winners fed back into main models",
    "baselines": ["equal_weight_scores_A", "equal_weight_scores_B", "equal_weight_scores_C", "equal_weight_eligible_pool"],
    "baseline_missing": "eligible current percentile centered at .5; missing value=0; same at-least-one-WQ-observed eligibility as main models",
    "monthly_style_snapshots": "2015-2022 last session each month; signal-close Top100 target exposures; development diagnostics, not held P&L",
    "smoothing_OOS": "Ridge A/B, all existing windows 1/3/5/10 and rebalance 1/5; no re-selection",
}


def paired_bootstrap(a, b, *, reps=2000, block=20, seed=5205):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape or a.ndim != 1 or len(a) < block or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("complete paired one-dimensional observations required")
    if (a <= -1).any() or (b <= -1).any():
        raise ValueError("simple returns must exceed -1")
    rng = np.random.default_rng(seed)
    n = len(a)
    starts = rng.integers(0, n, (reps, (n+block-1)//block))
    positions = ((starts[:, :, None]+np.arange(block)) % n).reshape(reps, -1)[:, :n]
    arithmetic = (b[positions]-a[positions]).mean(axis=1)*252
    cagr = np.expm1(np.log1p(b[positions]).mean(axis=1)*252)-np.expm1(np.log1p(a[positions]).mean(axis=1)*252)
    def ci(x):
        return [float(z) for z in np.quantile(x, [.025, .975])]
    return {"days": n, "mean_net_difference_annualized": float((b-a).mean()*252),
            "mean_diff_ci_low": ci(arithmetic)[0], "mean_diff_ci_high": ci(arithmetic)[1],
            "cagr_difference": float(np.expm1(np.log1p(b).mean()*252)-np.expm1(np.log1p(a).mean()*252)),
            "cagr_diff_ci_low": ci(cagr)[0], "cagr_diff_ci_high": ci(cagr)[1]}


def uncertainty(out):
    rows, halves = [], []
    for model in ("ridge", "lasso", "xgboost"):
        def read(f):
            return pd.read_csv(out/"backtests"/f"{model}_{f}_daily.csv", dtype={"date": str}).set_index("date")
        a, b = read("A"), read("B")
        if not a.index.equals(b.index):
            raise ValueError("paired dates differ")
        rows.append({"model": model, **paired_bootstrap(a.net_return, b.net_return)})
        halfkey = a.index.str[:4]+np.where(a.index.str[4:6].astype(int) <= 6, "H1", "H2")
        for half in halfkey.unique():
            mask = halfkey == half
            for family, frame in (("A", a), ("B", b)):
                data = frame.loc[mask]
                halves.append({"model": model, "family": family, "half": half, **performance(data.net_return, data.turnover)})
    pd.DataFrame(rows).to_csv(out/"supplement/paired_uncertainty.csv", index=False)
    pd.DataFrame(halves).to_csv(out/"supplement/halfyear_performance.csv", index=False)


def factor_diagnostics(root, out, selection):
    load = lambda name: pd.read_pickle(root/"panels"/f"{name}.pkl")
    u = load("universe").loc["20150101":"20221231"]
    cp, seg = load("close_adj").loc[:"20221231"], load("adjustment_segment").loc[:"20221231"]
    names = selection["wq"]+selection["fundamental"]
    controls = ["C01_EP_ratio", "C02_BP_ratio", "C03_SPQ_ratio", "C04_ROEQ_ratio"]
    horizon_rows, comparisons = [], []
    for horizon in POLICY["horizons"]:
        target, ends = close_forward_returns(cp, horizon, seg)
        target = target.reindex(u.index)
        for name in names+(controls if horizon == 10 else []):
            f = pd.read_pickle(out/"factors"/f"{name}.pkl").reindex(u.index)*selection["directions"][name]
            s = information_series(f, target, u)
            for phase, lo, hi in POLICY["factor_phases"]:
                block = s.loc[lo:hi].copy()
                block.loc[~ends.reindex(block.index).le(hi), ["ic", "rankic"]] = np.nan
                horizon_rows.append({"factor": name, "horizon": horizon, "phase": phase,
                                     "ic": float(block.ic.mean()), "rankic": float(block.rankic.mean()),
                                     "rankicir_daily": float(block.rankic.mean()/block.rankic.std()),
                                     "coverage": float(block.coverage.mean()), "days": int(block.rankic.count())})
            if horizon == 10:
                s.to_parquet(out/"supplement"/f"{name}_2015_2022_ic.parquet")
        print(json.dumps({"stage": "horizon_diagnostics", "horizon": horizon}), flush=True)
        if horizon == 10:
            for i, control in enumerate(controls):
                name = selection["fundamental"][i]
                f = pd.read_pickle(out/"factors"/f"{name}.pkl").reindex(u.index)*selection["directions"][name]
                c = pd.read_pickle(out/"factors"/f"{control}.pkl").reindex(u.index)*selection["directions"][control]
                pair = f.notna() & c.notna()
                fs, cs = information_series(f.where(pair), target, u), information_series(c.where(pair), target, u)
                for phase, lo, hi in POLICY["factor_phases"]:
                    dates = (u.index >= lo) & (u.index <= hi) & ends.reindex(u.index).le(hi)
                    fslice, cslice = fs.loc[dates], cs.loc[dates]
                    comparisons.append({"reconstructed": name, "ratio": control, "phase": phase,
                                        "reconstructed_rankic": float(fslice.rankic.mean()),
                                        "ratio_rankic": float(cslice.rankic.mean()),
                                        "difference": float((fslice.rankic-cslice.rankic).mean()),
                                        "common_pairs_per_day": float(fslice.pairs.mean())})
    pd.DataFrame(horizon_rows).to_csv(out/"supplement/factor_horizons.csv", index=False)
    pd.DataFrame(comparisons).to_csv(out/"supplement/reconstruction_vs_ratio.csv", index=False)


def portfolios(root, out, selection):
    folder = out/"supplement/portfolios"
    folder.mkdir(exist_ok=True)
    rule = json.loads((out/"frozen_model_choices.json").read_text())["trading"]
    market = Market(root)
    index = market.index[(market.index >= "20230101") & (market.index <= "20251231")]
    observed = np.load(out/"design/observed.npy", mmap_mode="r")
    allowed = pd.DataFrame(observed, index=market.index, columns=market.columns).loc[index]
    names = selection["wq"]+selection["fundamental"]
    sums = {key: pd.DataFrame(0., index=index, columns=market.columns) for key in ("A", "B", "C")}
    results = []
    def save(name, score, **extra):
        p = folder/f"{name}_daily.csv"
        if p.exists() and (folder/f"{name}_summary.json").exists():
            results.append(json.loads((folder/f"{name}_summary.json").read_text()))
            return
        daily, weights, events, result = market.backtest(score, "20230101", "20251231", weights=True, **rule, **extra)
        daily.to_csv(p)
        weights.to_pickle(folder/f"{name}_weights.pkl")
        events.to_csv(folder/f"{name}_events.csv", index=False)
        result = {k: None if isinstance(v, float) and not np.isfinite(v) else v for k, v in result.items()}
        result = {"strategy": name, **result}
        write_json(result, folder/f"{name}_summary.json")
        results.append(result)
        print(json.dumps({"stage": "supplement_portfolio", "strategy": name, "annual_return": result["annual_return"]}), flush=True)
    for name in names:
        f = pd.read_pickle(out/"factors"/f"{name}.pkl").loc[index]*selection["directions"][name]
        f = f.where(allowed & pd.DataFrame(market.universe, index=market.index, columns=market.columns).loc[index])
        z = (rank(f)-.5).fillna(0)
        sums["B"] += z/len(names)
        key = "A" if name in selection["wq"] else "C"
        sums[key] += z/len(selection["wq"] if key == "A" else selection["fundamental"])
        save(name, f)
    for key in ("A", "B", "C"):
        save(f"equal_weight_scores_{key}", sums[key].where(allowed))
    baseline = pd.DataFrame(1., index=index, columns=market.columns).where(allowed)
    save("equal_weight_eligible_pool", baseline, all_eligible=True)
    pd.DataFrame(results).to_csv(out/"supplement/portfolio_summary.csv", index=False)
    smooth = []
    for family in ("A", "B"):
        score = pd.read_pickle(out/"predictions"/f"ridge_{family}_2023_2025.pkl")
        for window in (1, 3, 5, 10):
            for frequency in (1, 5):
                daily, _, _, result = market.backtest(score, "20230101", "20251231", smooth=window, rebalance_every=frequency)
                gross = performance(daily.gross_return)
                smooth.append({"family": family, **result, "gross_annual_return": gross["annual_return"],
                               "sum_daily_costs": float(daily.cost.sum())})
    pd.DataFrame(smooth).to_csv(out/"supplement/smoothing_descriptive.csv", index=False)


def monthly_styles(root, out, selection):
    u = pd.read_pickle(root/"panels/universe.pkl")
    dates = u.loc["20150101":"20221231"].groupby(lambda x: x[:6]).tail(1).index
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=4):
        styles, meta = build_exposures(ResearchStore(root/"panels"), dates)
    industry = styles.pop("industry").astype(object).fillna("SIC2:Unknown")
    names = selection["wq"]+selection["fundamental"]
    exposures, relationships, industries = [], [], []
    for name in names:
        f = pd.read_pickle(out/"factors"/f"{name}.pkl").loc[dates]*selection["directions"][name]
        f = f.where(u.reindex(dates).eq(1))
        target = pd.DataFrame(False, index=f.index, columns=f.columns)
        tie_ranks = stable_tie_ranks(f.columns)
        for i in range(len(f)):
            values = f.iloc[i].to_numpy()
            chosen = top_positions(values, np.isfinite(values), tie_ranks)
            target.iloc[i, chosen] = True
        w = target.div(target.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
        for style, panel in styles.items():
            for day in dates:
                coverage = float(w.loc[day].where(panel.loc[day].notna(), 0).sum())
                exposure = float((w.loc[day]*panel.loc[day].fillna(0)).sum()) if coverage else np.nan
                exposures.append({"factor": name, "date": day, "phase": "2015-2021" if day < "20220101" else "2022", "style": style,
                                  "target_exposure": exposure, "known_exposure_weight": coverage})
            paired = f.notna() & panel.notna()
            r = rank(f.where(paired)).corrwith(rank(panel.where(paired)), axis=1)
            for phase, lo, hi in (("2015-2021", "20150101", "20211231"), ("2022", "20220101", "20221231")):
                relationships.append({"factor": name, "style": style, "phase": phase, "mean_monthend_rank_correlation": float(r.loc[lo:hi].mean())})
        for day in dates:
            weighted = w.loc[day].groupby(industry.loc[day]).sum()
            for key, value in weighted.items():
                if value > 0:
                    industries.append({"factor": name, "date": day, "industry": key, "weight": float(value)})
        print(json.dumps({"stage": "monthly_styles", "factor": name}), flush=True)
    pd.DataFrame(exposures).to_csv(out/"supplement/monthly_target_style_exposures.csv", index=False)
    pd.DataFrame(relationships).to_csv(out/"supplement/factor_style_correlations.csv", index=False)
    pd.DataFrame(industries).to_csv(out/"supplement/monthly_target_industries.csv", index=False)
    write_json({"basis": "signal-close Top100 target, monthly snapshots; not realized performance or risk-adjusted alpha",
                "snapshots": len(dates), "style_warmup": "2015 Momentum can be unavailable; known exposure weights separately recorded", "style_metadata": meta}, out/"supplement/monthly_styles_metadata.json")


def run(database, output, stage="all"):
    root, out = Path(database), Path(output)
    (out/"supplement").mkdir(exist_ok=True)
    bind_inputs(root, out, pd.read_pickle(root/'panels/universe.pkl'))
    original = out/"supplement/protocol.json"
    if original.exists() and json.loads(original.read_text()) != POLICY:
        raise ValueError("Supplement protocol differs")
    write_json(POLICY, original)
    selection = json.loads((out/"factor_selection.json").read_text())
    tasks = {"uncertainty": lambda: uncertainty(out), "factors": lambda: factor_diagnostics(root, out, selection),
             "portfolios": lambda: portfolios(root, out, selection), "styles": lambda: monthly_styles(root, out, selection)}
    for name, task in tasks.items():
        if stage in ("all", name):
            task()
            print(json.dumps({"stage": "supplement_complete", "part": name}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--database", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--stage", default="all", choices=["all", "uncertainty", "factors", "portfolios", "styles"])
    args = p.parse_args()
    run(args.database, args.output, args.stage)
