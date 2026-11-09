"""Reuse original Barra-style joint WLS; explain ledger returns, not labels.

SPY is a PERFORMANCE benchmark, not a reconstructed SPY holdings risk portfolio.
Specific includes unexplained returns AND classroom settlement/marking effects;
it is not claimed as investable, risk-adjusted alpha.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from usq.data import PanelStore
from usq.barra.model import build_model
from usq.barra.core.model import design_matrix
from usq.barra.core.attribution import carino_link
from .screen import write_json


class ResearchStore(PanelStore):
    def load(self, name, reference=None, numeric=True):
        x = super().load(name, reference, numeric)
        if name == "industry":
            x = x.astype(object).fillna("SIC2:Unknown")
        return x


def run(database, output, build_only=False, supplement=False):
    from threadpoolctl import threadpool_limits
    out = Path(output)
    risk = out/("supplement/barra_style" if supplement else "barra_style")
    risk.mkdir(exist_ok=True)
    print(json.dumps({"stage": "barra_build_start", "range": ["20230101", "20251231"]}), flush=True)
    with threadpool_limits(limits=4):
        model = build_model(ResearchStore(Path(database)/"panels"), out/"barra_style/model", start="20230101", end="20251231")
    write_json({"style_engine": "repository reference_barra_v1, joint market+SIC2+style WLS",
                "industry_unknown": "explicit Unknown category in research view only; source unmodified",
                "specific": "ledger gross return minus modeled market/industry/style returns; includes settlement and stale-mark effects",
                "SPY": "performance comparison only, no claim of SPY look-through active style attribution",
                "missing_styles": model.meta["missing_styles"],
                "successful_dates": len(model.factor_returns), "styles": model.styles}, risk/"research_adapter.json")
    print(json.dumps({"stage": "barra_build_complete", "dates": len(model.factor_returns), "styles": model.styles}), flush=True)
    if build_only:
        return
    if len(model.factor_returns) != len(model.alignment):
        raise ValueError("Some risk dates could not be estimated")
    names = pd.read_csv(out/("supplement/portfolio_summary.csv" if supplement else "backtests/summary.csv")).strategy
    source = out/("supplement/portfolios" if supplement else "backtests")
    summaries, exposure_means = [], []
    industries = [n for n in model.factor_returns.columns if n not in model.styles and n != "Market"]
    for name in names:
        w = pd.read_pickle(source/f"{name}_weights.pkl")
        daily = pd.read_csv(source/f"{name}_daily.csv", dtype={"date": str}).set_index("date")
        contributions, exposures, missing_styles = [], [], []
        for day in w.index:
            prev = model.alignment.loc[day, "exposure_date"]
            held = w.loc[day]
            held = held[held != 0]
            exp = model.exposures.loc[prev].reindex(held.index)
            X = design_matrix(exp, model.styles, industries)
            b = X.T@held
            f = model.factor_returns.loc[day].reindex(b.index).fillna(0.)
            c = b*f
            c["Specific_or_unexplained"] = daily.loc[day, "gross_return"]-c.sum()
            c["Costs"] = -daily.loc[day, "cost"]
            c["SPY_benchmark_subtraction"] = -daily.loc[day, "benchmark_return"]
            contributions.append(c.rename(day))
            exposures.append(b.rename(day))
            missing_styles.append({"date": day, "held_count": len(held),
                                   "zero_filled_style_weight": float(held[exp[model.styles].isna().any(axis=1)].sum())})
        contrib = pd.DataFrame(contributions).fillna(0.)
        expo = pd.DataFrame(exposures).fillna(0.)
        expected = daily.net_return-daily.benchmark_return
        error = float((contrib.sum(axis=1)-expected).abs().max())
        if error > 1e-9:
            raise AssertionError(f"Attribution reconciliation failed: {name} {error}")
        linked = carino_link(contrib, daily.net_return, daily.benchmark_return)
        grouped = {"Market": linked.get("Market", 0.),
                   "Industry": linked.reindex(industries).fillna(0.).sum(),
                   "Style": linked.reindex(model.styles).fillna(0.).sum(),
                   "Specific_or_unexplained": linked["Specific_or_unexplained"],
                   "Costs": linked["Costs"],
                   "SPY_benchmark_subtraction": linked["SPY_benchmark_subtraction"]}
        cumulative_excess = float(daily.net_nav.iloc[-1]-daily.benchmark_nav.iloc[-1])
        if abs(sum(grouped.values())-cumulative_excess) > 1e-8:
            raise AssertionError("Linked contributions do not reconcile")
        folder = risk/name
        folder.mkdir(exist_ok=True)
        contrib.to_csv(folder/"daily_contributions.csv")
        expo.to_csv(folder/"daily_exposures.csv")
        linked.to_csv(folder/"linked_factor_contributions.csv", header=["contribution"])
        pd.DataFrame(missing_styles).to_csv(folder/"exposure_coverage.csv", index=False)
        summaries.append({"strategy": name, **{k: float(v) for k, v in grouped.items()},
                          "cumulative_return_minus_spy": cumulative_excess, "max_daily_reconciliation_error": error})
        exposure_means.append({"strategy": name, **expo[model.styles].mean().to_dict()})
        print(json.dumps({"stage": "barra_attribution_complete", "strategy": name, "reconciliation_error": error}), flush=True)
    pd.DataFrame(summaries).to_csv(risk/"attribution_summary.csv", index=False)
    pd.DataFrame(exposure_means).to_csv(risk/"mean_style_exposures.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--supplement", action="store_true")
    args = parser.parse_args()
    run(args.database, args.output, args.build_only, args.supplement)
