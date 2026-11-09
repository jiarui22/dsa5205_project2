"""Expanding-from-2015 Ridge/Lasso/XGBoost; 2022 validation, 2023-25 OOS."""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .factors import rank
from .ledger import Market
from .screen import information_series, write_json


MODEL_POLICY = {
    "seed": 5205, "training_start": "20150101", "window": "expanding, never rolling fixed length",
    "label_purge": "label_end strictly before prediction half start; no incomplete targets",
    "target": "raw 10-session close_adj return, no neutralization, clipping or standardization",
    "features": "current eligible cross-sectional percentile minus .5, plus missing indicators; missing percentile=0",
    "sample_rule": "same pool and dates for A/B; at least one observed selected WQ factor; finite mature target in training",
    "constant_predictions": "undefined IC and RankIC; all-equal scores imply cash, never SID-order stock selection; invalid validation candidates cannot win",
    "min_validation_rankic_day_fraction": .8,
    "parameter_selection": "maximize pooled 2022 daily RankIC on A (price-volume only); reuse parameters unchanged for B",
    "grids": {"ridge": [{"alpha_per_sample": .001}, {"alpha_per_sample": .01}],
              "lasso": [{"alpha": .0001}, {"alpha": .0005}],
              "xgboost": [{"max_depth": 2}, {"max_depth": 3}]},
    "xgboost_fixed": {"n_estimators": 120, "learning_rate": .05, "subsample": .8,
                      "colsample_bytree": 1., "reg_lambda": 10., "min_child_weight": 100,
                      "tree_method": "hist", "n_jobs": 4, "random_state": 5205},
    "trading_selection": "maximize 2022 net Sharpe of Ridge A; same rule for every final model; ties prefer less turnover",
    "smooth_candidates": [1, 3, 5, 10], "rebalance_candidates": [1, 5],
    "top_n": 100, "main_cost_bps": 10, "cost_sensitivity_bps": [0, 5, 10, 20, 30],
    "execution": "T after-close signal, T+1 close execution, T+2 first held return; no future-based reselection",
    "marking": "close_adj; stale nontrading marks held pending quote; terminal source return if present, else last-mark cash-settlement assumption; event exceptions saved",
    "benchmark": "SPY adjusted close, investment starts on same close as strategy",
    "test_halves": ["2023H1", "2023H2", "2024H1", "2024H2", "2025H1", "2025H2"],
}


def halves(year):
    return [(f"{year}H1", f"{year}0101", f"{year}0630"),
            (f"{year}H2", f"{year}0701", f"{year}1231")]


class Design:
    def __init__(self, database, output):
        self.root, self.out = Path(database), Path(output)
        selection = json.loads((self.out/"factor_selection.json").read_text())
        self.names = selection["wq"] + selection["fundamental"]
        if not selection["wq"]:
            raise ValueError("No WQ factor passed the frozen criteria")
        self.nq = len(selection["wq"])
        self.u = pd.read_pickle(self.root/"panels/universe.pkl")
        self.index, self.columns = self.u.index, self.u.columns
        self.y = pd.read_pickle(self.root/"panels/ylabel.pkl").to_numpy()
        self.end = pd.read_csv(self.root/"panels/label_end.csv", dtype=str).set_index("date")["label_end"].reindex(self.index)
        cache = self.out/"design"
        cache.mkdir(exist_ok=True)
        shape = (len(self.index), len(self.columns), 2*len(self.names))
        signature = {"features": self.names, "shape": list(shape), "policy": MODEL_POLICY["features"]}
        if not (cache/"meta.json").exists():
            arr = np.lib.format.open_memmap(cache/"features.npy", mode="w+", dtype="float32", shape=shape)
            observed = np.zeros(self.u.shape, dtype=bool)
            for i, name in enumerate(self.names):
                f = pd.read_pickle(self.out/"factors"/f"{name}.pkl").where(self.u.eq(1))
                z = (rank(f)-.5)*selection["directions"][name]
                arr[:, :, 2*i] = z.fillna(0).to_numpy(dtype="float32")
                arr[:, :, 2*i+1] = z.isna().to_numpy(dtype="float32")
                if i < self.nq:
                    observed |= z.notna().to_numpy()
                print(json.dumps({"stage": "feature_cache", "factor": name}), flush=True)
            arr.flush()
            del arr
            np.save(cache/"observed.npy", observed)
            write_json(signature, cache/"meta.json")
        elif json.loads((cache/"meta.json").read_text()) != signature:
            raise ValueError("Feature cache differs; use a new experiment directory")
        self.x = np.load(cache/"features.npy", mmap_mode="r")
        self.observed = np.load(cache/"observed.npy", mmap_mode="r")

    def training(self, start, family):
        dates = (self.index >= "20150101") & (self.index < start) & self.end.lt(start).fillna(False).to_numpy()
        valid = dates[:, None] & self.u.eq(1).to_numpy() & self.observed & np.isfinite(self.y)
        positions = np.flatnonzero(valid)
        nf = 2*self.nq if family == "A" else 2*len(self.names)
        # Slice columns first to avoid materializing unused B columns for A.
        x = np.asarray(self.x[:, :, :nf].reshape(-1, nf)[positions], dtype="float32", order="C")
        y = np.asarray(self.y.ravel()[positions], dtype="float32")
        used_dates = np.flatnonzero(valid.any(axis=1))
        meta = {"samples": len(y), "feature_count": nf, "train_first": self.index[used_dates[0]],
                "train_last_signal": self.index[used_dates[-1]],
                "train_last_label_end": str(self.end.iloc[used_dates].max()),
                "forecast_start": start, "family": family}
        if not meta["train_last_label_end"] < start:
            raise AssertionError("Future target leaked into training")
        return x, y, meta

    def predict(self, model, start, end, family):
        positions = np.flatnonzero((self.index >= start) & (self.index <= end))
        nf = 2*self.nq if family == "A" else 2*len(self.names)
        rows = []
        for i in positions:
            p = model.predict(np.asarray(self.x[i, :, :nf], dtype="float32", order="C"))
            p = np.where(self.u.iloc[i].eq(1).to_numpy() & self.observed[i], p, np.nan)
            rows.append(p.astype("float32"))
        return pd.DataFrame(rows, index=self.index[positions], columns=self.columns)


def fit_model(kind, params, x, y):
    from threadpoolctl import threadpool_limits
    from sklearn.linear_model import Lasso, Ridge
    if kind == "ridge":
        model = Ridge(alpha=params["alpha_per_sample"]*len(y), solver="cholesky")
    elif kind == "lasso":
        model = Lasso(alpha=params["alpha"], max_iter=500, tol=1e-5, selection="cyclic")
    elif kind == "xgboost":
        from xgboost import XGBRegressor
        model = XGBRegressor(**MODEL_POLICY["xgboost_fixed"], **params, objective="reg:squarederror")
    else:
        raise ValueError(kind)
    with threadpool_limits(limits=4):
        model.fit(x, y)
    return model


def run(database, output):
    out = Path(output)
    protocol_path = out/"model_protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != MODEL_POLICY:
        raise ValueError("Model protocol changed; use a new experiment output")
    write_json(MODEL_POLICY, protocol_path)  # before opening any 2022 outcomes
    for folder in ("predictions", "models", "backtests", "validation"):
        (out/folder).mkdir(exist_ok=True)
    design = Design(database, output)
    market = None
    chosen, validation_rows = {}, []
    training_records = []
    for kind, grid in MODEL_POLICY["grids"].items():
        best, best_metric = None, -np.inf
        for candidate, params in enumerate(grid):
            pieces = []
            for half, start, end in halves(2022):
                path = out/"validation"/f"{kind}_grid{candidate}_{half}.pkl"
                meta_path = path.with_suffix(".json")
                if path.exists() and meta_path.exists():
                    score = pd.read_pickle(path)
                    meta = json.loads(meta_path.read_text())
                else:
                    clock = time.time()
                    x, y, meta = design.training(start, "A")
                    model = fit_model(kind, params, x, y)
                    score = design.predict(model, start, end, "A")
                    meta.update(model=kind, params=params, half=half, seconds=round(time.time()-clock, 2))
                    score.to_pickle(path)
                    write_json(meta, meta_path)
                    del model, x, y
                    gc.collect()
                pieces.append(score)
                training_records.append(meta)
                print(json.dumps({"stage": "validation_fit", **meta}), flush=True)
            scores = pd.concat(pieces)
            target = pd.DataFrame(design.y, index=design.index, columns=design.columns).reindex(scores.index)
            target = target.where(design.end.reindex(scores.index).le("20221231"), axis=0)
            evaluation = information_series(scores, target, design.u.reindex(scores.index))
            metric = float(evaluation.rankic.mean())
            eligible_days = target.notna().sum(axis=1).ge(30)
            fraction = float(evaluation.rankic.notna().sum()/eligible_days.sum())
            validation_rows.append({"model": kind, "candidate": candidate, "params": json.dumps(params), "rankic": metric, "defined_rankic_day_fraction": fraction})
            if np.isfinite(metric) and fraction >= MODEL_POLICY["min_validation_rankic_day_fraction"] and metric > best_metric:
                best, best_metric = candidate, metric
        if best is None:
            raise ValueError(f"Every {kind} candidate has undefined validation RankIC")
        chosen[kind] = {"candidate": best, "params": grid[best], "validation_rankic": best_metric}
    pd.DataFrame(validation_rows).to_csv(out/"validation/hyperparameters.csv", index=False)
    write_json(chosen, out/"validation/chosen_hyperparameters.json")
    ridge2022 = pd.concat([pd.read_pickle(out/"validation"/f"ridge_grid{chosen['ridge']['candidate']}_{half}.pkl") for half, _, _ in halves(2022)])
    market = Market(database)
    rules = []
    for smooth in MODEL_POLICY["smooth_candidates"]:
        for rebalance in MODEL_POLICY["rebalance_candidates"]:
            daily, _, events, summary = market.backtest(ridge2022, "20220101", "20221231", smooth=smooth, rebalance_every=rebalance)
            rules.append(summary)
    rule_table = pd.DataFrame(rules).sort_values(["sharpe", "mean_daily_turnover"], ascending=[False, True])
    rule_table.to_csv(out/"validation/trading_rules.csv", index=False)
    rule = {"smooth": int(rule_table.iloc[0].smooth), "rebalance_every": int(rule_table.iloc[0].rebalance_every)}
    write_json({"hyperparameters": chosen, "trading": rule, "frozen_before_test": True}, out/"frozen_model_choices.json")
    print(json.dumps({"stage": "validation_complete", "choices": chosen, "trading": rule}), flush=True)
    results, all_evaluations, annuals = [], [], []
    for kind in MODEL_POLICY["grids"]:
        for family in ("A", "B"):
            name = f"{kind}_{family}"
            pieces = []
            for year in (2023, 2024, 2025):
                for half, start, end in halves(year):
                    path = out/"predictions"/f"{name}_{half}.pkl"
                    meta_path = path.with_suffix(".json")
                    if path.exists() and meta_path.exists():
                        score, meta = pd.read_pickle(path), json.loads(meta_path.read_text())
                    else:
                        import joblib
                        clock = time.time()
                        x, y, meta = design.training(start, family)
                        model = fit_model(kind, chosen[kind]["params"], x, y)
                        score = design.predict(model, start, end, family)
                        meta.update(model=kind, params=chosen[kind]["params"], half=half, seconds=round(time.time()-clock, 2))
                        score.to_pickle(path)
                        joblib.dump(model, out/"models"/f"{name}_{half}.joblib", compress=3)
                        write_json(meta, meta_path)
                        del model, x, y
                        gc.collect()
                    pieces.append(score)
                    training_records.append(meta)
                    print(json.dumps({"stage": "test_fit", **meta}), flush=True)
            scores = pd.concat(pieces)
            scores.to_pickle(out/"predictions"/f"{name}_2023_2025.pkl")
            target = pd.DataFrame(design.y, index=design.index, columns=design.columns).reindex(scores.index)
            evaluation = information_series(scores, target, design.u.reindex(scores.index))
            evaluation.to_parquet(out/"evaluation"/f"{name}_test_daily.parquet")
            all_evaluations.append({"strategy": name, "ic": float(evaluation.ic.mean()), "rankic": float(evaluation.rankic.mean())})
            daily, weights, events, summary = market.backtest(scores, "20230101", "20251231", weights=True, **rule)
            daily.to_csv(out/"backtests"/f"{name}_daily.csv")
            weights.to_pickle(out/"backtests"/f"{name}_weights.pkl")
            events.to_csv(out/"backtests"/f"{name}_events.csv", index=False)
            results.append({"strategy": name, **summary})
            for year, block in daily.groupby(daily.index.str[:4]):
                from usq.backtest.metrics import performance
                ann = performance(block.net_return, block.turnover)
                ann["strategy"], ann["year"] = name, year
                ann["spy_total_return"] = float((1+block.benchmark_return).prod()-1)
                annuals.append(ann)
            costs = []
            for bps in MODEL_POLICY["cost_sensitivity_bps"]:
                _, _, _, cost_summary = market.backtest(scores, "20230101", "20251231", cost_bps=bps, **rule)
                costs.append({"strategy": name, **cost_summary})
            pd.DataFrame(costs).to_csv(out/"backtests"/f"{name}_costs.csv", index=False)
            group_daily = {}
            for group in range(1, 6):
                gd, _, _, _ = market.backtest(scores, "20230101", "20251231", group=group, groups=5, **rule)
                group_daily[f"G{group}"] = gd.net_return
            pd.DataFrame(group_daily).to_csv(out/"backtests"/f"{name}_groups.csv")
            # Checkpoint completed strategies; a slow XGB fit cannot hide finished results.
            pd.DataFrame(results).to_csv(out/"backtests/summary.csv", index=False)
            pd.DataFrame(annuals).to_csv(out/"backtests/annual.csv", index=False)
            pd.DataFrame(all_evaluations).to_csv(out/"evaluation/model_test_ic.csv", index=False)
            print(json.dumps({"stage": "strategy_complete", "strategy": name, **summary}), flush=True)
    write_json(training_records, out/"training_history.json")
    write_json({"complete": True, "strategies": [r["strategy"] for r in results],
                "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest()}, out/"model_run_complete.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--database", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    run(args.database, args.output)
