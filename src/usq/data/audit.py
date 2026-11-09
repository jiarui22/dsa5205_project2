from pathlib import Path
import numpy as np
import pandas as pd
from .panels import PanelStore, date_index
from usq.utils import json_dump


def audit_data(store: PanelStore, output):
    """Production data acceptance, independent calendar required from the data team."""
    ref = store.load("close")
    calfile = store.root / "calendar.csv"
    if not calfile.exists():
        raise ValueError("calendar.csv required: independent US exchange trading dates, including warmup")
    calendar = date_index(pd.read_csv(calfile, dtype=str)["date"])
    if not calendar.equals(ref.index):
        raise ValueError("panel dates differ from calendar.csv")
    if ref.index.min() > "20150105" or ref.index.max() < "20251231":
        raise ValueError("dataset does not cover 2015-2025; supply warmup and censor unmatured final labels")
    universe = store.load("universe", ref)
    if not universe.isin([0, 1]).all().all():
        raise ValueError("universe must be explicit 0/1, without unknown/NaN states")
    mask = universe.eq(1)
    rows = {}
    for name in ("open", "high", "low", "close", "close_adj", "volume", "adj_factor", "market_cap"):
        x = store.load(name, ref)
        invalid = x.isna() | ((x < 0) if name == "volume" else (x <= 0))
        bad = int((invalid & mask).sum().sum())
        rows[name] = {"bad_in_universe": bad, "coverage_in_universe": float((x.notna() & mask).sum().sum() / max(1, mask.sum().sum()))}
    if store.path("vwap").exists():
        vwap = store.load("vwap", ref)
        rows["vwap"] = {"required": False, "available": True,
            "bad_in_universe": int(((vwap.isna() | vwap.le(0)) & mask).sum().sum())}
    else:
        rows['vwap'] = {'required': False, 'available': False,
            'bad_in_universe': None, 'note': 'Optional factor input; do not infer from unrelated source amount/volume.'}
    industry = store.load("industry", ref, numeric=False)
    rows["industry"] = {"bad_in_universe": int((industry.isna() & mask).sum().sum())}
    high, low = store.load("high", ref), store.load("low", ref)
    rows["ohlc"] = {"bad_in_universe": int(((high < low) & mask).sum().sum())}
    report = {"start": ref.index.min(), "end": ref.index.max(), "dates": len(ref), "stocks": ref.shape[1],
              "checks": rows, "annual_trading_dates": pd.Series(ref.index).str[:4].value_counts().sort_index().to_dict(),
              "passed": not any(x["bad_in_universe"] for x in rows.values() if x.get('required', True)),
              "external_acceptance_still_required": ["calendar provenance", "dead securities", "PIT vintages", "adjustment/dividend methodology", "stable ID history"]}
    json_dump(report, output)
    if not report["passed"]: raise ValueError(f"data audit failed; see {output}")
    return report
