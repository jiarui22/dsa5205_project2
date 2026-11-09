"""Factor-only diagnostics: ten-day forward-return quintiles, not a P&L curve."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .factors import rank


def run(database, output):
    out, root = Path(output), Path(database)/"panels"
    selection = json.loads((out/"factor_selection.json").read_text())
    names = selection["wq"]+selection["fundamental"]
    names += ["C01_EP_ratio", "C02_BP_ratio", "C03_SPQ_ratio", "C04_ROEQ_ratio"]
    u = pd.read_pickle(root/"universe.pkl")
    y = pd.read_pickle(root/"ylabel.pkl")
    ends = pd.read_csv(root/"label_end.csv", dtype=str).set_index("date").label_end
    records = []
    for name in names:
        f = pd.read_pickle(out/"factors"/f"{name}.pkl")*selection["directions"][name]
        valid = u.eq(1) & np.isfinite(f) & np.isfinite(y)
        bins = np.ceil(rank(f.where(valid))*5).clip(1, 5)
        for phase, lo, hi in (("2015-2018", "20150101", "20181231"), ("2019-2021", "20190101", "20211231"), ("2022", "20220101", "20221231")):
            dates = (y.index >= lo) & (y.index <= hi) & ends.reindex(y.index).le(hi)
            for group in range(1, 6):
                values = y.where(bins.eq(group)).loc[dates]
                # Equal day weighting, not large-universe-day weighting.
                records.append({"factor": name, "phase": phase, "group": group,
                                "mean_forward_10d_return": float(values.mean(axis=1).mean()),
                                "mean_stocks": float(values.count(axis=1).mean()),
                                "basis": "overlapping forward labels, descriptive only, no execution or transaction costs"})
        print(json.dumps({"stage": "factor_quintiles", "factor": name}), flush=True)
    pd.DataFrame(records).to_csv(out/"evaluation/factor_forward_quintiles.csv", index=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--database", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    run(a.database, a.output)
