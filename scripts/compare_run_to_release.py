"""Compare freshly computed aggregate results with the upstream release, without refitting."""
from __future__ import annotations
import argparse
from io import StringIO
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = 'a4a31d6f822f6e2ab1099993266454799d9e8c31'


def original(name):
    return subprocess.run(['git', 'show', f'{BASE}:results/{name}'], cwd=ROOT,
                          check=True, capture_output=True, text=True).stdout


def compare(work):
    work = Path(work)
    checks = {}
    entries = [
        ('portfolio_summary.csv', work/'portfolios/portfolio_summary.csv', ['case', 'phase'],
         ['annual_return', 'annual_volatility', 'sharpe', 'max_drawdown', 'mean_daily_turnover',
          'mean_cash', 'spy_annual_return']),
        ('WQ_screen.csv', work/'factor_screen/WQ_screen.csv', ['factor'],
         ['direction', 'early_rankic', 'stability_rankic_signed', 'stability_coverage']),
    ]
    for name, path, keys, columns in entries:
        before = pd.read_csv(StringIO(original(name))).set_index(keys).sort_index()
        after = pd.read_csv(path).set_index(keys).sort_index()
        pd.testing.assert_index_equal(before.index, after.index)
        differences = {}
        for col in columns:
            a, b = before[col].to_numpy(float), after[col].to_numpy(float)
            np.testing.assert_allclose(a, b, rtol=1e-8, atol=1e-10, equal_nan=True)
            finite = np.isfinite(a) & np.isfinite(b)
            differences[col] = float(np.max(np.abs(a[finite]-b[finite]))) if finite.any() else 0
        checks[name] = {'matched': True, 'rows': len(after), 'maximum_absolute_differences': differences}
    for name in ('frozen_hyperparameters.json', 'frozen_family.json'):
        before, after = json.loads(original(name)), json.loads((work/'portfolios/learners'/name).read_text())
        if name == 'frozen_hyperparameters.json':
            for model in before:
                assert before[model]['candidate'] == after[model]['candidate']
                assert before[model]['params'] == after[model]['params']
        else:
            assert before == after
        checks[name] = {'matched': True}
    record = {'upstream_commit': BASE, 'passed': True, 'checks': checks,
              'note': 'Agreement reproduces historical results; it does not create a fresh holdout.'}
    (work/'release_comparison.json').write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2))
    return record


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work', required=True, type=Path)
    compare(p.parse_args().work)
