"""Benchmark-relative return attribution from stored daily portfolio returns.

The module fits and selects nothing.  It re-expresses the nominated candidate's
return difference to SPY as a chain of matched comparisons that already exist in
the final replay, and adds market-model and tracking-error diagnostics.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from .supplement import paired_bootstrap

SESSIONS = 252
BLOCK = 20
HAC_LAGS = 10
# Candidate -> same weighting, whole pool -> capitalization weights -> ETF.
CHAIN = ['fundamental_sqrt_cap', 'pool_sqrt_cap', 'pool_cap', 'SPY']
LINKS = {'fundamental_sqrt_cap-pool_sqrt_cap': 'selection',
         'pool_sqrt_cap-pool_cap': 'weighting',
         'pool_cap-SPY': 'universe'}
WEIGHTINGS = ['equal', 'sqrt_cap', 'cap']
MARKET_MODEL_CASES = ['fundamental_sqrt_cap', 'fundamental_cap', 'fundamental_equal',
                      'pool_sqrt_cap', 'pool_cap']
PHASES = ['validation', 'evaluation']


def cagr(returns):
    """Annualized compound return under the replay's 252-session convention."""
    return float(np.expm1(np.log1p(np.asarray(returns, float)).mean()*SESSIONS))


def total_return(returns):
    return float(np.prod(1+np.asarray(returns, float))-1)


def chain_attribution(series, path):
    """Telescoping CAGR differences along ``path``; the links sum to first minus last."""
    if len(path) < 2:
        raise ValueError('an attribution path needs at least two portfolios')
    growth = {name: cagr(series[name]) for name in path}
    return [{'link': f'{upper}-{lower}', 'upper': upper, 'lower': lower,
             'cagr_difference': growth[upper]-growth[lower]}
            for upper, lower in zip(path[:-1], path[1:])]


def active_statistics(portfolio, benchmark):
    """Arithmetic active return, tracking error, and the excess needed for t = 2."""
    diff = np.asarray(portfolio, float)-np.asarray(benchmark, float)
    tracking = float(diff.std(ddof=1)*np.sqrt(SESSIONS))
    years = len(diff)/SESSIONS
    return {'active_mean_annualized': float(diff.mean()*SESSIONS), 'tracking_error': tracking,
            'information_ratio': float(diff.mean()/diff.std(ddof=1)*np.sqrt(SESSIONS)),
            'detectable_excess': float(2*tracking/np.sqrt(years))}


def capm(portfolio, market, lags=HAC_LAGS):
    """Market-model OLS with a Bartlett-kernel (Newey-West) alpha t-statistic.

    Cash earns zero in the replay ledger, so raw returns are used on both sides.
    """
    y, x = np.asarray(portfolio, float), np.asarray(market, float)
    if y.shape != x.shape or y.ndim != 1 or len(y) <= lags+2:
        raise ValueError('paired one-dimensional return series required')
    design = np.column_stack([np.ones(len(x)), x])
    coefficients = np.linalg.solve(design.T@design, design.T@y)
    residual = y-design@coefficients
    scores = design*residual[:, None]
    meat = scores.T@scores
    for lag in range(1, lags+1):
        cross = scores[lag:].T@scores[:-lag]
        meat = meat+(1-lag/(lags+1))*(cross+cross.T)
    bread = np.linalg.inv(design.T@design)
    covariance = bread@meat@bread
    return {'alpha_annualized': float(coefficients[0]*SESSIONS), 'beta': float(coefficients[1]),
            'alpha_t_hac': float(coefficients[0]/np.sqrt(covariance[0, 0])),
            'residual_volatility': float(residual.std(ddof=2)*np.sqrt(SESSIONS)),
            'correlation': float(np.corrcoef(y, x)[0, 1])}


def load_phase(results, phase, cases):
    """Daily net returns for ``cases`` plus SPY on the phase's common sessions."""
    frames = {}
    for case in cases:
        daily = pd.read_csv(Path(results)/'portfolios'/case/phase/'daily.csv', dtype={'date': str})
        frames[case] = daily.set_index('date')
    reference = frames[cases[0]]
    for case, daily in frames.items():
        if not daily.index.equals(reference.index):
            raise ValueError(f'{case} is not aligned with {cases[0]} in {phase}')
        if not np.allclose(daily.benchmark_return, reference.benchmark_return, atol=1e-12):
            raise ValueError(f'{case} carries a different SPY series in {phase}')
    returns = pd.DataFrame({case: daily.net_return for case, daily in frames.items()})
    return returns.assign(SPY=reference.benchmark_return)


def link_row(returns, upper, lower, label):
    """One matched comparison: point difference, paired interval, active statistics."""
    interval = paired_bootstrap(returns[lower], returns[upper], block=BLOCK)
    return {'link': label, 'upper': upper, 'lower': lower,
            'cagr_difference': cagr(returns[upper])-cagr(returns[lower]),
            'ci_low': interval['cagr_diff_ci_low'], 'ci_high': interval['cagr_diff_ci_high'],
            **active_statistics(returns[upper], returns[lower])}


def build(results, tables=None):
    """Write the attribution tables beside the other aggregate evidence."""
    results = Path(results)
    tables = Path(tables) if tables is not None else results/'tables'
    tables.mkdir(parents=True, exist_ok=True)
    cases = sorted(set(CHAIN[:-1]+MARKET_MODEL_CASES+[f'{kind}_{w}' for kind in ('fundamental', 'pool') for w in WEIGHTINGS]))
    attribution, annual, selection, market_model = [], [], [], []
    for phase in PHASES:
        returns = load_phase(results, phase, cases)
        for link in chain_attribution(returns, CHAIN):
            attribution.append({'phase': phase, **link_row(returns, link['upper'], link['lower'], LINKS[link['link']])})
        attribution.append({'phase': phase, **link_row(returns, CHAIN[0], CHAIN[-1], 'total')})
        for year, block in returns.groupby(returns.index.str[:4]):
            for upper, lower, label in [(u, l, LINKS[f'{u}-{l}']) for u, l in zip(CHAIN[:-1], CHAIN[1:])]+[(CHAIN[0], CHAIN[-1], 'total')]:
                annual.append({'phase': phase, 'year': int(year), 'link': label,
                               'return_difference': total_return(block[upper])-total_return(block[lower])})
        for weighting in WEIGHTINGS:
            selection.append({'phase': phase, 'weighting': weighting,
                              **link_row(returns, f'fundamental_{weighting}', f'pool_{weighting}', 'selection')})
        for case in MARKET_MODEL_CASES:
            market_model.append({'phase': phase, 'case': case, 'cagr': cagr(returns[case]),
                                 **capm(returns[case], returns['SPY']),
                                 **active_statistics(returns[case], returns['SPY'])})
    frames = {'attribution': pd.DataFrame(attribution), 'annual': pd.DataFrame(annual),
              'selection': pd.DataFrame(selection), 'capm': pd.DataFrame(market_model)}
    names = {'attribution': 'attribution.csv', 'annual': 'attribution_annual.csv',
             'selection': 'selection_by_weighting.csv', 'capm': 'capm.csv'}
    for key, frame in frames.items():
        frame.to_csv(tables/names[key], index=False)
    return frames


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='results/project2_final')
    arguments = parser.parse_args()
    for name, frame in build(arguments.output).items():
        print(f'\n[{name}]\n{frame.round(4).to_string(index=False)}')
