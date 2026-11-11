"""Holdings-based risk attribution for the validation-selected strategy."""
from pathlib import Path
from math import erfc, sqrt

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from usq.barra.core.attribution import carino_link
from usq.barra.core.model import design_matrix
from usq.barra.model import build_model
from .risk import ResearchStore
from .study_periods import PORTFOLIO_PHASES


def residual_alpha(returns, lags=10):
    """Mean of daily net residual returns, with Bartlett HAC inference."""
    x = np.asarray(returns, dtype=float)
    n = len(x)
    if n <= lags + 1 or not np.isfinite(x).all():
        raise ValueError('Insufficient finite residual returns for HAC inference')
    mean = x.mean()
    centered = x - mean
    long_run_variance = centered @ centered / n
    for lag in range(1, lags + 1):
        long_run_variance += 2 * (1 - lag / (lags + 1)) * (centered[lag:] @ centered[:-lag]) / n
    se = sqrt(max(float(long_run_variance), 0) / n)
    t = mean / se if se > 0 else np.nan
    return {
        'sessions': n, 'HAC_lags': lags, 'annualized_alpha': 252 * mean,
        'annualized_HAC_se': 252 * se, 'HAC_t': t,
        'p_value_two_sided': erfc(abs(t) / sqrt(2)),
        'ci95_lower': 252 * (mean - 1.96 * se), 'ci95_upper': 252 * (mean + 1.96 * se),
    }


def build_selected_attribution(database, portfolio, output):
    """Use previous-close exposures and actual beginning-of-day ledger weights."""
    portfolio, output = Path(portfolio), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    daily = pd.read_csv(portfolio / 'daily.csv', dtype={'date': str}).set_index('date')
    weights = pd.read_pickle(portfolio / 'weights.pkl')
    if not weights.index.equals(daily.index):
        raise ValueError('Holdings and daily ledger dates differ')

    start, end = PORTFOLIO_PHASES['evaluation']
    with threadpool_limits(limits=4):
        model = build_model(ResearchStore(Path(database) / 'panels'), output / 'model',
                            start=start, end=end)
    missing_dates = daily.index.difference(model.factor_returns.index)
    if len(missing_dates):
        raise ValueError(f'Risk model could not estimate {len(missing_dates)} ledger dates')
    factors = model.factor_returns.reindex(daily.index)
    styles = model.styles
    industries = [name for name in factors if name.startswith('Industry:')]
    factor_names = ['Market', *industries, *styles]
    if set(factor_names) != set(factors):
        raise ValueError('Unexpected factor-return columns')
    exposures, coverage = [], []
    for date, row in weights.iterrows():
        previous = model.alignment.loc[date, 'exposure_date']
        assert previous < date
        held = row[row.ne(0)].astype(float)
        exp = model.exposures.loc[previous].reindex(held.index)
        # Missing style exposures enter the portfolio projection at standardized zero.
        # Track any held weight for which the style or industry contribution is unavailable.
        design = design_matrix(exp, styles, industries)
        exposures.append((design.T @ held).rename(date))
        missing_industry = ~exp.industry.isin(factors.loc[date, industries].dropna().index)
        coverage.append({
            'date': date,
            'missing_style_weight': held[exp[styles].isna().any(axis=1)].sum(),
            'missing_industry_factor_weight': held[missing_industry].sum(),
        })
    exposure = pd.DataFrame(exposures).reindex(columns=factor_names)
    contrib = exposure * factors.reindex(columns=factor_names).fillna(0)
    systematic = contrib.sum(axis=1)
    contrib['Specific / unexplained'] = daily.gross_return - systematic
    contrib['Trading costs'] = -daily.cost
    net_residual = daily.net_return - systematic
    np.testing.assert_allclose(contrib.sum(axis=1), daily.net_return, rtol=0, atol=1e-12)
    linked = carino_link(contrib, daily.net_return)
    np.testing.assert_allclose(linked.sum(), (1 + daily.net_return).prod() - 1, rtol=0, atol=1e-9)

    components = pd.DataFrame({
        'linked_contribution': [linked['Market'], linked[industries].sum(), linked[styles].sum(),
                                linked['Specific / unexplained'], linked['Trading costs']],
    }, index=pd.Index(['Market', 'Industry', 'Style', 'Specific / unexplained', 'Trading costs'],
                      name='component'))
    factor_table = pd.DataFrame({'mean_exposure': exposure.mean(),
                                 'linked_contribution': linked.reindex(factor_names)})
    factor_table.index.name = 'factor'
    alpha = pd.DataFrame([residual_alpha(net_residual)])
    diagnostics = pd.DataFrame(coverage).set_index('date')
    diagnostics['cross_section_r2'] = model.diagnostics.reindex(daily.index).r2.astype(float)
    diagnostics['n_obs'] = model.diagnostics.reindex(daily.index).n_obs.astype(int)
    diagnostics['net_residual'] = net_residual
    for name, frame in [('components', components), ('factors', factor_table), ('alpha', alpha),
                        ('daily_exposures', exposure), ('daily_contributions', contrib),
                        ('coverage', diagnostics)]:
        frame.to_csv(output / f'{name}.csv', index=name != 'alpha')
    return components, factor_table, alpha, diagnostics, model.meta
