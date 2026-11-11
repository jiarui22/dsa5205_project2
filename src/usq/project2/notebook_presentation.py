"""Notebook display helpers; these functions never fit models or change results."""
from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

MODEL_NAMES = {'fundamental': 'Six-factor score', 'ridge': 'Ridge',
               'lasso': 'Lasso', 'xgboost': 'XGBoost', 'SPY': 'SPY'}
COLORS = {'fundamental': '#3574af', 'ridge': '#2b927a', 'lasso': '#ba8060',
          'xgboost': '#8e5ead', 'SPY': '#555555'}
STYLE_NAMES = {'ResidualVolatility': 'Residual volatility', 'NonLinearSize': 'Nonlinear size',
               'BookToPrice': 'Book-to-price', 'EarningsYield': 'Earnings yield'}
PERIODS = ['2023 (Validation)', '2024', '2025', '2024-2025']
PERIOD_LABELS = ['2023\nvalidation', '2024', '2025', '2024–2025']
PERCENT_COLUMNS = {'annual_return', 'annual_volatility', 'max_drawdown', 'gross_cagr',
    'mean_daily_turnover', 'mean_cash', 'spy_annual_return', 'coverage', 'stability_coverage',
    'annualized_alpha', 'annualized_HAC_se', 'ci95_lower', 'ci95_upper',
    'linked_contribution', 'cagr', 'alpha_annualized', 'residual_volatility',
    'active_mean_annualized', 'tracking_error', 'cagr_difference', 'ci_low', 'ci_high'}


def format_table(frame, caption=None, percent=()):
    """Format a copy for display, keeping IC and R² in decimal units."""
    frame = frame.to_frame('Value') if isinstance(frame, pd.Series) else frame
    if len(frame) > 40:
        caption = (caption or 'Research table') + f' — first/last 20 of {len(frame):,} rows; full records remain in the run CSV.'
        frame = pd.concat([frame.head(20), frame.tail(20)])
    formats = {}
    for col in frame:
        if col == 'year' or any(token in str(col) for token in ('_date', '_signal', '_start', '_end')):
            formats[col] = lambda value: str(int(value)) if isinstance(value, (int, float, np.integer, np.floating)) else str(value)
        elif col in {'cagr_difference', 'ci_low', 'ci_high', 'linked_contribution'}:
            formats[col] = lambda value: f'{value * 100:+.2f} pp'
        elif col in PERCENT_COLUMNS or col in percent:
            formats[col] = '{:.2%}'
        elif pd.api.types.is_bool_dtype(frame[col]):
            continue
        elif pd.api.types.is_integer_dtype(frame[col]):
            formats[col] = '{:,.0f}'
        elif pd.api.types.is_float_dtype(frame[col]):
            formats[col] = '{:.6f}' if col == 'oos_r2' else '{:.4f}'
    style = frame.style.format(formats, na_rep='—').set_table_styles([
        {'selector': 'caption', 'props': [('text-align', 'left'), ('font-weight', 'bold'),
                                         ('font-size', '1em'), ('margin-bottom', '0.5em')]},
        {'selector': 'th', 'props': [('text-align', 'left')]},
        {'selector': 'td', 'props': [('text-align', 'right')]},
    ])
    return style.set_caption(caption) if caption else style


def _figure(number, title, shape=(1, 1), size=(11, 3.8), **kwargs):
    fig, axes = plt.subplots(*shape, figsize=size, layout='constrained', **kwargs)
    fig.suptitle(f'Figure {number}. {title}', fontsize=13, fontweight='bold')
    return fig, axes


def _dates(ax, dates):
    ax.set_xlim(dates.min(), dates.max())
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=7))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.set_xlabel('Date')


def plot_universe(counts):
    fig, ax = _figure(1, 'Daily eligible universe, 2015–2025', size=(11, 3.4))
    dates = pd.to_datetime(counts.index)
    ax.plot(dates, counts, color=COLORS['fundamental'])
    ax.set_ylabel('Eligible securities (count)'); _dates(ax, dates)
    return fig


def plot_fundamentals(development):
    fig, ax = _figure(2, 'Fundamental-factor evidence, 2020–2022')
    ax.bar(range(6), development.rankic, color=COLORS['fundamental'])
    ax.set_xticks(range(6), [f'F{i}' for i in range(1, 7)])
    ax.set(xlabel='Fundamental factor', ylabel='Mean signed daily RankIC (decimal)')
    ax.axhline(0, color='black', lw=.7)
    return fig


def plot_cv(view):
    fig, ax = _figure(3, 'Feature-family comparison in CV, 2020–2022')
    best = view.query('eligible').groupby(['model', 'feature_set']).mean_rankic.max().unstack()
    best = best.reindex(index=['ridge', 'lasso', 'xgboost'], columns=['fundamental', 'combined', 'engineered'])
    best.rename(index=MODEL_NAMES, columns={'fundamental': 'Fundamentals',
        'combined': 'Fundamentals + WQ', 'engineered': 'Plus engineered features'}).plot.bar(
            ax=ax, color=['#3574af', '#2b927a', '#ba8060'])
    ax.set(xlabel='Learner', ylabel='Best CV mean daily RankIC (decimal)')
    ax.tick_params(axis='x', rotation=0); ax.axhline(0, color='black', lw=.7)
    ax.legend(title='Input family', fontsize=9, title_fontsize=9)
    return fig


def plot_wealth(daily, selection):
    fig, axes = _figure(4, 'Frozen strategy representatives and SPY, 2024–2025',
                        shape=(2, 1), size=(11, 6.3), sharex=True)
    ref = daily(selection['selected_case']); dates = pd.to_datetime(ref.index)
    for model, case in selection['selected_by_model'].items():
        if case:
            axes[0].plot(dates, daily(case).net_nav, label=MODEL_NAMES[model], color=COLORS[model])
    spy = (1 + ref.benchmark_return).cumprod()
    axes[0].plot(dates, spy, '--', color=COLORS['SPY'], label='SPY')
    axes[0].set_ylabel('Net wealth (initial capital = 1)'); axes[0].legend(ncol=4, fontsize=9)
    for nav, label, color in [(ref.net_nav, 'Validation-selected strategy', COLORS['fundamental']),
                              (spy, 'SPY', COLORS['SPY'])]:
        axes[1].plot(dates, nav / nav.cummax().clip(lower=1) - 1, label=label, color=color)
    axes[1].set_ylabel('Drawdown (%)'); axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1].legend(fontsize=9); _dates(axes[1], dates)
    return fig


def plot_periods(r2, period_metrics, selection):
    fig, grid = _figure(5, 'Prediction and portfolio diagnostics by period',
                        shape=(2, 2), size=(12, 8.6))
    axes = grid.ravel()
    pred = r2.assign(label=r2.model.map(MODEL_NAMES)).pivot(index='period', columns='label', values='oos_r2').reindex(PERIODS)
    order = [MODEL_NAMES[m] for m in ('fundamental', 'ridge', 'lasso', 'xgboost')]
    colors = {MODEL_NAMES[m]: c for m, c in COLORS.items()}
    pred.reindex(columns=order).plot.bar(ax=axes[0], color=colors, width=.76)
    axes[0].set(title='Prediction OOS R²', ylabel='OOS R² (decimal)')
    reps = {case: MODEL_NAMES[m] for m, case in selection['selected_by_model'].items() if case}
    reps['SPY'] = 'SPY'
    portfolios = period_metrics.assign(label=period_metrics.case.map(reps))
    cols = [name for name in order if name in reps.values()] + ['SPY']
    for metric, title, ylabel, ax in zip(
        ['annual_return', 'sharpe', 'max_drawdown'], ['Net CAGR', 'Net Sharpe', 'Maximum drawdown'],
        ['Annualized return (%)', 'Sharpe ratio (annualized)', 'Drawdown (%)'], axes[1:]):
        portfolios.pivot(index='period', columns='label', values=metric).reindex(
            index=PERIODS, columns=cols).plot.bar(ax=ax, color=colors, width=.76)
        ax.set(title=title, ylabel=ylabel)
    for ax in axes:
        ax.set_xlabel(''); ax.axhline(0, color='black', lw=.7)
        ax.set_xticklabels(PERIOD_LABELS, rotation=0, fontsize=9)
        ax.legend(ncol=2, fontsize=8)
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    axes[3].yaxis.set_major_formatter(PercentFormatter(1))
    axes[2].get_legend().remove(); axes[3].get_legend().remove()
    return fig


def plot_robustness(ev):
    fig, axes = _figure(6, 'Six-factor score / market cap, 2024–2025',
                        shape=(1, 2), size=(12, 4.1))
    axes[0].plot([10, 20, 30], [ev.loc[c, 'annual_return'] for c in
        ['fundamental_cap', 'cost_20', 'cost_30']], marker='o', label='Market-cap portfolio')
    axes[0].axhline(ev.loc['fundamental_cap', 'spy_annual_return'], color=COLORS['SPY'], ls='--', label='SPY')
    axes[0].set(title='Trading-cost sensitivity', xlabel='Cost per traded dollar (bps)', ylabel='Net CAGR (%)')
    axes[0].set_xticks([10, 20, 30]); axes[0].legend(fontsize=8)
    cases = ['fundamental_cap'] + [f'without_F0{i}' for i in range(1, 7)]
    axes[1].bar(range(7), ev.loc[cases, 'annual_return'], color=['#17365d'] + ['#3c8d95'] * 6)
    axes[1].set_xticks(range(7), ['All six'] + [f'−F{i}' for i in range(1, 7)])
    axes[1].set(title='Leave-one-factor-out sensitivity', xlabel='Included factor set', ylabel='Net CAGR (%)')
    for ax in axes: ax.yaxis.set_major_formatter(PercentFormatter(1))
    return fig


def plot_reference_attribution(attribution):
    fig, axes = _figure(7, 'Market-cap portfolio attribution versus SPY, 2024–2025',
                        shape=(1, 2), size=(12, 4.1))
    z = attribution['attribution'].query("phase == 'evaluation'").set_index('link'); start = 0
    for i, key in enumerate(['selection', 'weighting', 'universe']):
        value = z.loc[key, 'cagr_difference'] * 100
        axes[0].bar(i, value, bottom=start, color='#387f83' if value >= 0 else '#b85c52')
        axes[0].text(i, start + value / 2, f'{value:+.2f}', ha='center', va='center', fontsize=9); start += value
    axes[0].bar(3, start, color='#17365d'); axes[0].set_xticks(range(4), ['Selection', 'Weighting', 'Universe', 'Total'])
    axes[0].set(title='Linked CAGR differences', ylabel='CAGR difference (pp)')
    annual = attribution['annual'].query("phase == 'evaluation'")
    annual.pivot(index='year', columns='link', values='return_difference').reindex(
        columns=['selection', 'weighting', 'universe', 'total']).mul(100).rename(columns=str.title).plot.bar(ax=axes[1])
    axes[1].set(title='Calendar-year return differences', xlabel='Year', ylabel='Total-return difference (pp)')
    axes[1].tick_params(axis='x', rotation=0); axes[1].legend(ncol=2, fontsize=8)
    return fig


def plot_styles(style_table):
    fig, axes = _figure(8, 'Validation-selected portfolio: style attribution, 2024–2025',
                        shape=(1, 2), size=(12, 4.8))
    table = style_table.sort_values('mean_exposure').rename(index=STYLE_NAMES)
    table.mean_exposure.plot.barh(ax=axes[0], color=COLORS['fundamental'])
    table.linked_contribution.mul(100).plot.barh(ax=axes[1], color=COLORS['ridge'])
    axes[0].set(title='Average held style exposure', xlabel='Standardized model units', ylabel='')
    axes[1].set(title='Linked style return contribution', xlabel='Contribution to cumulative return (pp)', ylabel='')
    for ax in axes: ax.axvline(0, color='black', lw=.8)
    return fig


def plot_reference_intervals(attribution):
    fig, ax = _figure(9, 'Market-cap portfolio differences: paired 95% bootstrap intervals, 2024–2025', size=(11, 4))
    ci = attribution['attribution'].query("phase == 'evaluation'")
    for i, row in enumerate(ci.itertuples()):
        ax.plot([row.ci_low * 100, row.ci_high * 100], [i, i], color='#387f83')
        ax.scatter(row.cagr_difference * 100, i, color='#17365d')
    ax.axvline(0, color='black', ls='--'); ax.set_yticks(range(len(ci)), ci.link.str.title())
    ax.set_xlabel('CAGR difference (pp); 20-session paired block bootstrap')
    return fig
