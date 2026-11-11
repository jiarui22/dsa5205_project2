"""Build report assets only from a verified, uninterrupted Notebook execution."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from usq.project2.supplement import paired_bootstrap

WEIGHTS = {'equal': 'Equal', 'sqrt_cap': 'Square-root cap', 'cap': 'Capitalization'}
MODELS = {'fundamental': 'Six-factor score', 'ridge': 'Ridge', 'lasso': 'Lasso', 'xgboost': 'XGBoost'}
STYLES = ['Size', 'Beta', 'Momentum', 'ResidualVolatility', 'NonLinearSize',
          'BookToPrice', 'Liquidity', 'EarningsYield', 'Leverage']


def escape(value):
    return str(value).replace('\\', r'\textbackslash{}').replace('_', r'\_').replace('%', r'\%').replace('&', r'\&')


def pct(value, digits=2):
    return '--' if pd.isna(value) else f'{100*value:.{digits}f}'


def number(value, digits=3):
    return '--' if pd.isna(value) else f'{value:.{digits}f}'


def parameter_label(kind, params):
    if kind == 'ridge':
        return rf'$\alpha/n={params["alpha_per_sample"]:g}$'
    if kind == 'lasso':
        alpha=f'{params["alpha"]:.6f}'.rstrip('0').rstrip('.')
        return rf'$\alpha={alpha}$'
    return (f'depth={params["max_depth"]}, lr={params["learning_rate"]:g}, '
            +rf'$\lambda={params["reg_lambda"]:g}$')


def write_table(path, headers, rows, align=None):
    align = align or 'l' + 'r' * (len(headers) - 1)
    body = '\n'.join(' & '.join(map(str, row)) + r' \\' for row in rows)
    path.write_text(r'\begin{tabular}{' + align + '}\n' + r'\toprule' + '\n'
                    + ' & '.join(headers) + r' \\' + '\n' + r'\midrule' + '\n'
                    + body + '\n' + r'\bottomrule' + '\n' + r'\end{tabular}' + '\n')


def render_report_figures(work, figs, evidence, selection, summary, intervals):
    """Replot the same run's aggregates at readable print sizes, without fitting."""
    cache=ROOT/'runs/report_runtime/mpl'
    cache.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR',str(cache))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    import matplotlib.dates as mdates
    portfolio=work/'portfolios'
    chosen=selection['selected_case']
    model_names={'fundamental':'Fundamental','ridge':'Ridge','lasso':'Lasso','xgboost':'XGBoost'}
    colors={'fundamental':'#3574af','ridge':'#2b927a','lasso':'#ba8060','xgboost':'#8e5ead','SPY':'#555555'}
    def daily(case,phase='evaluation'):
        return pd.read_csv(portfolio/'portfolios'/case/phase/'daily.csv',dtype={'date':str}).set_index('date')
    def save(name,fig):
        fig.tight_layout(pad=.7)
        fig.savefig(figs/f'{name}.pdf',bbox_inches='tight')
        plt.close(fig)
    with plt.rc_context({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':9.5,
                         'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,
                         'legend.fontsize':7.5,'axes.spines.top':False,'axes.spines.right':False,
                         'axes.grid':True,'grid.alpha':.18,'pdf.fonttype':42}):
        counts=pd.read_pickle(work/'database/panels/universe.pkl').loc['20150101':].eq(1).sum(axis=1)
        counts.rename('eligible_stocks').to_csv(evidence/'universe_counts.csv',index_label='date')
        fig,ax=plt.subplots(figsize=(7.1,2.4));ax.plot(pd.to_datetime(counts.index),counts,lw=1)
        ax.set(ylabel='Eligible stocks',title='Daily supplied universe');save('nb_universe',fig)
        stats=pd.read_csv(portfolio/'factor_statistics.csv').query("phase=='development'")
        fig,ax=plt.subplots(figsize=(7.1,2.6));ax.bar(range(6),stats.rankic,color='#3574af')
        ax.set_xticks(range(6),[f'F{i}' for i in range(1,7)]);ax.axhline(0,color='black',lw=.7)
        ax.set(ylabel='Signed RankIC',title='Fundamental evidence, 2020-2022');save('nb_fundamentals',fig)
        cv=pd.read_csv(portfolio/'learners/CV_comparison.csv').query('eligible')
        best=cv.groupby(['model','feature_set']).mean_rankic.max().unstack().reindex(['ridge','lasso','xgboost'])
        fig,ax=plt.subplots(figsize=(7.1,2.9));best.plot.bar(ax=ax,color=['#3574af','#2b927a','#ba8060'])
        ax.set(xlabel='',ylabel='Mean daily RankIC');ax.tick_params(axis='x',rotation=0)
        ax.axhline(0,color='black',lw=.7);ax.legend(title='Input family',ncol=3,fontsize=7.5,title_fontsize=8)
        save('nb_cv',fig)
        ref=daily(chosen);dates=pd.to_datetime(ref.index)
        fig,axes=plt.subplots(2,1,figsize=(7.1,4.2),sharex=True)
        for model,case in selection['selected_by_model'].items():
            if case:axes[0].plot(dates,daily(case).net_nav,label=model_names[model],color=colors[model],lw=1.1)
        spy=(1+ref.benchmark_return).cumprod()
        axes[0].plot(dates,spy,'k--',label='SPY',lw=1.1);axes[0].legend(ncol=4);axes[0].set_ylabel('Net wealth')
        for nav,label,color in [(ref.net_nav,'Selected','#3574af'),(spy,'SPY','#555555')]:
            axes[1].plot(dates,nav/nav.cummax().clip(lower=1)-1,label=label,color=color,lw=1)
        axes[1].yaxis.set_major_formatter(PercentFormatter(1));axes[1].set_ylabel('Drawdown');axes[1].legend(ncol=2)
        axes[1].xaxis.set_major_locator(mdates.MonthLocator(interval=6));axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        save('nb_wealth',fig)
        r2=pd.read_csv(evidence/'period_oos_r2.csv')
        periods=['2023 (Validation)','2024','2025','2024-2025'];short=['2023\nvalidation','2024','2025','2024-25']
        metrics=pd.read_csv(evidence/'period_metrics.csv')
        reps={case:model_names[model] for model,case in selection['selected_by_model'].items() if case};reps['SPY']='SPY'
        color_by_label={model_names[k]:v for k,v in colors.items() if k in model_names};color_by_label['SPY']=colors['SPY']
        fig,grid=plt.subplots(2,2,figsize=(7.1,5.5));axes=grid.ravel()
        pred=r2.assign(label=r2.model.map(model_names)).pivot(index='period',columns='label',values='oos_r2').reindex(periods)
        pred.plot.bar(ax=axes[0],color=color_by_label,width=.76);axes[0].set_title('Prediction OOS $R^2$')
        view=metrics.assign(label=metrics.case.map(reps))
        for key,title,ax in zip(['annual_return','sharpe','max_drawdown'],['Net CAGR','Net Sharpe','Max drawdown'],axes[1:]):
            view.pivot(index='period',columns='label',values=key).reindex(periods).plot.bar(ax=ax,color=color_by_label,width=.76)
            ax.set_title(title)
        for ax in axes:
            ax.set_xlabel('');ax.axhline(0,color='black',lw=.6);ax.set_xticklabels(short,rotation=0,fontsize=7.5)
            ax.legend(ncol=2,fontsize=6.8)
        axes[1].yaxis.set_major_formatter(PercentFormatter(1));axes[3].yaxis.set_major_formatter(PercentFormatter(1))
        axes[2].get_legend().remove();axes[3].get_legend().remove();save('nb_periods',fig)
        ev=summary.query("phase=='evaluation'").set_index('case')
        fig,axes=plt.subplots(1,2,figsize=(7.1,3.1))
        axes[0].plot([10,20,30],[ev.loc[c,'annual_return'] for c in ['fundamental_sqrt_cap','cost_20','cost_30']],marker='o')
        axes[0].axhline(ev.loc['fundamental_sqrt_cap','spy_annual_return'],color='black',ls='--',label='SPY')
        axes[0].set(title='Reference: cost sensitivity',xlabel='bps per traded dollar',ylabel='Net CAGR');axes[0].legend()
        cases=['fundamental_sqrt_cap']+[f'without_F0{i}' for i in range(1,7)]
        axes[1].bar(range(7),[ev.loc[c,'annual_return'] for c in cases],color=['#17365d']+['#3c8d95']*6)
        axes[1].set_xticks(range(7),['Six']+[f'-F{i}' for i in range(1,7)],fontsize=8);axes[1].set_title('Reference: factor ablation')
        for ax in axes:ax.yaxis.set_major_formatter(PercentFormatter(1))
        save('nb_stress',fig)
        attr={'attribution':pd.read_csv(work/'attribution/attribution.csv'),
              'annual':pd.read_csv(work/'attribution/attribution_annual.csv')}
        links=['selection','weighting','universe'];z=attr['attribution'].query("phase=='evaluation'").set_index('link')
        fig,axes=plt.subplots(1,2,figsize=(7.1,3.1));start=0
        for i,key in enumerate(links):
            value=z.loc[key,'cagr_difference']*100;axes[0].bar(i,value,bottom=start,color='#387f83' if value>=0 else '#b85c52')
            axes[0].text(i,start+value/2,f'{value:+.2f}',ha='center',va='center',fontsize=8);start+=value
        axes[0].bar(3,start,color='#17365d');axes[0].set_xticks(range(4),['Selection','Weighting','Universe','Total'],rotation=20,fontsize=7.5)
        axes[0].set_ylabel('CAGR difference (pp)')
        annual=attr['annual'].query("phase=='evaluation'").pivot(index='year',columns='link',values='return_difference')
        annual.reindex(columns=links+['total']).plot.bar(ax=axes[1],width=.8)
        axes[1].set(xlabel='',ylabel='Annual return difference');axes[1].tick_params(axis='x',rotation=0)
        axes[1].yaxis.set_major_formatter(PercentFormatter(1));axes[1].legend(ncol=2,fontsize=6.8)
        save('nb_reference_attribution',fig)
        styles=pd.read_csv(work/'risk_attribution/factors.csv').set_index('factor').loc[STYLES]
        names={'ResidualVolatility':'Residual vol.','NonLinearSize':'Nonlinear size','BookToPrice':'Book-to-price','EarningsYield':'Earnings yield'}
        fig,axes=plt.subplots(1,2,figsize=(7.1,3.4))
        for col,title,ax in zip(['mean_exposure','linked_contribution'],['Mean style exposure','Linked style contribution'],axes):
            values=styles[col].sort_values();ax.barh([names.get(k,k) for k in values.index],values,color='#3574af' if col=='mean_exposure' else '#2b927a')
            ax.axvline(0,color='black',lw=.6);ax.set_title(title);ax.tick_params(axis='y',labelsize=8)
        axes[0].set_xlabel('Standardized model units');axes[1].xaxis.set_major_formatter(PercentFormatter(1));save('nb_styles',fig)
        fig,ax=plt.subplots(figsize=(7.1,2.5))
        for i,row in enumerate(intervals.itertuples()):
            ax.plot([100*row.cagr_diff_ci_low,100*row.cagr_diff_ci_high],[i,i],color='#387f83',lw=2)
            ax.scatter(100*row.cagr_difference,i,color='#17365d',s=25)
        ax.axvline(0,color='black',ls='--',lw=.8);ax.set_yticks(range(len(intervals)),[f'{b}-session blocks' for b in intervals.block])
        ax.set_xlabel('Selected minus SPY CAGR (percentage points)');save('nb_selected_intervals',fig)


def build(work, executed, report):
    work, executed, report = map(Path, (work, executed, report))
    nb = json.loads(executed.read_text())
    verification = nb.get('metadata', {}).get('clean_run_verification', {})
    if verification.get('passed') is not True:
        raise ValueError('Report generation requires a successful full fresh-kernel run.')
    code = [c for c in nb['cells'] if c['cell_type'] == 'code']
    if [c['execution_count'] for c in code] != list(range(1, len(code) + 1)):
        raise ValueError('Nonsequential or missing code-cell execution counts.')
    if any(o.get('output_type') == 'error' for c in code for o in c.get('outputs', [])):
        raise ValueError('Notebook contains error outputs.')
    portfolio = work / 'portfolios'
    learners = portfolio / 'learners'
    summary = pd.read_csv(portfolio / 'portfolio_summary.csv')
    selection = json.loads((learners / 'frozen_family.json').read_text())
    parameters = json.loads((learners / 'frozen_hyperparameters.json').read_text())
    selected = selection['selected_case']
    evaluation = summary.query("phase=='evaluation'").set_index('case')
    chosen = evaluation.loc[selected]
    history = pd.read_csv(learners / 'training_history.csv', dtype={'forecast_start': str, 'train_last_label_end': str})
    if history.reused_verified_fit.any() or len(history) != 396:
        raise ValueError('Expected 396 fresh model fits without checkpoint reuse.')
    if not history.train_last_label_end.lt(history.forecast_start).all():
        raise ValueError('A model used an unmatured label.')
    audit = json.loads((portfolio / 'portfolio_audit.json').read_text())
    if not audit['passed'] or audit['portfolios'] != 38:
        raise ValueError('All 38 portfolio ledgers must pass accounting audit.')
    report.mkdir(parents=True, exist_ok=True)
    tabs, figs, evidence = (report / name for name in ('tables', 'figures', 'notebook_evidence'))
    for directory in (tabs, figs, evidence):
        directory.mkdir(exist_ok=True)
    # Store compact run-bound inputs, never individual-security panels or holdings.
    for src in [portfolio/'portfolio_summary.csv', portfolio/'factor_statistics.csv',
                portfolio/'portfolio_audit.json', learners/'CV_comparison.csv',
                learners/'frozen_hyperparameters.json', learners/'frozen_family.json',
                learners/'training_history.csv', work/'factor_screen/WQ_screen.csv',
                work/'factor_screen/WQ_selection.json', work/'data_complete.json',
                work/'input_audit.csv', work/'financial_audit.json']:
        shutil.copy2(src, evidence / src.name)
    for name in ('period_metrics.csv', 'period_oos_r2.csv'):
        shutil.copy2(ROOT/'results'/name, evidence/name)
    for name in ('alpha', 'components', 'factors', 'coverage'):
        shutil.copy2(work/'risk_attribution'/f'{name}.csv', evidence/f'barra_{name}.csv')
    # Require every expected plot output; print-sized figures are replotted below.
    image_names = {6: ['nb_universe'], 10: ['nb_fundamentals'], 12: ['nb_cv'],
                   16: ['nb_wealth', 'nb_periods'], 18: ['nb_stress'],
                   20: ['nb_reference_attribution'], 22: ['nb_styles'], 24: ['nb_reference_intervals']}
    for i, names in image_names.items():
        outputs = [o['data']['image/png'] for o in nb['cells'][i].get('outputs', [])
                   if 'image/png' in o.get('data', {})]
        if len(outputs) != len(names):
            raise ValueError(f'Unexpected figure outputs in cell {i}: {len(outputs)}')
    model_cases = {case: (model, weighting)
                   for model in MODELS for weighting in WEIGHTS
                   for case in [(f'fundamental_{weighting}' if model == 'fundamental'
                                 else f'ml_{model}' if weighting == 'sqrt_cap'
                                 else f'ml_{model}_{weighting}') ]}
    labels = {case: f'{MODELS[m]} / {WEIGHTS[w]}' for case, (m, w) in model_cases.items()}
    macro = {
        'SelectedName': escape(labels[selected]), 'SelectedCase': escape(selected),
        'SelectedCAGR': pct(chosen.annual_return), 'SelectedVol': pct(chosen.annual_volatility),
        'SelectedSharpe': number(chosen.sharpe), 'SelectedMDD': pct(chosen.max_drawdown),
        'SelectedTurnover': pct(chosen.mean_daily_turnover), 'SPYCAGR': pct(chosen.spy_annual_return),
        'ActiveCAGR': pct(chosen.annual_return - chosen.spy_annual_return),
        'FitCount': str(len(history)), 'CodeCells': str(len(code)),
        'LedgerError': f'{audit["maximum_independent_holdings_error"]:.2e}',
        'WQList': ', '.join(json.loads((work/'factor_screen/WQ_selection.json').read_text())['wq']),
    }
    def daily(case, phase='evaluation'):
        return pd.read_csv(portfolio/'portfolios'/case/phase/'daily.csv', dtype={'date': str}).set_index('date')
    reference = daily(selected)
    if not reference.index.is_unique or not np.isfinite(reference.net_return).all():
        raise ValueError('Invalid selected-strategy daily series.')
    spy = reference.benchmark_return
    annualized = lambda r: float(np.expm1(np.log1p(r).mean()*252))
    risk_alpha = pd.read_csv(work/'risk_attribution/alpha.csv').iloc[0]
    macro.update({'ResidualAlpha': pct(risk_alpha.annualized_alpha),
                  'ResidualLow': pct(risk_alpha.ci95_lower), 'ResidualHigh': pct(risk_alpha.ci95_upper),
                  'ResidualT': number(risk_alpha.HAC_t)})
    (report/'notebook_numbers.tex').write_text('\n'.join(
        '\\newcommand{\\'+key+'}{'+value+'}' for key,value in macro.items())+'\n')
    write_table(tabs/'nb_hyperparameters.tex', ['Model', 'Selected inputs', 'Frozen parameters', 'CV RankIC'],
        [[MODELS[k], escape(v['params']['feature_set']),
          parameter_label(k,v['params']),
          number(v['CV_mean_rankic'], 4)] for k,v in parameters.items()], 'llll')
    validation = summary.query("phase=='validation'").set_index('case')
    write_table(tabs/'nb_validation.tex', ['Score / weighting', r'CAGR (\%)', 'Sharpe', r'MDD (\%)', r'Cash (\%)'],
        [[escape(labels[c])+(' *' if c==selected else ''), pct(validation.loc[c,'annual_return']),
          number(validation.loc[c,'sharpe']), pct(validation.loc[c,'max_drawdown']),
          pct(validation.loc[c,'mean_cash'])] for c in model_cases])
    representatives = [c for c in selection['selected_by_model'].values() if c]
    rows = [[escape(labels[c]), pct(evaluation.loc[c,'annual_return']), pct(evaluation.loc[c,'annual_volatility']),
             number(evaluation.loc[c,'sharpe']), pct(evaluation.loc[c,'max_drawdown']),
             pct(evaluation.loc[c,'mean_daily_turnover'])] for c in representatives]
    spy_nav=(1+spy).cumprod()
    rows.append(['SPY', pct(annualized(spy)), pct(spy.std(ddof=1)*np.sqrt(252)),
                 number(spy.mean()/spy.std(ddof=1)*np.sqrt(252)),
                 pct((spy_nav/spy_nav.cummax().clip(lower=1)-1).min()), '--'])
    write_table(tabs/'nb_evaluation.tex', ['Portfolio', r'CAGR (\%)', r'Vol. (\%)', 'Sharpe', r'MDD (\%)', r'Turn. (\%)'], rows)
    periods=pd.read_csv(evidence/'period_metrics.csv')
    period_index=periods.assign(period=periods.period.astype(str)).set_index(['period','case'])
    prose={}
    for year,tag in [('2024','First'),('2025','Second')]:
        prose[tag+'YearCAGR']=pct(period_index.loc[(year,selected),'annual_return'])
        prose[tag+'YearSPY']=pct(period_index.loc[(year,'SPY'),'annual_return'])
    for kind,tag in [('ridge','Ridge'),('xgboost','Tree')]:
        case=selection['selected_by_model'].get(kind)
        prose[tag+'EvalCAGR']=pct(evaluation.loc[case,'annual_return']) if case else '--'
    with (report/'notebook_numbers.tex').open('a') as f:
        f.write('\n'.join('\\newcommand{\\'+key+'}{'+value+'}'
                          for key,value in prose.items())+'\n')
    write_table(tabs/'nb_annual.tex', ['Period', 'Portfolio', r'CAGR (\%)', 'Sharpe', r'MDD (\%)'],
        [[escape(r.period), 'Selected' if r.case==selected else 'SPY', pct(r.annual_return),
          number(r.sharpe), pct(r.max_drawdown)] for r in periods.itertuples() if r.case in [selected,'SPY']])
    screen=pd.read_csv(work/'factor_screen/WQ_screen.csv').set_index('factor')
    wqs=json.loads((work/'factor_screen/WQ_selection.json').read_text())['wq']
    write_table(tabs/'nb_wq.tex', ['Factor', 'Direction', '2018--19 RankIC', r'Coverage (\%)', 'Positive years'],
        [[n, int(screen.loc[n,'direction']), number(screen.loc[n,'stability_rankic_signed'],4),
          pct(screen.loc[n,'stability_coverage']), int(screen.loc[n,'positive_stability_years'])] for n in wqs])
    fundamentals=pd.read_csv(portfolio/'factor_statistics.csv').query("phase=='development'")
    write_table(tabs/'nb_fundamentals.tex', ['Signal', 'Direction', '2020--22 RankIC', r'Coverage (\%)'],
        [[r.factor[:3], int(r.direction), number(r.rankic,4), pct(r.coverage)] for r in fundamentals.itertuples()])
    forecast=history.query("tag=='forecast' and model=='ridge'")
    write_table(tabs/'nb_windows.tex', ['Forecast half', 'First training signal', 'Last training signal', 'Last label end', 'Rows'],
        [[r.half, r.train_first, r.train_last_signal, r.train_last_label_end, f'{r.samples:,}'] for r in forecast.itertuples()])
    r2=pd.read_csv(evidence/'period_oos_r2.csv')
    r2['period']=r2.period.astype(str)
    rv=r2.pivot(index='period',columns='model',values='oos_r2').reindex(['2023 (Validation)','2024','2025','2024-2025'])
    write_table(tabs/'nb_r2.tex', ['Period', 'Direct score', 'Ridge', 'Lasso', 'XGBoost'],
        [[escape(p)]+[number(row.get(k,np.nan),6) for k in MODELS] for p,row in rv.iterrows()])
    write_table(tabs/'nb_weighting.tex', ['Opportunity set', r'Equal CAGR (\%)', r'Sqrt-cap CAGR (\%)', r'Cap CAGR (\%)'],
        [[name]+[pct(evaluation.loc[f'{prefix}_{w}','annual_return']) for w in WEIGHTS]
         for prefix,name in [('fundamental','Six-factor Top100'),('pool','Full eligible pool')]])
    comparisons=[]
    for block in [10,20,40]:
        ci=paired_bootstrap(spy,reference.net_return,block=block)
        comparisons.append({'block':block,**ci})
    intervals=pd.DataFrame(comparisons);intervals.to_csv(evidence/'selected_spy_intervals.csv',index=False)
    write_table(tabs/'nb_intervals.tex', ['Block (sessions)', 'CAGR gap (pp)', r'Lower 95\% (pp)', r'Upper 95\% (pp)'],
        [[int(r.block),pct(r.cagr_difference),pct(r.cagr_diff_ci_low),pct(r.cagr_diff_ci_high)] for r in intervals.itertuples()])
    row=intervals.query('block==20').iloc[0]
    with (report/'notebook_numbers.tex').open('a') as f:
        f.write(r'\newcommand{\SelectedCILow}{'+pct(row.cagr_diff_ci_low)+'}\n')
        f.write(r'\newcommand{\SelectedCIHigh}{'+pct(row.cagr_diff_ci_high)+'}\n')
    coverage=pd.read_csv(work/'risk_attribution/coverage.csv')
    write_table(tabs/'nb_risk_coverage.tex', ['Diagnostic', 'Value'], [
        ['Mean held weight with missing style', pct(coverage.missing_style_weight.mean())+r'\%'],
        ['Mean held weight without an industry return', pct(coverage.missing_industry_factor_weight.mean())+r'\%'],
        ['Median cross-sectional regression $R^2$', number(coverage.cross_section_r2.median())],
        ['Median daily regression stock count', f'{int(coverage.n_obs.median()):,}'],
    ],'lr')
    components=pd.read_csv(work/'risk_attribution/components.csv')
    linked=components.set_index('component').linked_contribution
    component_macros={tag:pct(linked.loc[name]) for name,tag in [
        ('Market','MarketLinked'),('Style','StyleLinked'),
        ('Specific / unexplained','SpecificLinked'),('Trading costs','CostLinked')]}
    with (report/'notebook_numbers.tex').open('a') as f:
        f.write('\n'.join('\\newcommand{\\'+key+'}{'+value+'}'
                          for key,value in component_macros.items())+'\n')
    write_table(tabs/'nb_components.tex', ['Component', 'Linked cumulative contribution (pp)'],
        [[escape(r.component),pct(r.linked_contribution)] for r in components.itertuples()])
    factors=pd.read_csv(work/'risk_attribution/factors.csv').set_index('factor')
    write_table(tabs/'nb_styles.tex', ['Style', 'Mean exposure', 'Linked contribution (pp)'],
        [[escape(k),number(factors.loc[k,'mean_exposure']),pct(factors.loc[k,'linked_contribution'])] for k in STYLES])
    data=json.loads((work/'data_complete.json').read_text())
    finance=data['financial_audit']
    write_table(tabs/'nb_audit.tex', ['Input audit', 'Count'], [
        ['Exchange sessions / security-ID union', f'{data["calendar_sessions"]:,} / {data["SID_union"]:,}'],
        ['Source files hash-verified', data['source_files_verified']],
        ['Material OHLC cells quarantined', data['quarantined_bars']],
        ['Quarantined cells inside the supplied universe', data['quarantined_eligible_bars']],
        ['Financial vintage events', f'{finance["events"]:,}'],
        ['Availability before fiscal period end', finance['events_before_period_end']],
    ],'lr')
    template=(report/'notebook_report_template.tex').read_text()
    render_report_figures(work,figs,evidence,selection,summary,intervals)
    (report/'Group_34_Report.tex').write_text(template)
    inventory={'source_commit':'a4a31d6f822f6e2ab1099993266454799d9e8c31',
               'notebook_sha256':hashlib.sha256(executed.read_bytes()).hexdigest(),
               'verified_code_cells':len(code), 'fresh_model_fits':len(history),
               'portfolio_ledgers':audit['portfolios'], 'selected_case':selected,
               'data_included':False, 'report_phase':'2024-2025',
               'historical_dates_previously_inspected':True,
               'figures':sorted(p.name for p in figs.glob('nb_*.pdf')),
               'tables':sorted(p.name for p in tabs.glob('nb_*.tex'))}
    (report/'notebook_report_inventory.json').write_text(json.dumps(inventory,indent=2))
    (report/'asset_inventory.json').write_text(json.dumps(inventory,indent=2))
    print(json.dumps(inventory,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work',required=True,type=Path)
    p.add_argument('--executed-notebook',required=True,type=Path)
    p.add_argument('--report-dir',default=ROOT/'reports/final',type=Path)
    a=p.parse_args();build(a.work,a.executed_notebook,a.report_dir)
