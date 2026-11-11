"""Refresh the English report from published results only; no model/data execution."""
from __future__ import annotations
import argparse
import base64
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
SOURCE_COMMIT = '743b82c5b85b763ea2dde54668bf80d1f026ab32'
MODELS = {'fundamental': 'Six-factor score', 'ridge': 'Ridge', 'lasso': 'Lasso', 'xgboost': 'XGBoost'}
WEIGHTS = {'equal': 'Equal', 'sqrt_cap': 'Square-root cap', 'cap': 'Capitalization'}
COLORS = {'fundamental': '#3574af', 'ridge': '#2b927a', 'lasso': '#ba8060', 'xgboost': '#8e5ead', 'SPY': '#555555'}
STYLES = ['Size','Beta','Momentum','ResidualVolatility','NonLinearSize','BookToPrice','Liquidity','EarningsYield','Leverage']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def esc(value):
    return str(value).replace('_', r'\_').replace('%', r'\%').replace('&', r'\&')


def pct(value, digits=2):
    return '--' if pd.isna(value) else f'{100 * float(value):.{digits}f}'


def num(value, digits=3):
    return '--' if pd.isna(value) else f'{float(value):.{digits}f}'


def table(path, headers, rows, align=None):
    align = align or 'l' + 'r' * (len(headers)-1)
    path.write_text('\\begin{tabular}{'+align+'}\n\\toprule\n'+
        ' & '.join(headers)+r' \\'+'\n\\midrule\n'+
        '\n'.join(' & '.join(map(str,r))+r' \\' for r in rows)+
        '\n\\bottomrule\n\\end{tabular}\n')


def saved_table(nb, cell, output):
    """Read a stored HTML table; do not execute its source cell."""
    html = ''.join(nb['cells'][cell]['outputs'][output]['data']['text/html'])
    soup = BeautifulSoup(html, 'html.parser').select_one('table')
    headers = [x.get_text(' ', strip=True) for x in soup.select('thead tr')[0].find_all('th')][1:]
    rows = [[td.get_text(' ', strip=True) for td in tr.find_all('td', recursive=False)]
            for tr in soup.select('tbody tr')]
    return pd.DataFrame(rows, columns=headers)


def verify_notebook(nb, expected):
    cells = [c for c in nb['cells'] if c['cell_type']=='code']
    assert [c['execution_count'] for c in cells] == list(range(1, expected+1))
    assert not [o for c in cells for o in c.get('outputs', []) if o['output_type']=='error']


def build(report):
    results = ROOT/'results'; report=Path(report)
    nb_path=ROOT/'notebooks/Equity_Strategy_Research.ipynb'
    data_nb_path=ROOT/'notebooks/Equity_Data_Preparation.ipynb'
    nb=json.loads(nb_path.read_text()); data_nb=json.loads(data_nb_path.read_text())
    verify_notebook(nb,13); verify_notebook(data_nb,10)
    audit=json.loads((results/'reproduction_audit.json').read_text())
    assert audit['strategy_notebook_complete'] and audit['data_notebook_complete']
    for path, expected in audit['code_sha256'].items():
        assert sha(ROOT/path)==expected, f'Recorded source mismatch: {path}'
    history=pd.read_csv(results/'training_history.csv',dtype={'train_last_label_end':str,'forecast_start':str})
    assert len(history)==audit['fresh_model_fits']==396 and not history.reused_verified_fit.any()
    assert history.train_last_label_end.lt(history.forecast_start).all()
    ledger=json.loads((results/'portfolio_audit.json').read_text())
    assert ledger['passed'] and ledger['portfolios']==38
    summary=pd.read_csv(results/'portfolio_summary.csv')
    selection=json.loads((results/'frozen_family.json').read_text())
    params=json.loads((results/'frozen_hyperparameters.json').read_text())
    chosen=selection['selected_case']; ev=summary.query("phase=='evaluation'").set_index('case')
    assert chosen==audit['selected_case']=='fundamental_cap'
    expected=audit['selected_metrics'][1]
    for key in ['annual_return','sharpe','max_drawdown']:
        assert np.isclose(ev.loc[chosen,key],expected[key],rtol=0,atol=1e-12)
    tabs=report/'tables'; figs=report/'figures'; evidence=report/'notebook_evidence'
    for d in [tabs,figs,evidence]:d.mkdir(parents=True,exist_ok=True)
    inputs={str(p.relative_to(ROOT)):sha(p) for p in sorted(results.rglob('*')) if p.is_file()}
    inputs[str(nb_path.relative_to(ROOT))]=sha(nb_path)
    inputs[str(data_nb_path.relative_to(ROOT))]=sha(data_nb_path)
    for path in sorted(results.rglob('*')):
        if path.is_file():
            dest=evidence/path.relative_to(results);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest)
    periods=pd.read_csv(results/'period_metrics.csv')
    r2=pd.read_csv(results/'period_oos_r2.csv')
    wq=pd.read_csv(results/'WQ_screen.csv').set_index('factor')
    wqs=json.loads((results/'WQ_selection.json').read_text())['wq']
    stats=pd.read_csv(results/'factor_statistics.csv').query("phase=='development'")
    risk=pd.read_csv(results/'barra_alpha.csv').iloc[0]
    components=pd.read_csv(results/'barra_components.csv').set_index('component').linked_contribution
    factors=pd.read_csv(results/'barra_factors.csv').set_index('factor')
    attr=saved_table(nb,20,0)
    attr.to_csv(evidence/'saved_notebook_attribution.csv',index=False)
    numeric=lambda text:float(text.replace(' pp','').replace('%',''))
    at=attr.query("phase=='evaluation'").set_index('link')
    # Points are identities from published CAGR summaries; intervals remain the saved outputs.
    points={'selection':ev.loc[chosen,'annual_return']-ev.loc['pool_cap','annual_return'],
        'weighting':ev.loc['pool_cap','annual_return']-ev.loc['pool_equal','annual_return'],
        'universe':ev.loc['pool_equal','annual_return']-ev.loc[chosen,'spy_annual_return'],
        'total':ev.loc[chosen,'annual_return']-ev.loc[chosen,'spy_annual_return']}
    assert np.isclose(sum(points[k] for k in ['selection','weighting','universe']),points['total'])
    for key,value in points.items(): assert abs(100*value-numeric(at.loc[key,'cagr_difference']))<=.0051
    ci_low=numeric(at.loc['total','ci_low']);ci_high=numeric(at.loc['total','ci_high'])
    macros={'SelectedName':'Six-factor score / Capitalization','SelectedCase':esc(chosen),
        'SelectedCAGR':pct(ev.loc[chosen,'annual_return']),'SelectedVol':pct(ev.loc[chosen,'annual_volatility']),
        'SelectedSharpe':num(ev.loc[chosen,'sharpe']),'SelectedMDD':pct(ev.loc[chosen,'max_drawdown']),
        'SelectedTurnover':pct(ev.loc[chosen,'mean_daily_turnover']),'SPYCAGR':pct(ev.loc[chosen,'spy_annual_return']),
        'ActiveCAGR':pct(points['total']),'FitCount':'396','CodeCells':'13','DataCodeCells':'10',
        'LedgerError':f'{ledger["maximum_independent_holdings_error"]:.2e}',
        'WQList':', '.join(wqs),'ResidualAlpha':pct(risk.annualized_alpha),'ResidualLow':pct(risk.ci95_lower),
        'ResidualHigh':pct(risk.ci95_upper),'ResidualT':num(risk.HAC_t),
        'RidgeEvalCAGR':pct(ev.loc[selection['selected_by_model']['ridge'],'annual_return']),
        'TreeEvalCAGR':pct(ev.loc[selection['selected_by_model']['xgboost'],'annual_return']),
        'SelectedCILow':num(ci_low,2),'SelectedCIHigh':num(ci_high,2),
        'CostTwentyCAGR':pct(ev.loc['cost_20','annual_return']),'CostThirtyCAGR':pct(ev.loc['cost_30','annual_return']),
        'WithoutFourCAGR':pct(ev.loc['without_F04','annual_return']),
        'WithoutFourSharpe':num(ev.loc['without_F04','sharpe']),
        'SelectionLink':pct(points['selection']),'WeightingLink':pct(points['weighting']),
        'UniverseLink':pct(points['universe'])}
    ix=periods.set_index(['period','case'])
    for year,tag in [('2024','First'),('2025','Second')]:
        macros[tag+'YearCAGR']=pct(ix.loc[(year,chosen),'annual_return'])
        macros[tag+'YearSPY']=pct(ix.loc[(year,'SPY'),'annual_return'])
    for name,tag in [('Market','MarketLinked'),('Style','StyleLinked'),('Specific / unexplained','SpecificLinked'),('Trading costs','CostLinked')]:
        macros[tag]=pct(components.loc[name])
    (report/'notebook_numbers.tex').write_text('\n'.join('\\newcommand{\\'+k+'}{'+v+'}' for k,v in macros.items())+'\n')
    model_cases={case:(m,w) for m in MODELS for w in WEIGHTS for case in [
        f'fundamental_{w}' if m=='fundamental' else f'ml_{m}' if w=='sqrt_cap' else f'ml_{m}_{w}']}
    labels={case:f'{MODELS[m]} / {WEIGHTS[w]}' for case,(m,w) in model_cases.items()}
    table(tabs/'nb_hyperparameters.tex',['Model','Selected inputs','Frozen parameters','CV RankIC'],[
        [MODELS[m],esc(p['params']['feature_set']),
         rf'$\alpha/n={p["params"]["alpha_per_sample"]:g}$' if m=='ridge' else
         rf'$\alpha={p["params"]["alpha"]:.4f}$' if m=='lasso' else
         f'depth={p["params"]["max_depth"]}, lr={p["params"]["learning_rate"]:g}, '+rf'$\lambda={p["params"]["reg_lambda"]:g}$',
         num(p['CV_mean_rankic'],4)] for m,p in params.items()],'llll')
    val=summary.query("phase=='validation'").set_index('case')
    table(tabs/'nb_validation.tex',['Score / weighting',r'CAGR (\%)','Sharpe',r'MDD (\%)',r'Cash (\%)'],[
        [esc(labels[c])+(' *' if c==chosen else ''),pct(val.loc[c,'annual_return']),num(val.loc[c,'sharpe']),pct(val.loc[c,'max_drawdown']),pct(val.loc[c,'mean_cash'])] for c in model_cases])
    reps=[c for c in selection['selected_by_model'].values() if c]
    rows=[[esc(labels[c]),pct(ev.loc[c,'annual_return']),pct(ev.loc[c,'annual_volatility']),num(ev.loc[c,'sharpe']),pct(ev.loc[c,'max_drawdown']),pct(ev.loc[c,'mean_daily_turnover'])] for c in reps]
    spy=ix.loc[('2024-2025','SPY')]
    rows.append(['SPY',pct(spy.annual_return),pct(spy.annual_volatility),num(spy.sharpe),pct(spy.max_drawdown),'--'])
    table(tabs/'nb_evaluation.tex',['Portfolio',r'CAGR (\%)',r'Vol. (\%)','Sharpe',r'MDD (\%)',r'Turn. (\%)'],rows)
    table(tabs/'nb_annual.tex',['Period','Portfolio',r'CAGR (\%)','Sharpe',r'MDD (\%)'],[
        [esc(r.period),'Selected' if r.case==chosen else 'SPY',pct(r.annual_return),num(r.sharpe),pct(r.max_drawdown)] for r in periods.itertuples() if r.case in [chosen,'SPY']])
    table(tabs/'nb_wq.tex',['Factor','Direction','2018--19 RankIC',r'Coverage (\%)','Positive years'],[
        [name,int(wq.loc[name,'direction']),num(wq.loc[name,'stability_rankic_signed'],4),pct(wq.loc[name,'stability_coverage']),int(wq.loc[name,'positive_stability_years'])] for name in wqs])
    table(tabs/'nb_fundamentals.tex',['Signal','Direction','2020--22 RankIC',r'Coverage (\%)'],[
        [r.factor[:3],r.direction,num(r.rankic,4),pct(r.coverage)] for r in stats.itertuples()])
    table(tabs/'nb_windows.tex',['Forecast half','First training signal','Last training signal','Last label end','Rows'],[
        [r.half,r.train_first,r.train_last_signal,r.train_last_label_end,f'{r.samples:,}'] for r in history.query("tag=='forecast' and model=='ridge'").itertuples()])
    rv=r2.pivot(index='period',columns='model',values='oos_r2').reindex(['2023 (Validation)','2024','2025','2024-2025'])
    table(tabs/'nb_r2.tex',['Period','Direct score','Ridge','Lasso','XGBoost'],[
        [esc(p)]+[num(row[m],6) for m in MODELS] for p,row in rv.iterrows()])
    table(tabs/'nb_weighting.tex',['Opportunity set',r'Equal CAGR (\%)',r'Sqrt-cap CAGR (\%)',r'Cap CAGR (\%)'],[
        [name]+[pct(ev.loc[f'{prefix}_{w}','annual_return']) for w in WEIGHTS]
        for prefix,name in [('fundamental','Six-factor Top100'),('pool','Full eligible pool')]])
    table(tabs/'nb_intervals.tex',['Comparison link','CAGR gap (pp)',r'Lower 95\% (pp)',r'Upper 95\% (pp)'],[
        [k.title(),pct(points[k]),at.loc[k,'ci_low'].replace(' pp',''),at.loc[k,'ci_high'].replace(' pp','')] for k in points])
    coverage=pd.read_csv(results/'barra_coverage.csv')
    table(tabs/'nb_risk_coverage.tex',['Diagnostic','Value'],[
        ['Mean held weight with missing style',pct(coverage.missing_style_weight.mean())+r'\%'],
        ['Mean held weight without an industry return',pct(coverage.missing_industry_factor_weight.mean())+r'\%'],
        ['Median cross-sectional regression $R^2$',num(coverage.cross_section_r2.median())],
        ['Median daily regression stock count',f'{int(coverage.n_obs.median()):,}']],'lr')
    table(tabs/'nb_components.tex',['Component','Linked cumulative contribution (pp)'],[[esc(k),pct(v)] for k,v in components.items()])
    table(tabs/'nb_styles.tex',['Style','Mean exposure','Linked contribution (pp)'],[
        [esc(k),num(factors.loc[k,'mean_exposure']),pct(factors.loc[k,'linked_contribution'])] for k in STYLES])
    source_inventory=pd.read_csv(results/'raw_data_preparation/source_inventory.csv')
    raw_quality=pd.read_csv(results/'raw_data_preparation/market_quality.csv')
    universe=json.loads((results/'raw_data_preparation/universe_audit.json').read_text())
    prepared=saved_table(nb,4,4).iloc[0]
    assert prepared.price_precision_policy=='canonical_total_return_float64_v1'
    table(tabs/'nb_audit.tex',['Input audit','Count'],[
        ['Exchange sessions / security-ID union',f'{int(prepared.calendar_sessions):,} / {int(prepared.SID_union):,}'],
        ['Raw manifest entries (WRDS and Compustat)',len(source_inventory)],
        ['Published-snapshot files hash-verified',int(prepared.source_files_verified)],
        ['Raw CRSP daily records inspected',f'{raw_quality.rows.sum():,}'],
        ['Quarantined output OHLC cells / eligible cells','2 / 1'],
        ['Daily universe membership records',f'{universe["membership_cells"]:,}'],
        ['Finite targets checked against source returns',f'{audit["source_return_label_audit"]["finite_targets_checked"]:,}'],
    ],'lr')
    make_figures(nb,report,results,selection,ev,periods,r2,stats,components,factors,points,at,numeric)
    (report/'Group_34_Report.tex').write_text((report/'notebook_report_template.tex').read_text())
    inventory={'source_repository':'UtAstN/dsa5205-project2-final','source_commit':SOURCE_COMMIT,
        'report_revision_date':'2026-11-12','strategy_notebook_sha256':sha(nb_path),'data_notebook_sha256':sha(data_nb_path),
        'recorded_run':audit['strategy_run'],'data_run':audit['data_run'],
        'model_fits_recorded':396,'portfolio_ledgers_recorded':38,'unit_tests_recorded':audit['unit_tests_passed'],
        'research_rerun_for_report':False,'models_refitted':False,'backtests_replayed':False,'bootstrap_resampled':False,
        'interval_source':'Stored strategy Notebook cell 20 HTML table, displayed precision; 20-session blocks.',
        'wealth_figure_source':'Stored strategy Notebook cell 16 PNG output; no unavailable daily series reconstructed.',
        'recorded_source_hashes_verified':True,'inputs':inputs,
        'figures':sorted(p.name for p in figs.glob('nb_*.pdf')),'tables':sorted(p.name for p in tabs.glob('nb_*.tex')),
        'assets':{str(p.relative_to(report)):sha(p) for p in sorted(report.rglob('nb_*')) if p.is_file()}}
    for name in ['asset_inventory.json','notebook_report_inventory.json']:
        (report/name).write_text(json.dumps(inventory,indent=2)+'\n')
    print(json.dumps({k:v for k,v in inventory.items() if k not in ['inputs','assets']},indent=2))


def make_figures(nb,report,results,selection,ev,periods,r2,stats,components,factors,points,at,numeric):
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'runs/report_runtime/mpl'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    figs=report/'figures'
    def save(name,fig):
        fig.tight_layout(pad=.8);fig.savefig(figs/f'{name}.pdf',bbox_inches='tight');plt.close(fig)
    with plt.rc_context({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':9.5,
        'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':7.5,
        'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.18,'pdf.fonttype':42}):
        counts=pd.read_csv(results/'raw_data_preparation/universe_daily_counts.csv',index_col=0)
        dates=pd.to_datetime(counts.index)
        keep=dates>=pd.Timestamp('2015-01-01')
        fig,ax=plt.subplots(figsize=(7.1,2.4));ax.plot(dates[keep],counts.loc[keep,'Selected'])
        ax.set(ylabel='Eligible securities',title='Daily research universe, 2015-2025');save('nb_universe',fig)
        fig,ax=plt.subplots(figsize=(7.1,2.6));ax.bar(range(6),stats.rankic,color=COLORS['fundamental'])
        ax.set_xticks(range(6),[f'F{i}' for i in range(1,7)]);ax.axhline(0,color='black',lw=.7)
        ax.set(ylabel='Signed RankIC (decimal)',title='Fundamental evidence, 2020-2022');save('nb_fundamentals',fig)
        cv=pd.read_csv(results/'CV_comparison.csv').query('eligible')
        cv=cv.groupby(['model','feature_set']).mean_rankic.max().unstack().reindex(index=['ridge','lasso','xgboost'],columns=['fundamental','combined','engineered'])
        fig,ax=plt.subplots(figsize=(7.1,2.9));cv.plot.bar(ax=ax,color=['#3574af','#2b927a','#ba8060'])
        ax.set(xlabel='',ylabel='Mean daily RankIC');ax.tick_params(axis='x',rotation=0);ax.axhline(0,color='black',lw=.7)
        ax.legend(title='Input family',ncol=3,fontsize=7,title_fontsize=8);save('nb_cv',fig)
        # The only raster figure is the saved wealth path; all other plots use stored aggregates.
        from reportlab.pdfgen.canvas import Canvas
        from reportlab.lib.utils import ImageReader
        png=[o['data']['image/png'] for o in nb['cells'][16]['outputs'] if 'image/png' in o.get('data',{})][0]
        raw=base64.b64decode(png);reader=ImageReader(BytesIO(raw));w,h=reader.getSize();width=7.1*72;height=width*h/w
        canvas=Canvas(str(figs/'nb_wealth.pdf'),pagesize=(width,height));canvas.drawImage(reader,0,0,width,height);canvas.save()
        period_order=['2023 (Validation)','2024','2025','2024-2025'];short=['2023\nvalidation','2024','2025','2024-25']
        names={**MODELS,'SPY':'SPY'};cols={names[k]:v for k,v in COLORS.items()}
        fig,grid=plt.subplots(2,2,figsize=(7.1,5.5));axes=grid.ravel()
        r2.assign(label=r2.model.map(names)).pivot(index='period',columns='label',values='oos_r2').reindex(period_order).plot.bar(ax=axes[0],color=cols,width=.76)
        axes[0].set_title('Prediction OOS $R^2$')
        reps={case:MODELS[m] for m,case in selection['selected_by_model'].items() if case};reps['SPY']='SPY'
        data=periods.assign(label=periods.case.map(reps))
        for metric,title,ax in zip(['annual_return','sharpe','max_drawdown'],['Net CAGR','Net Sharpe','Max drawdown'],axes[1:]):
            data.pivot(index='period',columns='label',values=metric).reindex(period_order).plot.bar(ax=ax,color=cols,width=.76);ax.set_title(title)
        for ax in axes:
            ax.set_xlabel('');ax.axhline(0,color='black',lw=.6);ax.set_xticklabels(short,rotation=0,fontsize=7.5);ax.legend(ncol=2,fontsize=6.8)
        axes[1].yaxis.set_major_formatter(PercentFormatter(1));axes[3].yaxis.set_major_formatter(PercentFormatter(1))
        axes[2].get_legend().remove();axes[3].get_legend().remove();save('nb_periods',fig)
        fig,axes=plt.subplots(1,2,figsize=(7.1,3.1))
        axes[0].plot([10,20,30],[ev.loc[c,'annual_return'] for c in ['fundamental_cap','cost_20','cost_30']],marker='o')
        axes[0].axhline(ev.loc['fundamental_cap','spy_annual_return'],color='#555555',ls='--',label='SPY')
        axes[0].set(title='Selected: cost sensitivity',xlabel='bps per traded dollar',ylabel='Net CAGR');axes[0].set_xticks([10,20,30]);axes[0].legend()
        cases=['fundamental_cap']+[f'without_F0{i}' for i in range(1,7)]
        axes[1].bar(range(7),ev.loc[cases,'annual_return'],color=['#17365d']+['#3c8d95']*6)
        axes[1].set_xticks(range(7),['Six']+[f'-F{i}' for i in range(1,7)]);axes[1].set_title('Selected: factor ablation')
        for ax in axes:ax.yaxis.set_major_formatter(PercentFormatter(1))
        save('nb_stress',fig)
        fig,ax=plt.subplots(figsize=(7.1,2.9));start=0
        for i,key in enumerate(['selection','weighting','universe']):
            value=points[key]*100;ax.bar(i,value,bottom=start,color='#387f83' if value>=0 else '#b85c52')
            ax.text(i,start+value/2,f'{value:+.2f}',ha='center',va='center',fontsize=9);start+=value
        ax.bar(3,start,color='#17365d');ax.text(3,start/2,f'{start:+.2f}',ha='center',va='center',color='white')
        ax.set_xticks(range(4),['Selection','Weighting','Universe','Total']);ax.set(ylabel='CAGR difference (pp)',title='Selected capitalization portfolio versus SPY, 2024-2025')
        save('nb_reference_attribution',fig)
        styles=factors.loc[STYLES];labels={'ResidualVolatility':'Residual vol.','NonLinearSize':'Nonlinear size','BookToPrice':'Book-to-price','EarningsYield':'Earnings yield'}
        fig,axes=plt.subplots(1,2,figsize=(7.1,3.4))
        for col,title,ax in zip(['mean_exposure','linked_contribution'],['Mean style exposure','Linked style contribution'],axes):
            values=styles[col].sort_values()
            if col=='linked_contribution':values=values.mul(100)
            ax.barh([labels.get(k,k) for k in values.index],values,color='#3574af' if col=='mean_exposure' else '#2b927a')
            ax.axvline(0,color='black',lw=.7);ax.set_title(title);ax.tick_params(axis='y',labelsize=8)
        axes[0].set_xlabel('Standardized model units');axes[1].set_xlabel('Linked contribution (pp)');save('nb_styles',fig)
        fig,ax=plt.subplots(figsize=(7.1,2.0));low=numeric(at.loc['total','ci_low']);high=numeric(at.loc['total','ci_high']);point=100*points['total']
        ax.plot([low,high],[0,0],color='#387f83',lw=2);ax.scatter(point,0,color='#17365d',s=35)
        ax.axvline(0,color='black',ls='--',lw=.8);ax.set_yticks([0],['Selected minus SPY'])
        ax.set_ylim(-.6,.6);ax.set_xlabel('CAGR difference (pp); 20-session paired 95% interval')
        ax.annotate(f'{point:+.2f} pp',xy=(point,0),xytext=(point,.18),ha='center',fontsize=9)
        save('nb_selected_intervals',fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-dir',type=Path,default=ROOT/'reports/final')
    build(parser.parse_args().report_dir)
