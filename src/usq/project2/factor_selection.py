"""Recompute the WQ development screen without loading historical experiments."""
from pathlib import Path
import gc
import json
import numpy as np
import pandas as pd
from .factors import WQ_IDS, price_inputs, rank, wq
from .screen import POLICY as BASE_POLICY, information_series
from .protocol import validate_target
from .study_periods import (TRAIN_START, WQ_DIRECTION_END, WQ_SCREEN_START,
                            WQ_SCREEN_END, VALIDATION_YEAR, PORTFOLIO_PHASES)

POLICY = {**BASE_POLICY,
    'direction_end': WQ_DIRECTION_END,
    'selection_start': WQ_SCREEN_START,
    'selection_end': WQ_SCREEN_END,
    'validation_year': VALIDATION_YEAR,
    'test_start': PORTFOLIO_PHASES['evaluation'][0],
    'test_end': PORTFOLIO_PHASES['evaluation'][1],
    'direction_rule': 'sign of mean 2015–2017 daily RankIC on mature labels',
    'selection_order': '2018–2019 signed RankIC descending, then correlation filter',
    'temporal_caveat': 'latest downloaded financial statements, not strict historical PIT',
}


def select_candidates(summary, ranked):
    """Greedy correlation filter, after the declared validity thresholds."""
    candidates=summary.loc[summary.passes_predeclared_rules].sort_values(
        ['stability_rankic_signed','factor'],ascending=[False,True])
    selected=[];accepted_ranks={};decisions=[]
    for row in candidates.itertuples():
        x=ranked(row.factor,row.direction)
        correlations={name:float(x.corrwith(other,axis=1).abs().mean())
                      for name,other in accepted_ranks.items()}
        highest=max(correlations.values(),default=0.)
        accepted=np.isfinite(highest) and highest<POLICY['max_abs_factor_corr'] and len(selected)<POLICY['target_wq_count']
        decisions.append({'factor':row.factor,'accepted':bool(accepted),
                          'max_abs_daily_rank_corr':highest,'correlations':correlations})
        if accepted:selected.append(row.factor);accepted_ranks[row.factor]=x
    return selected,decisions


def screen_wq(database,output):
    root=Path(database);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    (output/'screen_protocol.json').write_text(json.dumps(POLICY,indent=2))
    u=pd.read_pickle(root/'panels/universe.pkl')
    def load(name):
        f=pd.read_pickle(root/'panels'/f'{name}.pkl')
        if not f.index.equals(u.index) or not f.columns.equals(u.columns):raise ValueError('Unaligned input')
        return f
    ends=validate_target(root,u)
    dates=u.index[(u.index>=TRAIN_START)&(u.index<=WQ_SCREEN_END)]
    y=load('ylabel').reindex(dates).where(ends.reindex(dates).le(WQ_SCREEN_END),axis=0)
    universe=u.reindex(dates);prices=price_inputs(load,u)
    rows=[];annual=[]
    for number in WQ_IDS:
        name=f'WQ{number:03d}'
        f=wq(number,prices).replace([np.inf,-np.inf],np.nan).astype('float32')
        stat=information_series(f.reindex(dates),y,universe)
        stat.loc[(stat.index<=WQ_DIRECTION_END)&ends.reindex(stat.index).gt(WQ_DIRECTION_END),['rankic','ic']]=np.nan
        direction=1 if stat.loc[:WQ_DIRECTION_END,'rankic'].mean()>=0 else -1
        stable=stat.loc[WQ_SCREEN_START:]
        yearly=stable.groupby(stable.index.str[:4]).rankic.mean()*direction
        score=float(stable.rankic.mean()*direction);coverage=float(stable.coverage.mean())
        positive=int(yearly.gt(0).sum())
        rows.append({'factor':name,'direction':direction,'early_rankic':float(stat.loc[:WQ_DIRECTION_END,'rankic'].mean()),
            'stability_rankic_signed':score,'stability_coverage':coverage,'positive_stability_years':positive,
            'passes_predeclared_rules':score>=POLICY['min_stability_rankic'] and coverage>=POLICY['min_coverage'] and positive>=POLICY['min_positive_stability_years']})
        annual.extend({'factor':name,'year':year,'signed_rankic':float(value)} for year,value in yearly.items())
        pd.DataFrame(rows).to_csv(output/'WQ_screen.csv',index=False)
        print(json.dumps({'stage':'WQ_screen','factor':name,'passed':bool(rows[-1]['passes_predeclared_rules'])}),flush=True)
        del f,stat;gc.collect()
    summary=pd.DataFrame(rows)
    sample=universe.loc[WQ_SCREEN_START:].iloc[::5]
    def ranked(name,direction):
        return rank(wq(int(name[2:]),prices).astype('float32').reindex(sample.index).where(sample.eq(1)))*direction
    selected,decisions=select_candidates(summary,ranked)
    result={'wq':selected,'directions':dict(zip(summary.factor,summary.direction.astype(int))),
            'decisions':decisions,'selection_period':'2018-2019','direction_period':'2015-2017',
            'candidate_count':len(WQ_IDS),'development_reused_for_ML_tuning':False}
    if not selected:raise ValueError('No WQ factors passed the declared screen')
    (output/'WQ_selection.json').write_text(json.dumps(result,indent=2))
    pd.DataFrame(annual).to_csv(output/'WQ_annual.csv',index=False)
    return summary,result
