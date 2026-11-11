"""Descriptive completion of the team plan; never reselect the frozen v2 main run."""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.linalg import lstsq
from threadpoolctl import threadpool_limits

from usq.backtest.metrics import performance
from usq.utils import json_dump
from .factors import rank
from .ledger import Market, stable_tie_ranks
from .protocol import bind_inputs
from .screen import information_series
from .supplement import paired_bootstrap

PHASES = [('early','20150101','20181231'), ('stability','20190101','20211231'),
          ('validation','20220101','20221231'), ('evaluation','20230101','20251231')]
TRADING_PHASES = PHASES[1:]


def extension_policy(baseline):
    return {
        'status':'descriptive extensions on already inspected historical sample; no main-model reselection',
        'baseline':str(baseline),'seed':5205,'target':'close_adj[T+11]/close_adj[T+1]-1','phases':[list(p) for p in PHASES],
        'deciles':{'phases':['stability','validation','evaluation'],'variants':['raw','raw_common','neutral_common'],
                   'grouping':'equal counts with fixed SHA256 SID ties; constant scores imply cash','cost_bps':10,
                   'rule':'frozen main smooth/rebalance','eligibility':'signal-date universe and at least one selected WQ observed, never future labels'},
        'neutralization':{'input':'fixed-direction contemporaneous percentile minus .5','controls':'full known SIC2 dummies + centered log positive signal-date cap',
                          'fit':'daily OLS, no outcome, >=30 pairs and n>rank+2','comparison':'raw percentile and residual on identical known-control observations','numerical_zero_tolerance':1e-10},
        'correlation':'pairwise valid daily cross-sectional Spearman; equal-date mean and pair counts; no labels',
        'pure_C':'same main eligibility and mature targets, six fundamental values+missing indicators only; family parameters frozen from A',
        'bootstrap_blocks':[10,20,40,60],'bootstrap_reps':2000,'event_haircuts':[0,.5,1],
        'output_preservation':'main baseline and raw data untouched'}


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def progress(stage, **data):
    print(json.dumps({'stage':stage,**data},ensure_ascii=False),flush=True)


def calendar_hac_t(series, lag=10):
    """NW variance of an available-date mean, preserving missing calendar slots."""
    x=np.asarray(series,dtype=float);ok=np.isfinite(x);n=int(ok.sum())
    if n<max(30,lag+2):return np.nan
    mean=x[ok].mean();z=np.where(ok,x-mean,0.)
    variance=float(z@z)
    for k in range(1,min(lag,len(z)-1)+1):
        variance+=2*(1-k/(lag+1))*float(z[k:]@z[:-k])
    return float(mean/np.sqrt(variance/n**2)) if variance>0 else np.nan


def summarize_information(daily):
    result={'ic_days':int(daily.ic.notna().sum()),'rankic_days':int(daily.rankic.notna().sum()),
            'ic':float(daily.ic.mean()),'rankic':float(daily.rankic.mean()),
            'mean_pairs':float(daily.pairs.mean()),'mean_coverage':float(daily.coverage.mean())}
    for key in ('ic','rankic'):
        s=daily[key];sd=s.std(ddof=1)
        result[key+'ir_daily']=float(s.mean()/sd) if np.isfinite(sd) and sd>0 else np.nan
        result[key+'_hac_t_lag10']=calendar_hac_t(s)
    return result


def neutralize_scores(factor, cap, industry, universe, min_pairs=30):
    """OLS of signed eligible percentile on same-date SIC2 dummies + log cap.

    Only valid signal-date controls and factor values enter each regression.
    There is no label input. Full industry dummies span an intercept.
    """
    for x in (cap,industry,universe):
        if not factor.index.equals(x.index) or not factor.columns.equals(x.columns):
            raise ValueError('Full neutralization axes differ')
    output=np.full(factor.shape,np.nan,dtype='float32');metadata=[]
    f=factor.to_numpy();c=cap.to_numpy();u=universe.to_numpy()==1;ind=industry.to_numpy(dtype=object)
    with threadpool_limits(limits=2):
        for i,day in enumerate(factor.index):
            valid=u[i]&np.isfinite(f[i])&np.isfinite(c[i])&(c[i]>0)&pd.notna(ind[i])
            ids=np.flatnonzero(valid);record={'date':day,'common_pairs':len(ids),'design_rank':0,'residual_std':np.nan,'max_abs_orthogonality':np.nan,'usable':False}
            if len(ids)<min_pairs:metadata.append(record);continue
            score=pd.Series(f[i,ids]).rank(pct=True,method='average').to_numpy()-.5
            code,categories=pd.factorize(ind[i,ids],sort=True)
            X=np.zeros((len(ids),len(categories)+1));X[np.arange(len(ids)),code]=1.
            size=np.log(c[i,ids].astype(float));scale=size.std()
            X[:,-1]=(size-size.mean())/scale if scale>1e-12 else 0.
            beta,_,rnk,_=lstsq(X,score,lapack_driver='gelsy')
            residual=score-X@beta;std=float(residual.std())
            record.update(design_rank=int(rnk),residual_std=std,max_abs_orthogonality=float(np.abs(X.T@residual).max()))
            if len(ids)>rnk+2 and std>=1e-10:
                output[i,ids]=residual.astype('float32');record['usable']=True
            metadata.append(record)
    return pd.DataFrame(output,index=factor.index,columns=factor.columns),pd.DataFrame(metadata)


class CachedMarket(Market):
    """Private bounded cache for immutable score objects in this orchestrator."""
    def _prepare_signal(self,scores,smooth):
        previous=getattr(self,'_prepared_cache',None)
        if previous is not None and previous[0] is scores and previous[1]==smooth:
            return previous[2]
        result=super()._prepare_signal(scores,smooth)
        self._prepared_cache=(scores,smooth,result)
        return result


class Context:
    def __init__(self,database,baseline,output):
        self.db,self.base,self.out=map(Path,(database,baseline,output))
        if self.out.resolve()==self.base.resolve() or self.base.resolve() in self.out.resolve().parents:
            raise ValueError('Extensions must use a separate output directory')
        self.out.mkdir(exist_ok=True,parents=True)
        policy=extension_policy(self.base);protocol=self.out/'extension_protocol.json'
        if protocol.exists():
            if json.loads(protocol.read_text())!=policy:raise ValueError('Different extension protocol; use a fresh output directory')
        else:json_dump(policy,protocol)
        self.u=pd.read_pickle(self.db/'panels/universe.pkl')
        self.ends=bind_inputs(self.db,self.base,self.u)
        self.y=pd.read_pickle(self.db/'panels/ylabel.pkl')
        self.selection=json.loads((self.base/'factor_selection.json').read_text())
        self.names=self.selection['wq']+self.selection['fundamental']
        self.rule=json.loads((self.base/'frozen_model_choices.json').read_text())['trading']
        self.observed=np.load(self.base/'design/observed.npy',mmap_mode='r')
        self.allowed=self.u.eq(1)&pd.DataFrame(self.observed,index=self.u.index,columns=self.u.columns)
        self._factors={}
        frozen={name:digest(self.base/name) for name in ['factor_selection.json','model_protocol.json','frozen_model_choices.json','input_signature.json','backtests/summary.csv']}
        path=self.out/'baseline_binding.json'
        if path.exists() and json.loads(path.read_text())!=frozen:raise ValueError('Frozen baseline changed')
        json_dump(frozen,path)
        for name in ['statistics','neutral','correlation','deciles','lasso','comparisons','pure_C','robustness','audit']:
            (self.out/name).mkdir(exist_ok=True)

    def factor(self,name):
        if name not in self._factors:
            f=pd.read_pickle(self.base/'factors'/f'{name}.pkl')
            self._factors[name]=f*self.selection['directions'][name]
        return self._factors[name]

    def information(self,factor):
        return information_series(factor,self.y,self.u)

    def phase_statistics(self,factor,variant,daily):
        rows=[]
        for phase,lo,hi in PHASES:
            part=daily.loc[lo:hi].copy();end=self.ends.reindex(part.index)
            bad=end.isna()|end.gt(hi)
            part.loc[bad,['ic','rankic','coverage']]=np.nan
            part.loc[bad,'pairs']=np.nan
            rows.append({'factor':factor,'variant':variant,'phase':phase,**summarize_information(part)})
        return rows


def statistics(ctx):
    cap=pd.read_pickle(ctx.db/'panels/market_cap.pkl')
    industry=pd.read_pickle(ctx.db/'panels/industry.pkl').astype(object)
    rows=[];yearrows=[];neutralmeta=[]
    for name in ctx.names:
        f=ctx.factor(name);daily=ctx.information(f)
        daily.to_parquet(ctx.out/'statistics'/f'{name}_raw_daily.parquet')
        rows.extend(ctx.phase_statistics(name,'raw',daily))
        for year,part in daily.groupby(daily.index.str[:4]):
            end=ctx.ends.reindex(part.index);part=part.copy();bad=end.isna()|end.gt(year+'1231')
            part.loc[bad,['ic','rankic','coverage','pairs']]=np.nan
            yearrows.append({'factor':name,'year':year,**summarize_information(part)})
        path=ctx.out/'neutral'/f'{name}.pkl'
        if path.exists():
            n=pd.read_pickle(path);meta=pd.read_csv(ctx.out/'neutral'/f'{name}_fit.csv')
        else:
            n,meta=neutralize_scores(f,cap,industry,ctx.u)
            n.to_pickle(path);meta.to_csv(ctx.out/'neutral'/f'{name}_fit.csv',index=False)
        neutralmeta.append({'factor':name,'usable_days':int(meta.usable.sum()),'max_abs_orthogonality':float(meta.max_abs_orthogonality.max())})
        common=f.where(n.notna());rawscore=rank(common)-.5
        for label,x in [('raw_common',rawscore),('neutral_common',n)]:
            d=ctx.information(x);d.to_parquet(ctx.out/'statistics'/f'{name}_{label}_daily.parquet')
            rows.extend(ctx.phase_statistics(name,label,d))
        progress('statistics_complete',factor=name)
    pd.DataFrame(rows).to_csv(ctx.out/'statistics/factor_summary.csv',index=False)
    pd.DataFrame(yearrows).to_csv(ctx.out/'statistics/factor_annual.csv',index=False)
    pd.DataFrame(neutralmeta).to_csv(ctx.out/'neutral/fit_summary.csv',index=False)


def correlations(ctx):
    records=[];dailyrows=[]
    for i,a in enumerate(ctx.names):
        for b in ctx.names[i:]:
            s=information_series(ctx.factor(a),ctx.factor(b),ctx.u)
            for phase,lo,hi in PHASES:
                p=s.loc[lo:hi]
                records.append({'factor_a':a,'factor_b':b,'phase':phase,'spearman':float(p.rankic.mean()),'mean_abs_spearman':float(p.rankic.abs().mean()),'mean_pairs':float(p.pairs.mean()),'defined_days':int(p.rankic.notna().sum())})
            tmp=s[['rankic','pairs']].loc['20150101':'20251231'].reset_index();tmp.columns=['date','rankic','pairs'];tmp['factor_a']=a;tmp['factor_b']=b;dailyrows.append(tmp)
        progress('correlation_row_complete',factor=a)
    result=pd.DataFrame(records);result.to_csv(ctx.out/'correlation/pairs.csv',index=False)
    pd.concat(dailyrows,ignore_index=True).to_parquet(ctx.out/'correlation/daily_pairs.parquet',index=False)
    for phase,_,_ in PHASES:
        p=result.loc[result.phase.eq(phase)]
        matrix=pd.DataFrame(np.nan,index=ctx.names,columns=ctx.names);counts=matrix.copy()
        for r in p.itertuples():
            matrix.loc[r.factor_a,r.factor_b]=matrix.loc[r.factor_b,r.factor_a]=r.spearman
            counts.loc[r.factor_a,r.factor_b]=counts.loc[r.factor_b,r.factor_a]=r.mean_pairs
        matrix.to_csv(ctx.out/'correlation'/f'{phase}_spearman.csv')
        counts.to_csv(ctx.out/'correlation'/f'{phase}_mean_pairs.csv')


def save_portfolio(ctx,market,score,folder,lo,hi,**kwargs):
    folder.mkdir(exist_ok=True,parents=True)
    daily,w,events,summary=market.backtest(score,lo,hi,weights=True,**ctx.rule,**kwargs)
    daily.to_csv(folder/'daily.csv');events.to_csv(folder/'events.csv',index=False)
    months=w.groupby(w.index.str[:6]).tail(1)
    v=months.to_numpy();ii,jj=np.where(v>0)
    pd.DataFrame({'date':months.index.to_numpy()[ii],'sid':months.columns.to_numpy()[jj],'weight':v[ii,jj]}).to_parquet(folder/'monthly_holdings.parquet',index=False)
    summary.update(gross_annual_return=performance(daily.gross_return)['annual_return'],sum_daily_costs=float(daily.cost.sum()),mean_cash=float(daily.cash_weight.mean()),max_weight=float(w.to_numpy().max()),mean_positions=float((w>0).sum(axis=1).mean()))
    json_dump(summary,folder/'summary.json')
    np.testing.assert_allclose(daily.net_return,daily.gross_return-daily.cost,atol=1e-12)
    np.testing.assert_allclose(daily.cost,daily.traded_notional*.001,atol=1e-12)
    return daily,summary


def deciles(ctx):
    market=CachedMarket(ctx.db);rows=[];monotonic=[]
    for name in ctx.names:
        raw=ctx.factor(name);n=pd.read_pickle(ctx.out/'neutral'/f'{name}.pkl')
        variants={'raw':raw.where(ctx.allowed),'raw_common':raw.where(ctx.allowed&n.notna()),'neutral_common':n.where(ctx.allowed)}
        for variant,score in variants.items():
            for phase,lo,hi in TRADING_PHASES:
                returns={};summaries=[]
                for group in range(1,11):
                    folder=ctx.out/'deciles'/name/variant/phase/f'G{group:02d}'
                    if (folder/'summary.json').exists():
                        summary=json.loads((folder/'summary.json').read_text());d=pd.read_csv(folder/'daily.csv',dtype={'date':str}).set_index('date')
                    else:
                        d,summary=save_portfolio(ctx,market,score,folder,lo,hi,group=group,groups=10,grouping='equal_count')
                    rows.append({'factor':name,'variant':variant,'phase':phase,'group':group,**summary})
                    summaries.append(summary);returns[group]=d.net_return
                r=pd.DataFrame(returns);r.to_csv(ctx.out/'deciles'/name/variant/phase/'group_returns.csv')
                delta=r[10]-r[1];pd.DataFrame({'G10_minus_G1_daily_diagnostic':delta}).to_csv(ctx.out/'deciles'/name/variant/phase/'high_low_diagnostic.csv')
                annual=pd.Series([s['annual_return'] for s in summaries],index=range(1,11))
                monotonic.append({'factor':name,'variant':variant,'phase':phase,'group_cagr_spearman':annual.corr(pd.Series(range(1,11),index=annual.index),method='spearman'),'adjacent_non_decreasing_fraction':float((annual.diff().dropna()>=0).mean()),'G10_minus_G1_arithmetic_annualized':float(delta.mean()*252),'G10_minus_G1_hac_t_lag10':calendar_hac_t(delta),'basis':'difference of two funded long-only sleeves; not an implemented short portfolio'})
            progress('decile_variant_complete',factor=name,variant=variant)
        # Release 100MB prepared score cache before the next factor.
        market._prepared_cache=None
    pd.DataFrame(rows).to_csv(ctx.out/'deciles/summary.csv',index=False)
    pd.DataFrame(monotonic).to_csv(ctx.out/'deciles/monotonicity.csv',index=False)


def matched_ratios(ctx):
    market=CachedMarket(ctx.db);rows=[]
    for name in ctx.selection['fundamental'][:4]:
        other=name.replace('reconstructed','ratio').replace('F0','C0')
        a,b=ctx.factor(name),ctx.factor(other)
        mask=ctx.allowed&np.isfinite(a)&np.isfinite(b)
        for factor,score in [(name,a.where(mask)),(other,b.where(mask))]:
            for phase,lo,hi in TRADING_PHASES:
                d,s=save_portfolio(ctx,market,score,ctx.out/'comparisons/ratios'/factor/phase,lo,hi)
                rows.append({'pair':name,'factor':factor,'phase':phase,**s})
        progress('ratio_pair_complete',factor=name)
    pd.DataFrame(rows).to_csv(ctx.out/'comparisons/ratio_portfolios.csv',index=False)


# Further stage functions are defined below; all share this immutable baseline.

def fixed_fundamental_models(ctx):
    import joblib
    from .models import Design,fit_model,halves
    from usq.barra.model import load_model
    from .risk import design_matrix,carino_link
    design=Design(ctx.db,ctx.base);offset=2*design.nq;nf=design.x.shape[-1]-offset
    choices=json.loads((ctx.base/'frozen_model_choices.json').read_text())['hyperparameters']
    history=[];summaries=[];effects=[];market=CachedMarket(ctx.db)
    root=ctx.out/'pure_C'
    for sub in ['models','predictions','portfolios','attribution']:(root/sub).mkdir(exist_ok=True)
    for kind in ['ridge','lasso','xgboost']:
        pieces=[]
        for year in [2023,2024,2025]:
            for half,start,end in halves(year):
                predpath=root/'predictions'/f'{kind}_{half}.pkl';metapath=predpath.with_suffix('.json')
                if predpath.exists() and metapath.exists():
                    score=pd.read_pickle(predpath);meta=json.loads(metapath.read_text());model=joblib.load(root/'models'/f'{kind}_{half}.joblib')
                else:
                    maskdates=(design.index>='20150101')&(design.index<start)&design.end.lt(start).fillna(False).to_numpy()
                    valid=maskdates[:,None]&design.u.eq(1).to_numpy()&design.observed&np.isfinite(design.y)
                    ids=np.flatnonzero(valid)
                    x=np.asarray(design.x[:,:,offset:].reshape(-1,nf)[ids],dtype='float32',order='C')
                    y=np.asarray(design.y.ravel()[ids],dtype='float32')
                    model=fit_model(kind,choices[kind]['params'],x,y)
                    dates=np.flatnonzero((design.index>=start)&(design.index<=end));pred=[]
                    for i in dates:
                        v=model.predict(np.asarray(design.x[i,:,offset:],dtype='float32',order='C'))
                        pred.append(np.where(design.u.iloc[i].eq(1).to_numpy()&design.observed[i],v,np.nan).astype('float32'))
                    score=pd.DataFrame(pred,index=design.index[dates],columns=design.columns)
                    used=design.index[valid.any(axis=1)]
                    meta={'model':kind,'family':'C','half':half,'samples':len(y),'feature_count':nf,'train_first':used[0],'train_last_signal':used[-1],'train_last_label_end':str(design.end.reindex(used).max()),'forecast_start':start,'params':choices[kind]['params'],'no_C_retuning':True}
                    assert meta['train_last_label_end']<start
                    score.to_pickle(predpath);json_dump(meta,metapath);joblib.dump(model,root/'models'/f'{kind}_{half}.joblib',compress=3)
                    del x,y;gc.collect()
                history.append(meta);pieces.append(score)
                vals=model.feature_importances_ if hasattr(model,'feature_importances_') else model.coef_
                for k,name in enumerate(ctx.selection['fundamental']):effects.append({'model':kind,'half':half,'factor':name,'value_effect':float(vals[2*k]),'missing_effect':float(vals[2*k+1])})
                progress('pure_C_fit',model=kind,half=half,samples=meta['samples'])
        score=pd.concat(pieces);score.to_pickle(root/'predictions'/f'{kind}_2023_2025.pkl')
        daily,w,events,summary=market.backtest(score,'20230101','20251231',weights=True,**ctx.rule)
        daily.to_csv(root/'portfolios'/f'{kind}_daily.csv');w.to_pickle(root/'portfolios'/f'{kind}_weights.pkl');events.to_csv(root/'portfolios'/f'{kind}_events.csv',index=False)
        ic=information_series(score,ctx.y.reindex(score.index),ctx.u.reindex(score.index))
        ic.to_parquet(root/'predictions'/f'{kind}_information.parquet')
        summaries.append({'strategy':kind+'_C',**summary,**summarize_information(ic)})
    pd.DataFrame(summaries).to_csv(root/'summary.csv',index=False);pd.DataFrame(effects).to_csv(root/'feature_effects.csv',index=False);json_dump(history,root/'training_history.json')
    risk=load_model(ctx.base/'barra_style/model');attrs=[]
    industries=[n for n in risk.factor_returns if n not in risk.styles and n!='Market']
    for kind in ['ridge','lasso','xgboost']:
        w=pd.read_pickle(root/'portfolios'/f'{kind}_weights.pkl');daily=pd.read_csv(root/'portfolios'/f'{kind}_daily.csv',dtype={'date':str}).set_index('date');parts=[];expos=[]
        for day in w.index:
            held=w.loc[day];held=held[held!=0]
            exp=risk.exposures.loc[risk.alignment.loc[day,'exposure_date']].reindex(held.index)
            b=design_matrix(exp,risk.styles,industries).T@held
            c=b*risk.factor_returns.loc[day].reindex(b.index).fillna(0)
            c['Specific_or_unexplained']=daily.loc[day,'gross_return']-c.sum();c['Costs']=-daily.loc[day,'cost'];c['SPY_benchmark_subtraction']=-daily.loc[day,'benchmark_return']
            parts.append(c.rename(day));expos.append(b.rename(day))
        c=pd.DataFrame(parts).fillna(0);e=pd.DataFrame(expos).fillna(0)
        err=float((c.sum(axis=1)-(daily.net_return-daily.benchmark_return)).abs().max());assert err<1e-9
        linked=carino_link(c,daily.net_return,daily.benchmark_return)
        c.to_csv(root/'attribution'/f'{kind}_daily.csv');e.to_csv(root/'attribution'/f'{kind}_exposures.csv');linked.to_csv(root/'attribution'/f'{kind}_linked.csv')
        attrs.append({'strategy':kind+'_C','Market':float(linked.get('Market',0)),'Industry':float(linked.reindex(industries).fillna(0).sum()),'Style':float(linked.reindex(risk.styles).fillna(0).sum()),'Specific_or_unexplained':float(linked['Specific_or_unexplained']),'Costs':float(linked['Costs']),'max_daily_error':err})
    pd.DataFrame(attrs).to_csv(root/'attribution/summary.csv',index=False)


def lasso_diagnostics(ctx):
    import joblib
    from .models import Design,fit_model,halves
    design=Design(ctx.db,ctx.base);nq=design.nq;records=[];coefrows=[];manualpieces=[]
    choice=json.loads((ctx.base/'frozen_model_choices.json').read_text())['hyperparameters']['lasso']['params']
    for half,start,end in halves(2022):
        x,y,meta=design.training(start,'A');model=fit_model('lasso',choice,x,y)
        score=design.predict(model,start,end,'A')
        records.append({'model':'lasso_A_validation','half':half,'nonzero_values':int(np.count_nonzero(model.coef_[::2])),'nonzero_missing':int(np.count_nonzero(model.coef_[1::2])),'constant_days':int(score.nunique(axis=1).le(1).sum()),'median_unique':float(score.nunique(axis=1).median()),'days':len(score)})
        for k,name in enumerate(ctx.selection['wq']):coefrows.append({'model':'lasso_A_validation','half':half,'factor':name,'value_coef':float(model.coef_[2*k]),'missing_coef':float(model.coef_[2*k+1])})
        del x,y;gc.collect()
    active=set();all_equivalent=True
    for path in sorted((ctx.base/'models').glob('lasso_*.joblib')):
        model=joblib.load(path);parts=path.stem.split('_');family,half=parts[1],parts[2]
        names=ctx.selection['wq'] if family=='A' else ctx.names
        score=pd.read_pickle(ctx.base/'predictions'/f'{path.stem}.pkl');levels=score.nunique(axis=1,dropna=True)
        d=pd.read_csv(ctx.base/'backtests'/f'lasso_{family}_daily.csv',dtype={'date':str}).set_index('date').reindex(score.index)
        records.append({'model':'lasso_'+family,'half':half,'nonzero_values':int(np.count_nonzero(model.coef_[::2])),'nonzero_missing':int(np.count_nonzero(model.coef_[1::2])),'constant_days':int(levels.le(1).sum()),'median_unique':float(levels.median()),'days':len(score),'mean_cash':float(d.cash_weight.mean()),'mostly_cash_days':int(d.cash_weight.gt(.99).sum())})
        for k,name in enumerate(names):
            coefrows.append({'model':'lasso_'+family,'half':half,'factor':name,'value_coef':float(model.coef_[2*k]),'missing_coef':float(model.coef_[2*k+1])})
            if model.coef_[2*k+1]!=0:active.add(name)
        if family=='A':
            dates=ctx.u.index.get_indexer(score.index)
            if np.count_nonzero(model.coef_[::2])==0:
                x=np.asarray(design.x[dates,:,:2*nq])
                manual=(x[:,:,1::2]@model.coef_[1::2]+model.intercept_).astype('float32')
                manual=np.where(ctx.allowed.reindex(score.index).to_numpy(),manual,np.nan)
                np.testing.assert_allclose(manual,score.to_numpy(),atol=1e-7,equal_nan=True)
                manualpieces.append(pd.DataFrame(manual,index=score.index,columns=score.columns))
            else:all_equivalent=False
    pd.DataFrame(records).to_csv(ctx.out/'lasso/diagnostics.csv',index=False);pd.DataFrame(coefrows).to_csv(ctx.out/'lasso/coefficients.csv',index=False)
    if all_equivalent:
        score=pd.concat(manualpieces).sort_index();market=CachedMarket(ctx.db)
        d,_,_,s=market.backtest(score,'20230101','20251231',**ctx.rule)
        reference=pd.read_csv(ctx.base/'backtests/lasso_A_daily.csv',dtype={'date':str}).set_index('date')
        np.testing.assert_allclose(d.net_return,reference.net_return,atol=1e-12)
        d.to_csv(ctx.out/'lasso/missingness_only_daily.csv');json_dump({'algebraic_equivalence':True,'max_daily_error':float((d.net_return-reference.net_return).abs().max()),**s},ctx.out/'lasso/missingness_only_summary.json')
    cp=pd.read_pickle(ctx.db/'panels/close_adj.pkl');cap=pd.read_pickle(ctx.db/'panels/market_cap.pkl');ind=pd.read_pickle(ctx.db/'panels/industry.pkl').astype(object)
    quoted=np.isfinite(cp.to_numpy())&(cp.to_numpy()>0);anyquote=quoted.any(axis=0);first=np.argmax(quoted,axis=0);first=np.where(anyquote,first,len(cp)+1)
    quote_age=np.arange(len(cp))[:,None]-first[None,:];quote_age=np.where(quote_age>=0,quote_age,np.nan)
    eligible=ctx.u.eq(1).to_numpy();streak=np.zeros(ctx.u.shape,dtype='int32');running=np.zeros(ctx.u.shape[1],dtype='int32')
    for i in range(len(ctx.u)):
        running=np.where(eligible[i],running+1,0);streak[i]=running
    profiles=[];industries=[]
    for name in sorted(active):
        k=ctx.names.index(name);missing=np.asarray(design.x[:,:,2*k+1])==1
        for state in [False,True]:
            for phase,lo,hi in PHASES:
                dates=np.flatnonzero((ctx.u.index>=lo)&(ctx.u.index<=hi));counts=[];ages=[];caps=[];streaks=[];left=[];sector_counts={};total=0
                for i in dates:
                    mask=ctx.allowed.iloc[i].to_numpy()&(missing[i]==state);n=int(mask.sum())
                    if not n:continue
                    counts.append(n);ages.append(float(np.nanmean(quote_age[i,mask])));streaks.append(float(streak[i,mask].mean()));left.append(float((first[mask]==0).mean()))
                    v=cap.iloc[i].to_numpy()[mask];v=v[np.isfinite(v)&(v>0)];caps.append(float(np.log(v).mean()) if len(v) else np.nan)
                    sector=ind.iloc[i,mask].fillna('Unknown').value_counts()
                    for key,nobs in sector.items():sector_counts[str(key)]=sector_counts.get(str(key),0)+int(nobs)
                    total+=n
                profiles.append({'factor':name,'missing':state,'phase':phase,'days_with_group':len(counts),'mean_group_size':np.mean(counts) if counts else np.nan,'mean_log_cap':np.nanmean(caps) if caps else np.nan,'mean_observed_quote_age_sessions':np.nanmean(ages) if ages else np.nan,'mean_consecutive_eligible_sessions':np.mean(streaks) if streaks else np.nan,'first_quote_left_censored_share':np.mean(left) if left else np.nan,'age_caveat':'observed panel quote age, not verified IPO/security age'})
                for sector,nobs in sector_counts.items():industries.append({'factor':name,'missing':state,'phase':phase,'industry':sector,'stock_day_count':nobs,'stock_day_share':nobs/total})
    pd.DataFrame(profiles).to_csv(ctx.out/'lasso/missingness_profiles.csv',index=False);pd.DataFrame(industries).to_csv(ctx.out/'lasso/missingness_industries.csv',index=False)
    json_dump({'active_missing_factors':sorted(active),'manual_missingness_only_equivalent':all_equivalent,'quote_age_left_censored':True,'not_a_new_independent_strategy':True},ctx.out/'lasso/metadata.json')
    progress('lasso_diagnostics_complete',active_missing_factors=sorted(active))


def robustness(ctx):
    rows=[];cash=[];smoothing=[];market=CachedMarket(ctx.db)
    for family in ['ridge','lasso','xgboost']:
        a=pd.read_csv(ctx.base/'backtests'/f'{family}_A_daily.csv',dtype={'date':str}).set_index('date')
        b=pd.read_csv(ctx.base/'backtests'/f'{family}_B_daily.csv',dtype={'date':str}).set_index('date')
        for block in [10,20,40,60]:rows.append({'model':family,'block':block,**paired_bootstrap(a.net_return,b.net_return,block=block)})
        for group in ['A','B']:
            score=pd.read_pickle(ctx.base/'predictions'/f'{family}_{group}_2023_2025.pkl')
            for smooth in [1,3,5,10]:
                for frequency in [1,5]:
                    d,_,_,s=market.backtest(score,'20230101','20251231',smooth=smooth,rebalance_every=frequency)
                    smoothing.append({'model':family,'family':group,**s,'gross_annual_return':performance(d.gross_return)['annual_return'],'sum_daily_costs':float(d.cost.sum())})
        progress('robustness_model_complete',model=family)
    pd.DataFrame(rows).to_csv(ctx.out/'robustness/block_intervals.csv',index=False)
    pd.DataFrame(smoothing).to_csv(ctx.out/'robustness/all_model_smoothing.csv',index=False)
    # Apply explicit hypothetical haircuts only to terminal events with no usable source payoff.
    original=market.source_return.copy();unknown=market.terminal&(~np.isfinite(original)|(original<-1));baseline_mark=np.nan_to_num(market.ret,nan=0.)
    for loss in [0.,.5,1.]:
        market.source_return=original.copy();market.source_return[unknown]=(1+baseline_mark[unknown])*(1-loss)-1
        for family in ['ridge','lasso','xgboost']:
            for group in ['A','B']:
                score=pd.read_pickle(ctx.base/'predictions'/f'{family}_{group}_2023_2025.pkl')
                d,_,e,s=market.backtest(score,'20230101','20251231',**ctx.rule)
                cash.append({'strategy':family+'_'+group,'unknown_terminal_haircut':loss,'total_unknown_terminal_source_rows':int(unknown.sum()),**s})
    market.source_return=original
    pd.DataFrame(cash).to_csv(ctx.out/'robustness/terminal_haircuts.csv',index=False)
    events=[]
    for path in list((ctx.base/'backtests').glob('*_events.csv'))+list((ctx.base/'supplement/portfolios').glob('*_events.csv')):
        try:e=pd.read_csv(path)
        except pd.errors.EmptyDataError:continue
        if e.empty:continue
        for policy,g in e.groupby('policy'):
            events.append({'strategy':path.stem.replace('_events',''),'policy':policy,'rows':len(g),'unique_dates':g.date.nunique(),'unique_sids':g.sid.nunique(),'max_held_weight':g.weight.max(),'sum_weight_times_event_return':float((g.weight*g['return']).sum()),'sum_weight':g.weight.sum(),'basis':'arithmetic event contributions, not annualized or Carino-linked'})
    pd.DataFrame(events).to_csv(ctx.out/'audit/event_impact.csv',index=False)
    source=json.loads((ctx.db/'metadata/acceptance.json').read_text())
    json_dump({'research_accepted':False,'known_limits':source.get('remaining_limits',[]),'provided_daily_members':source.get('daily_members_2015_2025'),'source_hashes_previously_verified':True,'unknown_terminal_payoffs_are_scenarios_not_recovered_facts':True},ctx.out/'audit/data_limits.json')
    screen=pd.read_csv(ctx.base/'evaluation/factor_screen.csv');wq=screen.loc[screen.family.eq('WQ')]
    pass50=wq.loc[(wq.stability_rankic_signed>=.01)&(wq.stability_coverage>=.5)&(wq.positive_stability_years>=2),'factor'].tolist()
    pass80=wq.loc[(wq.stability_rankic_signed>=.01)&(wq.stability_coverage>=.8)&(wq.positive_stability_years>=2),'factor'].tolist()
    json_dump({'candidates_passing_50':pass50,'candidates_passing_80':pass80,'same_candidate_set':set(pass50)==set(pass80),'dedup_and_models_unchanged':set(pass50)==set(pass80),'not_used_to_reselect_main':True},ctx.out/'robustness/coverage_thresholds.json')


def neutral_style_profiles(ctx):
    """Month-end beginning holdings, aligned with prior-date risk exposures."""
    from usq.barra.model import load_model
    from .risk import design_matrix
    model=load_model(ctx.base/'barra_style/model');industries=[k for k in model.factor_returns.columns if k not in model.styles and k!='Market']
    rows=[];industryrows=[]
    for name in ctx.names:
        for variant in ['raw_common','neutral_common']:
            folder=ctx.out/'deciles'/name/variant/'evaluation/G10'
            holdings=pd.read_parquet(folder/'monthly_holdings.parquet')
            d=pd.read_csv(folder/'daily.csv',dtype={'date':str}).set_index('date');days=d.groupby(d.index.str[:6]).tail(1).index
            for day in days:
                w=holdings.loc[holdings.date.eq(day)].set_index('sid').weight
                prior=model.alignment.loc[day,'exposure_date'];e=model.exposures.loc[prior].reindex(w.index)
                exposure=design_matrix(e,model.styles,industries).T@w
                missing=float(w[e[model.styles].isna().any(axis=1)].sum())
                rows.append({'factor':name,'variant':variant,'date':day,'exposure_date':prior,'invested_weight':float(w.sum()),'zero_filled_style_weight':missing,**exposure.reindex(model.styles).to_dict()})
                industryrows.extend({'factor':name,'variant':variant,'date':day,'industry':k,'weight':float(exposure[k])} for k in industries)
    result=pd.DataFrame(rows);result.to_csv(ctx.out/'neutral/monthly_G10_style_exposures.csv',index=False)
    means=result.groupby(['factor','variant'])[['invested_weight','zero_filled_style_weight']+model.styles].mean().reset_index();means.to_csv(ctx.out/'neutral/monthly_G10_mean_styles.csv',index=False)
    pd.DataFrame(industryrows).to_csv(ctx.out/'neutral/monthly_G10_industries.csv',index=False)
    json_dump({'dates':36,'basis':'month-end beginning-of-day G10 holdings; lagged risk exposures; not daily mean','factor_score_neutrality_does_not_imply_ranked_portfolio_neutrality':True},ctx.out/'neutral/style_profile_metadata.json')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database',required=True);parser.add_argument('--baseline',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--stage',choices=['statistics','correlations','deciles','ratios','pure-C','lasso','robustness','styles','all'],default='all')
    args=parser.parse_args();ctx=Context(args.database,args.baseline,args.output)
    tasks={'statistics':statistics,'correlations':correlations,'deciles':deciles,'ratios':matched_ratios,'pure-C':fixed_fundamental_models,'lasso':lasso_diagnostics,'robustness':robustness,'styles':neutral_style_profiles}
    for name,fn in tasks.items():
        if args.stage in (name,'all'):
            started=time.time();fn(ctx);progress('extension_stage_complete',part=name,seconds=round(time.time()-started,2))


if __name__=='__main__':main()
