"""Purged expanding-time CV and rolling forecasts on six accounting signals."""
from __future__ import annotations
import argparse,gc,json,os,subprocess,sys,time,warnings
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from .factors import rank,price_inputs,wq
from .models import fit_model,halves,MODEL_POLICY
from .screen import information_series
from .research_cache import atomic_json,bundle_complete,finish_bundle
from .extensions import digest
from .strategy_study import PORTFOLIO_WEIGHTINGS

FEATURE_SETS = ('fundamental', 'combined', 'engineered')
BASE_GRIDS = {
    'ridge': [{'alpha_per_sample': a} for a in [1., .1, .01, .001]],
    'lasso': [{'alpha': a} for a in [.001, .0005, .0001, .00001, .000001]],
    'xgboost': [{'max_depth': d, 'learning_rate': lr, 'reg_lambda': reg}
                for d in [2, 3, 4] for lr in [.03, .08] for reg in [1., 10.]]}
GRIDS = {kind: [dict(params, feature_set=group) for group in FEATURE_SETS for params in grid]
         for kind, grid in BASE_GRIDS.items()}


def feature_columns(names, group):
    if group not in FEATURE_SETS:
        raise ValueError(f'Unknown feature set: {group}')
    return [i for i, name in enumerate(names)
            if group == 'engineered' or name.startswith('F')
            or (group == 'combined' and name.startswith('WQ'))]


def engineered_panels(study, universe):
    """Yield causal, cross-sectionally comparable signals; no fitted transforms."""
    p = price_inputs(study.load, universe)
    observed = np.zeros(universe.shape, dtype=bool)
    for name in study.selection['wq']:
        f = wq(int(name[2:]), p)
        observed |= np.isfinite(f)
        yield name, f * study.selection['directions'].get(name,1), 'combined'
    eligible = universe.eq(1) & observed
    for name in study.selection['fundamental']:
        f = pd.read_pickle(study.out/'factor_panels'/f'{name}.pkl') * study.selection['directions'][name]
        z = rank(f.where(eligible)) - .5
        yield name, f, 'fundamental'
        # Changes are measured in ranks so accounting units cannot dominate.
        yield 'change20_' + name, z - z.shift(20), 'engineered'
    close = p['close']; seg = study.load('adjustment_segment')
    for window in [5, 20, 60]:
        yield f'momentum{window}', (close / close.shift(window) - 1).where(
            seg.eq(seg.shift(window)) & seg.gt(0)), 'engineered'
    yield 'volatility20', p['returns'].rolling(20, min_periods=20).std(), 'engineered'
    yield 'volume_surprise20', p['volume'] / p['volume'].rolling(20, min_periods=20).mean().replace(0, np.nan) - 1, 'engineered'
    yield 'size', np.log(study.load('market_cap').where(lambda x: x > 0)), 'engineered'

from .study_periods import (TRAIN_START, CV_YEARS, VALIDATION_YEAR, EVALUATION_YEARS,
                            FORECAST_WINDOW_YEARS, PORTFOLIO_PHASES)

PORTFOLIO_CANDIDATES = {
    model: {
        weight: (
            f'fundamental_{weight}' if model == 'fundamental'
            else f'ml_{model}' if weight == 'sqrt_cap'
            else f'ml_{model}_{weight}'
        )
        for weight in PORTFOLIO_WEIGHTINGS
    }
    for model in ('fundamental', *GRIDS)
}

POLICY={'features':'fundamental vs fundamental+WQ vs engineered; contemporaneous percentile ranks and missing flags',
        'training_start':TRAIN_START,'CV_years':list(CV_YEARS),
        'CV':'3 expanding annual folds, fresh semiannual fits inside each validation year',
        'purge':'training label_end strictly before forecast_start; scored labels mature by fold end',
        'CV_selection':'equal-fold mean daily RankIC, all folds >=80% defined days; tie earlier candidate (more shrinkage/shallower tree)',
        'grids':GRIDS,'XGB_fixed':{**MODEL_POLICY['xgboost_fixed'], 'n_estimators':200},
        'lasso_solver':{'dtype':'float64','precompute':True,'max_iter':5000,'tol':1e-5,'unconverged':'exclude candidate from selection'},
        'feature_selection':'feature set and hyperparameters jointly selected only by 2020-2022 CV',
        'engineering':'screened WQ signals; six fundamentals; 20-session fundamental rank changes; momentum5/20/60; volatility20; volume surprise20; log cap',
        'model_selection':'2023 net Sharpe across four score sources and three weightings; mean cash <=5%; tie lower turnover then case',
        'evaluation':'freeze parameters and selected score and weighting before 2024-2025 evaluation; report all candidates',
        'execution':'same Top100, buffer150, every10, target cap2%, seed5205, 10bps; equal, square-root capitalization or capitalization weights',
        'WQ_selection':'2018-2019 scores precede CV scoring',
        'forecast_window_years':FORECAST_WINDOW_YEARS}


def purged_dates(index,ends,start,train_start=TRAIN_START):
    return (np.asarray(index)>=train_start)&(np.asarray(index)<start)&(np.asarray(ends)<start)&(np.asarray(ends)!='')


def choose_parameters(rows):
    table=pd.DataFrame(rows);selected={}
    for kind,grid in GRIDS.items():
        candidates=[]
        for k in range(len(grid)):
            z=table.loc[table.model.eq(kind)&table.candidate.eq(k)]
            if len(z)!=len(CV_YEARS) or set(z.year)!=set(CV_YEARS):raise ValueError('Incomplete CV folds')
            eligible=np.isfinite(z.rankic).all() and z.defined_fraction.ge(.8).all() and ('fit_converged' not in z or z.fit_converged.all())
            if eligible:candidates.append((float(z.rankic.mean()),-k,k))
        if not candidates:raise ValueError(f'No nondegenerate CV candidate for {kind}; do not select on evaluation')
        k=max(candidates)[2];selected[kind]={'candidate':k,'params':grid[k],'CV_mean_rankic':max(candidates)[0]}
    return selected


def choose_family(rows):
    z=pd.DataFrame(rows)
    good=z.loc[np.isfinite(z.sharpe)&z.mean_cash.le(.05)].sort_values(['sharpe','mean_daily_turnover','case'],ascending=[False,True,True])
    if good.empty:raise ValueError('No valid validation family')
    return str(good.iloc[0]['case'])


def select_portfolio_specification(summary):
    validation = summary.loc[summary.phase.eq('validation')]
    case_to_model = {case: model for model, cases in PORTFOLIO_CANDIDATES.items()
                     for case in cases.values()}
    comparison = validation.loc[validation.case.isin(case_to_model)]
    if len(comparison) != len(case_to_model) or set(comparison.case) != set(case_to_model):
        raise ValueError('Incomplete or duplicate validation portfolio candidates')
    winner = choose_family(comparison)
    eligible = comparison.loc[np.isfinite(comparison.sharpe) & comparison.mean_cash.le(.05)]
    by_model = {}
    for model, cases in PORTFOLIO_CANDIDATES.items():
        rows = eligible.loc[eligible.case.isin(cases.values())]
        by_model[model] = choose_family(rows) if not rows.empty else None
    return {
        'selected_case': winner,
        'selected_by_model': by_model,
        'selection_period': str(VALIDATION_YEAR),
        'criterion': POLICY['model_selection'],
        'evaluation_used_for_selection': False,
    }


class ReturnModelExperiment:
    def __init__(self,study,refit=False):
        self.refit=True
        self.study=study;self.out=study.out/'learners';self.out.mkdir(exist_ok=True)
        self.design=self.out/'design';self.design.mkdir(exist_ok=True)
        p=self.out/'protocol.json'
        if p.exists() and json.loads(p.read_text())!=POLICY:raise ValueError('ML policy changed; use a fresh learners directory')
        atomic_json(p,POLICY)
        self.names=study.selection['fundamental'];self.rows=[]
        self.ends=np.asarray(study.ends.fillna(''),dtype='U8')
        self.u=study.u
        self.signature={'database':digest(study.db/'checksums.sha256'),'signals':study.selection,
                        'implementation':digest(Path(__file__)),'fit_implementation':digest(Path(__file__).with_name('models.py')),
                        'factors':[digest(study.out/'factor_panels'/f'{n}.pkl') for n in self.names]}
        self.features_ready=False

    def prepare_features(self):
        required=['features.npy','target.npy','allowed.npy','axes.json']
        (self.design/'complete.json').unlink(missing_ok=True)
        # One fresh design contains all feature blocks; workers select columns.
        value_count = len(self.study.selection['wq']) + 18
        shape=(*self.u.shape, value_count * 2);temp=self.design/'features.tmp.npy'
        x=np.lib.format.open_memmap(temp,mode='w+',dtype='float32',shape=shape)
        prices=price_inputs(self.study.load,self.u);observed=np.zeros(self.u.shape,dtype=bool)
        for name in self.study.selection['wq']:observed|=np.isfinite(wq(int(name[2:]),prices))
        del prices
        cap=self.study.load('market_cap');allowed=self.u.eq(1)&observed&np.isfinite(cap)&cap.gt(0)
        rank_sample=self.u.eq(1)&observed
        names=[];manifest=[]
        for k,(name,f,group) in enumerate(engineered_panels(self.study,self.u)):
            z=rank(f.replace([np.inf,-np.inf],np.nan).where(rank_sample))-.5
            x[:,:,2*k]=z.fillna(0).to_numpy('float32');x[:,:,2*k+1]=z.isna().to_numpy('float32')
            names.extend([name, name+'_missing'])
            manifest.append({'name':name,'block':group,'transformation':'eligible daily percentile minus .5; neutral zero + missing flag'})
            del f,z
        assert len(names)==shape[2]
        pd.DataFrame(manifest).to_csv(self.out/'feature_manifest.csv',index=False)
        x.flush();del x;temp.replace(self.design/'features.npy')
        np.save(self.design/'target.npy',self.study.targets.to_numpy('float32'))
        np.save(self.design/'allowed.npy',allowed.to_numpy())
        atomic_json(self.design/'axes.json',{'index':self.u.index.tolist(),'columns':self.u.columns.tolist(),'label_end':self.ends.tolist(),'features':names})
        finish_bundle(self.design,'complete.json',self.signature,required)
        del cap,allowed;gc.collect()
        self.features_ready=True

    def fit_piece(self,kind,params,half,start,end,tag):
        self.prepare_features() if not self.features_ready else None
        folder=self.out/'fits'/tag/f'{kind}_{half}';folder.mkdir(parents=True,exist_ok=True)
        train_start = (f'{int(start[:4])-FORECAST_WINDOW_YEARS}{start[4:]}'
                       if tag == 'forecast' else TRAIN_START)
        signature={'train_start':train_start,'design':self.signature,'kind':kind,'params':params,'half':half,'start':start,'end':end}
        required=['prediction.pkl','model.joblib','meta.json']
        (folder/'complete.json').unlink(missing_ok=True)
        job={'design':str(self.design.resolve()),'folder':str(folder.resolve()),'kind':kind,'params':params,'half':half,'start':start,'end':end,'signature':signature,'tag':tag,'train_start':train_start}
        atomic_json(folder/'job.json',job)
        subprocess.run([sys.executable,'-u','-m','usq.project2.model_selection','--worker',str((folder/'job.json').resolve())],check=True,env={**os.environ,'PYTHONPATH':str(Path(__file__).resolve().parents[2])})
        meta=json.loads((folder/'meta.json').read_text());assert meta['train_last_label_end']<start
        self.rows.append({**meta,'tag':tag,'reused_verified_fit':False});pd.DataFrame(self.rows).drop_duplicates(['tag','model','half']).to_csv(self.out/'training_history.csv',index=False)
        return pd.read_pickle(folder/'prediction.pkl')

    def diagnostics(self,score,end):
        label_end = pd.Series(self.ends, index=self.u.index).reindex(score.index)
        mature = label_end.notna() & label_end.ne('') & label_end.le(end)
        y = self.study.targets.reindex(score.index).where(mature, axis=0)
        u=self.u.reindex(score.index)
        d=information_series(score,y,u)
        possible=(np.isfinite(y)&u.eq(1)).sum(axis=1).ge(30)
        return {'rankic':float(d.rankic.mean()),'ic':float(d.ic.mean()),'defined_fraction':float(d.rankic.notna().sum()/possible.sum()),'scored_days':int(d.rankic.notna().sum())}

    def cross_validate(self):
        self.prepare_features();rows=[]
        for kind,grid in GRIDS.items():
            for k,params in enumerate(grid):
                for year in CV_YEARS:
                    score=pd.concat([self.fit_piece(kind,params,half,start,end,f'cv_grid{k}') for half,start,end in halves(year)])
                    rows.append({'model':kind,'candidate':k,'params':json.dumps(params),'feature_set':params['feature_set'],'fit_converged':all(r['fit_converged'] for r in self.rows[-2:]),'year':year,**self.diagnostics(score,f'{year}1231')})
                    pd.DataFrame(rows).to_csv(self.out/'CV_scores.csv',index=False)
                    print(json.dumps({'stage':'CV_fold','model':kind,'candidate':k,'year':year,**rows[-1]},allow_nan=True),flush=True)
        scores=pd.DataFrame(rows)
        summary=scores.groupby(['model','candidate','feature_set']).agg(mean_rankic=('rankic','mean'),worst_fold_rankic=('rankic','min'),fold_dispersion=('rankic','std'),minimum_defined_fraction=('defined_fraction','min'),finite_folds=('rankic','count'),fit_converged=('fit_converged','all')).reset_index()
        summary['eligible']=summary.finite_folds.eq(len(CV_YEARS))&summary.minimum_defined_fraction.ge(.8)&summary.fit_converged
        summary.to_csv(self.out/'CV_comparison.csv',index=False)
        chosen=choose_parameters(rows);p=self.out/'frozen_hyperparameters.json'
        if p.exists() and json.loads(p.read_text())!=chosen:raise ValueError('CV selection changed')
        atomic_json(p,chosen);return pd.DataFrame(rows),chosen

    def save_prediction_r2(self, kind, phase, score, benchmark, phase_end):
        # Compare model and historical-mean predictions on identical finite pairs.
        label_end = pd.Series(self.ends, index=self.u.index).reindex(score.index)
        mature = label_end.notna() & label_end.ne('') & label_end.le(phase_end)
        target = self.study.targets.reindex(score.index).where(mature, axis=0)
        prediction = score.to_numpy(dtype=np.float64)
        actual = target.to_numpy(dtype=np.float64)
        baseline = benchmark.reindex(score.index).to_numpy()[:, None]
        valid = np.isfinite(actual) & np.isfinite(prediction) & np.isfinite(baseline)
        daily = pd.DataFrame({
            'pairs': valid.sum(axis=1),
            'model_sse': np.where(valid, (actual - prediction)**2, 0).sum(axis=1),
            'benchmark_sse': np.where(valid, (actual - baseline)**2, 0).sum(axis=1),
        }, index=score.index)
        years = daily.groupby(daily.index.str[:4]).sum()
        years.loc['Full period'] = daily.sum()
        years['oos_r2'] = 1 - years.model_sse / years.benchmark_sse.where(years.benchmark_sse.gt(0))
        years.to_csv(self.out / f'{phase}_{kind}_oos_r2.csv', index_label='period')

    def fundamental_prediction_r2(self):
        """Calibrate the existing rank score to return units using past data."""
        study = self.study
        ranks = []
        for name in self.names:
            factor = pd.read_pickle(study.out / 'factor_panels' / f'{name}.pkl')
            signed = factor * study.selection['directions'][name]
            ranks.append((rank(signed.where(study.rank_eligible)) - .5).fillna(0))
        cap = study.load('market_cap')
        scores = (sum(ranks) / len(ranks)).where(
            study.rank_eligible & np.isfinite(cap) & cap.gt(0)
        )
        # Calibrate precisely the score used by the uncalibrated trading rule.
        pd.testing.assert_frame_equal(scores.reindex(study.score.index), study.score)
        target = study.targets.to_numpy(dtype=np.float32)
        for phase, years in (('validation', (VALIDATION_YEAR,)), ('evaluation', EVALUATION_YEARS)):
            predictions, benchmarks = [], []
            for year in years:
                for _, start, end in halves(year):
                    lower = f'{int(start[:4]) - FORECAST_WINDOW_YEARS}{start[4:]}'
                    train = purged_dates(self.u.index, self.ends, start, lower)
                    x = scores.to_numpy()[train]
                    y = target[train]
                    valid = np.isfinite(x) & np.isfinite(y)
                    x, y = x[valid].astype(np.float64), y[valid].astype(np.float64)
                    mean_x, mean_y = x.mean(), y.mean()
                    centered = x - mean_x
                    variance = np.dot(centered, centered)
                    if variance <= 0 or not np.isfinite(mean_y):
                        raise ValueError('Degenerate fundamental-score calibration sample')
                    slope = np.dot(centered, y - mean_y) / variance
                    predicted = mean_y + slope * (scores.loc[start:end] - mean_x)
                    predictions.append(predicted)
                    benchmarks.append(pd.Series(mean_y, index=predicted.index))
            self.save_prediction_r2('fundamental', phase, pd.concat(predictions),
                                    pd.concat(benchmarks), PORTFOLIO_PHASES[phase][1])

    def forecast(self,phase):
        chosen=json.loads((self.out/'frozen_hyperparameters.json').read_text())
        if phase=='evaluation' and not (self.out/'frozen_family.json').exists():raise ValueError('Freeze 2023 family choice before evaluation')
        years=[VALIDATION_YEAR] if phase=='validation' else EVALUATION_YEARS;metrics=[]
        for kind in GRIDS:
            predictions, benchmarks = [], []
            for year in years:
                for half, start, end in halves(year):
                    pred = self.fit_piece(kind, chosen[kind]['params'], half, start, end, 'forecast')
                    predictions.append(pred)
                    benchmarks.append(pd.Series(self.rows[-1]['training_target_mean'], index=pred.index))
            scores = pd.concat(predictions)
            benchmark = pd.concat(benchmarks)
            score=scores.where(self.study.common.reindex(scores.index))
            for weighting, case in PORTFOLIO_CANDIDATES[kind].items():
                retain = phase == 'evaluation' and case == json.loads(
                    (self.out / 'frozen_family.json').read_text())['selected_case']
                self.study.save(case, phase, score, weighting=weighting, retain=retain)
            phase_end = PORTFOLIO_PHASES[phase][1]
            metrics.append({'model':kind,'phase':phase,**self.diagnostics(score,phase_end)})
            self.save_prediction_r2(kind, phase, score, benchmark, phase_end)
        pd.DataFrame(metrics).to_csv(self.out/f'{phase}_forecast_diagnostics.csv',index=False)
        if phase=='validation':
            summary=pd.DataFrame(self.study.rows)
            record = select_portfolio_specification(summary)
            p=self.out/'frozen_family.json'
            if p.exists() and json.loads(p.read_text())!=record:raise ValueError('Frozen family changed')
            atomic_json(p,record)
        return pd.DataFrame(metrics)


def worker(path):
    j=json.loads(Path(path).read_text());d=Path(j['design']);out=Path(j['folder']);axes=json.loads((d/'axes.json').read_text())
    idx=pd.Index(axes['index']);columns=pd.Index(axes['columns']);ends=np.asarray(axes['label_end'],dtype='U8')
    x=np.load(d/'features.npy',mmap_mode='r');y=np.load(d/'target.npy',mmap_mode='r');allowed=np.load(d/'allowed.npy',mmap_mode='r')
    dates=purged_dates(idx,ends,j['start'],j.get('train_start',TRAIN_START));valid=dates[:,None]&allowed&np.isfinite(y)
    positions=np.flatnonzero(valid);used=np.flatnonzero(valid.any(axis=1));assert len(positions)>0
    params=dict(j['params']);group=params.pop('feature_set','fundamental')
    names=axes.get('features',[f'F{i}' for i in range(x.shape[2])])
    columns_used=feature_columns(names,group)
    xx=np.array(x.reshape(-1,x.shape[2])[np.ix_(positions,columns_used)],dtype='float64' if j['kind']=='lasso' else 'float32',order='F' if j['kind']=='lasso' else 'C');yy=np.asarray(y.ravel()[positions],dtype='float64' if j['kind']=='lasso' else 'float32')
    training_target_mean = float(yy.mean(dtype=np.float64))
    from sklearn.exceptions import ConvergenceWarning
    clock=time.time()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', ConvergenceWarning)
        model=fit_model(j['kind'], ({'n_estimators':200, **params} if j['kind']=='xgboost' else params),xx,yy)
    converged=not any(issubclass(w.category,ConvergenceWarning) for w in caught)
    messages=[str(w.message) for w in caught]
    del xx,yy,positions;gc.collect()
    if not converged and not j['signature'].get('allow_unconverged',False) and j.get('tag','').startswith('forecast'):
        raise RuntimeError('Forecast fit did not converge; do not publish results')
    forecast=np.flatnonzero((idx>=j['start'])&(idx<=j['end']));rows=[]
    for i in forecast:
        pred=model.predict(np.asarray(x[i][:,columns_used],dtype='float32'));rows.append(np.where(allowed[i],pred,np.nan).astype('float32'))
    score=pd.DataFrame(rows,index=idx[forecast],columns=columns)
    meta={'model':j['kind'],'half':j['half'],'params':json.dumps(j['params']),'samples':int(valid.sum()),'training_target_mean':training_target_mean,'features':len(columns_used),'feature_set':group,'fit_converged':converged,
          'solver_iterations':int(np.max(model.n_iter_)) if getattr(model,'n_iter_',None) is not None else None,
          'solver_dual_gap':float(model.dual_gap_) if hasattr(model,'dual_gap_') else None,'warnings':messages,
          'train_first':str(idx[used[0]]),'train_last_signal':str(idx[used[-1]]),'train_last_label_end':str(max(ends[used])),
          'forecast_start':j['start'],'forecast_end':j['end'],'seconds':time.time()-clock,
          'nonzero_value_coefficients':int(np.count_nonzero(model.coef_[::2])) if hasattr(model,'coef_') else None,
          'nonzero_missing_coefficients':int(np.count_nonzero(model.coef_[1::2])) if hasattr(model,'coef_') else None}
    assert meta['train_last_label_end']<j['start']
    importance=model.coef_ if hasattr(model,'coef_') else model.feature_importances_
    pd.DataFrame({'feature':[names[i] for i in columns_used],
                  'value':importance,'kind':'coefficient' if hasattr(model,'coef_') else 'tree_gain'}).to_csv(out/'feature_importance.csv',index=False)
    score.to_pickle(out/'prediction.pkl');joblib.dump(model,out/'model.joblib',compress=3);atomic_json(out/'meta.json',meta)
    finish_bundle(out,'complete.json',j['signature'],['prediction.pkl','model.joblib','meta.json'])
    print(json.dumps({'stage':'fit_complete',**meta}),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--worker');a=p.parse_args()
    if a.worker:worker(a.worker)
if __name__=='__main__':main()
