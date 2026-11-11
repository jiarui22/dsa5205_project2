"""Lightweight score-level fusion; no refits, no new factor or trading selection."""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .factors import rank
from .research_portfolio import ResearchMarket
from .research_cache import atomic_json,bundle_complete,finish_bundle
from .screen import information_series
from .extensions import digest,progress,summarize_information
from .supplement import paired_bootstrap

PHASES={'validation':('20220101','20221231'),'evaluation':('20230101','20251231')}
LAMBDAS=[0.,.25,.5];LAGS=[0,5];COSTS=[10,20,30]
POLICY={'version':1,'status':'retrospective experiment; 2022 reused, no fresh blind holdout',
 'formula':'(1-lambda)*same-date fundamental percentile + lambda*same-date price-Aprime prediction percentile',
 'lambdas':LAMBDAS,'financial_extra_sessions':LAGS,'cost_bps':COSTS,'phases':PHASES,
 'component_samples':'same finite F/P and positive-cap current eligible/observed-WQ cross-section for all lambda; no future target gating',
 'fundamental_component':'same six-factor fixed-direction average with neutral-zero missing scores; reuse audited lag0/lag5 reconstructed factors',
 'price_component':'reuse previously fitted A-prime halfyear predictions; no financial inputs, no refits',
 'portfolio':{'top_n':100,'buffer_n':150,'rebalance_every':10,'weighting':'sqrt_cap','max_target_weight':.02,'tie_seed':5205},
 'pool_benchmark':'same eligible samples, next-close timing, rebalance10 and costs; cap weighted without per-name2% cap; not SPY replication',
 'selection':'2022 only; maximize worst-case paired net active IR versus matching pool-cap across all6 cost/lag scenarios; then mean IR; then smaller lambda; each mean cash<=5%',
 'evaluation':'freeze lambda before evaluation stage; publish all18 configs in both phases; never pick from evaluation winner',
 'uncertainty':'evaluation lambda .25/.5 versus lambda0 in each matched lag/cost; paired circular bootstrap 2000 seed5205 blocks10/20/40',
 'storage':'daily ledgers, aggregate diagnostics and sparse monthly holdings only; no training matrices'}


def fuse_scores(fundamental,price,weight):
    if not 0<=weight<=1:raise ValueError('Lambda must be in [0,1]')
    if not fundamental.index.equals(price.index) or not fundamental.columns.equals(price.columns):raise ValueError('Fusion axes differ')
    common=np.isfinite(fundamental)&np.isfinite(price)
    return ((1-weight)*rank(fundamental.where(common))+weight*rank(price.where(common))).where(common)


def select_lambda(metrics):
    validation=metrics.loc[metrics.phase.eq('validation')&metrics.kind.eq('strategy')]
    expected={(lag,cost) for lag in LAGS for cost in COSTS};records=[]
    for weight in LAMBDAS:
        z=validation.loc[validation['lambda'].eq(weight)]
        if len(z)!=6 or set(zip(z.lag,z.cost_bps))!=expected:raise ValueError('Incomplete/duplicate validation grid')
        valid=np.isfinite(z.net_active_IR_vs_pool).all() and z.mean_cash.le(.05).all()
        records.append({'lambda':weight,'eligible':bool(valid),'worst_IR':float(z.net_active_IR_vs_pool.min()),'mean_IR':float(z.net_active_IR_vs_pool.mean()),'mean_validation_CAGR':float(z.annual_return.mean())})
    ranking=pd.DataFrame(records);good=ranking.loc[ranking.eligible].sort_values(['worst_IR','mean_IR','lambda'],ascending=[False,False,True])
    if good.empty:raise ValueError('No eligible validation candidate')
    return float(good.iloc[0]['lambda']),ranking


class LateFusion:
    def __init__(self,database,baseline,exploration,deep,output):
        self.db,self.base,self.exp,self.deep,self.out=map(Path,(database,baseline,exploration,deep,output))
        if any(self.out.resolve()==p.resolve() or p.resolve() in self.out.resolve().parents for p in [self.db,self.base,self.exp,self.deep]):raise ValueError('Independent output directory required')
        self.out.mkdir(parents=True,exist_ok=True);policy=json.loads(json.dumps(POLICY));path=self.out/'protocol.json'
        if path.exists() and json.loads(path.read_text())!=policy:raise ValueError('Changed protocol; use new output')
        if not path.exists():atomic_json(path,policy)
        for sub in ['portfolios','audit','analysis']:(self.out/sub).mkdir(exist_ok=True)
        self.selection=json.loads((self.base/'factor_selection.json').read_text());self.choice=json.loads((self.exp/'frozen_exploration_choice.json').read_text())
        files={'baseline_input':self.base/'input_signature.json','selection':self.base/'factor_selection.json','model_choices':self.base/'frozen_model_choices.json','exploration_choice':self.exp/'frozen_exploration_choice.json','deep_independent_verification':self.deep/'audit/independent_verification.json','snapshot_checksums':self.db/'checksums.sha256'}
        self.pred_paths=[]
        for year in [2022,2023,2024,2025]:
            for half in [f'{year}H1',f'{year}H2']:
                path=self.exp/'predictions'/f'extra_{self.choice["objective"]}_A_{half}.pkl';self.pred_paths.append(path)
                files[f'price/{half}']=path;files[f'price_meta/{half}']=path.with_suffix('.json')
        for lag in LAGS:
            for n in self.selection['fundamental']:files[f'financial/lag{lag}/{n}']=self.deep/'financial_delay'/f'lag{lag}'/(n+'.pkl')
        binding={k:digest(p) for k,p in files.items()};p=self.out/'input_binding.json'
        if p.exists() and json.loads(p.read_text())!=binding:raise ValueError('Changed owning inputs')
        if not p.exists():atomic_json(p,binding)
        self.binding=binding
        if not json.loads((self.deep/'audit/independent_verification.json').read_text())['passed']:raise ValueError('Audited delayed factors required')
        self.market=ResearchMarket(self.db);self.u=pd.read_pickle(self.db/'panels/universe.pkl')
        observed=np.load(self.base/'design/observed.npy',mmap_mode='r');self.allowed=self.u.eq(1)&pd.DataFrame(observed,index=self.u.index,columns=self.u.columns)
        self.dates=self.u.loc['20220101':'20251231'].index
        self.price=pd.concat([pd.read_pickle(p) for p in self.pred_paths]).sort_index().reindex(self.dates)
        for path in self.pred_paths:
            meta=json.loads(path.with_suffix('.json').read_text())
            assert meta['train_last_label_end']<meta['forecast_start'] and meta['family']=='A'
            assert meta['feature_count']==2*(len(self.selection['wq'])+4)
        assert self.price.columns.equals(self.u.columns)
        positive=pd.DataFrame(np.isfinite(self.market.cap)&(self.market.cap>0),index=self.u.index,columns=self.u.columns)
        self.original_common=(self.allowed&positive).reindex(self.dates)
        self.common=self.original_common&np.isfinite(self.price)
        # With these source models every eligible A row has a prediction. If this
        # changes, stop rather than silently changing the lambda0 opportunity set.
        np.testing.assert_array_equal(self.common,self.original_common)
        self.price=self.price.where(self.common);self.baseline_score=pd.DataFrame(1.,index=self.dates,columns=self.u.columns).where(self.common)
        self.fundamental={};self.y=pd.read_pickle(self.db/'panels/ylabel.pkl').reindex(self.dates)
        self.ends=pd.read_csv(self.db/'panels/label_end.csv',dtype=str,index_col=0).iloc[:,0].reindex(self.dates)
        cp=pd.read_pickle(self.db/'panels/close_adj.pkl').astype(float);seg=pd.read_pickle(self.db/'panels/adjustment_segment.pkl')
        asset=(cp.ffill()/cp.ffill().shift()-1).where(seg.eq(seg.where(seg>0).ffill().shift())&seg.gt(0))
        terminal=pd.read_pickle(self.db/'panels/delisting_event.pkl').eq(1);vendor=pd.read_pickle(self.db/'panels/daily_return.pkl')
        self.asset=asset.where(~(terminal&np.isfinite(vendor)&vendor.ge(-1)),vendor).fillna(0)
        p=self.out/'portfolio_summary.csv';self.rows=pd.read_csv(p).to_dict('records') if p.exists() else []
        self.score_hash=None;self.signal_rows=[];self.component_rows=[]

    def fundamental_score(self,lag):
        if lag not in self.fundamental:
            s=pd.DataFrame(0.,index=self.dates,columns=self.u.columns);names=self.selection['fundamental']
            for name in names:
                f=pd.read_pickle(self.deep/'financial_delay'/f'lag{lag}'/(name+'.pkl')).reindex(self.dates)*self.selection['directions'][name]
                s+=(rank(f.where(self.allowed.reindex(self.dates)))-.5).fillna(0)/len(names)
            self.fundamental[lag]=s.where(self.common)
        return self.fundamental[lag]

    @staticmethod
    def case(weight,lag,cost):return f'lambda{int(weight*100):03}_lag{lag}_cost{cost}'

    def save(self,phase,kind,case,score,cost,weight=None,lag=None,benchmark=None):
        folder=self.out/'portfolios'/kind/case/phase;folder.mkdir(parents=True,exist_ok=True)
        if self.score_hash is None or self.score_hash[0] is not score:
            h=hashlib.sha256();h.update(json.dumps([score.index.tolist(),score.columns.tolist()]).encode());h.update(score.to_numpy().tobytes(order='C'));self.score_hash=(score,h.hexdigest())
        sig={'phase':phase,'kind':kind,'case':case,'score':self.score_hash[1],'cost_bps':cost,'binding':self.binding,'portfolio':POLICY['portfolio']}
        required=['daily.csv','summary.json','events.csv','monthly_holdings.parquet']
        if bundle_complete(folder,'complete.json',sig,required):
            summary=json.loads((folder/'summary.json').read_text());daily=pd.read_csv(folder/'daily.csv',dtype={'date':str}).set_index('date')
        else:
            (folder/'complete.json').unlink(missing_ok=True);lo,hi=PHASES[phase]
            options={'all_eligible':True,'weighting':'cap'} if kind=='benchmark' else {'buffer_n':150,'weighting':'sqrt_cap','max_weight':.02}
            daily,w,events,summary=self.market.backtest(score,lo,hi,cost_bps=cost,rebalance_every=10,weights=True,**options)
            np.testing.assert_allclose(daily.net_return,daily.gross_return-daily.cost,atol=1e-12);np.testing.assert_allclose(daily.cost,daily.traded_notional*cost/10000,atol=1e-12)
            np.testing.assert_allclose((1+daily.net_return).cumprod(),daily.net_nav,atol=1e-11)
            independent=(w.astype(float)*self.asset.reindex(w.index)).sum(axis=1);error=float((independent-daily.gross_return).abs().max());assert error<2e-7
            summary.update(kind=kind,case=case,phase=phase,**{'lambda':weight},lag=lag,mean_cash=float(daily.cash_weight.mean()),gross_cagr=float((1+daily.gross_return).prod()**(252/len(daily))-1),max_held_weight=float(w.to_numpy().max()),independent_gross_error=error)
            if benchmark is not None:
                assert benchmark.index.equals(daily.index)
                a=daily.net_return-benchmark.net_return;te=float(a.std(ddof=1)*np.sqrt(252))
                summary.update(net_active_IR_vs_pool=float(a.mean()*252/te) if te>1e-12 else np.nan,tracking_error_vs_pool=te,pool_cap_CAGR=float((1+benchmark.net_return).prod()**(252/len(benchmark))-1))
            daily.to_csv(folder/'daily.csv');events.to_csv(folder/'events.csv',index=False)
            m=w.groupby(w.index.str[:6]).tail(1);ii,jj=np.where(m.to_numpy()>0)
            pd.DataFrame({'date':m.index.to_numpy()[ii],'sid':m.columns.to_numpy()[jj],'weight':m.to_numpy()[ii,jj]}).to_parquet(folder/'monthly_holdings.parquet',index=False)
            atomic_json(folder/'summary.json',summary);finish_bundle(folder,'complete.json',sig,required)
        self.rows=[r for r in self.rows if (r['kind'],r['case'],r['phase'])!=(kind,case,phase)]+[summary]
        tmp=self.out/'portfolio_summary.csv.tmp';pd.DataFrame(self.rows).to_csv(tmp,index=False);tmp.replace(self.out/'portfolio_summary.csv')
        return daily,summary

    def lambda0_check(self,phase,lag,cost,daily):
        if lag==0:folder=self.deep/'portfolios/costs'/f'simple_C_sqrt_cap_{cost}bps'/phase
        elif cost==10:folder=self.deep/'portfolios/financial_delay'/f'C_lag{lag}'/phase
        else:return
        previous=pd.read_csv(folder/'daily.csv',dtype={'date':str}).set_index('date')
        np.testing.assert_allclose(daily.net_return,previous.net_return,atol=1e-12)

    def phase(self,phase):
        if phase=='evaluation' and not (self.out/'frozen_lambda.json').exists():raise ValueError('Freeze validation selection before evaluation')
        benchmarks={}
        for cost in COSTS:benchmarks[cost]=self.save(phase,'benchmark',f'pool_cap_cost{cost}',self.baseline_score,cost)[0]
        lo,hi=PHASES[phase];target=self.y.loc[lo:hi].where(self.ends.loc[lo:hi].le(hi),axis=0);universe=self.u.reindex(target.index)
        for lag in LAGS:
            fund=self.fundamental_score(lag)
            corr=information_series(fund.loc[lo:hi],self.price.loc[lo:hi],self.common.loc[lo:hi].astype('uint8'))
            self.component_rows.append({'phase':phase,'lag':lag,'mean_daily_rank_correlation_F_P':float(corr.rankic.mean()),'mean_common_pairs':float(corr.pairs.mean())})
            for weight in LAMBDAS:
                score=fuse_scores(fund,self.price,weight)
                diagnostics=information_series(score.reindex(target.index),target,universe)
                self.signal_rows.append({'phase':phase,'lag':lag,'lambda':weight,**summarize_information(diagnostics)})
                for cost in COSTS:
                    case=self.case(weight,lag,cost);daily,_=self.save(phase,'strategy',case,score,cost,weight,lag,benchmarks[cost])
                    if weight==0:self.lambda0_check(phase,lag,cost,daily)
                progress('late_fusion_config_complete',phase=phase,lag=lag,weight=weight)
                del score;gc.collect()
        pd.DataFrame(self.signal_rows).to_csv(self.out/'analysis/signal_diagnostics.csv',index=False);pd.DataFrame(self.component_rows).to_csv(self.out/'analysis/component_correlation.csv',index=False)
        if phase=='validation':
            chosen,ranking=select_lambda(pd.DataFrame(self.rows));ranking.to_csv(self.out/'validation_selection.csv',index=False)
            frozen={'lambda':chosen,'criterion':POLICY['selection'],'evaluation_used_for_selection':False,'validation_scenarios_per_lambda':6,'comparison_baseline_lambda':0.0}
            p=self.out/'frozen_lambda.json'
            if p.exists() and json.loads(p.read_text())!=frozen:raise ValueError('Frozen lambda changed')
            atomic_json(p,frozen);progress('late_fusion_lambda_frozen',**frozen)

    def analysis(self):
        table=pd.DataFrame(self.rows);table['CAGR_minus_SPY']=table.annual_return-table.spy_annual_return
        strategy=table.loc[table.kind.eq('strategy')].copy();strategy['CAGR_minus_pool']=strategy.annual_return-strategy.pool_cap_CAGR
        pairs=[];annual=[];returns={}
        for lag in LAGS:
            for cost in COSTS:
                a=pd.read_csv(self.out/'portfolios/strategy'/self.case(0,lag,cost)/'evaluation/daily.csv',dtype={'date':str}).set_index('date').net_return
                for weight in LAMBDAS:
                    case=self.case(weight,lag,cost);d=pd.read_csv(self.out/'portfolios/strategy'/case/'evaluation/daily.csv',dtype={'date':str}).set_index('date');returns[case]=d.net_return
                    for year,g in d.groupby(d.index.str[:4]):annual.append({'case':case,'lambda':weight,'lag':lag,'cost_bps':cost,'year':year,'net_total_return':float((1+g.net_return).prod()-1),'SPY_total_return':float((1+g.benchmark_return).prod()-1)})
                    if weight:
                        for block in [10,20,40]:pairs.append({'lambda':weight,'lag':lag,'cost_bps':cost,'block':block,**paired_bootstrap(a,d.net_return,block=block)})
        strategy.to_csv(self.out/'analysis/strategy_comparison.csv',index=False);pd.DataFrame(pairs).to_csv(self.out/'analysis/paired_intervals.csv',index=False);pd.DataFrame(annual).to_csv(self.out/'analysis/annual.csv',index=False);pd.DataFrame(returns).to_csv(self.out/'analysis/evaluation_daily.csv')

    def verify(self):
        table=pd.read_csv(self.out/'portfolio_summary.csv');assert len(table)==42 and len(table.loc[table.kind.eq('strategy')])==36
        assert table.independent_gross_error.max()<2e-7 and not table.duplicated(['kind','case','phase']).any()
        chosen,_=select_lambda(table);frozen=json.loads((self.out/'frozen_lambda.json').read_text());assert frozen['lambda']==chosen and not frozen['evaluation_used_for_selection']
        for row in json.loads((self.base/'run_manifest.json').read_text())['files']:assert digest(self.base/row['path'])==row['sha256'],row['path']
        result={'passed':True,'strategy_portfolios':36,'matching_pool_benchmarks':6,'new_model_fits':0,'reused_price_prediction_halves':8,'baseline_artifacts_unchanged':535,'maximum_holdings_gross_error':float(table.independent_gross_error.max()),'selected_lambda':chosen,'selection_only_2022':True,'all_lambda0_existing_controls_match':True,'new_large_training_matrices':0,'status':'retrospective comparison, not fresh blind holdout'}
        atomic_json(self.out/'audit/verification.json',result);progress('late_fusion_verified',**result)


def main():
    p=argparse.ArgumentParser();p.add_argument('--database',required=True);p.add_argument('--baseline',required=True);p.add_argument('--exploration',required=True);p.add_argument('--deep',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();e=LateFusion(a.database,a.baseline,a.exploration,a.deep,a.output);e.phase('validation');e.phase('evaluation');e.analysis();e.verify()


if __name__=='__main__':main()
