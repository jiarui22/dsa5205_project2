"""Factor estimation and chronological portfolio experiments."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .final_study import FinalStudy,composite
from .screen import information_series
from .protocol import validate_target
from .study_periods import CV_YEARS, PORTFOLIO_PHASES

PORTFOLIO_WEIGHTINGS = ('equal', 'sqrt_cap', 'cap')

RESEARCH_SETTINGS = {'top_n': 100,
 'buffer_n': 150,
 'rebalance_every': 10,
 'cost_bps': 10,
 'max_target_weight': 0.02,
 'tie_seed': 5205,
 'baseline_case': 'fundamental_sqrt_cap',
 'signals': {'wq': ['WQ040', 'WQ019', 'WQ026'],
             'fundamental': ['F01_EP_reconstructed',
                             'F02_BP_reconstructed',
                             'F03_SPQ_reconstructed',
                             'F04_ROEQ_reconstructed',
                             'F05_FCF_yield',
                             'F06_earnings_surprise'],
             'directions': {'F01_EP_reconstructed': 1,
                            'F02_BP_reconstructed': -1,
                            'F03_SPQ_reconstructed': -1,
                            'F04_ROEQ_reconstructed': 1,
                            'F05_FCF_yield': 1,
                            'F06_earnings_surprise': 1,
                            'WQ040': 1,
                            'WQ019': 1,
                            'WQ026': 1}}}

class StrategyStudy(FinalStudy):
    phases = PORTFOLIO_PHASES

    def factor_observer(self,name,frame):
        folder = self.out/'factor_panels'
        folder.mkdir(exist_ok=True)
        frame.to_pickle(folder/f'{name}.pkl')
        if not hasattr(self,'factor_statistics'):
            self.factor_statistics=[]
            self.targets=self.load('ylabel')
            self.ends=validate_target(self.db,self.u)
        direction=self.selection['directions'][name]
        for phase,start,end in [('direction','20150101','20181231'),('development',f'{CV_YEARS[0]}0101',f'{CV_YEARS[-1]}1231'),
                                *((name, *dates) for name, dates in self.phases.items())]:
            y=self.targets.loc[start:end].where(self.ends.loc[start:end].le(end),axis=0)
            stat=information_series((frame*direction).reindex(y.index),y,self.u.reindex(y.index))
            self.factor_statistics.append({'factor':name,'phase':phase,'direction':direction,'rankic':float(stat.rankic.mean()),'ic':float(stat.ic.mean()),'coverage':float(stat.coverage.mean()),'defined_days':int(stat.rankic.notna().sum())})
        # The direction was calibrated on 2015--2018; check it against actual data.
        first=self.factor_statistics[-4]
        assert first['rankic']>=0, 'Direction disagrees with the declared calibration period'
        pd.DataFrame(self.factor_statistics).to_csv(self.out/'factor_statistics.csv',index=False)

    def replay_phase(self,phase):
        pool=pd.DataFrame(1.,index=self.dates,columns=self.u.columns).where(self.common)
        selection_file = self.out / 'learners/frozen_family.json'
        selected_case = (json.loads(selection_file.read_text())['selected_case']
                         if phase == 'evaluation' and selection_file.exists() else None)
        for weighting in PORTFOLIO_WEIGHTINGS:
            self.save('pool_'+weighting,phase,pool,buffer=None,weighting=weighting,maximum=None,all_eligible=True)
            case = 'fundamental_' + weighting
            retain = phase == 'evaluation' and (weighting == 'sqrt_cap' or case == selected_case)
            self.save(case, phase, self.score, weighting=weighting, retain=retain)

    def robustness(self):
        for cost in [20,30]:self.save('cost_'+str(cost),'evaluation',self.score,cost=cost,weighting='cap')
        for name in self.parts:self.save('without_'+name[:3],'evaluation',composite(self.parts,name).where(self.common),weighting='cap')

    def audit(self):
        summaries=pd.read_csv(self.out/'portfolio_summary.csv');checks=[]
        for row in summaries.itertuples():
            d=pd.read_csv(self.out/'portfolios'/row.case/row.phase/'daily.csv')
            assert d.date.is_unique and d.date.is_monotonic_increasing
            np.testing.assert_allclose(d.net_return,d.gross_return-d.cost,rtol=0,atol=1e-12)
            np.testing.assert_allclose(d.cost,d.traded_notional*row.cost_bps/10000,rtol=0,atol=1e-12)
            np.testing.assert_allclose((1+d.net_return).cumprod(),d.net_nav,rtol=0,atol=1e-11)
            assert np.isfinite(d.net_return).all() and d.cash_weight.ge(-1e-8).all()
            checks.append({'case':row.case,'phase':row.phase,'sessions':len(d),'accounting_passed':True})
        expected = 38 if any(str(r['case']).startswith('ml_') for r in self.rows) else 20
        assert len(checks)==expected
        record={'passed':True,'portfolios':len(checks),'maximum_independent_holdings_error':float(summaries.independent_gross_error.max()),'checks':checks}
        (self.out/'portfolio_audit.json').write_text(json.dumps(record,indent=2));return record


def create_experiment_config(database,work,wq_selection=None):
    work=Path(work);work.mkdir(parents=True,exist_ok=True)
    from copy import deepcopy
    cfg=deepcopy(RESEARCH_SETTINGS);cfg['database']=str(Path(database).resolve())
    cfg['portfolio_phases'] = {name: list(dates) for name, dates in PORTFOLIO_PHASES.items()}
    if wq_selection is not None:
        cfg['signals']['wq']=wq_selection['wq']
        cfg['signals']['directions'].update(wq_selection['directions'])
    p=work/'config.json';p.write_text(json.dumps(cfg,indent=2))
    return p
