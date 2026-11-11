import numpy as np
import pandas as pd
import pytest
from usq.project2.model_selection import purged_dates,choose_parameters,choose_family,GRIDS


def test_purge_excludes_unmatured_overlap_and_tail():
    idx=np.array(['20141231','20181201','20181214','20181231','20190102'])
    ends=np.array(['20150116','20181217','20190102','','20190118'])
    np.testing.assert_array_equal(purged_dates(idx,ends,'20190101'),[False,True,False,False,False])


def cv_rows():
    return [{'model':m,'candidate':k,'year':y,'rankic':.01 if k==0 else .02,'defined_fraction':1.} for m,g in GRIDS.items() for k in range(len(g)) for y in [2020,2021,2022]]


def test_cv_requires_all_folds_and_rejects_constant_candidate():
    rows=cv_rows();selected=choose_parameters(rows)
    assert all(x['candidate']==1 for x in selected.values())
    # A degenerate fold invalidates the entire apparently strongest candidate.
    for r in rows:
        if r['candidate']>0 and r['year']==2021:r['defined_fraction']=0
    assert all(x['candidate']==0 for x in choose_parameters(rows).values())
    with pytest.raises(ValueError,match='Incomplete'):
        choose_parameters(rows[:-1])


def test_validation_family_uses_net_sharpe_and_cash_constraint():
    rows=[{'case':'fundamental_sqrt_cap','sharpe':.4,'mean_daily_turnover':.01,'mean_cash':0},
          {'case':'ml_ridge','sharpe':.5,'mean_daily_turnover':.02,'mean_cash':0},
          {'case':'ml_lasso','sharpe':1.,'mean_daily_turnover':.001,'mean_cash':.5}]
    assert choose_family(rows)=='ml_ridge'
    # Deterministic turnover tie-break; no evaluation return is accepted.
    rows[0]['sharpe']=.5
    assert choose_family(rows)=='fundamental_sqrt_cap'


def test_feature_groups_use_wq_and_exclude_engineering_from_baselines():
    from usq.project2.model_selection import feature_columns
    names=['WQ040','WQ040_missing','F01_EP','F01_EP_missing','change20_F01_EP','momentum20']
    assert feature_columns(names,'fundamental')==[2,3]
    assert feature_columns(names,'combined')==[0,1,2,3]
    assert feature_columns(names,'engineered')==list(range(6))


def test_engineered_features_do_not_change_when_future_inputs_change(tmp_path):
    from types import SimpleNamespace
    from usq.project2.model_selection import engineered_panels
    from usq.project2.factors import price_inputs
    from pathlib import Path
    dates=pd.Index(pd.bdate_range('2015-01-01',periods=85).strftime('%Y%m%d'))
    columns=['A','B'];u=pd.DataFrame(1,index=dates,columns=columns)
    rng=np.random.default_rng(5205)
    close=pd.DataFrame(np.exp(rng.normal(0,.01,(85,2)).cumsum(axis=0))*20,index=dates,columns=columns)
    panels={k:close.copy() for k in ['open','high','low','close']}
    panels.update(volume=u*100,returns=close.pct_change(),adj_factor=u.copy(),adjustment_segment=u.copy(),market_cap=close*1000)
    folder=tmp_path/'factor_panels';folder.mkdir()
    f=close.copy();f.to_pickle(folder/'F01_EP.pkl')
    study=SimpleNamespace(load=lambda k:panels[k],out=tmp_path,
        selection={'wq':['WQ040'],'fundamental':['F01_EP'],'directions':{'F01_EP':1}})
    first={name:frame.copy() for name,frame,_ in engineered_panels(study,u)}
    for k in ['open','high','low','close','volume','returns','market_cap']:panels[k].iloc[65:]*=10
    f.iloc[65:]*=-5;f.to_pickle(folder/'F01_EP.pkl')
    second={name:frame.copy() for name,frame,_ in engineered_panels(study,u)}
    for name in first:pd.testing.assert_frame_equal(first[name].iloc[:65],second[name].iloc[:65])


def test_wq_screen_excludes_failed_and_redundant_candidates():
    from usq.project2.factor_selection import select_candidates
    table=pd.DataFrame([
        {'factor':'WQ040','direction':1,'stability_rankic_signed':.03,'passes_predeclared_rules':True},
        {'factor':'WQ006','direction':1,'stability_rankic_signed':.02,'passes_predeclared_rules':True},
        {'factor':'WQ026','direction':1,'stability_rankic_signed':.01,'passes_predeclared_rules':False}])
    ranks=pd.DataFrame([[1,2,3],[3,1,2]],columns=list('ABC'))
    selected,decisions=select_candidates(table,lambda *_:ranks)
    assert selected==['WQ040']
    assert decisions[1]['accepted'] is False
    assert decisions[1]['max_abs_daily_rank_corr']>.99


def test_unconverged_cv_candidate_cannot_win():
    rows=cv_rows()
    for row in rows:row['fit_converged']=row['candidate']==0
    assert all(choice['candidate']==0 for choice in choose_parameters(rows).values())


def test_fixed_window_excludes_old_rows_and_labels_maturing_at_refit():
    index = ['20181231', '20190102', '20231213', '20231214', '20240102']
    label_end = ['20190116', '20190117', '20231229', '20240101', '20240117']
    actual = purged_dates(index, label_end, '20240101', train_start='20190101')
    np.testing.assert_array_equal(actual, [False, True, True, False, False])


def test_prediction_r2_uses_matched_pairs_and_mature_labels(tmp_path):
    from types import SimpleNamespace
    from usq.project2.model_selection import ReturnModelExperiment

    index = pd.Index(['20240102', '20250102', '20251231'])
    targets = pd.DataFrame({'A': [3., 5., 100.], 'B': [10., 4., 100.]}, index=index)
    score = pd.DataFrame({'A': [2., 4., 0.], 'B': [np.nan, 3., 0.]}, index=index)
    baseline = pd.Series([1., 2., 0.], index=index)
    experiment = SimpleNamespace(study=SimpleNamespace(targets=targets), out=tmp_path,
                                 u=targets, ends=['20240118', '20250117', ''])
    ReturnModelExperiment.save_prediction_r2(experiment, 'ridge', 'evaluation', score,
                                             baseline, '20251231')
    full = pd.read_csv(tmp_path / 'evaluation_ridge_oos_r2.csv').set_index('period').loc['Full period']
    assert full.pairs == 3
    assert full.oos_r2 == pytest.approx(1 - 3 / 17)


def test_portfolio_selection_compares_weightings_and_ignores_benchmarks_and_test():
    from usq.project2.model_selection import PORTFOLIO_CANDIDATES, select_portfolio_specification

    rows = [
        {'case': case, 'phase': 'validation', 'sharpe': 0.7, 'mean_cash': 0.01,
         'mean_daily_turnover': 0.01}
        for cases in PORTFOLIO_CANDIDATES.values() for case in cases.values()
    ]
    for row in rows:
        if row['case'] == 'ml_xgboost_cap':
            row['sharpe'] = 1.2
        if row['case'].startswith('ml_lasso'):
            row.update(sharpe=2.0, mean_cash=1.0)
    rows.extend([
        dict(rows[0], case='pool_cap', sharpe=3.0),
        dict(rows[0], phase='evaluation', sharpe=4.0),
    ])
    record = select_portfolio_specification(pd.DataFrame(rows))
    assert record['selected_case'] == 'ml_xgboost_cap'
    assert record['selected_by_model']['lasso'] is None
    assert record['selected_by_model']['xgboost'] == 'ml_xgboost_cap'
    with pytest.raises(ValueError, match='Incomplete'):
        select_portfolio_specification(pd.DataFrame(rows[1:]))
