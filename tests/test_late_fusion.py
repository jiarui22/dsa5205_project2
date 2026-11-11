import numpy as np
import pandas as pd
import pytest
from usq.project2.late_fusion import fuse_scores,select_lambda


def test_fusion_uses_common_percentiles_and_preserves_zero_lambda_order():
    f=pd.DataFrame([[10.,20.,20.,40.],[4.,3.,2.,1.]])
    p=pd.DataFrame([[4.,2.,3.,1.],[10.,20.,30.,np.nan]])
    common=np.isfinite(f)&np.isfinite(p)
    pd.testing.assert_frame_equal(fuse_scores(f,p,0),f.where(common).rank(axis=1,pct=True))
    expected=.75*f.where(common).rank(axis=1,pct=True)+.25*p.where(common).rank(axis=1,pct=True)
    pd.testing.assert_frame_equal(fuse_scores(f,p,.25),expected.where(common))
    pd.testing.assert_frame_equal(fuse_scores(f*100,p+1e6,.25),fuse_scores(f,p,.25))
    changed=p.copy();changed.iloc[1]*=100
    pd.testing.assert_series_equal(fuse_scores(f,p,.5).iloc[0],fuse_scores(f,changed,.5).iloc[0])
    with pytest.raises(ValueError):fuse_scores(f,p,1.1)
    with pytest.raises(ValueError):fuse_scores(f,p.rename(columns={0:'different'}),.5)


def test_lambda_selection_is_worst_case_validation_only_and_requires_complete_grid():
    rows=[]
    for phase in ['validation','evaluation']:
        for weight in [0.,.25,.5]:
            for lag in [0,5]:
                for cost in [10,20,30]:
                    ir=({0.:.2,.25:.3,.5:.8}[weight] if phase=='validation' else 100*weight)
                    if weight==.5 and phase=='validation' and cost==30:ir=-.5
                    rows.append({'kind':'strategy','phase':phase,'lambda':weight,'lag':lag,'cost_bps':cost,'net_active_IR_vs_pool':ir,'mean_cash':.01,'annual_return':.1})
    table=pd.DataFrame(rows);choice,_=select_lambda(table);assert choice==.25
    table.loc[table.phase.eq('evaluation'),'net_active_IR_vs_pool']=-1e9
    assert select_lambda(table)[0]==.25
    with pytest.raises(ValueError):select_lambda(table.drop(index=0))
    table.loc[(table.phase=='validation')&(table['lambda']==.25),'mean_cash']=.2
    assert select_lambda(table)[0]==0
