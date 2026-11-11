import json
import numpy as np
import pandas as pd
import pytest

from usq.database.raw_preparation import audit_market_sources, audit_published_prices
from usq.data.adjustments import canonical_close, PRICE_PRECISION_POLICY
from usq.data.labels import close_forward_returns


def test_raw_key_audit_detects_duplicates_across_batch_boundaries(tmp_path):
    raw=tmp_path/'raw';raw.mkdir();out=tmp_path/'out';out.mkdir()
    rows=150001
    frame=pd.DataFrame({'date':pd.Timestamp('2020-01-02'),
        'permno':np.arange(1,rows+1),'prc':10.,'openprc':10.,'askhi':11.,'bidlo':9.,'vol':100.})
    frame.loc[rows-1,'permno']=1
    frame.to_parquet(raw/'crsp_dsf_2014_2024.parquet',index=False)
    frame.iloc[:1].assign(date=pd.Timestamp('2025-01-02')).to_parquet(raw/'crsp_dsf_v2_2025.parquet',index=False)
    with pytest.raises(ValueError,match='Duplicate raw'):
        audit_market_sources(raw,out)
    assert pd.read_csv(out/'market_quality.csv').duplicate_keys.sum()==1


def test_published_price_audit_rejects_label_drift(tmp_path):
    root=tmp_path/'snapshot';(root/'panels').mkdir(parents=True);(root/'metadata').mkdir()
    dates=pd.Index(pd.bdate_range('2020-01-02',periods=30).strftime('%Y%m%d'),name='date')
    p=pd.DataFrame({'A':100.+np.arange(30)},index=dates)
    returns=p.pct_change();seg=p*0+1
    adjusted=canonical_close(p,returns,seg);target,_=close_forward_returns(adjusted,10,seg)
    for name,value in dict(close=p,daily_return=returns,adjustment_segment=seg,
            close_adj=adjusted,ylabel=target,universe=seg).items():
        value.to_pickle(root/'panels'/f'{name}.pkl')
    (root/'metadata/preprocessing_complete.json').write_text(json.dumps({'price_precision_policy':PRICE_PRECISION_POLICY}))
    _,coverage=audit_published_prices(root)
    assert coverage.finite_targets.iloc[-11:].eq(0).all()
    target.iloc[0,0]+=1e-6;target.to_pickle(root/'panels/ylabel.pkl')
    with pytest.raises(AssertionError):audit_published_prices(root)
