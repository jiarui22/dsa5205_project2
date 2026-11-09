import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from usq.database.common import calendar, save_panel, save_parquet, dump
from usq.database.fundamentals import normalize_records, feature_events, expand_events, primary_registry, CORE
from usq.database.market import sina_frame, yahoo_frame, split_multiplier
from usq.database.reader import ResearchDatabase


def source_rows():
    dates=pd.date_range('2020-03-31',periods=8,freq='QE')
    d=pd.DataFrame({'gvkey':['000001']*8,'datadate':dates.strftime('%Y-%m-%d'),
        'period_end':dates,'fyearq':dates.year,'fqtr':[(i%4)+1 for i in range(8)],'fyr':12,
        'rdq':(dates+pd.Timedelta(days=30)).strftime('%Y-%m-%d'),
        'fdateq':(dates+pd.Timedelta(days=50)).strftime('%Y-%m-%d'),
        'source_row':np.arange(8),'niq':np.arange(1,9,dtype=float),'revtq':np.arange(11,19,dtype=float),
        'saleq':np.arange(11,19,dtype=float),'revenue_q':np.arange(11,19,dtype=float),
        'oancfy':[10,30,60,100,20,60,120,200],'capxy':[1,3,6,10,2,6,12,20],
        'cshoq':2.,'ceqq':5.,'atq':10.,'ltq':5.,'cheq':1.,'dlttq':1.,'dlcq':1.,
        'priusa':'01','conm':'EXAMPLE INC','conml':'Example Inc.','tic':'EXM','cusip':'123456789',
        'cik':'123','exchg':'14','fic':'USA','sic':'3500','naics':'333000','ipodate':'2010-01-01',
        'dldte':None,'costat':'A','source_vintage':'test_hash'})
    return d


def test_independent_calendar_special_closures_and_full_range():
    dates=calendar('2014-01-01','2026-01-30')
    assert len(dates)==3038 and dates[0]=='20140102' and dates[-1]=='20260130'
    assert '20181205' not in dates and '20250109' not in dates
    assert '20210618' in dates and '20220620' not in dates


def test_quarterly_and_ytd_are_not_summed_twice():
    dates=calendar('2019-01-01','2023-12-31')
    _,f=normalize_records(source_rows(),dates)
    assert f.loc[3,'earnings_ttm']==pytest.approx(10e6)
    assert f.loc[3,'revenue_ttm']==pytest.approx(50e6)
    assert f.loc[3,'operating_cash_flow_ttm']==pytest.approx(100e6)
    assert f.loc[4,'operating_cash_flow_q']==pytest.approx(20e6)
    assert f.loc[7,'operating_cash_flow_ttm']==pytest.approx(200e6)
    assert f.loc[:2,'earnings_ttm'].isna().all()


def test_no_announcement_backfill_and_old_release_cannot_replace_new_period():
    dates=pd.Index(['20200102','20200103','20200106','20200107'],name='date')
    sids=pd.Index(['X'],name='sid')
    e=pd.DataFrame({'sid':['X'],'available_date':pd.to_datetime(['2020-01-06']),
                    'period_end':pd.to_datetime(['2019-12-31']),'value':[3.]})
    panel=expand_events(e,dates,sids)
    assert panel.iloc[:2].isna().all().all() and panel.iloc[2,0]==3
    r=pd.DataFrame({'sid':['X','X'],'period_end':pd.to_datetime(['2019-12-31','2019-09-30']),
        'source_row':[1,2],'availability_basis':['rdq','rdq'],
        'available_day':[(pd.Timestamp('2020-01-03')-pd.Timestamp('1970-01-01')).days,
                         (pd.Timestamp('2020-01-07')-pd.Timestamp('1970-01-01')).days],
        'shares_out':[10.,1.]})
    events=feature_events(r,'shares_out')
    assert len(events)==1 and events.value.iloc[0]==10


def test_duplicate_fiscal_period_is_quarantined_not_first_row_selected():
    d=source_rows(); duplicate=d.iloc[[0]].copy();duplicate['source_row']=99;duplicate['niq']=999
    raw,good=normalize_records(pd.concat([d,duplicate],ignore_index=True),calendar('2019-01-01','2023-12-31'))
    assert raw.period_conflict.sum()==2
    assert pd.Timestamp('2020-03-31') not in set(good.period_end)
    assert good.loc[good.datadate.eq('2020-12-31'),'earnings_ttm'].isna().all()


def test_ttm_waits_for_late_constituent_and_fiscal_gap_breaks_window():
    d=source_rows();d.loc[2,'rdq']='2021-02-15'
    _,f=normalize_records(d,calendar('2019-01-01','2023-12-31'))
    assert pd.to_datetime(f.loc[3,'earnings_ttm_available_day'],unit='D')==pd.Timestamp('2021-02-16')
    d=source_rows().drop(index=2).reset_index(drop=True)
    _,f=normalize_records(d,calendar('2019-01-01','2023-12-31'))
    assert f.loc[f.datadate.eq('2020-12-31'),'earnings_ttm'].isna().all()


def test_raw_amount_zero_missing_and_real_vwap():
    raw=b'var=([{"d":"2020-01-02","o":"10","h":"12","l":"9","c":"11","v":"100","a":"1050"},{"d":"2020-01-03","o":"10","h":"12","l":"9","c":"11","v":"100","a":"0"}]);'
    d=sina_frame(raw)
    assert d.loc['20200102','vwap']==10.5
    assert pd.isna(d.loc['20200103','amount']) and pd.isna(d.loc['20200103','vwap'])


def test_split_synchronized_market_cap_no_artificial_split_drop():
    events={'splits':{'1':{'date':int(pd.Timestamp('2020-08-31 13:30',tz='UTC').timestamp()),'numerator':4,'denominator':1}}}
    multiplier=split_multiplier(events,['20200828','20200831'])
    shares=4e9;prices=np.array([500.,125.])
    cap=shares*prices*multiplier
    np.testing.assert_allclose(cap,[2e12,2e12])


def test_reader_validates_complete_axes_and_indices_separately(tmp_path):
    (tmp_path/'panels').mkdir();(tmp_path/'metadata').mkdir()
    dates=pd.Index(['20200102','20200103'],name='date');sids=pd.Index(['CNA:000001:01'],name='sid')
    pd.DataFrame({'date':dates}).to_csv(tmp_path/'panels/calendar.csv',index=False)
    save_parquet(pd.DataFrame({'sid':sids,'column_order':[0]}),tmp_path/'metadata/security_master.parquet')
    x=pd.DataFrame([[1.],[2.]],index=dates,columns=sids)
    entry=save_panel(x,tmp_path/'panels/close.pkl',dates,sids)
    indices=pd.DataFrame([[100.],[101.]],index=dates,columns=['^GSPC'])
    indices.to_pickle(tmp_path/'panels/indices_close.pkl')
    dump({'research_accepted':False,'stock_fields':{'close':entry},'fundamental_fields':[],
          'index_fields':{'close':{'columns':['^GSPC']}}},tmp_path/'metadata/snapshot.json')
    db=ResearchDatabase(tmp_path)
    assert db.load('close','2020-01-03','2020-01-03').shape==(1,1)
    assert db.indices().columns.tolist()==['^GSPC']
    assert db.load('close').dtypes.iloc[0]==np.dtype('float32')
    with pytest.raises(ValueError):ResearchDatabase(tmp_path,require_accepted=True)
    with pytest.raises(KeyError):db.load('close',sids=['UNKNOWN'])
    x.iloc[:1].to_pickle(tmp_path/'panels/close.pkl')
    with pytest.raises(ValueError):db.load('close',start='20200102',end='20200102')


def test_universe_cannot_have_unknown_and_axis_intersection_forbidden(tmp_path):
    dates=pd.Index(['20200102'],name='date');sids=pd.Index(['X'],name='sid')
    with pytest.raises(ValueError):save_panel(pd.DataFrame([[np.nan]],index=dates,columns=sids),tmp_path/'u.pkl',dates,sids,'mask')
    with pytest.raises(ValueError):save_panel(pd.DataFrame([[1]],index=dates,columns=['Y']),tmp_path/'x.pkl',dates,sids)


def test_long_history_keeps_actual_pre_panel_availability():
    d=source_rows()
    _,f=normalize_records(d,calendar('2021-01-01','2023-12-31'))
    assert f.loc[0,'available_ts']==pd.Timestamp('2020-05-01')


def test_pool_does_not_use_next_year_announcement_to_qualify_current_year():
    d=source_rows()
    _,f=normalize_records(d,calendar('2019-01-01','2023-12-31'))
    master,_,_=primary_registry(d,f,research_start='2020-01-01',research_end='2020-12-31')
    assert len(master)==0  # First complete fiscal four-quarter TTM is released in 2021.


def test_missing_constituent_date_does_not_make_ttm_known():
    d=source_rows();d.loc[2,['rdq','fdateq']]=None
    _,f=normalize_records(d,calendar('2019-01-01','2023-12-31'))
    f['sid']='X'
    events=feature_events(f,'earnings_ttm')
    assert not events.period_end.eq(pd.Timestamp('2020-12-31')).any()


def test_industry_forward_only_not_current_classification_backcast():
    from usq.database.industry import expand_industry
    dates=pd.Index(['20200102','20200103','20200106'],name='date')
    sids=pd.Index(['X'],name='sid')
    records=pd.DataFrame({'sid':['X'],'period_end':pd.to_datetime(['2019-12-31']),
        'available_date':pd.to_datetime(['2020-01-03']),'value':['SIC2:35']})
    frame=expand_industry(records,dates,sids)
    assert pd.isna(frame.iloc[0,0]) and frame.iloc[1,0]=='SIC2:35'
    assert isinstance(frame.dtypes.iloc[0],pd.CategoricalDtype)


def test_raw_market_cache_hash_is_checked(tmp_path):
    import gzip
    from usq.database.market import read_cached_raw
    from usq.database.common import digest
    p=tmp_path/'raw.gz';p.write_bytes(gzip.compress(b'original',mtime=0))
    dump({'sha256':digest(p)},p.with_suffix('.json'))
    assert read_cached_raw(p)==b'original'
    p.write_bytes(gzip.compress(b'modified',mtime=0))
    with pytest.raises(ValueError):read_cached_raw(p)
