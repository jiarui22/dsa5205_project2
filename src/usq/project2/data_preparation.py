"""Audit supplied data and create a quarantined, next-close research snapshot."""
from pathlib import Path
import json,shutil
import numpy as np
import pandas as pd
from usq.database.common import digest,save_panel
from usq.database.complete import write_checksums
from usq.database.close_labels import build_close_labels
from usq.data.labels import close_forward_returns,daily_close_returns


def material_ohlc(open_,high,low,close):
    error=np.maximum(low-np.minimum(open_,close),np.maximum(open_,close)-high)
    return (error>.01001)&(error/close.abs()>.001)


def prepare_research_data(source,work):
    source,work=Path(source),Path(work);work.mkdir(parents=True,exist_ok=True)
    out=work/'database';marker=work/'data_complete.json'
    binding={'source_checksums':digest(source/'checksums.sha256'),'implementation':digest(Path(__file__))}
    if out.exists() or marker.exists():
        raise RuntimeError('Clean-run policy: choose a new experiment directory; prior outputs cannot be reused.')
    source_count=0
    for line in (source/'checksums.sha256').read_text().splitlines():
        h,name=line.split('  ',1);assert digest(source/name)==h,name;source_count+=1
    load=lambda n:pd.read_pickle(source/'panels'/f'{n}.pkl')
    u=load('universe');dates,sids=u.index,u.columns
    assert dates.is_unique and sids.is_unique and dates.is_monotonic_increasing
    assert u.isin([0,1]).all().all()
    bars={n:load(n) for n in ['open','high','low','close']}
    bad=material_ohlc(bars['open'],bars['high'],bars['low'],bars['close'])
    bad=bad|((bars['high']<bars['low'])&bars['high'].notna()&bars['low'].notna())
    assert int(bad.sum().sum())==2,'Unreviewed source anomalies: inspect before changing the agreed quarantine policy.'
    audits=[]
    fields=json.loads((source/'metadata/snapshot.json').read_text())['stock_fields']
    for name in fields:
        f=load(name);assert f.index.equals(dates) and f.columns.equals(sids),name
        numeric=all(pd.api.types.is_numeric_dtype(x) for x in f.dtypes)
        audits.append({'field':name,'rows':len(f),'securities':len(f.columns),
            'missing_cells':int(f.isna().sum().sum()),'infinite_cells':int(np.isinf(f.to_numpy()).sum()) if numeric else 0,
            'eligible_coverage':float((f.notna()&u.eq(1)).to_numpy().sum()/u.eq(1).to_numpy().sum())})
        del f
    pd.DataFrame(audits).to_csv(work/'input_audit.csv',index=False)
    vintages=pd.read_parquet(source/'metadata/fundamental_vintages.parquet')
    available=pd.to_datetime(vintages.available_date);period=pd.to_datetime(vintages.period_end)
    assert (available>=period).all()
    vintages.groupby(['field','availability_basis']).size().rename('events').reset_index().to_csv(work/'financial_availability.csv',index=False)
    financial={'events':len(vintages),'fields':int(vintages.field.nunique()),'events_before_period_end':int((available<period).sum()),'missing_available_dates':int(available.isna().sum())}
    (work/'financial_audit.json').write_text(json.dumps(financial,indent=2))
    del vintages
    # Start with a complete independent target snapshot, never alter source files.
    build_close_labels(source,out)
    meta=json.loads((out/'metadata/snapshot.json').read_text());entries=meta['stock_fields']
    affected=['open','high','low','close','open_adj','high_adj','low_adj','close_adj','vwap','vwap_adj','vwap_local_pair','amount','amount_source_volume','volume','market_cap','market_cap_split_synced']
    for name in affected+['trade_available']:
        if name not in entries:continue
        f=pd.read_pickle(out/'panels'/f'{name}.pkl')
        f=f.mask(bad,0 if name=='trade_available' else np.nan)
        entries[name].update(save_panel(f,out/'panels'/f'{name}.pkl',dates,sids,'mask' if name=='trade_available' else 'numeric'))
    cp=pd.read_pickle(out/'panels/close_adj.pkl');seg=pd.read_pickle(out/'panels/adjustment_segment.pkl')
    target,end=close_forward_returns(cp,10,seg);returns=daily_close_returns(cp,seg)
    for name,value in [('returns',returns),('forward_10d_raw',target),('ylabel',target),('label_10',target),('label_available',target.notna().astype('uint8'))]:
        entries[name].update(save_panel(value,out/'panels'/f'{name}.pkl',dates,sids,'mask' if name=='label_available' else 'numeric'))
    end.to_csv(out/'panels/label_end.csv',index_label='date');meta['label_end_sha256']=digest(out/'panels/label_end.csv')
    meta['cleaning_policy']='Two reviewed material OHLC violations quarantined across price-derived inputs and trading availability; source files unchanged.'
    (out/'metadata/snapshot.json').write_text(json.dumps(meta,indent=2))
    coverage=float((target.notna()&u.eq(1)).loc['20150101':].to_numpy().sum()/u.loc['20150101':].eq(1).to_numpy().sum())
    for file in ['acceptance.json','label_audit.json']:
        p=out/'metadata'/file;v=json.loads(p.read_text());v['label_coverage_on_supplied_universe']=coverage;v['label_coverage_on_supplied_2015_2025_universe']=coverage;v['label_missing_pct_on_supplied_universe']=100*(1-coverage);v['material_OHLC_quarantined']=int(bad.to_numpy().sum());p.write_text(json.dumps(v,indent=2))
    write_checksums(out)
    checks={'passed':True,'source_files_verified':source_count,'calendar_sessions':len(dates),'SID_union':len(sids),'quarantined_bars':int(bad.to_numpy().sum()),'quarantined_eligible_bars':int((bad&u.eq(1)).to_numpy().sum()),'last_11_targets_censored':bool(target.tail(11).isna().all().all()),'financial_audit':financial,'binding':binding}
    assert checks['last_11_targets_censored']
    # Recheck supplied data, including original targets, after the entire operation.
    for line in (source/'checksums.sha256').read_text().splitlines():
        h,name=line.split('  ',1);assert digest(source/name)==h,name
    marker.write_text(json.dumps(checks,indent=2));return out
