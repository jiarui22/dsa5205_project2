"""Publish strategy inputs from one raw-data preparation run, without copying a legacy snapshot."""
from pathlib import Path
from datetime import datetime, timezone
import gzip
import json
import shutil
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from .common import digest, calendar, compact_dates, save_panel, save_parquet
from .complete import write_checksums, quarterly_view
from .fundamentals import expand_events
from .wrds_reconstruction import prepare_links, interval_segments, dated_join
from usq.data.labels import NEXT_CLOSE_LABEL_POLICY, close_forward_returns
from usq.data.adjustments import PRICE_PRECISION_POLICY, adjusted_bars, observed_returns


def publish_strategy_snapshot(raw, prepared, data_root):
    raw, prepared, data_root = map(Path, [raw, prepared, data_root])
    destination = data_root/'processed'/prepared.name/'research_snapshot'
    u = pd.read_parquet(prepared/'reconstructed_universe.parquet')
    required_bytes = u.size*8*40+1024**3
    if shutil.disk_usage(data_root).free < required_bytes:
        raise OSError(28,f'Insufficient free space to publish a fresh strategy snapshot; need {required_bytes/1024**3:.1f} GiB.')
    destination.mkdir(parents=True, exist_ok=False)
    panels, metadata = destination/'panels', destination/'metadata'
    panels.mkdir(); metadata.mkdir()
    dates = calendar('2014-01-01','2025-12-31')
    sids = pd.Index(u.columns, name='sid')
    u.index = pd.Index(u.index.strftime('%Y%m%d'),name='date')
    u.columns = sids
    if not u.index.equals(dates):
        raise ValueError('Reconstructed universe differs from the independent calendar')
    master = pd.read_parquet(prepared/'financial_security_registry.parquet').set_index('sid').loc[sids].reset_index()
    master['column_order'] = np.arange(len(sids))
    save_parquet(master,metadata/'security_master.parquet')
    pd.DataFrame({'date':dates}).to_csv(panels/'calendar.csv',index=False)
    entries = {}
    precise_fields = {'open', 'high', 'low', 'close', 'open_adj', 'high_adj', 'low_adj',
        'close_adj', 'adj_factor', 'volume', 'crsp_shares_out', 'crsp_permno',
        'market_cap', 'market_cap_split_synced', 'daily_return', 'returns',
        'ylabel', 'forward_10d_raw', 'label_10'}
    def put(name, frame, kind='numeric', unit='USD'):
        precision = 'float64' if name in precise_fields else 'float32'
        entries[name] = save_panel(frame,panels/f'{name}.pkl',dates,sids,kind,numeric_dtype=precision)
        entries[name].update(source='Rebuilt from raw inputs in the current preparation run',unit=unit)
    put('universe',u.astype('uint8'),'mask','0/1')

    # Read all source marks for the independently selected SID set. No reference
    # panel contributes prices, returns, identifiers, eligibility or financials.
    names = ['prc','openprc','askhi','bidlo','vol','shrout','ret']
    arrays = {n:np.full(u.shape,np.nan,dtype='float64') for n in names}
    permno = np.full(u.shape,np.nan,dtype='float32')
    observed = np.zeros(u.shape,bool); terminal = np.zeros(u.shape,bool)
    links = prepare_links(raw); links = links.loc[links.sid.isin(sids)]
    segments, conflicts = interval_segments(links,'permno','linkdt','linkenddt',['sid'])
    if len(conflicts):
        raise ValueError('Ambiguous raw security links cannot be published')
    dt = pd.to_datetime(dates,format='%Y%m%d')
    for filename in ['crsp_dsf_2014_2024.parquet','crsp_dsf_v2_2025.parquet']:
        cols = ['date','permno',*names]
        ciz = 'v2' in filename
        if ciz: cols += ['securitysubtype']
        for batch in pq.ParquetFile(next(raw.rglob(filename))).iter_batches(batch_size=150000,columns=cols):
            d = dated_join(batch.to_pandas(),segments)
            d = d.loc[d.sid.isin(sids)]
            i,j = dt.get_indexer(d.date),sids.get_indexer(d.sid)
            if d.duplicated(['date','sid']).any() or observed[i,j].any():
                raise ValueError('Duplicate raw SID/session mapping')
            observed[i,j] = True; permno[i,j] = d.permno
            for field in names: arrays[field][i,j] = d[field]
            if ciz: terminal[i,j] = d.securitysubtype.eq('UNK') | d.prc.le(0)
    # CIZ can post its final payoff on the first session after a link ends.
    v2 = pd.read_parquet(next(raw.rglob('crsp_dsf_v2_2025.parquet')))
    v2 = v2.loc[v2.securitysubtype.eq('UNK')|v2.prc.le(0)]
    events = []
    joined = v2.merge(links,on='permno',how='inner')
    endpos = dt.searchsorted(joined.linkenddt,side='right')
    nextday = np.full(len(joined),np.datetime64('NaT','ns'),dtype='datetime64[ns]')
    valid = endpos<len(dt); nextday[valid] = dt.to_numpy()[endpos[valid]]
    inside = joined.date.between(joined.linkdt,joined.linkenddt)
    joined = joined.loc[inside|joined.date.eq(pd.to_datetime(nextday))].drop_duplicates(['date','sid','permno'])
    if joined.duplicated(['date','sid']).any(): raise ValueError('Ambiguous CIZ terminal identity')
    for row in joined.itertuples():
        i,j = dt.get_loc(row.date),sids.get_loc(row.sid)
        successor = observed[i,j] and permno[i,j]!=row.permno
        if not successor:
            for field in names: arrays[field][i,j] = getattr(row,field)
            permno[i,j] = row.permno; observed[i,j] = True; terminal[i,j] = True
        events.append(dict(sid=row.sid,date=dates[i],source='CIZ embedded return',successor_preserved=bool(successor)))
    signed = arrays['prc']; close = np.abs(signed.astype(float))
    good = np.isfinite(close)&(close>0)&observed&~terminal
    close[~good] = np.nan
    daily = arrays['ret'].astype(float);daily[~np.isfinite(daily)|(daily < -1)] = np.nan
    legacy = pd.read_parquet(next(raw.rglob('crsp_dsedelist.parquet')))
    legacy = legacy.loc[legacy.dlstcd.ge(200)&pd.to_datetime(legacy.dlstdt).between('2014-01-01','2024-12-31')]
    legacy = legacy.merge(links,on='permno',how='inner')
    legacy = legacy.loc[legacy.dlstdt.between(legacy.linkdt,legacy.linkenddt)].drop_duplicates(['sid','dlstdt'])
    for row in legacy.itertuples():
        i,j = dt.searchsorted(row.dlstdt,side='right'),sids.get_loc(row.sid)
        if i>=len(dt): continue
        if observed[i,j]:
            status = 'source_observation_preserved'
        else:
            terminal[i,j] = True
            value = float(row.dlret) if pd.notna(row.dlret) and row.dlret>=-1 else np.nan
            daily[i,j] = value;permno[i,j] = row.permno
            status = 'reported_terminal_return' if np.isfinite(value) else 'unresolved_terminal_return'
        events.append(dict(sid=row.sid,date=dates[i],source='SIZ delisting ledger',status=status))
    save_parquet(pd.DataFrame(events),metadata/'terminal_events.parquet')
    segment = np.zeros(u.shape,dtype='int32');counter=np.zeros(len(sids),dtype='int32')
    for i in range(len(dates)):
        continuing = np.zeros(len(sids),bool) if i==0 else good[i]&good[i-1]&(permno[i]==permno[i-1])&np.isfinite(arrays['ret'][i])&(arrays['ret'][i]>-1)
        starts = good[i]&~continuing;counter[starts]+=1
        segment[i,good[i]]=counter[good[i]]
    def frame(value):return pd.DataFrame(value,index=dates,columns=sids)
    bars={'close':frame(close)}
    for dest,field in [('open','openprc'),('high','askhi'),('low','bidlo')]:
        value=np.abs(arrays[field].astype(float));value[(value<=0)|~np.isfinite(value)|~good]=np.nan;bars[dest]=frame(value)
    from usq.project2.data_preparation import material_ohlc
    bad=material_ohlc(bars['open'],bars['high'],bars['low'],bars['close']) | bars['high'].lt(bars['low'])
    bi,bj=np.where(bad.to_numpy())
    raw_exceptions=pd.DataFrame({'date':dates[bi],'sid':sids[bj],'quarantined':True})
    raw_exceptions.to_csv(metadata/'ohlc_quarantine.csv',index=False)
    seg=frame(segment)
    bars={field:value.mask(bad) for field,value in bars.items()}
    adjusted=adjusted_bars(bars,frame(daily),seg)
    fac=adjusted['close']/bars['close']
    put('adj_factor',fac,unit='ratio');put('adjustment_segment',seg,unit='segment ID')
    for field,value in bars.items():
        put(field,value,unit='USD/share');put(field+'_adj',adjusted[field],unit='USD/share')
    vol=arrays['vol'].astype(float);vol[(vol<0)|~good|~np.isfinite(vol)]=np.nan
    shares=arrays['shrout'].astype(float)*1000;shares[(shares<=0)|~good|~np.isfinite(shares)]=np.nan
    cap=frame(close*shares).mask(bad)
    put('volume',frame(vol).mask(bad),unit='shares');put('crsp_shares_out',frame(shares).mask(bad),unit='shares')
    put('crsp_permno',frame(permno),unit='PERMNO');put('market_cap',cap);put('market_cap_split_synced',cap)
    put('daily_return',frame(daily),unit='decimal return');put('delisting_event',frame(terminal.astype('uint8')),'mask','0/1')
    trade=frame(good&(signed>0)&(vol>0)&~terminal).mask(bad,False)
    put('trade_available',trade.astype('uint8'),'mask','0/1')
    cp=pd.read_pickle(panels/'close_adj.pkl');segs=pd.read_pickle(panels/'adjustment_segment.pkl')
    target,end=close_forward_returns(cp,10,segs)
    put('returns',observed_returns(cp,frame(daily),segs),unit='decimal return')
    for field in ['ylabel','forward_10d_raw','label_10']:put(field,target,unit='decimal ten-session return')
    put('label_available',target.notna().astype('uint8'),'mask','0/1');end.to_csv(panels/'label_end.csv',index_label='date')
    (panels/'label_metadata.json').write_text(json.dumps(NEXT_CLOSE_LABEL_POLICY,indent=2))

    financial=pd.read_parquet(prepared/'fundamental_events.parquet');financial=financial.loc[financial.sid.isin(sids)].copy()
    fields=financial.field.unique().tolist()
    for field in fields:put(field,expand_events(financial.loc[financial.field.eq(field)],dates,sids))
    financial['period_end']=compact_dates(financial.period_end);financial['available_date']=compact_dates(financial.available_date)
    financial['source']='Compustat North America quarterly raw archive'
    save_parquet(financial,metadata/'fundamental_vintages.parquet')
    quarter=pd.read_parquet(prepared/'fundamental_quarterly_records.parquet');quarter=quarter.loc[quarter.sid.isin(sids)]
    qcols=[c for c in quarter if not c.endswith('_available_day')]
    known,unavailable=quarterly_view(quarter,qcols,fields,dates[-1])
    save_parquet(known,metadata/'fundamental_quarterly.parquet');save_parquet(unavailable,metadata/'financial_records_unavailable.parquet')
    industry=pd.read_parquet(prepared/'industry_events.parquet');industry=industry.loc[industry.sid.isin(sids)]
    save_parquet(industry,metadata/'industry_history.parquet')
    groups={sid:g.sort_values('available_date') for sid,g in industry.groupby('sid')};columns={}
    for sid in sids:
        values=np.full(len(dates),None,dtype=object)
        if sid in groups:
            g=groups[sid];pos=np.searchsorted(g.available_date.to_numpy(),dt.to_numpy(),side='right')-1;ok=pos>=0;values[ok]=g.value.to_numpy()[pos[ok]]
        columns[sid]=pd.Categorical(values)
    put('industry',pd.DataFrame(columns,index=dates).rename_axis(columns='sid'),'industry','SIC2')
    put('free_cash_flow_ttm',pd.read_pickle(panels/'operating_cash_flow_ttm.pkl')-pd.read_pickle(panels/'capital_expenditures_ttm.pkl'))
    benchmark=data_root/'raw/benchmark/SPY.gz'
    source=json.loads(gzip.decompress(benchmark.read_bytes()))['chart']['result'][0]
    index=pd.to_datetime(source['timestamp'],unit='s',utc=True).tz_convert('America/New_York').tz_localize(None).normalize()
    quote=pd.DataFrame(source['indicators']['quote'][0],index=index)
    quote['adjclose']=source['indicators']['adjclose'][0]['adjclose'];quote=quote.loc[~quote.index.duplicated(keep='last')].sort_index()
    index_fields={}
    for field,series in [('close',quote.close),('price_close',quote.close),('total_return_close',quote.adjclose),('total_return_open',quote.open*quote.adjclose/quote.close)]:
        series=series.reindex(dt)
        if series.isna().any():raise ValueError('Raw SPY response does not cover every research session')
        value=pd.DataFrame({'SPY':series.to_numpy()},index=dates).astype('float32');value.to_pickle(panels/f'indices_{field}.pkl')
        index_fields[field]={'columns':['SPY'],'sha256':digest(panels/f'indices_{field}.pkl'),'shape':list(value.shape)}
    audit={'pipeline_complete':True,'producer_run':prepared.name,'calendar_sessions':len(dates),'SID_union':len(sids),
        'price_precision_policy':PRICE_PRECISION_POLICY,'price_arithmetic_dtype':'float64',
        'adjustment_implementation_sha256':digest(Path(__file__).parents[1]/'data/adjustments.py'),
        'quarantined_bars':int(bad.to_numpy().sum()),'quarantined_eligible_bars':int((bad&u.eq(1)).to_numpy().sum()),
        'last_11_targets_censored':bool(target.tail(11).isna().all().all()),
        'construction_uses_reference_snapshot':False,'sina_used_for_strategy':False,
        'builder_sha256':digest(Path(__file__))}
    manifest={'schema_version':2,'status':'raw_reconstructed_next_close_snapshot','research_accepted':False,
        'created_utc':datetime.now(timezone.utc).isoformat(),'stock_fields':entries,'fundamental_fields':fields,
        'label_policy':NEXT_CLOSE_LABEL_POLICY,'preprocessing':audit,'index_fields':index_fields,
        'raw_SPY_sha256':digest(benchmark),
        'limitations':['Latest Compustat values are not a historical revisions archive.','FIC and PRIUSA are static source descriptors.','Missing terminal payoffs remain explicitly unresolved.']}
    (metadata/'snapshot.json').write_text(json.dumps(manifest,indent=2))
    (metadata/'preprocessing_complete.json').write_text(json.dumps(audit,indent=2))
    write_checksums(destination)
    return destination


def activate_snapshot(data_root,snapshot,producer):
    data_root,snapshot,producer=map(Path,[data_root,snapshot,producer])
    if not json.loads((producer/'complete.json').read_text()).get('complete'):
        raise ValueError('Only a completed data notebook run may activate its snapshot')
    activation=snapshot/'metadata/activation.json'
    if activation.exists():
        raise FileExistsError('Snapshot already activated; start a new preparation run.')
    activation.write_text(json.dumps({'producer_complete':True,'producer_run':producer.name},indent=2))
    write_checksums(snapshot)
    record={'snapshot':str(snapshot.relative_to(data_root)),'producer':str(producer.resolve().relative_to(data_root.resolve().parent)),
            'checksums_sha256':digest(snapshot/'checksums.sha256')}
    pointer=data_root/'processed/latest_snapshot.json';temp=pointer.with_suffix('.tmp')
    temp.write_text(json.dumps(record,indent=2));temp.replace(pointer)
