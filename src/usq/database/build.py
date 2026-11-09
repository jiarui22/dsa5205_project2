from __future__ import annotations

from pathlib import Path
from urllib.parse import quote
import gzip
import json
import os
import platform
import shutil
from importlib.metadata import version
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from .common import calendar, digest, dump, save_panel, save_parquet, compact_dates
from .fundamentals import CORE, STOCK_MAP, feature_events, expand_events
from .market import cached_stock, yahoo_frame, split_multiplier, read_cached_raw, UPSTREAM
from .industry import industry_events, expand_industry

FUNDAMENTAL_FIELDS=list(STOCK_MAP)+['earnings_ttm','revenue_ttm','operating_cash_flow_q',
    'operating_cash_flow_ttm','capital_expenditures_q','capital_expenditures_ttm']
MARKET_FIELDS=['open','high','low','close','volume','amount','vwap','adj_factor','daily_return','market_cap']


def _empty(dates,sids):
    return pd.DataFrame(np.full((len(dates),len(sids)),np.nan,dtype='float32'),index=dates,columns=sids)


def _known_periods(events, day):
    if events.empty:return np.full(len(day),np.datetime64('NaT'),dtype='datetime64[ns]')
    group=events.sort_values('available_date')
    pos=np.searchsorted(group.available_date.to_numpy(),day,side='right')-1
    out=np.full(len(day),np.datetime64('NaT'),dtype='datetime64[ns]')
    valid=pos>=0;out[valid]=group.period_end.to_numpy()[pos[valid]]
    return out


def _write_vintages(features,path):
    writer=None;temp=path.with_suffix('.tmp.parquet')
    try:
        for field in FUNDAMENTAL_FIELDS:
            events=feature_events(features,field)
            events['field']=field
            events['source']='Compustat North America fundq'
            events['source_vintage']=features.source_vintage.iloc[0]
            events['period_end']=compact_dates(events.period_end)
            events['available_date']=compact_dates(events.available_date)
            cols=['sid','period_end','available_date','field','value','source','source_vintage','availability_basis','source_row']
            table=pa.Table.from_pandas(events[cols],preserve_index=False)
            if writer is None:writer=pq.ParquetWriter(temp,table.schema,compression='zstd')
            writer.write_table(table)
            print(json.dumps({'stage':'financial_long_table','field':field,'rows':len(events)}),flush=True)
    finally:
        if writer is not None:writer.close()
    temp.replace(path)


def build_snapshot(cfg):
    out=Path(cfg['output']);stage=Path(cfg['stage_dir'])
    if (out/'metadata'/'snapshot.json').exists():
        raise FileExistsError('Snapshot already exists. Use a NEW output directory; no silent overwrite.')
    dates=calendar(cfg['calendar_start'],cfg['calendar_end'])
    master=pd.read_parquet(stage/'security_candidates.parquet').sort_values('column_order')
    features=pd.read_parquet(stage/'financial_features.parquet')
    # Refuse inconsistent caches/pools instead of shrinking their intersection.
    if set(features.sid)!=set(master.sid):raise ValueError('Feature security set differs from frozen master')
    sids=pd.Index(master.sid.astype(str),name='sid')
    day=pd.to_datetime(dates,format='%Y%m%d').to_numpy()
    if shutil.disk_usage(out.parent).free<8*1024**3:raise OSError('Reserve at least 8 GiB before building')
    panels=out/'panels';metadata=out/'metadata';panels.mkdir(parents=True,exist_ok=True);metadata.mkdir(exist_ok=True)
    pd.DataFrame({'date':dates}).to_csv(panels/'calendar.csv',index=False)
    save_parquet(master,metadata/'security_master.parquet')
    dump({'scope':'Frozen source-primary-issue candidate union; NOT historically verified membership',
          'count':len(master),'sid_order_sha256':__import__('hashlib').sha256('\n'.join(sids).encode()).hexdigest(),
          'rule':'At least one usable six-core-feature record in the target interval; membership is evaluated forward, not by future survival.',
          'source_id':'CNA:GVKEY:PRIUSA','historical_identity_verified':False},metadata/'frozen_pool.json')
    _write_vintages(features,metadata/'fundamental_vintages.parquet')
    entries={};core_mask=np.ones((len(dates),len(sids)),dtype=bool)
    for field in FUNDAMENTAL_FIELDS:
        events=feature_events(features,field)
        frame=expand_events(events,dates,sids,max_age_days=cfg.get('max_financial_age_days',550))
        if field in CORE:
            valid=frame.notna().to_numpy()
            if field in ['shares_out','total_assets']:valid &= frame.to_numpy()>0
            if field=='total_liabilities':valid &= frame.to_numpy()>=0
            core_mask &= valid
        entries[field]=save_panel(frame,panels/(field+'.pkl'),dates,sids)
        print(json.dumps({'stage':'panel','field':field,'shape':list(frame.shape),'coverage':entries[field]['coverage']}),flush=True)
        del frame
    # Candidate daily membership uses only released information plus source IPO/delist bounds.
    # Security-type and historical ticker mapping remain NOT VERIFIED, explicitly below.
    for i,row in enumerate(master.itertuples()):
        if pd.notna(row.ipodate):core_mask[:,i] &= dates>=pd.Timestamp(row.ipodate).strftime('%Y%m%d')
        if pd.notna(row.dldte):core_mask[:,i] &= dates<pd.Timestamp(row.dldte).strftime('%Y%m%d')
    mask=pd.DataFrame(core_mask.astype('uint8'),index=dates,columns=sids)
    entries['universe']=save_panel(mask,panels/'universe.pkl',dates,sids,'mask')
    sic_events,industry_provenance=industry_events(cfg,master,dates)
    save_parquet(sic_events,metadata/'industry_history.parquet')
    industry=expand_industry(sic_events,dates,sids)
    entries['industry']=save_panel(industry,panels/'industry.pkl',dates,sids,'industry');del industry
    dump(industry_provenance,metadata/'industry_source.json')
    # A per-security cache is small: populate full-width fields without retaining ten
    # complete stock matrices in RAM simultaneously.
    source=out/'raw'/'market';market_cache={};market_notes=[]
    ambiguous_tickers=set(master.loc[master.tic.duplicated(keep=False)&master.tic.notna(),'tic'])
    for row in master.itertuples():
        if row.tic in ambiguous_tickers:
            market_notes.append({'sid':row.sid,'ticker':row.tic,'quarantined':'ticker assigned to multiple issuer IDs; require explicit historical mapping'})
            continue
        data,yahoo,meta,events=cached_stock(str(row.tic),source)
        if data.empty:continue
        data=data.loc[(data.index>=dates[0])&(data.index<=dates[-1])].copy()
        off_calendar=data.index.difference(dates)
        if len(off_calendar):
            market_notes.append({'sid':row.sid,'ticker':row.tic,'off_calendar_dates':off_calendar.tolist()})
            # Explicit quarantine of invalid sessions, not a hidden global-axis intersection.
            data=data.loc[~data.index.isin(off_calendar)]
        if pd.notna(row.ipodate):data=data.loc[data.index>=pd.Timestamp(row.ipodate).strftime('%Y%m%d')]
        if pd.notna(row.dldte):data=data.loc[data.index<pd.Timestamp(row.dldte).strftime('%Y%m%d')]
        aligned=data.reindex(dates)
        aligned['adj_factor']=np.nan
        if not yahoo.empty:
            adj=yahoo.adjusted_close.reindex(dates)
            aligned['adj_factor']=adj/aligned.close.where(aligned.close.gt(0))
        # No fill_method: a missing trade/price must not create a zero daily return.
        aligned['daily_return']=(aligned.close*aligned.adj_factor).pct_change(fill_method=None)
        market_cache[row.sid]=(aligned,events)
        market_notes.append({'sid':row.sid,'ticker':row.tic,'source_rows':int(aligned.close.notna().sum()),
                'yahoo_instrument':meta.get('instrumentType'),'split_events_verified':bool(meta),
                'security_identity_status':'source-primary candidate; historical mapping not independently verified'})
    shares=pd.read_pickle(panels/'shares_out.pkl')
    shares_events=feature_events(features,'shares_out')
    share_periods={sid:g for sid,g in shares_events.groupby('sid')}
    for field in MARKET_FIELDS:
        frame=_empty(dates,sids)
        for sid,(data,events) in market_cache.items():
            if field=='market_cap':
                shares_series=shares[sid].to_numpy(dtype='float64')
                known=_known_periods(share_periods.get(sid,pd.DataFrame()),day)
                period_keys=pd.DatetimeIndex(known).strftime('%Y%m%d').fillna('').to_numpy()
                current_mult=split_multiplier(events,dates.to_numpy())
                old_mult=split_multiplier(events,period_keys)
                # CSHOQ already represents ADR/ADS equivalents per vendor definition;
                # never apply the ADR ratio a second time.
                scaled=shares_series*current_mult/old_mult
                frame[sid]=(data.close.to_numpy(dtype='float64')*scaled).astype('float32')
            else:frame[sid]=data[field].to_numpy(dtype='float32')
        entries[field]=save_panel(frame,panels/(field+'.pkl'),dates,sids)
        print(json.dumps({'stage':'panel','field':field,'covered_securities':len(market_cache),'coverage':entries[field]['coverage']}),flush=True)
        del frame
    del shares
    dump(market_notes,metadata/'market_normalization.json')
    index_entries={};cols=pd.Index(cfg.get('indices',[]),name='index_symbol')
    index_frame=pd.DataFrame(np.nan,index=dates,columns=cols,dtype='float32')
    for symbol in cols:
        key=quote(symbol,safe='').replace('%','_');path=source/'yahoo'/(key+'.gz')
        if not path.exists():continue
        frame,meta,events=yahoo_frame(read_cached_raw(path))
        if meta.get('instrumentType')!='INDEX':raise ValueError('Non-index in index input')
        index_frame[symbol]=frame.close_split_adjusted.reindex(dates).astype('float32')
    p=panels/'indices_close.pkl';index_frame.to_pickle(p,protocol=4)
    index_entries['close']={'sha256':digest(p),'shape':list(index_frame.shape),'columns':cols.tolist(),
                            'source':'Yahoo price-index close, not total-return index; VIX is a volatility index'}
    source_meta=json.loads((stage/'quarterly_selected.json').read_text())
    raw_manifest=[]
    for raw_path in sorted(source.rglob('*.gz')):
        sidecar=json.loads(raw_path.with_suffix('.json').read_text())
        if digest(raw_path)!=sidecar['sha256']:raise ValueError('Modified raw market cache; refuse to build')
        raw_manifest.append({'path':str(raw_path.relative_to(out)),'sha256':digest(raw_path),
            'bytes':raw_path.stat().st_size,**sidecar})
    dump(raw_manifest,metadata/'raw_market_manifest.json')
    snapshot={'schema_version':1,'status':'prototype_incomplete','research_accepted':False,
       'research_start':cfg['research_start'],'research_end':cfg['research_end'],
       'calendar_start':dates[0],'calendar_end':dates[-1],'calendar_sessions':len(dates),
       'security_candidates':len(sids),'stocks_with_market_sample':len(market_cache),
       'tables':{name:{'sha256':digest(metadata/name),'rows':pq.read_metadata(metadata/name).num_rows}
           for name in ['security_master.parquet','fundamental_vintages.parquet','industry_history.parquet']},
       'stock_fields':entries,'fundamental_fields':FUNDAMENTAL_FIELDS,'index_fields':index_entries,
       'calendar':{'library':'exchange_calendars','version':version('exchange_calendars'),'calendar':'XNYS',
          'independent_from_stock_quotes':True,'sha256':digest(panels/'calendar.csv'),
          'references':['https://www.nyse.com/markets/hours-calendars','https://github.com/gerrymanoim/exchange_calendars']},
       'source':source_meta,'upstream_market_project':UPSTREAM,
       'raw_market_manifest_sha256':digest(metadata/'raw_market_manifest.json'),
       'fundamentals_policy':{'vintage':'latest downloaded snapshot; historical revisions unavailable; NOT strict PIT',
           'announcement':'valid rdq, otherwise valid fdateq fallback; first strictly following exchange session',
           'ttm':'four consecutive fiscal quarters, with every constituent disclosure available',
           'ffill':'forward only; no future backfill; reject stale period age > configured limit',
           'max_period_age_days':cfg.get('max_financial_age_days',550)},
       'universe_policy':'PROVISIONAL: source primary US issue + six released core features + source IPO/delist bounds; historical type/ID mapping unverified. No filtering by future price completeness.',
       'market_cap_policy':'USD estimate: last disclosed company CSHOQ*1e6 times raw primary-issue close; synchronize known split events after share-count period. CSHOQ already ADR-adjusted. Multi-class/ADR equivalence still requires validation.',
       'adjustment_policy':'adj_factor = Yahoo adjusted_close / Sina raw close, where both exist; dividend+split adjustment proxy; no invented factor=1 for unavailable delisted histories',
       'industry_policy':'Historical annual SICH aggregated to SIC2, released-final-date proxy then forward fill; NO backcast of current GICS',
       'environment':{'python':platform.python_version(),'platform':platform.platform(),
          'packages':{x:version(x) for x in ['numpy','pandas','pyarrow','exchange_calendars','requests']}}}
    dump(snapshot,metadata/'snapshot.json')
    rows=[]
    for name in entries:
        if name in FUNDAMENTAL_FIELDS:
            source_name='Compustat fundq';unit='shares' if name=='shares_out' else 'USD';notes='latest-download release proxy, forward fill; not historical PIT'
        elif name=='industry':source_name='Compustat annual historical SICH';unit='SIC2 string';notes=snapshot['industry_policy']
        elif name=='universe':source_name='forward eligibility rule';unit='0/1 uint8';notes=snapshot['universe_policy']
        else:
            source_name='Sina daily OHLCV/amount; Yahoo adjustment where available'
            unit={'volume':'shares','amount':'USD','market_cap':'USD','adj_factor':'ratio','daily_return':'decimal daily return'}.get(name,'USD/share')
            notes=snapshot['market_cap_policy'] if name=='market_cap' else snapshot['adjustment_policy'] if name in ['adj_factor','daily_return'] else 'source zero amount treated as missing; vwap=amount/volume, never OHLC approximation'
        rows.append({'field':name,'source':source_name,'unit':unit,'notes':notes})
    pd.DataFrame(rows).to_csv(metadata/'data_dictionary.csv',index=False)
    report=audit_snapshot(out)
    dump(report,metadata/'acceptance.json')
    print(json.dumps({'stage':'complete_prototype','root':str(out),'research_accepted':report['passed'],
                      'candidate_columns':len(sids),'market_sample_securities':len(market_cache)}),flush=True)
    return report


def audit_snapshot(root):
    from .reader import ResearchDatabase
    db=ResearchDatabase(root);root=Path(root)
    import pyarrow.compute as pc
    for name,entry in db.meta.get('tables',{}).items():
        table_path=root/'metadata'/name
        if digest(table_path)!=entry['sha256']:raise ValueError(f'Table checksum mismatch: {name}')
        ids=pc.unique(pq.read_table(table_path,columns=['sid'])['sid']).to_pylist()
        if not set(ids).issubset(set(db.sids)):raise ValueError(f'Company/security outside frozen pool: {name}')
    if digest(root/'panels'/'calendar.csv')!=db.meta['calendar']['sha256']:
        raise ValueError('Calendar checksum mismatch')
    manifest_path=root/'metadata'/'raw_market_manifest.json'
    if manifest_path.exists():
        if digest(manifest_path)!=db.meta['raw_market_manifest_sha256']:
            raise ValueError('Raw market manifest checksum mismatch')
        for entry in json.loads(manifest_path.read_text()):
            if digest(root/entry['path'])!=entry['sha256']:
                raise ValueError('Raw market file checksum mismatch')
    universe=db.load('universe');mask=universe.eq(1)
    if not universe.isin([0,1]).all().all():raise ValueError('Unknown universe state')
    research=(universe.index>=pd.Timestamp(db.meta['research_start']).strftime('%Y%m%d')) & (universe.index<=pd.Timestamp(db.meta['research_end']).strftime('%Y%m%d'))
    rows={};axes=True
    for name in db.meta['stock_fields']:
        frame=db.load(name,verify_hash=True)  # Whole axes checked BEFORE any subset.
        bad=frame.isna()
        if name in ['open','high','low','close','vwap','adj_factor','market_cap']:bad|=frame.le(0)
        if name=='volume':bad|=frame.lt(0)
        denom=int(mask.loc[research].sum().sum())
        rows[name]={'missing_or_invalid_in_candidate_universe':int((bad&mask).loc[research].sum().sum()),
                    'coverage_in_candidate_universe':float(((~bad)&mask).loc[research].sum().sum()/max(1,denom)),
                    'shape':list(frame.shape),'full_axes_identical':True}
    indices=db.indices()
    counts=universe.loc[research].sum(axis=1)
    yearly={}
    for year,g in counts.groupby(counts.index.str[:4]):
        yearly[year]={'min_candidate_universe':int(g.min()),'median_candidate_universe':float(g.median()),'max_candidate_universe':int(g.max())}
    blockers=['Historical security type, primary-issue validity, ticker changes and delisted identities not fully independently verified.',
              'Market data is a bounded pilot, NOT a complete 4000+ stock market snapshot.',
              'Latest fundamental snapshot is not a historical PIT archive; date fallbacks and revisions must be disclosed.',
              'Delisted stock OHLCV may be available while adjustment/dividend/terminal-event data remain missing.']
    return {'passed':False,'status':'prototype_incomplete','mechanical_axes_passed':axes,
       'research_start':db.meta['research_start'],'research_end':db.meta['research_end'],
       'calendar_sessions':len(db.dates),'candidate_security_columns':len(db.sids),
       'stock_fields':rows,'candidate_universe_by_year':yearly,
       'indices':{'columns':list(indices.columns),'missing_cells':int(indices.isna().sum().sum())},
       'blockers':blockers,'pickle_trust':'Only team-controlled artifacts; never load untrusted pickles.'}
