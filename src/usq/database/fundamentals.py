"""Compustat fundq ingestion, explicit exceptions, and release-ordered features.

This is a latest-download classroom approximation, NOT a historical PIT archive.
RDQ is a quarterly earnings announcement proxy, not field-by-field availability.
"""
from __future__ import annotations

from pathlib import Path
import json
import re
import zipfile
import numpy as np
import pandas as pd
from .common import digest, dump, save_parquet, calendar, timestamps_from_days

CORE=['shares_out','book_equity','earnings_ttm','revenue_ttm','total_assets','total_liabilities']
STOCK_MAP={'shares_out':'cshoq','book_equity':'ceqq','total_assets':'atq','total_liabilities':'ltq',
           'cash':'cheq','long_term_debt':'dlttq','short_term_debt':'dlcq'}
NUMERIC='cshoq ceqq niq revtq saleq atq ltq oancfy capxy niy revty saley cheq dlttq dlcq ajexq'.split()
FIELDS=('gvkey datadate fyearq fqtr fyr datafqtr datacqtr rdq fdateq pdateq finalq updq srcq rp '
        'conm conml tic cusip cik priusa exchg fic loc sic naics gsector indfmt consol datafmt curcdq '
        'ipodate dldte costat '+ ' '.join(NUMERIC)).split()
FUND_RE=r'\bETF\b|\bETN\b|exchange.traded fund|\bmutual fund\b|\bclosed.end\b|\bfund\b'


def ingest(source, target, *, min_year=2010, chunksize=25000):
    """Read selected columns in chunks; keep a provenance-verified immutable stage."""
    source=Path(source); target=Path(target)
    meta_path=target.with_suffix('.json')
    if target.exists():
        old=json.loads(meta_path.read_text())
        if old['source_sha256']!=digest(source):
            raise ValueError('Source changed; use a new stage/snapshot, never silently reuse')
        return pd.read_parquet(target),old
    source_hash=digest(source); archive=None
    if source.suffix=='.zip':
        archive=zipfile.ZipFile(source)
        members=[x for x in archive.infolist() if x.filename.endswith('.csv') and not x.filename.startswith('__MACOSX/')]
        if len(members)!=1: raise ValueError('Expected one CSV in source ZIP')
        stream=archive.open(members[0])
        member=members[0].filename
    else:
        stream=source.open('rb'); member=source.name
    frames=[]; offset=0; kept=0
    try:
        reader=pd.read_csv(stream,usecols=lambda x:x in FIELDS,dtype=str,
            chunksize=chunksize,keep_default_na=False,na_values=['','.', '._']+['.'+x for x in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'])
        for i,ch in enumerate(reader):
            ch['source_row']=np.arange(offset,offset+len(ch),dtype='int64'); offset+=len(ch)
            mask=(ch.datadate.ge(f'{min_year}-01-01') & ch.exchg.isin(['11','12','14']) &
                  ch.indfmt.eq('INDL') & ch.consol.eq('C') & ch.datafmt.eq('STD') & ch.curcdq.eq('USD'))
            z=ch.loc[mask].copy()
            if not z.empty:
                name=z.conml.fillna(z.conm).fillna('')
                isfund=z.sic.isin(['6722','6726']) | name.str.contains(FUND_RE,case=False,regex=True)
                z=z.loc[~isfund]
                frames.append(z); kept+=len(z)
            if i%10==0: print(json.dumps({'stage':'source','scanned':offset,'candidate_rows':kept}),flush=True)
    finally:
        stream.close()
        if archive is not None: archive.close()
    frame=pd.concat(frames,ignore_index=True)
    frame.gvkey=frame.gvkey.str.zfill(6)
    for c in NUMERIC+['fyearq','fqtr','fyr']:
        frame[c]=pd.to_numeric(frame[c],errors='coerce')
    frame['period_end']=pd.to_datetime(frame.datadate,errors='coerce')
    frame['revenue_q']=frame.revtq.combine_first(frame.saleq)
    frame['source_vintage']=source_hash
    save_parquet(frame,target)
    meta={'source_sha256':source_hash,'source_filename':source.name,'archive_member':member,
          'scanned_rows':offset,'candidate_rows':len(frame),'stage_sha256':digest(target),
          'notes':['Source exchange and primary-issue descriptors are not a reconstructed historical security master.',
                   'Investment funds excluded using source SIC and legal-name evidence.',
                   'Source file remains unchanged; selected-column chunk ingestion.']}
    dump(meta,meta_path)
    return frame,meta


def normalize_records(raw, dates, *, allow_final_date_fallback=True):
    d=raw.copy()
    release=pd.to_datetime(d.rdq,errors='coerce')
    lag=(release-d.period_end).dt.days
    valid=lag.between(0,365)
    d['availability_basis']=np.where(valid,'rdq_next_session','unusable_date')
    release=release.where(valid)
    if allow_final_date_fallback:
        final=pd.to_datetime(d.fdateq,errors='coerce')
        use=release.isna() & (final-d.period_end).dt.days.between(0,365)
        release=release.where(~use,final)
        d.loc[use,'availability_basis']='fdateq_next_session_fallback_not_announcement'
    # Only dates are provided: conservative first session strictly AFTER the date.
    # Preserve actual pre-warmup and post-panel release dates in the long table;
    # never relabel an old release as the panel's first session.
    lower=min(pd.to_datetime(dates[0]),release.min())
    upper=max(pd.to_datetime(dates[-1]),release.max())+pd.Timedelta(days=14)
    release_sessions=calendar(lower.strftime('%Y-%m-%d'),upper.strftime('%Y-%m-%d'))
    sessions=pd.to_datetime(release_sessions,format='%Y%m%d').to_numpy()
    values=release.to_numpy(dtype='datetime64[ns]')
    pos=np.searchsorted(sessions,values,side='right')
    accepted=release.notna().to_numpy() & (pos<len(sessions))
    avail=np.full(len(d),np.datetime64('NaT','ns'),dtype='datetime64[ns]')
    avail[accepted]=sessions[pos[accepted]]
    d['release_date']=release
    d['available_ts']=pd.to_datetime(avail)
    d['date_usable']=accepted
    # Different fiscal-year/quarter representations can carry different flow values.
    # Quarantine ALL such duplicate period keys instead of picking the first row.
    d['period_conflict']=d.duplicated(['gvkey','period_end'],keep=False)
    usable=d.loc[~d.period_conflict & d.period_end.notna() & d.fqtr.isin([1,2,3,4])].copy()
    usable=usable.sort_values(['gvkey','period_end','source_row']).reset_index(drop=True)
    g=usable.groupby('gvkey',sort=False)
    ordinal=usable.fyearq*4+usable.fqtr
    gap=(usable.period_end-g.period_end.shift()).dt.days
    consecutive=(ordinal-(g.fyearq.shift()*4+g.fqtr.shift())).eq(1)&gap.between(60,120)
    usable['chain']=consecutive
    valid4=consecutive & g['chain'].shift(1).eq(True) & g['chain'].shift(2).eq(True)
    usable['available_day']=(usable.available_ts-pd.Timestamp('1970-01-01')).dt.days
    # Missing constituent announcements MUST make a TTM availability unusable.
    release4=g.available_day.rolling(4,min_periods=4).max().reset_index(level=0,drop=True)
    for dest,source in STOCK_MAP.items():
        usable[dest]=usable[source]*1e6
    for dest,source in [('earnings_ttm','niq'),('revenue_ttm','revenue_q')]:
        usable[dest]=(g[source].rolling(4,min_periods=4).sum().reset_index(level=0,drop=True)*1e6).where(valid4)
        usable[dest+'_available_day']=release4.where(usable[dest].notna())
    for stem,source in [('operating_cash_flow','oancfy'),('capital_expenditures','capxy')]:
        prev=g[source].shift()
        same_fy=usable.fyearq.eq(g.fyearq.shift()) & usable.fyr.eq(g.fyr.shift()) & consecutive
        quarter=(usable[source]-prev).where(same_fy)
        quarter=quarter.where(~usable.fqtr.eq(1),usable[source])
        base_day=usable.available_day.astype('float64')
        prior_day=g.available_day.shift().astype('float64')
        qday=base_day.where(usable.fqtr.eq(1),
            pd.concat([base_day,prior_day],axis=1).max(axis=1,skipna=False))
        usable[stem+'_q']=quarter*1e6
        usable[stem+'_q_available_day']=qday.where(quarter.notna())
        usable[stem+'_ttm']=(usable.groupby('gvkey')[stem+'_q'].rolling(4,min_periods=4).sum()
                            .reset_index(level=0,drop=True)).where(valid4)
        usable[stem+'_ttm_available_day']=(qday.groupby(usable.gvkey).rolling(4,min_periods=4).max()
            .reset_index(level=0,drop=True)).where(usable[stem+'_ttm'].notna())
    return d,usable


def feature_events(records, field):
    col=field+'_available_day'
    day=records[col] if col in records else records.available_day
    events=records[['sid','period_end','source_row','availability_basis']].copy()
    events['value']=records[field]
    events['available_date']=timestamps_from_days(day)
    events=events.loc[events.value.notna() & events.available_date.notna()].copy()
    events=events.sort_values(['sid','available_date','period_end','source_row'])
    # Several fiscal periods released together: explicitly choose newest period.
    events=events.drop_duplicates(['sid','available_date'],keep='last')
    newest=events.groupby('sid').period_end.cummax()
    events=events.loc[events.period_end.eq(newest)]
    return events


def expand_events(events, dates, sids, *, max_age_days=550):
    """As-of/ffill only. Exact daily age gate; no backfill or hidden column pruning."""
    array=np.full((len(dates),len(sids)),np.nan,dtype='float32')
    day=pd.to_datetime(dates,format='%Y%m%d').to_numpy(dtype='datetime64[ns]')
    positions={sid:i for i,sid in enumerate(sids)}
    for sid,group in events.groupby('sid',sort=False):
        if sid not in positions: raise ValueError('Event outside frozen security pool')
        group=group.sort_values('available_date')
        when=group.available_date.to_numpy(dtype='datetime64[ns]')
        pos=np.searchsorted(when,day,side='right')-1
        valid=pos>=0; safe=np.maximum(pos,0)
        periods=group.period_end.to_numpy(dtype='datetime64[ns]')[safe]
        age=(day-periods).astype('timedelta64[D]').astype(float)
        valid &= (age>=0)&(age<=max_age_days)
        values=group.value.to_numpy(dtype='float64')[safe]
        array[valid,positions[sid]]=values[valid]
    return pd.DataFrame(array,index=dates,columns=sids)


def primary_registry(raw, features, *, research_start='2015-01-01', research_end='2025-12-31'):
    """Source primary US issue identifiers, not a claim of historical ticker mapping.

    The source PRIUSA tag is a Compustat issue identifier. GVKEY alone is NEVER
    passed off as a security identifier. Missing/ambiguous tags are quarantined.
    Eligibility remains date-dependent; the frozen column pool is a historical union.
    """
    tags=raw.loc[raw.priusa.notna(),['gvkey','priusa']].drop_duplicates()
    ambiguous=tags.loc[tags.duplicated('gvkey',keep=False),'gvkey'].unique()
    tags=tags.loc[~tags.gvkey.isin(ambiguous)].copy()
    tags['priusa']=tags.priusa.astype(str).str.zfill(2)
    tags['sid']='CNA:'+tags.gvkey+':'+tags.priusa
    features=features.merge(tags,on='gvkey',how='inner',validate='many_to_one',suffixes=('','_tag'))
    # A company enters the column union only if its six required features can be
    # known within the target research window. Do NOT require future-year survival.
    disclosed=pd.concat([features.available_day,features.earnings_ttm_available_day,
                         features.revenue_ttm_available_day],axis=1).max(axis=1,skipna=False)
    known=timestamps_from_days(disclosed)
    ipo=pd.to_datetime(features.ipodate,errors='coerce')
    delist=pd.to_datetime(features.dldte,errors='coerce')-pd.Timedelta(days=1)
    entry=pd.concat([known,ipo,pd.Series(pd.Timestamp(research_start),index=features.index)],axis=1).max(axis=1)
    exit_=pd.concat([features.period_end+pd.Timedelta(days=550),delist,
                    pd.Series(pd.Timestamp(research_end),index=features.index)],axis=1).min(axis=1)
    sessions=pd.to_datetime(calendar(research_start,research_end),format='%Y%m%d').to_numpy()
    pos=np.searchsorted(sessions,entry.to_numpy(),side='left')
    possible=pos<len(sessions)
    safe=np.minimum(pos,len(sessions)-1)
    possible &= sessions[safe]<=exit_.to_numpy()
    core_any=features[CORE].notna().all(axis=1) & known.notna() & possible
    core_any &= features.shares_out.gt(0) & features.total_assets.gt(0) & features.total_liabilities.ge(0)
    wanted=set(features.loc[core_any,'gvkey'])
    features=features.loc[features.gvkey.isin(wanted)].copy()
    last=raw.loc[raw.gvkey.isin(wanted)].sort_values(['period_end','source_row']).groupby('gvkey').tail(1)
    master=last[['gvkey','conm','conml','tic','cusip','cik','exchg','fic','sic','naics','ipodate','dldte','costat']].merge(tags,on='gvkey',validate='one_to_one')
    master=master.sort_values('sid').reset_index(drop=True)
    master['column_order']=np.arange(len(master))
    master['security_id_basis']='Compustat GVKEY + source PRIUSA issue identifier'
    master['historical_primary_issue_mapping_verified']=False
    master['historical_security_type_verified']=False
    master['ticker_mapping_status']='source_descriptive_ticker_not_historical_validity_mapping'
    return master,features,{'ambiguous_primary_tag_companies':len(ambiguous),
          'missing_primary_tag_companies':int(raw.loc[raw.priusa.isna(),'gvkey'].nunique()),
          'eligible_security_candidates':len(master)}
