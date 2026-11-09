"""Historical SIC from the original annual source; never backcast current GICS."""
from pathlib import Path
import sqlite3
import numpy as np
import pandas as pd
from .common import digest, dump, save_parquet, calendar


def industry_events(cfg, master, dates):
    stage=Path(cfg['stage_dir']); cache=stage/'historical_sic.parquet'
    wanted=set(master.gvkey)
    if cache.exists():
        annual=pd.read_parquet(cache)
        provenance={'kind':'cached_source_selected_annual_sic','sha256':digest(cache)}
    elif cfg.get('annual_source_dta'):
        source=Path(cfg['annual_source_dta']); frames=[]
        fields=['gvkey','datadate','fdate','pdate','sich','indfmt','consol','datafmt','curcd','exchg']
        for ch in pd.read_stata(source,columns=fields,chunksize=25000,convert_categoricals=False):
            ch.gvkey=ch.gvkey.astype(str).str.zfill(6)
            z=ch.loc[ch.gvkey.isin(wanted)&ch.indfmt.eq('INDL')&ch.consol.eq('C')&ch.datafmt.eq('STD')&ch.curcd.eq('USD')&
                     ch.exchg.isin([11,12,14])&ch.datadate.dt.year.between(2010,2024)].copy()
            for c in ['datadate','fdate','pdate']:z[c]=pd.to_datetime(z[c],errors='coerce').dt.strftime('%Y-%m-%d')
            frames.append(z[['gvkey','datadate','fdate','pdate','sich']])
        annual=pd.concat(frames,ignore_index=True)
        save_parquet(annual,cache)
        provenance={'kind':'annual_historical_sich','filename':source.name,'sha256':digest(source)}
        dump(provenance,stage/'historical_sic_source.json')
    elif cfg.get('annual_stage'):
        source=Path(cfg['annual_stage'])
        con=sqlite3.connect(f'file:{source}?mode=ro',uri=True)
        annual=pd.read_sql_query("SELECT gvkey,datadate,fdate,pdate,sich FROM source_records WHERE indfmt='INDL' AND curcd='USD'",con)
        con.close(); annual.gvkey=annual.gvkey.astype(str).str.zfill(6)
        annual=annual.loc[annual.gvkey.isin(wanted)]
        save_parquet(annual,cache)
        provenance={'kind':'trusted_local_readonly_annual_stage','filename':source.name,'sha256':digest(source),
             'original_source':'North America annual data.dta, supplied previously; reproduce from annual_source_dta on another computer'}
        dump(provenance,stage/'historical_sic_source.json')
    else:
        return pd.DataFrame(columns=['sid','period_end','available_date','value']),{'kind':'unavailable_no_historical_source'}
    if (stage/'historical_sic_source.json').exists():
        import json
        provenance=json.loads((stage/'historical_sic_source.json').read_text())
    annual=annual.loc[annual.gvkey.isin(wanted)].copy()
    annual['period_end']=pd.to_datetime(annual.datadate,errors='coerce')
    final=pd.to_datetime(annual.fdate,errors='coerce'); preliminary=pd.to_datetime(annual.pdate,errors='coerce')
    valid=(final-annual.period_end).dt.days.between(0,365)
    release=final.where(valid)
    release=release.fillna(preliminary.where((preliminary-annual.period_end).dt.days.between(0,365)))
    annual['release']=release
    annual=annual.loc[annual.sich.notna() & annual.release.notna()]
    if annual.duplicated(['gvkey','period_end']).any():
        annual=annual.loc[~annual.duplicated(['gvkey','period_end'],keep=False)]
    annual=annual.merge(master[['gvkey','sid']],on='gvkey',validate='many_to_one')
    lo=min(pd.to_datetime(dates[0]),annual.release.min())
    hi=max(pd.to_datetime(dates[-1]),annual.release.max())+pd.Timedelta(days=14)
    sessions=pd.to_datetime(calendar(lo.strftime('%Y-%m-%d'),hi.strftime('%Y-%m-%d')),format='%Y%m%d').to_numpy()
    pos=np.searchsorted(sessions,annual.release.to_numpy(),side='right')
    annual=annual.loc[pos<len(sessions)].copy();pos=pos[pos<len(sessions)]
    annual['available_date']=pd.to_datetime(sessions[pos])
    annual['value']='SIC2:'+pd.to_numeric(annual.sich).astype('int64').astype(str).str.zfill(4).str[:2]
    annual=annual.sort_values(['sid','available_date','period_end']).drop_duplicates(['sid','available_date'],keep='last')
    latest=annual.groupby('sid').period_end.cummax()
    return annual.loc[annual.period_end.eq(latest),['sid','period_end','available_date','value']],provenance


def expand_industry(events, dates, sids):
    # Factorized string categories reduce disk/RAM versus repeated Python strings.
    out={};day=pd.to_datetime(dates,format='%Y%m%d').to_numpy()
    groups={sid:g.sort_values('available_date') for sid,g in events.groupby('sid')}
    categories=sorted(events.value.dropna().unique())
    for sid in sids:
        codes=np.full(len(dates),-1,dtype='int16')
        if sid in groups:
            g=groups[sid];pos=np.searchsorted(g.available_date.to_numpy(),day,side='right')-1
            valid=pos>=0; values=pd.Categorical(g.value,categories=categories).codes
            codes[valid]=values[pos[valid]]
        out[sid]=pd.Categorical.from_codes(codes,categories=categories)
    return pd.DataFrame(out,index=dates).rename_axis(columns='sid')
