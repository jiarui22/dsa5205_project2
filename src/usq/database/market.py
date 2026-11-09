"""Raw snapshot adapters following global-stock-data's documented endpoints.

No credentials, no anti-bot bypass, no retry rotation. Rate-limit responses stop
the batch. Historical ticker identity remains a separate acceptance requirement.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote
from zoneinfo import ZoneInfo
import gzip
import json
import re
import time
import requests
import numpy as np
import pandas as pd
from .common import digest, dump

UPSTREAM={'repository':'https://github.com/simonlin1212/global-stock-data',
          'commit':'b7f22029660769baa29a4403cc61c285f35f8905','sections':['Layer 2.1','Yahoo chart API']}


class RateLimited(RuntimeError):pass


def read_cached_raw(path):
    path=Path(path)
    meta=json.loads(path.with_suffix('.json').read_text())
    if digest(path)!=meta['sha256']:
        raise ValueError(f'Raw snapshot checksum mismatch: {path.name}')
    return gzip.decompress(path.read_bytes())


def _raw(vendor,symbol,root,start,end):
    key=quote(symbol,safe='').replace('%','_')
    path=Path(root)/vendor/(key+'.gz')
    info=path.with_suffix('.json')
    if path.exists():
        meta=json.loads(info.read_text())
        if digest(path)!=meta['sha256']:raise ValueError('Raw market snapshot hash mismatch')
        if meta['requested_start']!=start or meta['requested_end']!=end:
            raise ValueError('Raw market snapshot range differs; use a new snapshot directory')
        return gzip.decompress(path.read_bytes()),meta
    if vendor=='sina':
        url='https://stock.finance.sina.com.cn/usstock/api/jsonp.php/var/US_MinKService.getDailyK'
        params={'symbol':symbol,'num':10000}
        headers={'Referer':'https://finance.sina.com.cn/','User-Agent':'Mozilla/5.0'}
    else:
        url='https://query2.finance.yahoo.com/v8/finance/chart/'+quote(symbol,safe='')
        params={'period1':int(pd.Timestamp(start,tz='UTC').timestamp()),
                'period2':int((pd.Timestamp(end,tz='UTC')+pd.Timedelta(days=2)).timestamp()),
                'interval':'1d','events':'div,split'}
        headers={'User-Agent':'Mozilla/5.0'}
    response=requests.get(url,params=params,headers=headers,timeout=35)
    if response.status_code in (403,429):
        raise RateLimited(f'{vendor} HTTP {response.status_code}; stop, do not evade provider restriction')
    # Retain explicit 404 responses too; an unavailable delisted stock is not an empty success.
    if response.status_code not in (200,404,422):response.raise_for_status()
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp')
    temp.write_bytes(gzip.compress(response.content,mtime=0));temp.replace(path)
    meta={'vendor':vendor,'symbol':symbol,'http_status':response.status_code,
          'url':url,'params':params,'requested_start':start,'requested_end':end,
          'fetched_utc':datetime.now(timezone.utc).isoformat(),'sha256':digest(path),'upstream':UPSTREAM}
    dump(meta,info)
    time.sleep(.35)
    return response.content,meta


def sina_frame(raw):
    text=raw.decode('utf-8',errors='replace')
    match=re.search(r'\((\[.*\])\)',text,re.S)
    rows=json.loads(match.group(1)) if match else []
    if not rows:return pd.DataFrame(columns=['open','high','low','close','volume','amount'])
    d=pd.DataFrame(rows).rename(columns={'d':'date','o':'open','h':'high','l':'low','c':'close','v':'volume','a':'amount'})
    for col in ['open','high','low','close','volume','amount']:
        d[col]=pd.to_numeric(d.get(col,pd.Series(index=d.index,dtype=float)),errors='coerce')
    if d.date.duplicated().any():raise ValueError('Duplicate source trading date; quarantine symbol')
    d.index=pd.Index(pd.to_datetime(d.date).dt.strftime('%Y%m%d'),name='date')
    d=d[['open','high','low','close','volume','amount']].sort_index()
    d['amount']=d.amount.where(d.amount.gt(0))  # Source zero frequently means unavailable amount.
    d['vwap']=d.amount/d.volume.where(d.volume.gt(0))
    bad=(d[['open','high','low','close']].le(0).any(axis=1) | d.high.lt(d.low) | d.volume.lt(0))
    if bad.any():d.loc[bad,:]=np.nan
    return d


def yahoo_frame(raw):
    obj=json.loads(raw);c=((obj.get('chart') or {}).get('result') or [None])[0]
    if not c:return pd.DataFrame(),{},(obj.get('chart') or {}).get('error')
    meta=c.get('meta',{}); ts=c.get('timestamp',[]); q=(c.get('indicators',{}).get('quote') or [{}])[0]
    tz=ZoneInfo(meta.get('exchangeTimezoneName','America/New_York'))
    days=[datetime.fromtimestamp(t,tz).strftime('%Y%m%d') for t in ts]
    adj=((c.get('indicators',{}).get('adjclose') or [{}])[0]).get('adjclose',[None]*len(ts))
    d=pd.DataFrame({'date':days,'close_split_adjusted':q.get('close',[None]*len(ts)),
                    'adjusted_close':adj}).set_index('date')
    d.index=pd.Index(d.index,name='date')
    if d.index.has_duplicates:raise ValueError('Duplicate Yahoo daily bars')
    return d.sort_index(),meta,c.get('events',{})


def fetch_snapshot(cfg):
    stage=Path(cfg['stage_dir']);out=Path(cfg['output']);root=out/'raw'/'market'
    master=pd.read_parquet(stage/'security_candidates.parquet')
    selected=master.copy()
    requested=cfg.get('market_symbols')
    if requested: selected=selected.loc[selected.tic.isin(requested)]
    limit=cfg.get('market_fetch_limit',24)
    if limit is not None and int(limit)>0:selected=selected.head(int(limit))
    # Data source explicit company-specific alias: never append ALL FB history.
    symbols=set(selected.tic.dropna().astype(str))
    if 'META' in symbols:symbols.add('FB')
    start=(pd.Timestamp(cfg['calendar_start'])-pd.Timedelta(days=730)).strftime('%Y-%m-%d')
    end=cfg['calendar_end']
    jobs=[('sina',s) for s in sorted(symbols)] + [('yahoo',s.replace('.','-')) for s in sorted(set(selected.tic.dropna().astype(str)))]
    jobs += [('yahoo',s) for s in cfg.get('indices',[])]
    reports=[]
    def job(pair):
        vendor,symbol=pair
        raw,meta=_raw(vendor,symbol,root,start,end)
        if vendor=='sina':
            frame=sina_frame(raw)
            return {'vendor':vendor,'symbol':symbol,'rows':len(frame),'http_status':meta['http_status']}
        frame,info,error=yahoo_frame(raw)
        return {'vendor':vendor,'symbol':symbol,'rows':len(frame),'http_status':meta['http_status'],
                'instrument_type':info.get('instrumentType'),'source_name':info.get('longName',info.get('shortName')),
                'error':error if meta['http_status']!=200 else None}
    with ThreadPoolExecutor(max_workers=min(int(cfg.get('market_fetch_workers',3)),4)) as pool:
        pending={pool.submit(job,pair):pair for pair in jobs}
        for future in as_completed(pending):
            pair=pending[future]
            try: result=future.result()
            except RateLimited:
                for f in pending:f.cancel()
                raise
            except Exception as exc:
                result={'vendor':pair[0],'symbol':pair[1],'error':f'{type(exc).__name__}: {exc}'}
            reports.append(result)
            print(json.dumps({'stage':'market_fetch',**result},ensure_ascii=False),flush=True)
    dump({'selection':'bounded source/identity pilot; not full-universe download' if limit else 'full candidate download',
          'candidate_securities':len(master),'selected_securities':len(selected),'reports':reports,
          'upstream':UPSTREAM},out/'metadata'/'market_fetch_report.json')
    return reports


def cached_stock(symbol,root):
    key=quote(symbol,safe='').replace('%','_')
    path=Path(root)/'sina'/(key+'.gz')
    if not path.exists():return pd.DataFrame(),pd.DataFrame(),{},{}
    d=sina_frame(read_cached_raw(path))
    if symbol=='META':
        # Official Meta change effective 2022-06-09; FB was reused by an ETF later.
        old=Path(root)/'sina'/'FB.gz'
        if old.exists():
            f=sina_frame(read_cached_raw(old))
            d=pd.concat([f.loc[f.index<'20220609'],d.loc[d.index>='20220609']]).sort_index()
        else:d=d.loc[d.index>='20220609']
    key=quote(symbol.replace('.','-'),safe='').replace('%','_')
    path=Path(root)/'yahoo'/(key+'.gz')
    if not path.exists():return d,pd.DataFrame(),{},{}
    y,meta,events=yahoo_frame(read_cached_raw(path))
    if meta and (meta.get('instrumentType')!='EQUITY' or meta.get('currency')!='USD'):
        # A reused ticker returning an ETF must not be merged into issuer history.
        return pd.DataFrame(),pd.DataFrame(),meta,{'quarantined':'non-equity or non-USD instrument'}
    return d,y,meta,events


def split_multiplier(events, days):
    items=(events or {}).get('splits',{})
    splits=[]
    for e in items.values():
        day=datetime.fromtimestamp(e['date'],ZoneInfo('America/New_York')).strftime('%Y%m%d')
        ratio=float(e['numerator'])/float(e['denominator'])
        if not np.isfinite(ratio) or ratio<=0:raise ValueError('Invalid split event')
        splits.append((day,ratio))
    splits.sort()
    out=np.ones(len(days),dtype='float64')
    for day,ratio in splits:out[np.asarray(days)>=day]*=ratio
    return out
