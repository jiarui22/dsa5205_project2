"""Retrieve Sina US daily bars and validate paired turnover/volume against CRSP."""
from pathlib import Path
import gzip
import json
import urllib.request
import numpy as np
import pandas as pd
from .wrds_reconstruction import interval_segments, dated_join
from .common import save_parquet


def fetch_daily(ticker, output):
    if not ticker or any(c not in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-_' for c in ticker):
        raise ValueError('Invalid ticker')
    url = ('https://stock.finance.sina.com.cn/usstock/api/jsonp.php/'
           'IO.XSRV2.CallbackList/US_MinKService.getDailyK?symbol='+ticker)
    req = urllib.request.Request(url, headers={'User-Agent':'Mozilla/5.0',
                                              'Referer':'https://finance.sina.com.cn/'})
    with urllib.request.urlopen(req, timeout=25) as response:
        body = response.read().decode('utf-8')
    # Parse JSON data, never evaluate the JSONP wrapper as executable code.
    data = json.loads(body[body.index('['):body.rindex(']')+1])
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    with gzip.open(out/(ticker+'.json.gz'), 'wt') as stream:
        json.dump({'url':url,'records':data},stream)
    d = pd.DataFrame(data).rename(columns={'d':'date','o':'open','h':'high',
        'l':'low','c':'close','v':'source_volume','a':'amount'})
    d['date'] = pd.to_datetime(d.date)
    for col in ['open','high','low','close','source_volume','amount']:
        d[col] = pd.to_numeric(d[col],errors='coerce') if col in d else np.nan
    d = d.loc[d.date.between('2014-01-01','2025-12-31')].copy()
    if d.date.duplicated().any():
        raise ValueError('Duplicate vendor dates')
    d['ticker'] = ticker
    return d


def validate_pair(d):
    matches = pd.Series(True,index=d.index)
    for field, source in [('open','openprc'),('high','askhi'),('low','bidlo'),('close','prc')]:
        error = (d[field]-d[source].abs()).abs()
        matches &= error.le(.01001) & error.div(d[source].abs()).le(.001)
    vwap = d.amount/d.source_volume
    good = (matches & d.prc.gt(0) & d.amount.gt(0) & d.source_volume.gt(0)
            & vwap.between(d.bidlo*.995,d.askhi*1.005))
    # The price denominator always comes from the same vendor as turnover.
    # CRSP volume is a separate coverage diagnostic, never this denominator.
    return matches, vwap.where(good)


def probe_sina(raw, output, tickers=('AAPL','MSFT','NVDA','AIR','SPY'), *, response_directory=None):
    raw, output = Path(raw), Path(output)
    responses = output/'sina_responses' if response_directory is None else Path(response_directory)
    if response_directory is not None:
        # Raw captures are immutable inputs, never a cache for a later clean run.
        responses.mkdir(parents=True, exist_ok=False)
    summaries, accepted, failures = [], [], []
    names = pd.read_parquet(next(raw.rglob('crsp_dsenames.parquet')))
    names = names.loc[names.ticker.isin(tickers)].copy()
    names['namedt'] = pd.to_datetime(names.namedt)
    names['nameendt'] = pd.to_datetime(names.nameendt).fillna(pd.Timestamp('2099-12-31'))
    segments, conflicts = interval_segments(names,'permno','namedt','nameendt',['ticker'])
    cols = ['date','permno','prc','openprc','askhi','bidlo','vol']
    old = pd.read_parquet(next(raw.rglob('crsp_dsf_2014_2024.parquet')),columns=cols,
                          filters=[('permno','in',names.permno.unique().tolist())])
    old = dated_join(old,segments)
    new = pd.read_parquet(next(raw.rglob('crsp_dsf_v2_2025.parquet')),columns=cols+['ticker'],
                          filters=[('ticker','in',list(tickers))])
    market = pd.concat([old,new],ignore_index=True)
    # Ticker reuse is addressed by historical names; simultaneous matches are withheld.
    market = market.loc[~market.duplicated(['date','ticker'],keep=False)]
    for ticker in tickers:
        try:
            d = fetch_daily(ticker, responses)
            d = d.merge(market.loc[market.ticker.eq(ticker)],on=['date','ticker'],how='left',validate='one_to_one')
            matches, d['vwap_local_pair'] = validate_pair(d)
            usable = d.vwap_local_pair.notna()
            summaries.append(dict(ticker=ticker,period_rows=len(d),positive_amount=int(d.amount.gt(0).sum()),
                crsp_price_matches=int(matches.sum()),accepted_vwap=int(usable.sum()),
                first_positive_amount=str(d.loc[d.amount.gt(0),'date'].min()),
                first_accepted_vwap=str(d.loc[usable,'date'].min()),
                volume_median_ratio=float((d.source_volume/d.vol).median())))
            accepted.append(d.loc[usable,['date','ticker','permno','amount','source_volume','vwap_local_pair']])
        except Exception as error:
            failures.append(dict(ticker=ticker,error=repr(error)))
    summary = pd.DataFrame(summaries)
    summary.to_csv(output/'sina_validation.csv',index=False)
    pd.DataFrame(failures).to_csv(output/'sina_failures.csv',index=False)
    if accepted:
        save_parquet(pd.concat(accepted,ignore_index=True),output/'validated_sina_pairs.parquet')
    return summary, failures


def reconcile_sina(raw, reference, output):
    from .wrds_reconstruction import prepare_links, _comparison
    raw,reference,output = map(Path,[raw,reference,output])
    if not (output/'validated_sina_pairs.parquet').exists():
        result = pd.DataFrame(columns=['field','common','matched','raw_only','reference_only','match_fraction_on_common'])
        result.to_csv(output/'sina_reconciliation.csv',index=False)
        return result
    d = pd.read_parquet(output/'validated_sina_pairs.parquet')
    links = prepare_links(raw)
    segments,_ = interval_segments(links,'permno','linkdt','linkenddt',['sid'])
    d = dated_join(d,segments)
    dates = pd.Index(pd.read_csv(reference/'panels/calendar.csv',dtype=str).date)
    master = pd.read_parquet(reference/'metadata/security_master.parquet').sort_values('column_order')
    d = d.loc[d.sid.isin(master.sid)].drop_duplicates(['date','sid'])
    i = dates.get_indexer(d.date.dt.strftime('%Y%m%d'));j = pd.Index(master.sid).get_indexer(d.sid)
    stats = []
    for col,field in [('amount','amount'),('source_volume','amount_source_volume'),('vwap_local_pair','vwap_local_pair')]:
        old = pd.read_pickle(reference/'panels'/f'{field}.pkl')
        z = _comparison(d[col],old.to_numpy()[i,j])
        z['reference_only'] = int(old.notna().to_numpy().sum())-z['common']
        stats.append(dict(field=field,**z))
    result = pd.DataFrame(stats).set_index('field')
    result['match_fraction_on_common'] = result.matched/result.common
    result.to_csv(output/'sina_reconciliation.csv')
    return result
