"""Reconstruct the supplied monthly research universe from raw CRSP records.

The selection algorithm was recovered from the team's exported tool transcript.
No strategy returns or reference membership enter selection.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from .common import calendar, save_parquet
from .wrds_reconstruction import prepare_links, interval_segments, dated_join


def monthly_liquidity_universe(close, dollar_volume, listed, *, keep=.8,
                               lookback=60, min_history=252, min_price=5.):
    """Select at the first signal close of each month, then apply daily validity."""
    if not 0 < keep <= 1:
        raise ValueError('keep must be in (0, 1]')
    if not (close.index.equals(dollar_volume.index) and close.columns.equals(dollar_volume.columns)
            and close.index.equals(listed.index) and close.columns.equals(listed.columns)):
        raise ValueError('Universe inputs must share complete axes')
    live = listed.fillna(False).astype(bool)
    valid = live & close.ge(min_price) & close.notna()
    count = close.notna().rolling(min_history, min_periods=min_history).sum()
    score = dollar_volume.where(dollar_volume > 0).rolling(lookback, min_periods=lookback).mean()
    eligible = valid & count.ge(min_history) & score.notna()
    result = pd.DataFrame(False, index=close.index, columns=close.columns)
    selected = pd.Series(False, index=close.columns)
    previous = None
    for date in close.index:
        period = date.to_period('M')
        if period != previous:
            s = score.loc[date].where(eligible.loc[date]).dropna()
            n = int(np.ceil(len(s) * keep))
            selected[:] = False
            selected.loc[s.sort_values(ascending=False, kind='stable').index[:n]] = True
            previous = period
        result.loc[date] = selected & valid.loc[date]
    return result


def reconstruct_universe(raw, reference, output):
    """Use all raw-derived candidate SIDs, not only the reference's ever-selected SIDs."""
    raw, output = Path(raw), Path(output)
    reference = Path(reference) if reference is not None else None
    master = pd.read_parquet(output/'financial_security_registry.parquet').sort_values('column_order')
    sids = pd.Index(master.sid, name='sid')
    dates = pd.to_datetime(calendar('2014-01-01','2025-12-31'),format='%Y%m%d')
    shape = (len(dates), len(sids))
    price = np.full(shape, np.nan, dtype='float64')
    volume = np.full(shape, np.nan, dtype='float64')
    common = np.zeros(shape, dtype=bool)
    observed = np.zeros(shape, dtype=bool)
    links = prepare_links(raw)
    links = links.loc[links.sid.isin(sids)]
    segments, conflicts = interval_segments(links,'permno','linkdt','linkenddt',['sid'])
    if len(conflicts):
        raise ValueError('Conflicting security links require review before universe reconstruction')
    names = pd.read_parquet(next(raw.rglob('crsp_dsenames.parquet')))
    names['namedt'] = pd.to_datetime(names.namedt)
    names['nameendt'] = pd.to_datetime(names.nameendt).fillna(pd.Timestamp('2099-12-31'))
    names = names.loc[(names.namedt <= '2024-12-31') & (names.nameendt >= '2014-01-01')]
    name_segments, conflicts = interval_segments(names,'permno','namedt','nameendt',['shrcd'])
    if len(conflicts):
        raise ValueError('Conflicting historical share codes require review')
    usa = master.fic.eq('USA').to_numpy()
    for filename in ['crsp_dsf_2014_2024.parquet','crsp_dsf_v2_2025.parquet']:
        ciz = 'v2' in filename
        cols = ['permno','date','prc','vol']
        if ciz:
            cols += ['securitytype','securitysubtype','sharetype']
        for batch in pq.ParquetFile(next(raw.rglob(filename))).iter_batches(batch_size=150000,columns=cols):
            d = dated_join(batch.to_pandas(),segments)
            d = d.loc[d.sid.isin(sids)].copy()
            if ciz:
                is_common = d.securitytype.eq('EQTY') & d.securitysubtype.eq('COM') & d.sharetype.eq('NS')
                terminal = d.securitysubtype.eq('UNK') | d.prc.le(0)
            else:
                d = dated_join(d,name_segments)
                is_common = d.shrcd.isin([10,11,18])
                terminal = pd.Series(False,index=d.index)
            i,j = dates.get_indexer(pd.to_datetime(d.date)),sids.get_indexer(d.sid)
            if (i<0).any() or (j<0).any() or d.duplicated(['date','sid']).any() or observed[i,j].any():
                raise ValueError('Duplicate or out-of-axis raw price mapping')
            observed[i,j] = True
            p = d.prc.abs().where(np.isfinite(d.prc)&d.prc.abs().gt(0)&~terminal)
            v = d.vol.where(np.isfinite(d.vol)&d.vol.ge(0)&p.notna())
            price[i,j] = p.to_numpy(dtype='float64')
            volume[i,j] = v.to_numpy(dtype='float64')
            common[i,j] = is_common.to_numpy() & usa[j]
    close = pd.DataFrame(price.astype('float64'),index=dates,columns=sids)
    dollars = close * volume.astype('float64')
    listed = pd.DataFrame(common,index=dates,columns=sids) & close.notna()
    rebuilt = monthly_liquidity_universe(close,dollars,listed)
    if reference is None:
        saved = rebuilt.loc[:,rebuilt.any(axis=0)]
        saved.to_parquet(output/'reconstructed_universe.parquet',compression='zstd')
        price_gate = listed & close.ge(5)
        history = close.notna().rolling(252,min_periods=252).sum().ge(252)
        liquid = dollars.where(dollars>0).rolling(60,min_periods=60).mean().notna()
        counts = pd.DataFrame({'Observed':close.notna().sum(axis=1),
            'Common equity':listed.sum(axis=1),'Price ≥ $5':price_gate.sum(axis=1),
            '252-session history':(price_gate&history).sum(axis=1),
            'Liquidity eligible':(price_gate&history&liquid).sum(axis=1),
            'Selected':rebuilt.sum(axis=1)})
        counts.to_csv(output/'universe_daily_counts.csv',index_label='date')
        truncation=[]
        for end in ['2017-06-30','2020-12-31','2022-12-30']:
            sub=monthly_liquidity_universe(close.loc[:end],dollars.loc[:end],listed.loc[:end])
            truncation.append(dict(end=end,differing_cells=int((sub!=rebuilt.loc[:end]).to_numpy().sum())))
        report=dict(raw_candidate_SIDs=len(sids),selected_SIDs=len(saved.columns),
            membership_cells=int(rebuilt.to_numpy().sum()),truncation=truncation,
            price_proxy='Unadjusted CRSP close × share volume',
            numeric_precision='float64',security_codes='SIZ SHRCD 10/11/18; CIZ EQTY/COM/NS',
            caveat='Compustat FIC and PRIUSA are static source descriptors.')
        (output/'universe_audit.json').write_text(json.dumps(report,indent=2))
        return report
    # Reference membership is loaded only after the independent selection is complete.
    old = pd.read_pickle(reference/'panels/universe.pkl').astype(bool)
    old.index = pd.to_datetime(old.index,format='%Y%m%d')
    all_sids = sids.union(old.columns,sort=False)
    a,b = rebuilt.reindex(columns=all_sids,fill_value=False),old.reindex(columns=all_sids,fill_value=False)
    if not a.index.equals(b.index):
        raise ValueError('Reference calendar differs from independent XNYS calendar')
    av,bv = a.to_numpy(),b.to_numpy()
    delta = av!=bv
    i,j = np.where(delta)
    pd.DataFrame({'date':a.index[i],'sid':all_sids[j], 'rebuilt':av[i,j],'reference':bv[i,j]}).to_csv(
        output/'universe_membership_differences.csv',index=False)
    saved = rebuilt.loc[:,rebuilt.any(axis=0)]
    saved.to_parquet(output/'reconstructed_universe.parquet',compression='zstd')
    daily = pd.DataFrame({'rebuilt':av.sum(axis=1),'reference':bv.sum(axis=1),
                          'extra':(av&~bv).sum(axis=1),'missing':(~av&bv).sum(axis=1)},index=a.index)
    daily.to_csv(output/'universe_reconciliation_daily.csv',index_label='date')
    # Freshly recompute on truncated inputs; a causal rule must preserve earlier decisions.
    truncation = []
    for end in ['2017-06-30','2020-12-31','2022-12-30']:
        sub = monthly_liquidity_universe(close.loc[:end],dollars.loc[:end],listed.loc[:end])
        truncation.append(dict(end=end,differing_cells=int((sub!=rebuilt.loc[:end]).to_numpy().sum())))
    report = dict(raw_candidate_SIDs=len(sids), rebuilt_ever_selected_SIDs=int(rebuilt.any(axis=0).sum()),
        reference_ever_selected_SIDs=int(old.any(axis=0).sum()), rebuilt_membership_cells=int(av.sum()),
        reference_membership_cells=int(bv.sum()), differing_cells=int(delta.sum()),
        extra_cells=int((av&~bv).sum()),missing_cells=int((~av&bv).sum()),
        exactly_reproduced=bool(not delta.any()),truncation=truncation,
        price_proxy='CRSP raw close * CRSP share volume; not Sina turnover',
        history_rule='All 252 most recent sessions have a close; not IPO age',
        timing='60-session positive-dollar-volume mean includes signal close; first session monthly selection; daily listed/price gate',
        tie_rule='Stable candidate-column order, ceiling(0.8 * eligible names)',
        caveat='Static Compustat FIC and PRIUSA descriptors are not a historical revisions archive.')
    (output/'universe_audit.json').write_text(json.dumps(report,indent=2))
    return report
