"""Independent raw-download reconstruction and reconciliation, not a certified PIT feed."""
from pathlib import Path
from datetime import datetime, timezone
import json
import shutil
import zipfile
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from .common import digest, calendar, save_parquet
from .fundamentals import normalize_records, feature_events, expand_events, ingest, primary_registry


def inventory(raw):
    rows = []
    for path in sorted(Path(raw).rglob('*.parquet')):
        f = pq.ParquetFile(path)
        rows.append(dict(file=path.name, bytes=path.stat().st_size,
                         rows=f.metadata.num_rows, columns=', '.join(f.schema.names),
                         sha256=digest(path)))
    return pd.DataFrame(rows)


def interval_segments(frame, key, start, end, values):
    """Resolve inclusive dated records; conflicting overlaps remain unmapped."""
    rows, conflicts = [], []
    for identity, group in frame.groupby(key, sort=False):
        events = {}
        for i, r in enumerate(group.to_dict('records')):
            lo, hi = pd.Timestamp(r[start]), pd.Timestamp(r[end])
            if pd.isna(lo) or pd.isna(hi) or lo > hi:
                continue
            value = tuple(None if pd.isna(r[c]) else r[c] for c in values)
            events.setdefault(lo, []).append((i, value))
            events.setdefault(hi + pd.Timedelta(days=1), []).append((i, None))
        active = {}
        dates = sorted(events)
        for j, day in enumerate(dates[:-1]):
            for i, value in events[day]:
                if value is None:
                    active.pop(i, None)
                else:
                    active[i] = value
            distinct = set(active.values())
            if not distinct:
                continue
            stop = dates[j+1] - pd.Timedelta(days=1)
            if len(distinct) > 1:
                conflicts.append({key: identity, 'start': day, 'end': stop})
                # Keep a null segment so asof cannot carry a preceding link into it.
                value = (None,) * len(values)
            else:
                value = next(iter(distinct))
            rows.append(dict(zip([key, 'start', 'end'] + values,
                                 [identity, day, stop] + list(value))))
    return pd.DataFrame(rows), pd.DataFrame(conflicts)


def dated_join(frame, segments, key='permno'):
    left = frame.sort_values('date').copy()
    right = segments.sort_values('start').copy()
    left[key] = left[key].astype('int64')
    right[key] = right[key].astype('int64')
    merged = pd.merge_asof(left, right, by=key, left_on='date', right_on='start')
    stale = merged.date.gt(merged.end) | merged.end.isna()
    merged.loc[stale, [c for c in segments if c not in [key, 'start', 'end']]] = np.nan
    return merged.drop(columns=['start', 'end'])


def prepare_links(raw):
    p = next(Path(raw).rglob('crsp_ccmxpf_linktable.parquet'))
    links = pd.read_parquet(p)
    # Reconstruct issue-level identifiers, including valid secondary issues.
    # P/C-only selection would silently lose J/N issues already in the reference.
    links = links.loc[links.linktype.isin(['LC', 'LU', 'LS'])
                      & links.lpermno.notna()].copy()
    links['gvkey'] = links.gvkey.astype(str).str.zfill(6)
    links['liid'] = links.liid.astype(str).str.zfill(2)
    links['sid'] = 'CNA:' + links.gvkey + ':' + links.liid
    links['permno'] = links.lpermno.astype('int64')
    links['linkdt'] = pd.to_datetime(links.linkdt).fillna(pd.Timestamp('1900-01-01'))
    links['linkenddt'] = pd.to_datetime(links.linkenddt).fillna(pd.Timestamp('2099-12-31'))
    links = links.loc[(links.linkdt <= '2025-12-31') & (links.linkenddt >= '2014-01-01')]
    return links


def _comparison(new, old):
    a, b = np.asarray(new, dtype=float), np.asarray(old, dtype=float)
    both = np.isfinite(a) & np.isfinite(b)
    # Float32 panel storage is allowed; missingness is reported separately.
    same = np.isclose(a[both], b[both], rtol=1e-6, atol=1e-6)
    return dict(common=int(both.sum()), matched=int(same.sum()),
                raw_only=int((np.isfinite(a) & ~np.isfinite(b)).sum()),
                reference_only=int((~np.isfinite(a) & np.isfinite(b)).sum()),
                absolute_error_sum=float(np.abs(a[both]-b[both]).sum()))


def reconstruct_market(raw, reference, output):
    """Stream all records; save the existing SID projection and independent funnels."""
    raw, reference, output = map(Path, [raw, reference, output])
    output.mkdir(parents=True, exist_ok=True)
    target = output/'market_projection.parquet'
    if target.exists():
        raise FileExistsError('Use a new output directory for a clean reconstruction.')
    if shutil.disk_usage(output).free < 650*1024**2:
        raise OSError('At least 650 MiB free is required for the compressed reconstruction.')
    links = prepare_links(raw)
    link_segments, conflicts = interval_segments(links, 'permno', 'linkdt', 'linkenddt', ['sid'])
    save_parquet(links, output/'dated_ccm_links.parquet')
    conflicts.to_csv(output/'ambiguous_ccm_intervals.csv', index=False)
    names = pd.read_parquet(next(raw.rglob('crsp_dsenames.parquet')))
    names['namedt'] = pd.to_datetime(names.namedt)
    names['nameendt'] = pd.to_datetime(names.nameendt).fillna(pd.Timestamp('2099-12-31'))
    names = names.loc[(names.namedt <= '2024-12-31') & (names.nameendt >= '2014-01-01')]
    name_segments, name_conflicts = interval_segments(
        names, 'permno', 'namedt', 'nameendt', ['shrcd', 'exchcd'])
    name_conflicts.to_csv(output/'ambiguous_name_intervals.csv', index=False)
    master = pd.read_parquet(reference/'metadata/security_master.parquet')
    sids = pd.Index(master.sort_values('column_order').sid)
    dates = pd.Index(pd.read_csv(reference/'panels/calendar.csv', dtype=str).date)
    fields = {'close':'prc', 'open':'openprc', 'high':'askhi', 'low':'bidlo',
              'volume':'vol', 'market_cap':'market_cap', 'daily_return':'ret'}
    originals = {f: pd.read_pickle(reference/'panels'/f'{f}.pkl').to_numpy()
                 for f in fields}
    original_u = pd.read_pickle(reference/'panels/universe.pkl').to_numpy()
    counts, comparisons, anomalies, files = [], [], [], []
    writer = None
    candidate_writer = None
    total = saved = 0
    tail_date, tail_sids = None, set()
    try:
        for filename in ['crsp_dsf_2014_2024.parquet', 'crsp_dsf_v2_2025.parquet']:
            path = next(raw.rglob(filename)); f = pq.ParquetFile(path)
            previous_date, previous_ids = None, set()
            columns = ['permno','date','prc','openprc','askhi','bidlo','vol','shrout','ret','retx','cfacpr','cfacshr']
            ciz = 'v2' in filename
            if ciz:
                columns += ['securitytype','securitysubtype','sharetype','primaryexch']
            for batch in f.iter_batches(batch_size=150000, columns=columns):
                d = batch.to_pandas(); d.date = pd.to_datetime(d.date)
                if not d.date.is_monotonic_increasing or (previous_date is not None and d.date.iloc[0] < previous_date):
                    raise ValueError('Raw dates are not sorted; inspect this export before streaming.')
                duplicate = d.duplicated(['date','permno'],keep=False)
                if previous_date is not None:
                    duplicate |= d.date.eq(previous_date) & d.permno.isin(previous_ids)
                if duplicate.any():
                    raise ValueError('Duplicate raw (date, PERMNO) observations.')
                last_date = d.date.iloc[-1]
                last_ids = set(d.loc[d.date.eq(last_date),'permno'])
                previous_ids = previous_ids | last_ids if last_date == previous_date else last_ids
                previous_date = last_date; total += len(d)
                d = dated_join(d, link_segments)
                if ciz:
                    common = d.securitytype.eq('EQTY') & d.securitysubtype.eq('COM') & d.sharetype.eq('NS') & d.primaryexch.isin(['N','A','Q'])
                else:
                    d = dated_join(d, name_segments)
                    common = d.shrcd.isin([10,11]) & d.exchcd.isin([1,2,3])
                d['quote_price'] = d.prc.lt(0)
                d['prc'] = d.prc.abs()
                d['market_cap'] = d.prc*d.shrout*1000
                valid_price = d.prc.gt(0) & ~d.quote_price
                tradable = common & valid_price & d.vol.gt(0)
                d['common_stock'] = common.astype('uint8')
                candidate = d.loc[tradable & d.sid.notna(),['date','sid','permno','market_cap']].copy()
                candidate['classification_basis'] = 'CIZ_incomplete_type_filter' if ciz else 'SIZ_SHRCD_10_11'
                candidate_table = pa.Table.from_pandas(candidate, preserve_index=False)
                if candidate_writer is None:
                    candidate_writer = pq.ParquetWriter(output/'universe_candidates.parquet',candidate_table.schema,compression='zstd')
                candidate_writer.write_table(candidate_table)
                funnel = pd.DataFrame({'date':d.date, 'raw_rows':1,
                    'common_stock':common.astype(int), 'positive_trade':tradable.astype(int),
                    'dated_ccm':(tradable & d.sid.notna()).astype(int)})
                for cap in [0,100000000,300000000,500000000,1000000000]:
                    funnel[f'cap_ge_{cap}'] = (tradable & d.sid.notna() & d.market_cap.ge(cap)).astype(int)
                counts.append(funnel.groupby('date').sum())
                p = d.loc[d.sid.isin(sids)].copy()
                p['source_format'] = 'CIZ' if ciz else 'SIZ'
                # The projection is for reconciliation, never a newly inferred universe.
                di = dates.get_indexer(p.date.dt.strftime('%Y%m%d'))
                si = sids.get_indexer(p.sid)
                assert (di >= 0).all() and (si >= 0).all()
                p['reference_universe'] = original_u[di,si].astype('uint8')
                dup = p.duplicated(['date','sid'], keep=False)
                if tail_date is not None:
                    dup |= p.date.eq(tail_date) & p.sid.isin(tail_sids)
                if dup.any():
                    raise ValueError('Multiple PERMNOs map to the same SID/date; inspect CCM before proceeding.')
                if len(p):
                    last = p.date.max()
                    same_day = set(p.loc[p.date.eq(last),'sid'])
                    tail_sids = tail_sids | same_day if last == tail_date else same_day
                    tail_date = last
                err = np.maximum(p.bidlo-np.minimum(p.openprc,p.prc), np.maximum(p.openprc,p.prc)-p.askhi)
                bad = ((err > .01001) & (err/p.prc > .001)) | p.askhi.lt(p.bidlo)
                if bad.any():
                    anomalies.append(p.loc[bad,['date','sid','permno','openprc','askhi','bidlo','prc']])
                for field, col in fields.items():
                    stat = _comparison(p[col], originals[field][di,si])
                    comparisons.append(dict(field=field, **stat))
                keep = ['date','sid','permno','source_format','prc','openprc','askhi','bidlo','vol','shrout','market_cap','ret','retx','cfacpr','cfacshr','quote_price','common_stock','reference_universe']
                table = pa.Table.from_pandas(p[keep], preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(target, table.schema, compression='zstd')
                writer.write_table(table); saved += len(p)
                if total % 1500000 == 0:
                    print(f'Market records scanned: {total:,}; projected: {saved:,}', flush=True)
            files.append(dict(file=filename, rows=f.metadata.num_rows))
    finally:
        if writer is not None:
            writer.close()
        if candidate_writer is not None:
            candidate_writer.close()
    daily = pd.concat(counts).groupby(level=0).sum().sort_index()
    daily['reference_universe'] = original_u.sum(axis=1)[dates.get_indexer(daily.index.strftime('%Y%m%d'))]
    daily.to_csv(output/'universe_funnel.csv', index_label='date')
    comp = pd.DataFrame(comparisons).groupby('field').sum()
    for field in comp.index:
        comp.loc[field,'reference_only'] = int(np.isfinite(originals[field]).sum())-comp.loc[field,'common']
    comp['match_fraction_on_common'] = comp.matched/comp.common
    comp['mean_absolute_error'] = comp.absolute_error_sum/comp.common
    comp.to_csv(output/'market_reconciliation.csv')
    pd.concat(anomalies, ignore_index=True).to_csv(output/'ohlc_exceptions.csv',index=False) if anomalies else pd.DataFrame().to_csv(output/'ohlc_exceptions.csv', index=False)
    delist = pd.read_parquet(next(raw.rglob('crsp_dsedelist.parquet')))
    delist = delist.loc[pd.to_datetime(delist.dlstdt).between('2014-01-01','2025-12-31')]
    save_parquet(delist, output/'delisting_events.parquet')
    # SIZ RET and DLRET are separate inputs. CIZ RET must not receive DLRET twice.
    report = dict(raw_rows=total, projected_rows=saved, ccm_conflict_intervals=len(conflicts),
                  name_conflict_intervals=len(name_conflicts), delisting_events=len(delist), files=files,
                  classification_limit='2025 lacks USIncFlg, IssuerType, ConditionalType, TradingStatusFlg and price flags; CIZ candidate counts are not equivalent to legacy SHRCD 10/11.',
                  status='reconciliation_only_not_a_replacement_research_snapshot')
    (output/'market_audit.json').write_text(json.dumps(report, indent=2))
    return daily, comp, report


def reconstruct_fundamentals(raw, reference, output):
    raw, reference, output = map(Path, [raw, reference, output])
    q = pd.read_parquet(next(raw.rglob('compustat_fundq_2014_2026.parquet')))
    q['gvkey'] = q.gvkey.astype(str).str.zfill(6)
    q['source_row'] = np.arange(len(q))
    q = q.loc[q.indfmt.eq('INDL') & q.datafmt.eq('STD') & q.consol.eq('C') & q.curcdq.eq('USD')].copy()
    q['period_end'] = pd.to_datetime(q.datadate)
    # These fields are absent in the supplied raw export, not zero-valued.
    for field, missing in [('fdateq',pd.NaT),('ltq',np.nan)]:
        if field not in q:
            q[field] = missing
    q['revenue_q'] = q.revtq.combine_first(q.saleq) if 'revtq' in q else q.saleq
    dates = pd.Index(pd.read_csv(reference/'panels/calendar.csv', dtype=str).date, name='date')
    sids = pd.Index(pd.read_parquet(reference/'metadata/security_master.parquet').sort_values('column_order').sid, name='sid')
    excluded, records = normalize_records(q, dates, allow_final_date_fallback=False)
    records['sid'] = records.gvkey  # Temporary company key; security mapping follows each field's release.
    links = prepare_links(raw)
    fields = ['shares_out','book_equity','earnings_ttm','revenue_ttm','total_assets','cash',
              'long_term_debt','short_term_debt','operating_cash_flow_q','operating_cash_flow_ttm',
              'capital_expenditures_q','capital_expenditures_ttm']
    records['earnings_q'] = records.niq*1e6
    records['revenue_q'] = records.revenue_q*1e6
    fields += ['earnings_q','revenue_q']
    quarterly_cols = ['gvkey','period_end','fyearq','fqtr','rdq','available_ts','availability_basis']+fields
    quarterly_cols += [c for c in records if c.endswith('_available_day')]
    save_parquet(records[quarterly_cols],output/'fundamental_quarterly_records.parquet')
    all_events, comparisons = [], []
    for field in fields:
        e = feature_events(records, field).rename(columns={'sid':'gvkey'})
        e = e.merge(links[['gvkey','sid','permno','linkdt','linkenddt']].drop_duplicates(), on='gvkey')
        e = e.loc[e.available_date.between(e.linkdt,e.linkenddt)
                  & e.available_date.le('2025-12-31')].copy()
        e = e.drop_duplicates(['sid','available_date','period_end','value'])
        ambiguous = e.duplicated(['sid','available_date'],keep=False)
        if ambiguous.any():
            raise ValueError(f'Ambiguous fundamental security events: {field}')
        e['field'] = field
        all_events.append(e[['sid','period_end','available_date','field','value','availability_basis','source_row']])
        panel = expand_events(e.loc[e.sid.isin(sids)], dates, sids)
        ref = pd.read_pickle(reference/'panels'/f'{field}.pkl')
        stat = _comparison(panel.to_numpy().ravel(), ref.to_numpy().ravel())
        comparisons.append(dict(field=field, **stat))
        a,b = panel.to_numpy(),ref.to_numpy()
        different = np.isfinite(a)&np.isfinite(b)&~np.isclose(a,b,rtol=1e-6,atol=1e-6)
        di,si = np.where(different)
        if len(di):
            order = np.argsort(np.abs(a[di,si]-b[di,si]))[-20:][::-1]
            pd.DataFrame({'date':dates[di[order]],'sid':sids[si[order]],
                'raw_rebuilt':a[di[order],si[order]],'reference':b[di[order],si[order]]}).to_csv(
                    output/f'financial_differences_{field}.csv',index=False)
    events = pd.concat(all_events, ignore_index=True)
    save_parquet(events, output/'fundamental_events.parquet')
    comp = pd.DataFrame(comparisons).set_index('field')
    comp['match_fraction_on_common'] = comp.matched/comp.common
    comp.to_csv(output/'fundamental_reconciliation.csv')
    audit = dict(raw_filtered_rows=len(q), conflicting_period_rows=int(excluded.period_conflict.sum()),
                 missing_release_rows=int((~excluded.date_usable).sum()), output_events=len(events),
                 missing_source_fields=['ltq','fdateq','revtq','ajexq'],
                 limitations=['RDQ is an earnings announcement proxy, not field-specific publication proof.',
                              'Latest-download values may contain historical restatements.',
                              '2014 start cannot reproduce fundamentals released before 2014.',
                              'Total liabilities are unavailable; debt is not substituted for liabilities.'])
    (output/'fundamental_audit.json').write_text(json.dumps(audit,indent=2))
    return comp, audit


def new_output(root):
    output = Path(root)/('wrds_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f'))
    output.mkdir(parents=True, exist_ok=False)
    return output


def reconstruct_quarterly_archive(source, reference, output):
    """Reproduce the reference financial projection from the identified original ZIP."""
    source,output = Path(source),Path(output)
    reference = Path(reference) if reference is not None else None
    target = output/'quarterly_selected.parquet'
    if target.exists():
        raise FileExistsError('Use a fresh output directory; do not reuse selected-column stages.')
    q,provenance = ingest(source,target,min_year=2010)
    dates = (pd.Index(pd.read_csv(reference/'panels/calendar.csv',dtype=str).date,name='date')
             if reference is not None else calendar('2014-01-01','2025-12-31'))
    sids = (pd.Index(pd.read_parquet(reference/'metadata/security_master.parquet').sort_values('column_order').sid,name='sid')
            if reference is not None else None)
    excluded,records = normalize_records(q,dates,allow_final_date_fallback=True)
    master,features,mapping = primary_registry(q,records)
    features['earnings_q'] = features.niq*1e6
    features['revenue_q'] = features.revenue_q*1e6
    fields = ['shares_out','book_equity','earnings_ttm','revenue_ttm','total_assets','total_liabilities',
              'cash','long_term_debt','short_term_debt','operating_cash_flow_q','operating_cash_flow_ttm',
              'capital_expenditures_q','capital_expenditures_ttm','earnings_q','revenue_q']
    # Preserve source features independently of whether an SID appears in the reference.
    save_parquet(features,output/'fundamental_quarterly_records.parquet')
    save_parquet(master,output/'financial_security_registry.parquet')
    all_events,comparison = [],[]
    for field in fields:
        e = feature_events(features,field)
        e = e.loc[e.available_date.le('2025-12-31')].copy()
        e['field'] = field
        e['source_vintage'] = provenance['source_sha256']
        all_events.append(e[['sid','period_end','available_date','field','value','availability_basis','source_row','source_vintage']])
        if reference is not None:
            panel = expand_events(e.loc[e.sid.isin(sids)],dates,sids)
            ref = pd.read_pickle(reference/'panels'/f'{field}.pkl')
            stats = _comparison(panel.to_numpy().ravel(),ref.to_numpy().ravel())
            stats['all_cells_identical'] = bool(np.array_equal(panel.to_numpy(),ref.to_numpy(),equal_nan=True))
            comparison.append(dict(field=field,**stats))
    events = pd.concat(all_events,ignore_index=True)
    save_parquet(events,output/'fundamental_events.parquet')
    comp = pd.DataFrame(comparison).set_index('field') if comparison else pd.DataFrame()
    if reference is not None:
        comp['match_fraction_on_common'] = comp.matched/comp.common
        comp.to_csv(output/'fundamental_reconciliation.csv')
        original_vintage = pd.read_parquet(reference/'metadata/fundamental_vintages.parquet',columns=['source_vintage']).source_vintage.unique()
    audit = dict(raw_scanned_rows=provenance['scanned_rows'],raw_filtered_rows=len(q),
        source_sha256=provenance['source_sha256'],
        conflicting_period_rows=int(excluded.period_conflict.sum()),
        missing_release_rows=int((~excluded.date_usable).sum()),output_events=len(events),
        total_fields=len(fields),mapping=mapping,
        limitations=['Source PRIUSA tags are not verified historical primary-issue status.',
                     'RDQ/FDATEQ are availability proxies, not field-specific publication proof.',
                     'Latest-download values may contain historical restatements.'])
    if reference is not None:
        audit.update(source_hash_matches_reference=bool(len(original_vintage)==1 and original_vintage[0]==provenance['source_sha256']),
                     exact_fields=int(comp.all_cells_identical.sum()))
    (output/'fundamental_audit.json').write_text(json.dumps(audit,indent=2))
    return comp,audit


def reconstruct_industry_archive(source, reference, output):
    """Stream historical SICH directly from the original North America Stata ZIP."""
    source,output = Path(source),Path(output)
    reference = Path(reference) if reference is not None else None
    master = pd.read_parquet(output/'financial_security_registry.parquet')
    wanted = set(master.gvkey)
    columns = ['gvkey','datadate','fdate','pdate','sich','indfmt','consol','datafmt','curcd']
    frames = []; scanned = 0
    with zipfile.ZipFile(source) as archive:
        members = [x for x in archive.namelist() if x.endswith('data.dta') and not x.startswith('__MACOSX/')]
        if len(members)!=1:
            raise ValueError('Expected one North America annual Stata source.')
        with archive.open(members[0]) as stream:
            # ZipExtFile is seekable: pandas reads selected rows without a 6.5 GB extraction.
            with pd.read_stata(stream,columns=columns,chunksize=10000,convert_categoricals=False) as reader:
                first_chunk = True
                for ch in reader:
                    if first_chunk:
                        # Load label metadata once through the public API. Otherwise
                        # pandas repeatedly seeks to the ZIP member's tail per chunk.
                        reader.value_labels()
                        first_chunk = False
                    scanned += len(ch)
                    ch.gvkey = ch.gvkey.astype(str).str.zfill(6)
                    keep = (ch.gvkey.isin(wanted)&ch.indfmt.eq('INDL')&ch.consol.eq('C')
                            & ch.datafmt.eq('STD')&ch.curcd.eq('USD')&ch.datadate.dt.year.between(2010,2024))
                    frames.append(ch.loc[keep,['gvkey','datadate','fdate','pdate','sich']].copy())
    annual = pd.concat(frames,ignore_index=True)
    annual['period_end'] = pd.to_datetime(annual.datadate)
    final = pd.to_datetime(annual.fdate,errors='coerce')
    preliminary = pd.to_datetime(annual.pdate,errors='coerce')
    release = final.where((final-annual.period_end).dt.days.between(0,365))
    release = release.fillna(preliminary.where((preliminary-annual.period_end).dt.days.between(0,365)))
    annual['release'] = release
    conflicts = annual.duplicated(['gvkey','period_end'],keep=False)
    annual = annual.loc[~conflicts & annual.sich.notna() & annual.release.notna()].copy()
    annual = annual.merge(master[['gvkey','sid']],on='gvkey',validate='many_to_one')
    dates = (pd.Index(pd.read_csv(reference/'panels/calendar.csv',dtype=str).date,name='date')
             if reference is not None else calendar('2014-01-01','2025-12-31'))
    sids = (pd.Index(pd.read_parquet(reference/'metadata/security_master.parquet').sort_values('column_order').sid,name='sid')
            if reference is not None else pd.Index(master.sid,name='sid'))
    lower = min(pd.to_datetime(dates[0]),annual.release.min())
    upper = max(pd.to_datetime(dates[-1]),annual.release.max())+pd.Timedelta(days=14)
    sessions = pd.to_datetime(calendar(str(lower.date()),str(upper.date())),format='%Y%m%d').to_numpy()
    positions = np.searchsorted(sessions,annual.release.to_numpy(),side='right')
    annual['available_date'] = pd.to_datetime(sessions[positions])
    annual['value'] = 'SIC2:'+pd.to_numeric(annual.sich).astype('int64').astype(str).str.zfill(4).str[:2]
    annual = annual.sort_values(['sid','available_date','period_end']).drop_duplicates(['sid','available_date'],keep='last')
    annual = annual.loc[annual.period_end.eq(annual.groupby('sid').period_end.cummax())
                        & annual.available_date.le('2025-12-31')]
    events = annual[['sid','period_end','available_date','value']]
    save_parquet(events,output/'industry_events.parquet')
    if reference is None:
        report = dict(source_sha256=digest(source),raw_rows=scanned,events=len(events),
            covered_securities=int(events.sid.nunique()),conflicts_excluded=int(conflicts.sum()),
            release_policy='Valid FDATE, otherwise PDATE; first strictly following NYSE session; forward only.',
            caveat='Historical revisions are unavailable; release dates are proxies.')
        (output/'industry_audit.json').write_text(json.dumps(report,indent=2))
        return report
    old = pd.read_pickle(reference/'panels/industry.pkl')
    day = pd.to_datetime(dates,format='%Y%m%d').to_numpy()
    matched = missing = cells = new_only = conflicting_values = 0
    differences = []
    groups = {sid:g for sid,g in events.groupby('sid')}
    for sid in sids:
        rebuilt = np.full(len(dates),'',dtype=object)
        if sid in groups:
            g = groups[sid]
            pos = np.searchsorted(g.available_date.to_numpy(),day,side='right')-1
            valid = pos>=0
            rebuilt[valid] = g.value.to_numpy()[pos[valid]]
        ref = old[sid].astype(object).fillna('').to_numpy()
        matched += int((rebuilt==ref).sum()); cells += len(ref)
        missing += int(((rebuilt=='')&(ref!='')).sum())
        new_only += int(((rebuilt!='')&(ref=='')).sum())
        conflicting_values += int(((rebuilt!='')&(ref!='')&(rebuilt!=ref)).sum())
        where = np.flatnonzero(rebuilt!=ref)
        if len(where):
            differences.append(dict(sid=sid,cells=len(where),first=dates[where[0]],last=dates[where[-1]],
                                    new_value=rebuilt[where[0]],reference_value=ref[where[0]]))
    pd.DataFrame(differences).to_csv(output/'industry_differences.csv',index=False)
    report = dict(source_sha256=digest(source),raw_rows=scanned,events=len(events),
        matching_cells=matched,total_cells=cells,all_cells_identical=matched==cells,reference_only=missing,
        new_only=new_only,conflicting_values=conflicting_values,
        conflicts_excluded=int(conflicts.sum()),
        release_policy='Valid FDATE, otherwise PDATE; first strictly following NYSE session; forward only.',
        caveat='Historical SICH is used; release dates are proxies and historical revisions remain unavailable.')
    (output/'industry_audit.json').write_text(json.dumps(report,indent=2))
    return report


def reconcile_delistings(raw, reference, output):
    """Identify the reference's next-session terminal postings without changing CIZ RET."""
    raw,reference,output = map(Path,[raw,reference,output])
    links = prepare_links(raw)
    segments,_ = interval_segments(links,'permno','linkdt','linkenddt',['sid'])
    d = pd.read_parquet(next(raw.rglob('crsp_dsedelist.parquet')))
    d['date'] = pd.to_datetime(d.dlstdt)
    d = d.loc[d.date.between('2014-01-01','2025-12-31')].copy()
    d = dated_join(d,segments).rename(columns={'date':'source_event_date'})
    dates = pd.Index(pd.read_csv(reference/'panels/calendar.csv',dtype=str).date)
    sessions = pd.to_datetime(dates,format='%Y%m%d').to_numpy()
    pos = np.searchsorted(sessions,d.source_event_date.to_numpy(),side='right')
    d = d.loc[pos<len(sessions)].copy()
    d['date'] = dates.to_numpy()[pos[pos<len(sessions)]]
    d = d.loc[d.sid.notna()]
    ambiguous = d.duplicated(['date','sid'],keep=False)
    d = d.loc[~ambiguous]
    terminal = pd.read_pickle(reference/'panels/delisting_event.pkl').eq(1)
    vendor = pd.read_pickle(reference/'panels/daily_return.pkl')
    i,j = np.where(terminal)
    observed = pd.DataFrame({'date':vendor.index[i],'sid':vendor.columns[j],
                             'reference_return':vendor.to_numpy()[i,j]})
    result = observed.merge(d[['date','sid','permno','source_event_date','dlret','dlstcd']],
                            on=['date','sid'],how='left',validate='one_to_one')
    result['dlret_matches_reference'] = np.isclose(result.dlret,result.reference_return,rtol=1e-6,atol=1e-6)
    save_parquet(result,output/'terminal_posting_reconciliation.parquet')
    report = dict(reference_terminal_rows=len(result),matched_source_dlret=int(result.dlret_matches_reference.sum()),
        unmatched_source_events=int(result.source_event_date.isna().sum()),
        convention='Reference terminal observations are compared with the first NYSE session strictly after DLSTDT.',
        caution='Posting convention is disclosed separately from the event date. Do not add DLRET to CIZ RET again.')
    (output/'delisting_audit.json').write_text(json.dumps(report,indent=2))
    return report


def reconstruct_price_targets(reference, output):
    """Create total-return-linked close marks without crossing identity/data gaps."""
    from usq.data.labels import close_forward_returns
    reference, output = Path(reference), Path(output)
    p = pd.read_parquet(output/'market_projection.parquet',columns=[
        'date','sid','permno','prc','ret','quote_price','openprc','askhi','bidlo'])
    dates = pd.Index(pd.read_csv(reference/'panels/calendar.csv',dtype=str).date,name='date')
    sids = pd.Index(pd.read_parquet(reference/'metadata/security_master.parquet').sort_values('column_order').sid,name='sid')
    p['session'] = dates.get_indexer(p.date.dt.strftime('%Y%m%d'))
    p = p.sort_values(['sid','session']).reset_index(drop=True)
    error = np.maximum(p.bidlo-np.minimum(p.openprc,p.prc),np.maximum(p.openprc,p.prc)-p.askhi)
    bad = ((error>.01001)&(error/p.prc>.001)) | p.askhi.lt(p.bidlo)
    good = p.prc.gt(0) & ~p.quote_price & ~bad
    previous = p.shift()
    linked = (good & good.shift(fill_value=False) & p.sid.eq(previous.sid)
              & p.permno.eq(previous.permno) & p.session.sub(previous.session).eq(1)
              & np.isfinite(p.ret) & p.ret.gt(-1))
    group = (~linked).cumsum()
    logret = np.log1p(p.ret.where(linked,0))
    anchor = np.log(p.prc.where(good)).groupby(group).transform('first')
    p['close_adj'] = np.exp(anchor+logret.groupby(group).cumsum()).where(good)
    p['segment'] = group.where(good,0).astype('int64')
    a = np.full((len(dates),len(sids)),np.nan,dtype='float32')
    seg = np.zeros(a.shape,dtype='float32')
    di,si = p.session.to_numpy(),sids.get_indexer(p.sid)
    a[di,si] = p.close_adj.to_numpy();seg[di,si] = p.segment.to_numpy()
    cp = pd.DataFrame(a,index=dates,columns=sids)
    segments = pd.DataFrame(seg,index=dates,columns=sids)
    target,end = close_forward_returns(cp,10,segments)
    assert target.tail(11).isna().all().all()
    # Compare returns rather than arbitrary adjusted-price anchor levels.
    oldcp = pd.read_pickle(reference/'panels/close_adj.pkl')
    oldseg = pd.read_pickle(reference/'panels/adjustment_segment.pkl')
    oldtarget,_ = close_forward_returns(oldcp,10,oldseg)
    stats = _comparison(target.to_numpy().ravel(),oldtarget.to_numpy().ravel())
    p['next_close_10d'] = target.to_numpy()[di,si]
    p['label_end'] = end.to_numpy()[di]
    save_parquet(p[['date','sid','permno','close_adj','segment','next_close_10d','label_end']],output/'price_targets.parquet')
    report = dict(formula='close_adj[t+11]/close_adj[t+1]-1',quarantined_bars=int(bad.sum()),
        quote_marks=int(p.quote_price.sum()),comparison=stats,
        note='Terminal events are retained separately. A target with no executable exit stays missing; it is not a zero return.')
    (output/'target_audit.json').write_text(json.dumps(report,indent=2))
    return report


def universe_diagnostics(output):
    rows = []
    source = pq.ParquetFile(Path(output)/'market_projection.parquet')
    for batch in source.iter_batches(columns=['date','market_cap','common_stock','reference_universe']):
        d = batch.to_pandas()
        d = d.loc[d.reference_universe.eq(1)]
        z = pd.DataFrame({'year':d.date.dt.year,'observations':1,
                          'outside_strict_common':d.common_stock.eq(0).astype(int)})
        for cap in [100000000,300000000,500000000]:
            z[f'below_{cap}'] = d.market_cap.lt(cap).astype(int)
        rows.append(z.groupby('year').sum())
    result = pd.concat(rows).groupby(level=0).sum()
    for col in result.columns.difference(['observations']):
        result[col] = result[col]/result.observations
    result.to_csv(Path(output)/'reference_universe_diagnostics.csv')
    return result
