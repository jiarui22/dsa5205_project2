"""Complete a trusted Team17 export using local, reconciled source material.

Never edits inputs or overwrites an existing snapshot. This is not a market
download, stock selection, factor experiment, or historical-revisions archive.
Run with ``python -m usq.database.complete --help``.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import gzip
import json
from pathlib import Path
import platform
import shutil

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .common import calendar, compact_dates, digest, dump, save_panel, save_parquet, timestamps_from_days
from .fundamentals import CORE, STOCK_MAP, expand_events, feature_events


RESTORE = ['shares_out', 'cash', 'short_term_debt', 'operating_cash_flow_q',
           'capital_expenditures_q', 'capital_expenditures_ttm', 'earnings_q', 'revenue_q']
RECONCILE = ['book_equity', 'earnings_ttm', 'revenue_ttm', 'total_assets',
             'total_liabilities', 'long_term_debt', 'operating_cash_flow_ttm']
FINANCIAL = list(STOCK_MAP) + ['earnings_ttm', 'revenue_ttm',
    'operating_cash_flow_q', 'operating_cash_flow_ttm',
    'capital_expenditures_q', 'capital_expenditures_ttm', 'earnings_q', 'revenue_q']


def write_checksums(root):
    """Whole-snapshot hashes, excluding only the checksum file itself."""
    root = Path(root)
    target = root / 'checksums.sha256'
    lines = [digest(path) + '  ' + path.relative_to(root).as_posix()
             for path in sorted(root.rglob('*')) if path.is_file() and path != target]
    target.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def explicit_subset(frame, dates, sids):
    """Check the entire source before a declared, exact output selection."""
    if frame.index.has_duplicates or frame.columns.has_duplicates:
        raise ValueError('Duplicate source panel axes')
    if not frame.index.is_monotonic_increasing:
        raise ValueError('Unsorted source dates')
    if len(dates.difference(frame.index)) or len(sids.difference(frame.columns)):
        raise ValueError('Missing requested dates/SIDs; never silently intersect')
    return frame.loc[dates, sids].rename_axis(index='date', columns='sid')


def raw_from_adjusted(adjusted, factor):
    if not adjusted.index.equals(factor.index) or not adjusted.columns.equals(factor.columns):
        raise ValueError('Inverse-adjustment axes differ')
    return adjusted.div(factor.where(factor.gt(0))).astype('float32')


def verified_amount_pair(original, current, ticker_matches):
    """Keep paired vendor amount/volume only with dated identity and OHLC checks.

    Never use the current CRSP volume to invert an unrelated vendor VWAP.
    This validation can retain source observations, not manufacture missing ones.
    """
    good = np.asarray(ticker_matches, dtype=bool).copy()
    for key in ['open', 'high', 'low', 'close']:
        old, now = np.asarray(original[key], float), np.asarray(current[key], float)
        error = np.abs(old - now)
        good &= np.isfinite(old) & np.isfinite(now) & (old > 0) & (now > 0)
        good &= (error <= .01001) & (error / np.where(now > 0, now, np.nan) <= .001)
    amount, volume = np.asarray(original['amount'], float), np.asarray(original['volume'], float)
    good &= np.isfinite(amount) & np.isfinite(volume) & (amount > 0) & (volume > 0)
    vwap = amount / np.where(volume > 0, volume, np.nan)
    good &= (vwap >= np.asarray(current['low'])*.995) & (vwap <= np.asarray(current['high'])*1.005)
    return (np.where(good, amount, np.nan), np.where(good, volume, np.nan),
            np.where(good, vwap, np.nan))


def quarterly_view(features, columns, fields, end):
    """Retain field-specific availability, including delayed TTM constituents.

    A quarter's own RDQ cannot stand in for the availability of every derived
    TTM/YTD-difference value on that row. This table is a source-event table,
    not an as-of panel: consumers must use each field's availability timestamp.
    """
    result = features[columns].copy()
    end = pd.Timestamp(end)
    for field in fields:
        day = features[field + '_available_day'] if field + '_available_day' in features else features.available_day
        when = timestamps_from_days(day)
        usable = when.notna() & (when <= end)
        result[field + '_available_ts'] = when.where(usable)
        result[field] = result[field].where(usable)
    known = features.available_ts.notna() & (features.available_ts <= end)
    return result.loc[known].copy(), result.loc[~known].copy()


def dated_permno(history, dates, sids):
    """Source history projection, not a newly independently verified CCM table."""
    day = pd.to_datetime(dates, format='%Y%m%d').to_numpy(dtype='datetime64[ns]')
    values = np.zeros((len(dates), len(sids)), dtype='int32')
    conflicts = np.zeros(values.shape, dtype=bool)
    positions = {sid: i for i, sid in enumerate(sids)}
    for sid, group in history.groupby('sid', sort=False):
        if sid not in positions:
            raise ValueError('History includes SID outside frozen pool')
        col = positions[sid]
        for row in group.itertuples():
            mask = (day >= np.datetime64(row.effective_start)) & (day <= np.datetime64(row.effective_end))
            conflicts[:, col] |= mask & (values[:, col] != 0) & (values[:, col] != row.permno)
            values[mask, col] = row.permno
    out = values.astype('float32')
    out[(values == 0) | conflicts] = np.nan
    return pd.DataFrame(out, index=dates, columns=sids), conflicts


def spy_frames(cache, dates):
    sidecar = json.loads(cache.with_suffix('.json').read_text())
    if digest(cache) != sidecar['sha256']:
        raise ValueError('SPY raw cache hash mismatch')
    obj = json.loads(gzip.decompress(cache.read_bytes()))['chart']['result'][0]
    if obj['meta']['symbol'] != 'SPY' or obj['meta'].get('currency') != 'USD':
        raise ValueError('Wrong benchmark identity or units')
    keys = pd.to_datetime(obj['timestamp'], unit='s', utc=True).tz_convert('America/New_York').strftime('%Y%m%d')
    if keys.has_duplicates:
        raise ValueError('Duplicate SPY source session')
    q = obj['indicators']['quote'][0]
    close = np.asarray(q['close'], dtype=float)
    adjclose = np.asarray(obj['indicators']['adjclose'][0]['adjclose'], dtype=float)
    factor = adjclose / np.where(close > 0, close, np.nan)
    frames = {}
    for name, values in [('total_return_close', adjclose),
                         ('total_return_open', np.asarray(q['open'], float) * factor),
                         ('close', close)]:
        source = pd.DataFrame({'SPY': values}, index=keys)
        if len(dates.difference(source.index)):
            raise ValueError('SPY source missing calendar dates')
        f = source.loc[dates].astype('float32').rename_axis(index='date', columns='index_symbol')
        if not np.isfinite(f.to_numpy()).all() or not f.gt(0).all().all():
            raise ValueError('Invalid SPY benchmark values')
        frames[name] = f
    return frames, sidecar


def complete(args):
    out = Path(args.output)
    if out.exists():
        raise FileExistsError('Use a new output directory; no snapshot overwrite')
    if shutil.disk_usage(out.parent).free < 6 * 1024**3:
        raise OSError('Reserve at least 6 GiB before completing this snapshot')
    team, parent, stage = Path(args.team), Path(args.parent), Path(args.stage)
    original_panels = team / 'us_stock/panels'
    original_meta = team / 'us_stock/metadata'
    inputs = {}

    def record(path, label):
        path = Path(path)
        inputs[label] = {'filename': path.name, 'bytes': path.stat().st_size, 'sha256': digest(path)}
        return path

    def load_team(field):
        path = original_panels / (field + '.pkl')
        frame = pd.read_pickle(path)
        if not frame.index.equals(source_dates) or not frame.columns.equals(source_sids):
            raise ValueError(f'Complete original panel axes differ: {field}')
        return explicit_subset(frame, dates, sids)

    def write_table(frame, name):
        if 'sid' in frame and not set(frame.sid).issubset(set(sids)):
            raise ValueError(f'Table outside frozen pool: {name}')
        save_parquet(frame, metadata / name)
        tables[name] = {'sha256': digest(metadata / name), 'rows': len(frame),
                        'columns': list(frame.columns)}

    def write_field(frame, name, source, unit, notes='', kind='numeric'):
        entries[name] = save_panel(frame, panels / (name + '.pkl'), dates, sids, kind)
        entries[name].update({'source': source, 'unit': unit, 'notes': notes})
        good = frame.notna().to_numpy(copy=True)
        if kind == 'numeric':
            good &= np.isfinite(frame.to_numpy())
            if name in {'open', 'high', 'low', 'close', 'open_adj', 'high_adj', 'low_adj',
                         'close_adj', 'adj_factor', 'vwap', 'vwap_adj', 'market_cap',
                         'market_cap_split_synced', 'amount', 'amount_source_volume'}:
                good &= frame.to_numpy() > 0
            elif name == 'volume':
                good &= frame.to_numpy() >= 0
        coverage[name] = {'missing_pct_on_supplied_universe': float(100*(umask & ~good).sum()/umask.sum())}
        print(json.dumps({'stage': 'write', 'field': name, **coverage[name]}), flush=True)

    # A deliberate scope decision, not an inferred intersection: retain precisely
    # the supplied universe axes, including 2014 price warm-up; stop at 2025-12-31.
    upath = record(args.universe, 'supplied_universe')
    spath = record(args.stocks, 'supplied_stocks')
    supplied = pd.read_parquet(upath)
    if supplied.isna().any().any() or not supplied.isin([0, 1]).all().all():
        raise ValueError('Unknown or invalid supplied universe')
    dates = pd.Index(supplied.index.strftime('%Y%m%d'), name='date')
    sids = pd.Index(supplied.columns.astype(str), name='sid')
    expected = calendar(pd.Timestamp(dates[0]).strftime('%Y-%m-%d'), pd.Timestamp(dates[-1]).strftime('%Y-%m-%d'))
    if not dates.equals(expected) or sids.has_duplicates:
        raise ValueError('Universe differs from independent calendar/unique SID axes')
    universe = pd.DataFrame(supplied.to_numpy(dtype='uint8'), index=dates, columns=sids)
    umask = universe.eq(1).to_numpy()
    summary = pd.read_parquet(spath)
    if summary.sid.duplicated().any() or set(summary.sid) != set(sids):
        raise ValueError('Stock summary SID set differs')
    summary = summary.set_index('sid').loc[sids].reset_index()
    if not np.array_equal(summary.eligible_days.to_numpy(), umask.sum(axis=0)):
        raise ValueError('Stock summary eligible days differ')
    source_close = pd.read_pickle(original_panels / 'close.pkl')
    source_dates, source_sids = source_close.index, source_close.columns
    explicit_subset(source_close, dates, sids)
    del source_close, supplied

    for p in sorted(original_panels.glob('*.pkl')):
        record(p, 'team/panels/' + p.name)
    source_snapshot = json.loads(record(original_meta / 'snapshot.json', 'team/snapshot').read_text())
    features = pd.read_parquet(record(stage / 'financial_features.parquet', 'financial_features_stage'))
    source_meta = json.loads(record(stage / 'quarterly_selected.json', 'quarterly_source_metadata').read_text())
    if set(features.source_vintage.dropna().unique()) != {source_meta['source_sha256']}:
        raise ValueError('Financial stage vintage is inconsistent')
    if len(set(sids) - set(features.sid)) or features.duplicated(['sid', 'period_end']).any():
        raise ValueError('Missing SID or ambiguous fiscal period in financial stage')
    # Preserve a verified, selected-column stage; verify it is the same stage used
    # by the local parent build, not an unrelated similarly named file.
    parent_snapshot = json.loads(record(parent / 'metadata/snapshot.json', 'parent/snapshot').read_text())
    if parent_snapshot['source']['source_sha256'] != source_meta['source_sha256']:
        raise ValueError('Parent and feature-stage source vintages differ')
    features = features.loc[features.sid.isin(sids)].copy()
    features['earnings_q'] = features.niq * 1e6
    features['revenue_q_source_musd'] = features.revenue_q
    features['revenue_q'] = features.revenue_q * 1e6
    reconciliation = []
    for field in RECONCILE:
        observed = load_team(field)
        rebuilt = expand_events(feature_events(features, field), dates, sids, max_age_days=550)
        same = np.array_equal(observed.to_numpy(), rebuilt.to_numpy(), equal_nan=True)
        reconciliation.append({'field': field, 'all_cells_identical': bool(same)})
        if not same:
            raise ValueError(f'Financial source reconciliation failed: {field}')
        del observed, rebuilt
    print(json.dumps({'stage': 'source_reconciliation', 'fields_identical': len(reconciliation)}), flush=True)

    panels, metadata = out / 'panels', out / 'metadata'
    panels.mkdir(parents=True); metadata.mkdir()
    entries, tables, coverage = {}, {}, {}
    pd.DataFrame({'date': dates}).to_csv(panels / 'calendar.csv', index=False)
    master = pd.read_parquet(record(original_meta / 'security_master.parquet', 'team/security_master'))
    if master.sid.duplicated().any() or len(set(sids) - set(master.sid)):
        raise ValueError('Invalid source master')
    master = master.set_index('sid').loc[sids].reset_index()
    master['column_order'] = np.arange(len(sids))
    write_table(master, 'security_master.parquet')
    write_table(summary, 'universe_summary.parquet')
    history = pd.read_parquet(record(original_meta / 'crsp_security_history.parquet', 'team/security_history'))
    history = history.loc[history.sid.isin(sids)].copy()
    write_table(history, 'crsp_security_history.parquet')

    for p in sorted(original_panels.glob('*.pkl')):
        name = p.stem
        frame = load_team(name)
        kind = 'industry' if name == 'industry' else 'mask' if name in {'trade_available', 'delisting_event'} else 'numeric'
        unit = 'SIC2 string' if kind == 'industry' else '0/1' if kind == 'mask' else 'USD' if name in FINANCIAL or name == 'market_cap_split_synced' else 'shares' if name in {'volume', 'crsp_shares_out'} else 'decimal daily return' if name == 'daily_return' else 'ratio' if name == 'adj_factor' else 'segment ID' if name == 'adjustment_segment' else 'USD/share'
        notes = source_snapshot['fundamentals_policy']['vintage'] if name in FINANCIAL else source_snapshot['industry_policy'] if name == 'industry' else source_snapshot['adjustment_policy'] if name in {'adj_factor', 'close_adj', 'open_adj', 'high_adj', 'low_adj', 'vwap_adj'} else 'Preserved source values; explicitly selected dates and SID columns.'
        write_field(frame, name, 'Team17 supplied ' + p.name, unit, notes, kind)
        del frame
        gc.collect()
    write_field(universe, 'universe', 'User-supplied universe_daily.parquet', '0/1',
                'Membership preserved unchanged. Generation rules not supplied; not a certified six-core complete/tradable mask.', 'mask')
    del universe
    for field in RESTORE:
        events = feature_events(features, field)
        frame = expand_events(events, dates, sids, max_age_days=550)
        unit = 'shares' if field == 'shares_out' else 'USD'
        write_field(frame, field, 'Compustat North America fundq; reconciled local stage', unit,
            'RDQ, otherwise valid FDATEQ proxy; first strictly following XNYS session; forward-only, period age <=550 calendar days. Latest download, not historical PIT.')
        # Where an old parent panel exists, exact equality is an additional check.
        oldpath = parent / 'panels' / (field + '.pkl')
        if oldpath.exists():
            old = explicit_subset(pd.read_pickle(record(oldpath, 'parent/panels/' + oldpath.name)), dates, sids)
            if not np.array_equal(old.to_numpy(), frame.to_numpy(), equal_nan=True):
                raise ValueError(f'Restored field differs from parent: {field}')
            del old
        del frame

    # Event tables are restricted to the frozen SID set, keep true pre-panel dates,
    # and never present post-snapshot announcements as available history.
    long_parts = []
    for field in FINANCIAL:
        event = feature_events(features, field)
        event = event.loc[event.available_date <= pd.Timestamp(dates[-1])].copy()
        event['field'] = field
        event['source'] = 'Compustat North America fundq'
        event['source_vintage'] = source_meta['source_sha256']
        event['period_end'] = compact_dates(event.period_end)
        event['available_date'] = compact_dates(event.available_date)
        long_parts.append(event[['sid', 'period_end', 'available_date', 'field', 'value',
                                'source', 'source_vintage', 'availability_basis', 'source_row']])
    write_table(pd.concat(long_parts, ignore_index=True), 'fundamental_vintages.parquet')
    del long_parts
    qcols = ['sid', 'gvkey', 'period_end', 'fyearq', 'fqtr', 'fyr', 'datafqtr', 'datacqtr',
             'rdq', 'fdateq', 'pdateq', 'release_date', 'available_ts', 'availability_basis',
             'date_usable', 'finalq', 'updq', 'srcq', 'rp', 'source_row', 'source_vintage',
             'ajexq'] + FINANCIAL
    qcols = list(dict.fromkeys(qcols))
    known = features.available_ts.notna() & (features.available_ts <= pd.Timestamp(dates[-1]))
    quarters, unavailable = quarterly_view(features, qcols, FINANCIAL, dates[-1])
    write_table(quarters, 'fundamental_quarterly.parquet')
    unavailable['exclusion_reason'] = np.where(unavailable.available_ts.isna(),
        'no usable announcement/final-date proxy', 'announcement after snapshot end')
    write_table(unavailable, 'financial_records_unavailable.parquet')
    write_table(features.loc[known, ['sid', 'gvkey', 'period_end', 'available_ts', 'ajexq',
        'source_row', 'source_vintage']], 'company_adjustment_factor_quarterly.parquet')
    industry_path = record(parent / 'metadata/industry_history.parquet', 'parent/industry_history')
    industry = pd.read_parquet(industry_path)
    industry = industry.loc[industry.sid.isin(sids) & (pd.to_datetime(industry.available_date) <= pd.Timestamp(dates[-1]))].copy()
    write_table(industry, 'industry_history.parquet')
    record(parent / 'metadata/industry_source.json', 'parent/industry_source')
    shutil.copy2(parent / 'metadata/industry_source.json', metadata / 'industry_source.json')
    del industry, quarters, unavailable, features
    gc.collect()

    factor = pd.read_pickle(panels / 'adj_factor.pkl')
    for raw, adjusted in [('open', 'open_adj'), ('high', 'high_adj'), ('low', 'low_adj'), ('vwap', 'vwap_adj')]:
        frame = raw_from_adjusted(pd.read_pickle(panels / (adjusted + '.pkl')), factor)
        write_field(frame, raw, 'Algebraic inverse of supplied ' + adjusted, 'USD/share',
            'Recovered as adjusted price / positive adj_factor; no new independent quote. Missing source values remain missing.')
        del frame
    del factor
    # Some original local sample quotes include real paired amount and volume.
    # Recover just those proven observations, not VWAP * primary CRSP volume.
    local_amount_stats = {'available': False, 'retained_cells': 0}
    if (parent / 'panels/amount.pkl').exists():
        original_quote = {}
        for name in ['open', 'high', 'low', 'close', 'amount', 'volume']:
            path = record(parent / 'panels' / (name + '.pkl'), 'parent/panels/' + name + '.pkl')
            original_quote[name] = explicit_subset(pd.read_pickle(path), dates, sids).to_numpy(copy=True)
        current_quote = {name: pd.read_pickle(panels / (name + '.pkl')).to_numpy(copy=True)
                         for name in ['open', 'high', 'low', 'close']}
        parent_master = pd.read_parquet(record(parent / 'metadata/security_master.parquet', 'parent/security_master'))
        ticker_map = parent_master.set_index('sid').tic
        ticker_matches = np.zeros(umask.shape, dtype=bool)
        day = pd.to_datetime(dates, format='%Y%m%d').to_numpy(dtype='datetime64[ns]')
        def ticker_key(value):
            return ''.join(c for c in str(value).upper() if c.isalnum()) if pd.notna(value) else ''
        for col in np.where(np.isfinite(original_quote['amount']).any(axis=0))[0]:
            sid = sids[col]
            expected_ticker = ticker_key(ticker_map[sid])
            for row in history.loc[history.sid.eq(sid)].itertuples():
                if expected_ticker and ticker_key(row.ticker) == expected_ticker:
                    ticker_matches[:, col] |= ((day >= np.datetime64(row.effective_start)) &
                                              (day <= np.datetime64(row.effective_end)))
        amount, source_volume, paired_vwap = verified_amount_pair(original_quote, current_quote, ticker_matches)
        local_amount_stats = {'available': True, 'candidate_cells': int(np.isfinite(original_quote['amount']).sum()),
            'retained_cells': int(np.isfinite(amount).sum()),
            'retained_sids': int(np.isfinite(amount).any(axis=0).sum()),
            'new_paired_vwap_cells_where_supplied_vwap_missing': int((np.isfinite(paired_vwap) & ~np.isfinite(pd.read_pickle(panels/'vwap.pkl').to_numpy())).sum())}
        for name, values, unit in [('amount', amount, 'USD'),
                                   ('amount_source_volume', source_volume, 'shares'),
                                   ('vwap_local_pair', paired_vwap, 'USD/share')]:
            write_field(pd.DataFrame(values, index=dates, columns=sids), name,
                'Verified paired original local Sina sample amount/volume', unit,
                'Sparse source recovery only. Dated ticker match, all four raw OHLC within both 0.1% and $0.01001, positive paired source amount/volume, implied VWAP within daily range +/-0.5%. No filling of the broader CRSP amount/VWAP gaps; vwap_local_pair is separate from preserved supplied VWAP.')
        del original_quote, current_quote, amount, source_volume, paired_vwap, ticker_matches, parent_master
        gc.collect()
    cap = pd.read_pickle(panels / 'market_cap_split_synced.pkl')
    write_field(cap, 'market_cap', 'Alias of supplied market_cap_split_synced', 'USD',
        '2014-2025 CRSP abs(PRC)*SHROUT*1000; security-level capitalization, not independently aggregated issuer/multi-class company capitalization.')
    del cap
    permno, conflicts = dated_permno(history, dates, sids)
    write_field(permno, 'crsp_permno', 'Date-valid projection of supplied CRSP security history', 'PERMNO identifier',
        'Do not join historical prices using static stocks.permno. Conflicting dates quarantined as NaN; not a recovered independent CCM input ledger.')
    if conflicts.any():
        r, c = np.where(conflicts)
        write_table(pd.DataFrame({'date': dates[r], 'sid': sids[c]}), 'security_history_conflicts.parquet')
    del permno

    complete_core = umask.copy()
    for field in CORE:
        frame = pd.read_pickle(panels / (field + '.pkl'))
        valid = np.isfinite(frame.to_numpy())
        if field in {'shares_out', 'total_assets'}:
            valid &= frame.to_numpy() > 0
        if field == 'total_liabilities':
            valid &= frame.to_numpy() >= 0
        complete_core &= valid
        del frame
    write_field(pd.DataFrame(complete_core.astype('uint8'), index=dates, columns=sids),
        'universe_core_complete', 'Supplied universe AND six currently valid core fields', '0/1',
        'Diagnostic supplementary mask only; does not replace supplied universe or add a tradability requirement.', 'mask')
    ocf = pd.read_pickle(panels / 'operating_cash_flow_ttm.pkl')
    capex = pd.read_pickle(panels / 'capital_expenditures_ttm.pkl')
    write_field(ocf - capex, 'free_cash_flow_ttm', 'operating_cash_flow_ttm - capital_expenditures_ttm', 'USD',
        'Both components must be available. Financial-sector economic applicability remains a factor-level decision.')
    del ocf, capex

    # This is an observation extract, deliberately NOT named the missing DLRET
    # event ledger: event date/source event identifiers cannot be recovered here.
    event = pd.read_pickle(panels / 'delisting_event.pkl').eq(1).to_numpy()
    rr, cc = np.where(event)
    returns = pd.read_pickle(panels / 'daily_return.pkl').to_numpy()
    close = pd.read_pickle(panels / 'close.pkl').to_numpy()
    obs = pd.DataFrame({'sid': sids[cc], 'observed_panel_date': dates[rr],
        'panel_daily_return': returns[rr, cc], 'panel_raw_close': close[rr, cc],
        'evidence': 'existing delisting_event=1; source event date/DLRET not recovered'})
    write_table(obs, 'delisting_observations.parquet')
    del returns, close, event

    index_entries = {}
    spy_cache = record(args.spy_cache, 'spy/raw_cache')
    record(spy_cache.with_suffix('.json'), 'spy/source_metadata')
    spy, spy_meta = spy_frames(spy_cache, dates)
    raw = out / 'raw/benchmark'; raw.mkdir(parents=True)
    shutil.copy2(spy_cache, raw / 'SPY.gz')
    shutil.copy2(spy_cache.with_suffix('.json'), raw / 'SPY.json')
    for field, frame in spy.items():
        path = panels / ('indices_' + field + '.pkl')
        frame.to_pickle(path, protocol=4)
        index_entries[field] = {'sha256': digest(path), 'shape': list(frame.shape),
            'columns': ['SPY'], 'source': 'Yahoo chart, cached source verified by hash',
            'notes': 'SPY ETF benchmark (not a stock-pool member). Adjusted close is a vendor total-return proxy; adjusted open = raw open * adjclose/raw close. Levels are vendor-normalized, not unadjusted trade prices.' if field != 'close' else 'SPY raw close, USD/share, not dividend-reinvested return.'}
    # Recover previously cached price indices separately from stock axes and SPY.
    oldindex = parent / 'panels/indices_close.pkl'
    if oldindex.exists():
        original = pd.read_pickle(record(oldindex, 'parent/indices_close'))
        if original.index.has_duplicates or len(dates.difference(original.index)):
            raise ValueError('Cached index calendar cannot cover snapshot')
        frame = original.loc[dates].astype('float32')
        frame.index = dates
        frame.to_pickle(panels / 'indices_price_close.pkl', protocol=4)
        index_entries['price_close'] = {'sha256': digest(panels / 'indices_price_close.pkl'),
            'shape': list(frame.shape), 'columns': list(frame.columns),
            'source': 'Local parent Yahoo cached price-index close; VIX is volatility index',
            'notes': 'Price indices, not total-return benchmarks; missing values not filled.'}

    counts = umask.sum(axis=1)
    research = dates >= '20150101'
    trades = pd.read_pickle(panels / 'trade_available.pkl').eq(1).to_numpy()
    acceptance = {'structural_passed': True, 'research_accepted': False,
        'source_values_preserved': True, 'source_files_unchanged': True,
        'calendar_independently_verified': True, 'all_stock_axes_identical': True,
        'source_reconciliation': reconciliation,
        'daily_members_2015_2025': {'min': int(counts[research].min()),
            'median': float(np.median(counts[research])), 'max': int(counts[research].max())},
        'provided_universe_missing_six_core_cells': int((umask & ~complete_core).sum()),
        'eligible_but_not_trade_available': int((umask & ~trades).sum()),
        'history_conflicting_cells': int(conflicts.sum()), 'coverage': coverage,
        'sparse_local_amount_recovery': local_amount_stats,
        'remaining_limits': [
            'Daily supplied universe is below 4000. Rules/code behind supplied membership were not provided; no future-filtering or full security-type certification.',
            'Latest Compustat download, not historical revisions/PIT. RDQ/final-date proxies do not prove field-by-field public availability.',
            'Reliable full-market VWAP remains incomplete. Only a sparse validated local sample amount/volume pair was recoverable; do not manufacture amount as VWAP times unrelated CRSP volume.',
            'Original CRSP input manifest, independent CCM links and full terminal-event ledger absent. Observation extract is not a recovered DLRET source ledger.',
            'Security-level CRSP market cap is not an aggregated multi-class issuer market cap.',
            'Output ends 2025-12-31. Old mixed-source January 2026 tail remains in original inputs, not included or silently bridged; censor final unmatured holding-period labels.',
            'No factor selection, model training or formal backtest has been run by this completion.']}
    # Every input is checked again, not only claimed unchanged in documentation.
    checks = [(upath, 'supplied_universe'), (spath, 'supplied_stocks'),
        (stage / 'financial_features.parquet', 'financial_features_stage'),
        (original_meta / 'snapshot.json', 'team/snapshot')]
    checks += [(p, 'team/panels/' + p.name) for p in sorted(original_panels.glob('*.pkl'))]
    for path, label in checks:
        if digest(path) != inputs[label]['sha256']:
            raise ValueError('Input changed during completion: ' + label)
    dump(inputs, metadata / 'input_manifest.json')
    dump(acceptance, metadata / 'acceptance.json')
    manifest = {'schema_version': 2, 'status': 'locally_completed_research_prototype',
        'research_accepted': False, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'research_start': '2015-01-01', 'research_end': '2025-12-31',
        'calendar_start': dates[0], 'calendar_end': dates[-1], 'calendar_sessions': len(dates),
        'security_candidates': len(sids), 'tables': tables, 'stock_fields': entries,
        'fundamental_fields': FINANCIAL + ['free_cash_flow_ttm'], 'index_fields': index_entries,
        'calendar': {'library': 'exchange_calendars', 'version': __import__('exchange_calendars').__version__,
            'calendar': 'XNYS', 'independent_from_stock_quotes': True,
            'sha256': digest(panels / 'calendar.csv')},
        'source': source_meta, 'fundamentals_policy': source_snapshot['fundamentals_policy'],
        'financial_quarterly_policy': 'Source-event rows, not an as-of table. Every financial field has field_available_ts; derived TTM/differenced flows can be later than the row own available_ts. Values not available by snapshot end are masked. Research daily use must use fundamental_vintages or released wide panels.',
        'adjustment_policy': source_snapshot['adjustment_policy'],
        'adjusted_price_usage_note': 'Preserved factors may have ex-post level normalization. Re-anchor adjusted price windows at the signal date or use scale-cancelling operators; do not treat cross-sectional absolute adjusted price levels as verified as-of prices.',
        'industry_policy': source_snapshot['industry_policy'], 'crsp_policy': source_snapshot['crsp_policy'],
        'crsp_policy_scope_note': 'crsp_policy is inherited upstream documentation, including old 7208-column/2026-tail/10d-label descriptions, not the current snapshot scope. Current axes and fields are authoritative: 4366 SIDs through 2025, no inherited target labels, no CRSP input manifest recovered.',
        'universe_policy': 'Preserve supplied daily universe. Explicit 4366-column output selection; 1347 other original price SIDs excluded, no unknown membership states. No new stock selection.',
        'time_scope_policy': '2014 warm-up retained; 2015-2025 research. January 2026 mixed-source tail explicitly excluded.',
        'market_cap_policy': entries['market_cap']['notes'],
        'input_manifest_sha256': digest(metadata / 'input_manifest.json'),
        'environment': {'python': platform.python_version(), 'pandas': pd.__version__,
            'numpy': np.__version__, 'pyarrow': __import__('pyarrow').__version__},
        'trusted_pickle_only': True}
    dump(manifest, metadata / 'snapshot.json')
    pd.DataFrame([{'field': k, 'source': v['source'], 'unit': v['unit'], 'notes': v['notes']}
        for k, v in entries.items()]).to_csv(metadata / 'data_dictionary.csv', index=False)
    daily = pd.DataFrame({'date': dates, 'universe': counts,
        'universe_core_complete': complete_core.sum(axis=1), 'eligible_and_trade': (umask & trades).sum(axis=1)})
    daily.to_csv(metadata / 'daily_coverage.csv', index=False)
    # Read back EVERY emitted stock panel and table, not just the source arrays.
    for name, entry in entries.items():
        path = panels / (name + '.pkl')
        frame = pd.read_pickle(path)
        if not frame.index.equals(dates) or not frame.columns.equals(sids) or digest(path) != entry['sha256']:
            raise ValueError('Output round-trip failure: ' + name)
        del frame
    for name, entry in tables.items():
        if pq.read_metadata(metadata / name).num_rows != entry['rows'] or digest(metadata / name) != entry['sha256']:
            raise ValueError('Output table verification failure: ' + name)
    size = sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
    summary_text = f'''# 本地数据补齐结果

新目录：`{out.name}`。原始输入未修改；本次是数据整合，不是研究结果。

- 股票面板：{len(entries)} 个字段，全部 {len(dates):,} 行 × {len(sids):,} 列，日期 {dates[0]}—{dates[-1]}。
- 新目录大小：约 {size/1024**3:.3f} GiB。
- 恢复：现金、股数、短期债务、季度现金流、季度/TTM 资本开支、单季度净利润和收入；并生成自由现金流 TTM。
- 恢复明细：财务公布日期/版本/来源记录、财务季度表、历史行业表及公司季度调整因子。季度表数值财务列统一美元或实际股数，每个财务字段单独带 field_available_ts，TTM/差分现金流不能只按本行季度公布日使用；超过快照截止时点的值已屏蔽。AJEXQ 为供应商季度调整因子，不是每日总回报复权因子。
- 补齐：独立 calendar.csv、明确 0/1 股票池、日期有效的 PERMNO 投影、原始 OHLC 的代数恢复、统一美元 market_cap、SPY 复权收盘/开盘基准、已有缓存价格指数、字段字典与新哈希清单。
- 旧本地样本的真实成交金额经历史代码、OHLC 和配对成交量核验，保留 {local_amount_stats['retained_cells']:,} 个股票日；只写可验证单元，不向其余股票强填。配对成交量使用 amount_source_volume，独立备查 VWAP 使用 vwap_local_pair，原有 vwap/vwap_adj 不变。
- 已核对：7 个已有财务字段与本地公布时点展开结果逐单元完全一致；恢复字段若有父库宽表，也逐单元完全一致；输出所有股票面板均重新读回并检查全轴和哈希。
- 股票池保留不变；额外 universe_core_complete 只供诊断，不替换原始股票池。
- 所有研究宽表和按 SID 组织的明细仅含这 {len(sids):,} 个 SID；SPY/指数另表，不参加股票列对齐。

## 仍需说明

每日股票数 {acceptance['daily_members_2015_2025']['min']:,}—{acceptance['daily_members_2015_2025']['max']:,}，未达到每日 4,000+。股票池生成规则未提供。

全市场真实 VWAP 缺失、完整退市事件账本和源文件沿革仍无法凭现有面板恢复；不能伪造量、价、金额或退出价。退市观察表只是原标记的抽取，不是原始退市事件账本。amount 为少量已核验本地原始金额，仍高度稀疏，不能认为全市场日成交额已经补齐。

财务采用已同意的最新下载数据 + 公布日代理后向前延续口径，并非严格历史 PIT。SPY 复权收盘为供应商总回报代理，非自建精确现金分红再投资账本。股票市值为证券级，不能不加处理就称为多类别股份公司总市值。

保留的股票复权绝对价可能带事后归一化锚点；量价因子应在信号日重锚窗口或使用能消去公共尺度的运算，不能直接把跨股票绝对复权价当成已验证的无未来信息价格来排序。snapshot.json 中继承的 crsp_policy 是上游说明，当前列集合、截止日期和字段清单优先；未继承旧版 10 日 VWAP 标签。

2026 年 1 月混合来源尾部仍保留在原始文件，未纳入这份 2014—2025 整合库；最后尚未到期的持有期标签必须剔除，不能虚构年末出场价格。

完整检查见 metadata/acceptance.json，字段口径见 metadata/data_dictionary.csv。research_accepted 仍为 false，不把结构检查通过等同于正式研究验收。
'''
    (out / '补齐说明.md').write_text(summary_text, encoding='utf-8')
    write_checksums(out)
    print(json.dumps({'stage': 'complete', 'output': str(out), 'fields': len(entries),
        'shape': [len(dates), len(sids)], 'bytes': size, 'research_accepted': False}), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['team', 'parent', 'stage', 'universe', 'stocks', 'spy-cache', 'output']:
        parser.add_argument('--' + name, required=True, type=Path)
    complete(parser.parse_args())


if __name__ == '__main__':
    main()
