"""Raw-download preparation stages used by the data notebook."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from .wrds_reconstruction import reconstruct_quarterly_archive, reconstruct_industry_archive
from .universe_reconstruction import reconstruct_universe


def build_financial_events(source, output):
    return reconstruct_quarterly_archive(source, None, output)[1]


def build_industry_events(source, output):
    return reconstruct_industry_archive(source, None, output)


def build_universe(raw, output):
    return reconstruct_universe(raw, None, output)


def audit_market_sources(raw, output):
    """Stream full raw files; retain anomalies and audit duplicate source keys."""
    from usq.project2.data_preparation import material_ohlc
    raw, output = Path(raw), Path(output)
    rows, exceptions = [], []
    for name in ['crsp_dsf_2014_2024.parquet','crsp_dsf_v2_2025.parquet']:
        path = next(raw.rglob(name))
        parquet = pq.ParquetFile(path)
        keys = np.empty(parquet.metadata.num_rows, dtype='uint64')
        offset = 0
        row = dict(file=name,rows=0,quote_marks=0,
            missing_or_zero_close=0,negative_volume=0,zero_volume=0,material_OHLC=0)
        for batch in parquet.iter_batches(batch_size=150000,
                columns=['date','permno','prc','openprc','askhi','bidlo','vol']):
            d=batch.to_pandas();row['rows']+=len(d)
            days=pd.to_datetime(d.date).to_numpy(dtype='datetime64[D]').astype('int64')
            if d.permno.isna().any() or (days<0).any() or not d.permno.between(0,2**31-1).all():
                raise ValueError('Invalid raw date or PERMNO key')
            keys[offset:offset+len(d)] = (d.permno.to_numpy('uint64') << np.uint64(32)) | days.astype('uint64')
            offset += len(d)
            c,o,h,l=(pd.to_numeric(d[f],errors='coerce').abs() for f in ['prc','openprc','askhi','bidlo'])
            bad=material_ohlc(o,h,l,c)|h.lt(l)
            row['quote_marks']+=int(d.prc.lt(0).sum())
            row['missing_or_zero_close']+=int((c.isna()|c.eq(0)).sum())
            row['negative_volume']+=int(d.vol.lt(0).sum());row['zero_volume']+=int(d.vol.eq(0).sum())
            row['material_OHLC']+=int(bad.sum())
            if bad.any():exceptions.append(d.loc[bad].assign(source_file=name))
        keys.sort()
        row['duplicate_keys']=int(np.unique(keys[1:][keys[1:]==keys[:-1]]).size)
        rows.append(row)
        del keys
    table=pd.DataFrame(rows);table.to_csv(output/'market_quality.csv',index=False)
    anomalies=pd.concat(exceptions,ignore_index=True) if exceptions else pd.DataFrame()
    anomalies.to_csv(output/'raw_ohlc_exceptions.csv',index=False)
    if table.duplicate_keys.sum():raise ValueError('Duplicate raw date–PERMNO keys require review')
    if table.negative_volume.sum():raise ValueError('Negative raw share volume requires review')
    return table,anomalies


def audit_published_prices(snapshot):
    """Verify the stored chain, label equation, timing and numeric precision."""
    from usq.data.adjustments import canonical_close, PRICE_PRECISION_POLICY
    from usq.data.labels import close_forward_returns
    root=Path(snapshot)
    load=lambda name:pd.read_pickle(root/'panels'/f'{name}.pkl')
    meta=json.loads((root/'metadata/preprocessing_complete.json').read_text())
    if meta['price_precision_policy']!=PRICE_PRECISION_POLICY:raise ValueError('Unexpected price precision')
    close,returns,segments=load('close'),load('daily_return'),load('adjustment_segment')
    canonical=canonical_close(close,returns,segments)
    pd.testing.assert_frame_equal(canonical,load('close_adj'),check_exact=True)
    target,end=close_forward_returns(canonical,10,segments)
    pd.testing.assert_frame_equal(target,load('ylabel'),check_exact=True)
    assert target.tail(11).isna().all().all()
    u=load('universe').eq(1)
    coverage=pd.DataFrame({'eligible':u.sum(axis=1),'finite_targets':(u&target.notna()).sum(axis=1)})
    coverage['coverage']=coverage.finite_targets/coverage.eligible.replace(0,np.nan)
    return meta,coverage
