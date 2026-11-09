from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def dump(value, path):
    from usq.utils import json_dump
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(target.suffix+'.tmp')
    json_dump(value, temp)
    temp.replace(target)


def save_parquet(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp.parquet')
    frame.to_parquet(temp, index=False, compression='zstd')
    temp.replace(path)


def date_key(value):
    return pd.Timestamp(value).strftime('%Y%m%d')


def compact_dates(values):
    """Vectorized ISO-day formatting; avoid per-row libc strftime overhead on macOS."""
    array=pd.to_datetime(values).to_numpy(dtype='datetime64[D]')
    strings=np.char.replace(array.astype(str),'-','')
    return pd.Series(strings,index=getattr(values,'index',None)).where(~np.isnat(array))


def timestamps_from_days(values):
    """Integer day conversion, preserving NaT without float-to-nanosecond rounding."""
    day=pd.Series(values,copy=False)
    safe=day.fillna(0).astype('int64')
    return pd.to_datetime(safe,unit='D',origin='unix',errors='coerce').where(day.notna())


def save_panel(frame, path, dates, sids, kind='numeric'):
    path = Path(path)
    if not frame.index.equals(dates) or not frame.columns.equals(sids):
        raise ValueError(f'Full axes mismatch: {path.name}; never take an intersection')
    if dates.has_duplicates or sids.has_duplicates or not dates.is_monotonic_increasing:
        raise ValueError('Axes must be unique and dates sorted')
    if kind == 'mask':
        if not frame.isin([0,1]).all().all():
            raise ValueError('Universe/masks must be explicit 0/1 with no unknown states')
        frame = frame.astype('uint8')
    elif kind == 'industry':
        # String categories, never numeric industry codes interpreted as quantities.
        for col in frame:
            frame[col] = frame[col].astype('category')
    else:
        frame = frame.astype('float32')
        if np.isinf(frame.to_numpy()).any():
            raise ValueError('Overflow/infinity in numeric panel')
    path.parent.mkdir(parents=True,exist_ok=True)
    temp = path.with_suffix('.tmp')
    frame.to_pickle(temp,protocol=4)
    temp.replace(path)
    return {'sha256':digest(path),'shape':list(frame.shape),'start':dates[0],'end':dates[-1],
            'dtype':kind if kind!='numeric' else 'float32',
            'coverage':float(frame.notna().to_numpy().mean())}


def calendar(start, end):
    import exchange_calendars as xcals
    cal = xcals.get_calendar('XNYS',start=start,end=end)
    days = pd.Index(cal.sessions.strftime('%Y%m%d'),name='date')
    for day,present in {'20181205':False,'20210618':True,'20220620':False,
                        '20250109':False,'20250120':False,'20260119':False}.items():
        if days[0] <= day <= days[-1] and (day in days)!=present:
            raise ValueError(f'Independent-calendar regression failed: {day}')
    return days
