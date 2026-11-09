from __future__ import annotations

import json
import re
from pathlib import Path
import pandas as pd
from .common import date_key, digest


class ResearchDatabase:
    """Lazy trusted-team pickle reader, checking COMPLETE axes before slicing.

    ``ready`` is not implied by successfully loading a prototype snapshot.
    Use ``require_accepted=True`` before feeding a snapshot into real research.
    """
    def __init__(self, root, *, require_accepted=False):
        self.root = Path(root)
        self.meta = json.loads((self.root/'metadata'/'snapshot.json').read_text())
        if require_accepted and not self.meta.get('research_accepted',False):
            raise ValueError('Snapshot is not research-accepted; read metadata/acceptance.json')
        self.dates = pd.Index(pd.read_csv(self.root/'panels'/'calendar.csv',dtype=str)['date'],name='date')
        master = pd.read_parquet(self.root/'metadata'/'security_master.parquet')
        self.security_master=master.sort_values('column_order').reset_index(drop=True)
        self.sids = pd.Index(self.security_master.sid.astype(str),name='sid')
        if self.dates.has_duplicates or self.sids.has_duplicates or not self.dates.is_monotonic_increasing:
            raise ValueError('Invalid canonical calendar/security axes')

    def _slice(self, frame, start, end, columns):
        lo = date_key(start) if start is not None else self.dates[0]
        hi = date_key(end) if end is not None else self.dates[-1]
        if lo>hi:
            raise ValueError('start must not be after end')
        if columns is None:
            cols=frame.columns
        else:
            cols=pd.Index([columns] if isinstance(columns,str) else list(columns)).astype(str)
            unknown=cols.difference(frame.columns)
            if len(unknown):
                raise KeyError(f'Unknown stable IDs/index symbols: {unknown.tolist()}')
            if cols.has_duplicates:
                raise ValueError('Duplicate requested columns')
        return frame.loc[(frame.index>=lo)&(frame.index<=hi),cols]

    def load(self, field, start=None, end=None, sids=None, *, verify_hash=False):
        if not re.fullmatch(r'[A-Za-z0-9_]+',field) or field.startswith('indices_'):
            raise ValueError('Invalid stock field; use indices() for indices')
        entry=self.meta['stock_fields'].get(field)
        if entry is None:
            raise KeyError(f'Unavailable stock field: {field}')
        path=self.root/'panels'/f'{field}.pkl'
        if verify_hash and digest(path)!=entry['sha256']:
            raise ValueError('Snapshot hash mismatch')
        frame=pd.read_pickle(path)  # Trusted team products ONLY.
        if not frame.index.equals(self.dates) or not frame.columns.equals(self.sids):
            raise ValueError('Full panel axes differ from the frozen universe/calendar')
        return self._slice(frame,start,end,sids)

    def stocks(self, fields, start=None, end=None, sids=None):
        fields=[fields] if isinstance(fields,str) else fields
        return {name:self.load(name,start,end,sids) for name in fields}

    def fundamentals(self, fields=None, start=None, end=None, sids=None):
        default=['shares_out','book_equity','earnings_ttm','revenue_ttm','total_assets','total_liabilities']
        fields=default if fields is None else ([fields] if isinstance(fields,str) else list(fields))
        if any(name not in self.meta['fundamental_fields'] for name in fields):
            raise KeyError('Not a fundamental field')
        return self.stocks(fields,start,end,sids)

    def industry(self, start=None, end=None, sids=None):
        return self.load('industry',start,end,sids)

    def indices(self, field='close', start=None, end=None, symbols=None):
        if not re.fullmatch(r'[A-Za-z0-9_]+',field):
            raise ValueError('Invalid index field')
        path=self.root/'panels'/f'indices_{field}.pkl'
        frame=pd.read_pickle(path)
        if not frame.index.equals(self.dates) or frame.columns.has_duplicates:
            raise ValueError('Index calendar differs or index columns duplicate')
        # Index codes deliberately do NOT undergo stock-security-column alignment.
        expected=self.meta.get('index_fields',{}).get(field,{}).get('columns')
        if expected is None or list(frame.columns)!=expected:
            raise ValueError('Index columns differ from index manifest')
        return self._slice(frame,start,end,symbols)

    def financial_history(self, start=None, end=None, sids=None):
        filters=[]
        if start is not None: filters.append(('available_date','>=',date_key(start)))
        if end is not None: filters.append(('available_date','<=',date_key(end)))
        if sids is not None:
            values=[sids] if isinstance(sids,str) else list(sids)
            unknown=pd.Index(values).difference(self.sids)
            if len(unknown):raise KeyError(f'Unknown stable IDs: {unknown.tolist()}')
            filters.append(('sid','in',values))
        return pd.read_parquet(self.root/'metadata'/'fundamental_vintages.parquet',filters=filters or None)
