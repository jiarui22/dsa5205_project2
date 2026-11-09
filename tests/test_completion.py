"""Small tests for source-preserving completion; no licensed source data needed."""
import gzip
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from usq.database.common import digest
from usq.database.complete import dated_permno, explicit_subset, raw_from_adjusted, spy_frames, verified_amount_pair, quarterly_view
from usq.database.fundamentals import expand_events, feature_events


class CompletionTests(unittest.TestCase):
    def setUp(self):
        self.dates = pd.Index(['20200102', '20200103', '20200106'], name='date')
        self.sids = pd.Index(['X', 'Y'], name='sid')
        self.frame = pd.DataFrame([[1., 2.], [3., 4.], [5., 6.]], index=self.dates, columns=self.sids)

    def test_selection_is_labelled_and_preserves_requested_order(self):
        f = explicit_subset(self.frame, self.dates[1:], self.sids[::-1])
        self.assertEqual(f.iloc[0].tolist(), [4., 3.])
        self.assertEqual(f.columns.tolist(), ['Y', 'X'])

    def test_missing_sid_is_error_not_intersection(self):
        with self.assertRaises(ValueError):
            explicit_subset(self.frame, self.dates, pd.Index(['X', 'Z']))

    def test_missing_date_is_error_not_intersection(self):
        with self.assertRaises(ValueError):
            explicit_subset(self.frame, pd.Index(['20200107']), self.sids)

    def test_duplicate_source_axis_is_error(self):
        bad = pd.concat([self.frame, self.frame.iloc[[0]]])
        with self.assertRaises(ValueError):
            explicit_subset(bad, self.dates, self.sids)

    def test_inverse_adjustment_no_fake_factor_or_backfill(self):
        factor = self.frame.copy()
        factor.iloc[:, :] = [[.5, 0], [2, np.nan], [1, 4]]
        got = raw_from_adjusted(self.frame, factor)
        self.assertEqual(got.iloc[0, 0], 2)
        self.assertTrue(pd.isna(got.iloc[0, 1]))
        self.assertTrue(pd.isna(got.iloc[1, 1]))
        self.assertEqual(got.iloc[2, 1], 1.5)
        self.assertEqual(str(got.dtypes.iloc[0]), 'float32')

    def test_inverse_adjustment_rejects_different_axes(self):
        with self.assertRaises(ValueError):
            raw_from_adjusted(self.frame, self.frame.iloc[:, ::-1])

    def test_amount_requires_paired_volume_identity_and_quote_controls(self):
        old = {name: np.full(5, value, dtype=float) for name, value in
               [('open', 10), ('high', 12), ('low', 9), ('close', 11), ('amount', 1050), ('volume', 100)]}
        now = {k: old[k].copy() for k in ['open', 'high', 'low', 'close']}
        old['volume'][1] = 0
        old['close'][2] = 13
        old['amount'][3] = 100000
        amount, volume, vwap = verified_amount_pair(old, now, np.array([True, True, True, True, False]))
        self.assertEqual(amount[0], 1050)
        self.assertEqual(volume[0], 100)
        self.assertEqual(vwap[0], 10.5)
        self.assertTrue(np.isnan(amount[1:]).all())

    def test_permno_uses_date_valid_history_not_static_last_value(self):
        h = pd.DataFrame({'sid': ['X', 'X', 'Y'], 'permno': [100, 200, 300],
            'effective_start': pd.to_datetime(['2020-01-02', '2020-01-06', '2020-01-02']),
            'effective_end': pd.to_datetime(['2020-01-03', '2020-01-06', '2020-01-06'])})
        f, conflict = dated_permno(h, self.dates, self.sids)
        self.assertEqual(f.X.tolist(), [100, 100, 200])
        self.assertFalse(conflict.any())

    def test_conflicting_history_is_missing_not_last_row_wins(self):
        h = pd.DataFrame({'sid': ['X', 'X'], 'permno': [100, 200],
            'effective_start': pd.to_datetime(['2020-01-02', '2020-01-03']),
            'effective_end': pd.to_datetime(['2020-01-06', '2020-01-03'])})
        f, conflict = dated_permno(h, self.dates, self.sids)
        self.assertTrue(conflict[1, 0])
        self.assertTrue(pd.isna(f.X.iloc[1]))
        self.assertTrue(f.Y.isna().all())

    def test_financial_forward_only_and_stale_gate(self):
        e = pd.DataFrame({'sid': ['X'], 'period_end': pd.to_datetime(['2019-12-31']),
            'available_date': pd.to_datetime(['2020-01-03']), 'value': [7.]})
        f = expand_events(e, self.dates, self.sids, max_age_days=4)
        self.assertTrue(pd.isna(f.X.iloc[0]))
        self.assertEqual(f.X.iloc[1], 7.)
        self.assertTrue(pd.isna(f.X.iloc[2]))
        self.assertTrue(f.Y.isna().all())

    def test_late_old_quarter_cannot_replace_newer_quarter(self):
        base = pd.Timestamp('1970-01-01')
        r = pd.DataFrame({'sid': ['X', 'X'],
            'period_end': pd.to_datetime(['2019-12-31', '2019-09-30']),
            'available_day': [(pd.Timestamp('2020-01-03')-base).days,
                              (pd.Timestamp('2020-01-06')-base).days],
            'source_row': [1, 2], 'availability_basis': ['rdq', 'rdq'], 'earnings_q': [10., 5.]})
        events = feature_events(r, 'earnings_q')
        self.assertEqual(events.value.tolist(), [10.])

    def test_quarterly_table_keeps_late_ttm_availability_and_censors_future_values(self):
        base = pd.Timestamp('1970-01-01')
        days = lambda x: (pd.Timestamp(x)-base).days
        f = pd.DataFrame({'sid': ['X', 'Y'], 'available_ts': pd.to_datetime(['2020-01-03', '2020-01-03']),
            'available_day': [days('2020-01-03')]*2, 'earnings_q': [1., 2.],
            'earnings_ttm': [10., 20.], 'earnings_ttm_available_day': [days('2020-01-06'), days('2021-01-04')]})
        q, quarantine = quarterly_view(f, list(f.columns), ['earnings_q', 'earnings_ttm'], '2020-12-31')
        self.assertTrue(quarantine.empty)
        self.assertEqual(q.loc[0, 'earnings_ttm_available_ts'], pd.Timestamp('2020-01-06'))
        self.assertTrue(pd.isna(q.loc[1, 'earnings_ttm']))
        self.assertTrue(pd.isna(q.loc[1, 'earnings_ttm_available_ts']))
        self.assertEqual(q.loc[1, 'earnings_q'], 2.)

    def make_spy(self, root, symbol='SPY'):
        path = Path(root) / 'SPY.gz'
        obj = {'chart': {'result': [{'meta': {'symbol': symbol, 'currency': 'USD'},
            'timestamp': [int(pd.Timestamp(d+' 14:30', tz='UTC').timestamp()) for d in ['2020-01-02', '2020-01-03', '2020-01-06']],
            'indicators': {'quote': [{'open': [10., 11., 12.], 'close': [11., 12., 13.]}],
                           'adjclose': [{'adjclose': [5.5, 6., 6.5]}]}}]}}
        path.write_bytes(gzip.compress(json.dumps(obj).encode(), mtime=0))
        path.with_suffix('.json').write_text(json.dumps({'sha256': digest(path)}))
        return path

    def test_spy_is_separate_index_axis_and_proper_adjusted_open(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.make_spy(tmp)
            frames, _ = spy_frames(path, self.dates)
            self.assertEqual(frames['total_return_close'].columns.tolist(), ['SPY'])
            self.assertEqual(frames['total_return_open'].SPY.tolist(), [5., 5.5, 6.])

    def test_spy_hash_and_identity_are_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.make_spy(tmp, symbol='WRONG')
            with self.assertRaises(ValueError):
                spy_frames(path, self.dates)
            path = self.make_spy(tmp)
            path.write_bytes(path.read_bytes() + b'changed')
            with self.assertRaises(ValueError):
                spy_frames(path, self.dates)


if __name__ == '__main__':
    unittest.main()
