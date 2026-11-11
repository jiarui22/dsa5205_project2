import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd

from usq.data import PanelStore
from usq.data.labels import close_forward_returns, daily_close_returns, prepare


class CloseLabelTests(unittest.TestCase):
    def setUp(self):
        self.dates = pd.bdate_range('2021-12-01', periods=30).strftime('%Y%m%d')
        self.price = pd.DataFrame({'A': 100*1.01**np.arange(30)}, index=self.dates)
        self.segment = self.price*0+1

    def test_next_close_ten_intervals_and_exact_exit(self):
        y, end = close_forward_returns(self.price, segment=self.segment)
        self.assertAlmostEqual(y.iloc[0, 0], 1.01**10-1)
        self.assertNotAlmostEqual(y.iloc[0, 0], .1)
        self.assertEqual(end.iloc[0], self.dates[11])
        self.assertTrue(y.tail(11).isna().all().all())
        self.assertTrue(end.tail(11).isna().all())
        p = self.price.copy(); p.iloc[0, 0] *= 2
        y2, _ = close_forward_returns(p, segment=self.segment)
        self.assertEqual(y2.iloc[0, 0], y.iloc[0, 0])  # signal-close move is not held
        p.iloc[1, 0] *= 3
        y3, _ = close_forward_returns(p, segment=self.segment)
        self.assertAlmostEqual(y3.iloc[0, 0], p.iloc[11, 0]/p.iloc[1, 0]-1)
        legacy, old_end = close_forward_returns(p, segment=self.segment, delay=0)
        self.assertAlmostEqual(legacy.iloc[0, 0], p.iloc[10, 0]/p.iloc[0, 0]-1)
        self.assertEqual(old_end.iloc[0], self.dates[10])

    def test_label_does_not_depend_on_future_beyond_exit(self):
        y, _ = close_forward_returns(self.price, segment=self.segment)
        p = self.price.copy(); p.iloc[12:, 0] *= 100
        changed, _ = close_forward_returns(p, segment=self.segment)
        self.assertEqual(y.iloc[0, 0], changed.iloc[0, 0])

    def test_unknown_or_changing_segment_blocks_return(self):
        for bad in [0, np.nan, 2]:
            seg = self.segment.copy(); seg.iloc[5, 0] = bad
            y, _ = close_forward_returns(self.price, segment=seg)
            self.assertTrue(pd.isna(y.iloc[0, 0]))

    def test_missing_end_price_is_not_zero_and_never_filled(self):
        for endpoint in (1, 11):
            for bad in [0, np.nan, np.inf]:
                p = self.price.copy(); p.iloc[endpoint, 0] = bad
                y, _ = close_forward_returns(p, segment=self.segment)
                self.assertTrue(pd.isna(y.iloc[0, 0]))

    def test_intermediate_price_missing_requires_segment_evidence(self):
        p = self.price.copy(); p.iloc[5, 0] = np.nan
        proven, _ = close_forward_returns(p, segment=self.segment)
        unproven, _ = close_forward_returns(p)
        self.assertTrue(pd.notna(proven.iloc[0, 0]))
        self.assertTrue(pd.isna(unproven.iloc[0, 0]))

    def test_segment_axes_cannot_silently_intersect(self):
        with self.assertRaises(ValueError):
            close_forward_returns(self.price, segment=self.segment.iloc[1:])

    def test_one_day_return_looks_backward_and_respects_segments(self):
        r = daily_close_returns(self.price, self.segment)
        self.assertTrue(pd.isna(r.iloc[0, 0]))
        self.assertAlmostEqual(r.iloc[1, 0], .01)
        seg = self.segment.copy(); seg.iloc[1, 0] = 2
        self.assertTrue(pd.isna(daily_close_returns(self.price, seg).iloc[1, 0]))

    def test_close_prepare_needs_no_vwap_amount_cap_industry_or_universe(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = PanelStore(tmp)
            store.save('close_adj', self.price)
            store.save('adjustment_segment', self.segment)
            prepare(store)
            y = store.load('ylabel')
            self.assertAlmostEqual(y.iloc[0, 0], 1.01**10-1, places=6)
            for field in ['label_10', 'forward_10d_raw']:
                np.testing.assert_array_equal(store.load(field).to_numpy(), y.to_numpy())
            self.assertFalse(store.path('vwap').exists())
            meta = json.loads((Path(tmp)/'label_metadata.json').read_text())
            self.assertFalse(meta['vwap_required'])
            self.assertFalse(meta['standardize'])
            self.assertIsNone(meta['neutralization'])
            self.assertEqual(meta['label_version'], 'next_close_10d_v2')
            self.assertEqual(meta['entry_offset'], 1)
            self.assertEqual(meta['exit_offset'], 11)

    def test_signal_to_entry_identity_change_is_rejected(self):
        seg = self.segment.copy()
        seg.iloc[1:] = 2
        y, _ = close_forward_returns(self.price, segment=seg)
        self.assertTrue(pd.isna(y.iloc[0, 0]))

    def test_delay_and_horizon_validation(self):
        for delay in (-1, True, 1.5):
            with self.assertRaises(ValueError):
                close_forward_returns(self.price, delay=delay)
        for horizon in (0, False, 1.5):
            with self.assertRaises(ValueError):
                close_forward_returns(self.price, horizon=horizon)

    def test_raw_y_cannot_be_silently_standardized(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                prepare(PanelStore(tmp), standardize=True)


if __name__ == '__main__':
    unittest.main()
