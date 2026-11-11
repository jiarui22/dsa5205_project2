import numpy as np
import pandas as pd
from usq.database.wrds_reconstruction import interval_segments, dated_join, _comparison
from usq.database.sina_validation import validate_pair


def test_dated_links_expire_and_conflicting_overlaps_are_unmapped():
    links = pd.DataFrame({'permno':[1,1], 'lo':pd.to_datetime(['2020-01-01','2020-01-03']),
        'hi':pd.to_datetime(['2020-01-04','2020-01-05']), 'sid':['A','B']})
    segments, conflicts = interval_segments(links,'permno','lo','hi',['sid'])
    dates = pd.date_range('2019-12-31','2020-01-06')
    result = dated_join(pd.DataFrame({'permno':1,'date':dates}),segments)
    assert len(conflicts) == 1
    assert result.sid.fillna('?').tolist() == ['?','A','A','?','?','B','?']


def test_identical_overlapping_links_do_not_create_ambiguity():
    links = pd.DataFrame({'permno':[1,1], 'lo':pd.to_datetime(['2020-01-01','2020-01-03']),
        'hi':pd.to_datetime(['2020-01-04','2020-01-05']), 'sid':['A','A']})
    segments, conflicts = interval_segments(links,'permno','lo','hi',['sid'])
    result = dated_join(pd.DataFrame({'permno':[1], 'date':pd.to_datetime(['2020-01-05'])}),segments)
    assert conflicts.empty and result.sid.iloc[0] == 'A'


def test_sina_vwap_uses_paired_vendor_volume_and_rejects_zero_amount():
    d = pd.DataFrame({'open':[10.,10.,10.], 'high':[11.,11.,11.], 'low':[9.,9.,9.],
        'close':[10.,10.,10.], 'openprc':[10.,10.,20.], 'askhi':[11.,11.,21.],
        'bidlo':[9.,9.,19.], 'prc':[10.,10.,20.], 'amount':[1000.,0.,1000.],
        'source_volume':[100.,100.,100.], 'vol':[200.,200.,200.]})
    matches, vwap = validate_pair(d)
    assert vwap.iloc[0] == 10
    assert vwap.iloc[1:].isna().all()
    assert matches.tolist() == [True,True,False]


def test_reconciliation_counts_missing_values_separately():
    result = _comparison([1.,np.nan,3.],[1.,2.,np.nan])
    assert result['matched'] == 1 and result['common'] == 1
    assert result['raw_only'] == 1 and result['reference_only'] == 1


def test_quote_based_crsp_mark_does_not_validate_executable_vwap():
    d = pd.DataFrame({'open':[10.], 'high':[11.], 'low':[9.], 'close':[10.],
        'openprc':[10.], 'askhi':[11.], 'bidlo':[9.], 'prc':[-10.],
        'amount':[1000.], 'source_volume':[100.]})
    _,vwap = validate_pair(d)
    assert vwap.isna().all()
