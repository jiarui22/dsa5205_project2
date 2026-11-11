"""Material quarantine must distinguish source anomalies from rounding noise."""
import numpy as np
import pandas as pd
from usq.project2.data_preparation import material_ohlc


def test_material_bar_rule_requires_absolute_and_relative_error():
    open_=pd.DataFrame([[1.02,5.72,10.0,10000.0]])
    high=pd.DataFrame([[10.73,6.09,11.0,11000.0]])
    low=pd.DataFrame([[9.75,5.7600007,9.0,10000.02]])
    close=pd.DataFrame([[10.12,5.99,8.999999,10001.0]])
    actual=material_ohlc(open_,high,low,close)
    np.testing.assert_array_equal(actual,[[True,True,False,False]])
    # Quarantine cannot turn missing OHLC fields into an executable observation.
    open_.iloc[0,0]=np.nan
    assert not material_ohlc(open_,high,low,close).iloc[0,0]


def test_low_disk_space_is_reported_before_creating_data_copy(tmp_path, monkeypatch):
    from collections import namedtuple
    import pytest
    from usq.project2 import data_preparation

    source = tmp_path / 'data'
    source.mkdir()
    (source / 'input.bin').write_bytes(b'input')
    work = tmp_path / 'run'
    work.mkdir()
    usage = namedtuple('Usage', 'total used free')
    monkeypatch.setattr(data_preparation.shutil, 'disk_usage', lambda path: usage(100, 100, 0))
    with pytest.raises(OSError, match='Insufficient space for a fresh experiment') as error:
        data_preparation.check_workspace_space(source, work, panel_cells=100)
    assert error.value.errno == 28
    assert not (work / 'database').exists()
