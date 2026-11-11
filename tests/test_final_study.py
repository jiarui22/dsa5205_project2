import numpy as np
import pandas as pd
from usq.project2.final_study import composite


def test_composite_neutral_missing_and_leave_one_out():
    parts = {'a': pd.DataFrame([[.4, 0], [.2, -.2]]),
             'b': pd.DataFrame([[0, -.4], [-.2, .2]])}
    np.testing.assert_allclose(composite(parts), [[.2, -.2], [0, 0]])
    pd.testing.assert_frame_equal(composite(parts, 'a'), parts['b'])
    # Changing later signals cannot change the earlier composite.
    changed = {k: v.copy() for k, v in parts.items()}
    changed['a'].iloc[1] = 100
    pd.testing.assert_series_equal(composite(parts).iloc[0], composite(changed).iloc[0])
