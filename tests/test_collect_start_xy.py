"""--collect-start-xy parses to the first episode's continuation point.

USER DECISION 2026-09-13: start the Town04 trace inside the city grid, because
random-spawn Town04 traces drove the outer highway once and never revisited
their own path, leaving SGM nothing to merge.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_empty_means_no_forced_start():
    import eval_placement as ep
    assert ep.parse_start_xy("") is None
    assert ep.parse_start_xy(None) is None


def test_parses_negative_coordinates():
    import eval_placement as ep
    out = ep.parse_start_xy("300,-172.5")
    assert np.allclose(out, [300.0, -172.5])


@pytest.mark.parametrize("bad", ["300", "a,b", "1,2,3"])
def test_rejects_malformed(bad):
    import eval_placement as ep
    with pytest.raises(ValueError):
        ep.parse_start_xy(bad)
