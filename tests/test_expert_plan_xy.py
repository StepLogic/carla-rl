"""BasicAgentExpert.plan_xy: the expert's current plan as a CARLA x/y polyline for
ViNT's DAgger aim labels -- the queued waypoints, extended back along the lane
(so the car's projection falls inside) and past the end."""
from collections import deque
from types import SimpleNamespace

import numpy as np

from vae_nav.expert_oracle import BasicAgentExpert


class _Wp:
    """A lane along +x from x = -2 to x = 30 (CARLA metres)."""

    def __init__(self, x):
        self.x = float(x)
        self.transform = SimpleNamespace(location=SimpleNamespace(x=self.x, y=5.0))

    def previous(self, d):
        return [_Wp(self.x - d)] if self.x - d >= -2.0 else []

    def next(self, d):
        return [_Wp(self.x + d)] if self.x + d <= 30.0 else []


def _expert(queue, planned=True):
    e = BasicAgentExpert.__new__(BasicAgentExpert)
    e._planned = planned
    e._agent = SimpleNamespace(get_local_planner=lambda: SimpleNamespace(
        _waypoints_queue=deque((w, "LANEFOLLOW") for w in queue)))
    return e


def test_the_plan_with_its_extensions():
    xy = _expert([_Wp(x) for x in (4.0, 6.0, 8.0)]).plan_xy(back_m=3.0, ahead_m=2.0, step_m=1.0)
    assert np.allclose(xy[:, 0], [1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 9.0, 10.0])
    assert np.allclose(xy[:, 1], 5.0)


def test_extensions_stop_where_the_lane_does():
    xy = _expert([_Wp(0.0), _Wp(29.0)]).plan_xy(back_m=5.0, ahead_m=5.0, step_m=1.0)
    assert xy[0, 0] == -2.0 and xy[-1, 0] == 30.0


def test_no_plan_is_an_empty_polyline():
    assert _expert([], planned=True).plan_xy().shape == (0, 2)
    assert _expert([_Wp(1.0)], planned=False).plan_xy().shape == (0, 2)
