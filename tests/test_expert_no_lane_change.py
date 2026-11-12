"""The expert never plans a lane change (user 2026-09-28: "i want no lane change").

On the real Town05 OpenDRIVE map (offline, no simulator): goals are sampled as
CarlaCore.reset_hero does; for every route where CARLA's planner (the one
BasicAgent uses) takes a lane change, lane_following_route must reach the goal
using next() alone, with no CHANGELANE option.
"""
import os
import random
import sys

import pytest

XODR = "/home/robotlab/Apps/CARLA_0.9.15/CarlaUE4/Content/Carla/Maps/OpenDrive/Town05.xodr"
carla = pytest.importorskip("carla")
if not os.path.exists(XODR):
    pytest.skip("Town05.xodr not installed", allow_module_level=True)
sys.path.append("/home/robotlab/Apps/CARLA_0.9.15/PythonAPI/carla")
sys.path.insert(0, "/home/robotlab/scratch/carla-rl/dependencies/rlib_integration")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

from agents.navigation.new_route_planner import GlobalRoutePlanner  # noqa: E402
from rlib_integration.carla_core import pick_origin  # noqa: E402

from vae_nav.expert_oracle import (lane_following_route,  # noqa: E402
                                   route_has_lane_change)


def test_lane_change_routes_are_replaced_by_lane_following_ones():
    m = carla.Map("Town05", open(XODR).read())
    allw = m.generate_waypoints(2)
    wps = [w for w in allw if not w.is_junction]
    jns = [w for w in allw if w.is_junction]
    band = [5.0 + 2.5 * i for i in range(79)]
    grp = GlobalRoutePlanner(m, 2.0)
    rng = random.Random(0)
    found = 0
    for _ in range(600):
        origin, maxf, _ = pick_origin(wps, jns, band, 0.5, 0.3, rng=rng)
        b = band if maxf is None else ([d for d in band if d <= maxf] or [maxf * 0.5])
        dest = None
        for _ in range(24):
            c = origin.next(rng.choice(b))
            if c:
                dest = c[-1]
                break
        if dest is None:
            continue
        route = grp.trace_route(origin.transform.location, dest.transform.location)
        if not route_has_lane_change(route):
            continue
        found += 1
        lf = lane_following_route(origin, dest.transform.location)
        assert lf, "no lane-following route for a goal sampled by next()"
        assert not route_has_lane_change(lf)
        end = lf[-1][0].transform.location
        assert end.distance(dest.transform.location) <= 3.0
        # consecutive waypoints are next() steps: never more than ~2 m apart
        for (a, _), (b2, _) in zip(lf, lf[1:]):
            assert a.transform.location.distance(b2.transform.location) < 2.6
    assert found >= 3, f"expected the audit's lane-change routes, found {found}"
