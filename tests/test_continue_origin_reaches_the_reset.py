"""The continuation origin must actually REACH the reset, not just be set.

MEASURED 2026-09-12 22:40: every v6n placement trace (Town02 s7-s10, Town04
s7-s10) was collected with RANDOM RESPAWNS despite COLLECT_CONTINUE=1. The
gap from each episode's first position to the previous episode's last was a
median 96-433 m, with at most 1 of 29 under 15 m.

CAUSE: CarlaEnv.reset (rlib_integration/carla_env.py:74) takes the experiment's
waypoints only when BOTH `origin` and `destination` are set and truthy;
otherwise it calls reset_hero, the random sampler. collect_trace set `origin`
alone. The episodes-from-trace path set both, which is why it worked there and
why the earlier test -- which only checked the helper's return value -- passed.

These tests pin the contract against a fake env that applies the same rule.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class FakeLoc:
    def __init__(self, x, y):
        self.x, self.y, self.z = x, y, 0.0


class FakeWP:
    def __init__(self, x, y, ahead=None):
        self.transform = type("T", (), {"location": FakeLoc(x, y)})()
        self._ahead = ahead

    def next(self, d):
        return [self._ahead] if self._ahead is not None else []


class FakeExperiment:
    pass


def carla_env_reset_rule(exp):
    """carla_env.py:74, verbatim in logic: both set and truthy, or random."""
    if (hasattr(exp, "origin") and hasattr(exp, "destination")
            and exp.origin and exp.destination):
        loc = exp.origin.transform.location
        return (loc.x, loc.y)
    return ("RANDOM",)


def test_setting_origin_alone_is_ignored_by_the_reset():
    """The bug, reproduced: origin without destination -> random spawn."""
    exp = FakeExperiment()
    exp.origin = FakeWP(10.0, 20.0)
    assert carla_env_reset_rule(exp) == ("RANDOM",)


def test_set_continue_origin_reaches_the_reset():
    import importlib
    import eval_placement as ep
    importlib.reload(ep)
    exp = FakeExperiment()
    wp = FakeWP(10.0, 20.0, ahead=FakeWP(40.0, 20.0))
    ep.set_continue_origin(exp, wp)
    assert carla_env_reset_rule(exp) == (10.0, 20.0)


def test_destination_falls_back_to_origin_at_a_dead_end():
    """No waypoint ahead (end of a lane) must still continue, not go random."""
    import importlib
    import eval_placement as ep
    importlib.reload(ep)
    exp = FakeExperiment()
    wp = FakeWP(5.0, 6.0, ahead=None)
    ep.set_continue_origin(exp, wp)
    assert carla_env_reset_rule(exp) == (5.0, 6.0)


def test_clear_removes_both_so_the_sampler_is_untouched_afterwards():
    import importlib
    import eval_placement as ep
    importlib.reload(ep)
    exp = FakeExperiment()
    ep.set_continue_origin(exp, FakeWP(1.0, 2.0, ahead=FakeWP(3.0, 4.0)))
    ep.clear_continue_origin(exp)
    assert not hasattr(exp, "origin") and not hasattr(exp, "destination")
    assert carla_env_reset_rule(exp) == ("RANDOM",)


def test_spawn_check_flags_a_silent_fallback():
    """A continuation whose spawn lands far from the waypoint is a failure."""
    import importlib
    import eval_placement as ep
    importlib.reload(ep)
    wp = FakeWP(10.0, 20.0)
    assert ep.continued_spawn_ok(np.array([11.0, 20.5]), wp) is True
    assert ep.continued_spawn_ok(np.array([400.0, -20.0]), wp) is False
