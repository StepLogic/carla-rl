"""BasicAgentExpert.reset must plan from where the car IS, not where it WAS.

CARLA's BasicAgent.set_destination(goal) traces the route from the local
planner's current target_waypoint -- the waypoint under the vehicle when the
agent was built, or wherever the last episode ended. The env builds one expert
and teleports the hero every episode, so every in-env route started at a stale
point (measured: route[0] fixed at the construction spawn, 86-326 m from the
real start, 9/9 oracle episodes lost). This pins the fix: trace from the
vehicle's current waypoint, install it with clean_queue=True, and never call
set_destination.
"""
from collections import deque
from types import SimpleNamespace

from vae_nav.expert_oracle import BasicAgentExpert


class _Loc:
    def __init__(self, x, y):
        self.x, self.y = x, y


class _FakeMap:
    def get_waypoint(self, loc):
        return SimpleNamespace(loc=loc)


class _FakeAgent:
    def __init__(self):
        self.calls = []
        self._lp = SimpleNamespace(
            _waypoints_queue=deque([("stale", "LANEFOLLOW")] * 5),
            _vehicle_controller=SimpleNamespace(
                _lon_controller=SimpleNamespace(_error_buffer=deque([1.0, 2.0])),
                _lat_controller=SimpleNamespace(_e_buffer=deque([0.5]))))

    def get_local_planner(self):
        return self._lp

    def trace_route(self, start_wp, end_wp):
        self.calls.append(("trace", start_wp.loc, end_wp.loc))
        return [("a", "LANEFOLLOW"), ("b", "LANEFOLLOW"), ("c", "LANEFOLLOW")]

    def set_global_plan(self, route, clean_queue=False, **_):
        self.calls.append(("set_global_plan", clean_queue))
        if clean_queue:
            self._lp._waypoints_queue.clear()
        self._lp._waypoints_queue.extend(route)

    def set_destination(self, *a, **k):
        raise AssertionError("set_destination plans from a stale waypoint")


def _expert(vehicle_loc):
    e = BasicAgentExpert.__new__(BasicAgentExpert)
    e._vehicle = SimpleNamespace(get_location=lambda: vehicle_loc)
    e._map = _FakeMap()
    e._agent = _FakeAgent()
    e._goal_loc = None
    e._planned = False
    return e


def test_reset_traces_from_current_vehicle_location_and_clears_old_queue():
    here, goal = _Loc(10.0, 20.0), _Loc(40.0, 20.0)
    e = _expert(here)
    n = e.reset(goal)
    assert n == 3, "route length must be the NEW plan, not old+new"
    assert e._agent.calls[0] == ("trace", here, goal)
    assert ("set_global_plan", True) in e._agent.calls
    assert list(e._agent.get_local_planner()._waypoints_queue) == [
        ("a", "LANEFOLLOW"), ("b", "LANEFOLLOW"), ("c", "LANEFOLLOW")]
    assert e._planned and e.last_plan_error is None


def test_reset_clears_pid_error_buffers():
    e = _expert(_Loc(0.0, 0.0))
    e.reset(_Loc(5.0, 0.0))
    ctrl = e._agent.get_local_planner()._vehicle_controller
    assert len(ctrl._lon_controller._error_buffer) == 0
    assert len(ctrl._lat_controller._e_buffer) == 0


def test_second_reset_after_teleport_plans_from_new_location():
    e = _expert(_Loc(0.0, 0.0))
    e.reset(_Loc(5.0, 0.0))
    far = _Loc(300.0, 300.0)
    e._vehicle = SimpleNamespace(get_location=lambda: far)   # teleported hero
    e.reset(_Loc(330.0, 300.0))
    traces = [c for c in e._agent.calls if c[0] == "trace"]
    assert traces[-1][1] is far


def test_final_approach_never_turns_around():
    fa = BasicAgentExpert._final_approach
    # goal 3 m straight ahead: crawl on, no steer
    steer, go = fa((0.0, 0.0), (1.0, 0.0), (3.0, 0.0))
    assert go and steer == 0.0
    # goal 3 m ahead, 0.5 m to the right: gentle steer, clipped
    steer, go = fa((0.0, 0.0), (1.0, 0.0), (3.0, 0.5))
    assert go and 0.0 < steer <= BasicAgentExpert.FINAL_APPROACH_MAX_STEER
    # goal beside the car (no longer ahead): stop, do not circle
    assert fa((0.0, 0.0), (1.0, 0.0), (0.0, 2.5)) == (0.0, False)
    # goal behind: stop
    assert fa((0.0, 0.0), (1.0, 0.0), (-2.0, 0.3)) == (0.0, False)
    # within the stop radius: stop
    assert fa((0.0, 0.0), (1.0, 0.0), (0.8, 0.0)) == (0.0, False)
