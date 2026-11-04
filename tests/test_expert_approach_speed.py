"""The expert's final approach slows BEFORE the goal (imitation-first plan, M0).

The design asks for a 6 km/h final approach, but BasicAgent drives its whole
route at the cruise speed and only the crawl after the route's end was slow.
The M0 benchmark (jobs 463141-463144) measured what that does: the car entered
the 2 m arrival circle at a median 11.6 km/h (118/152 arrivals above the
10 km/h limit) and was at a median 9.73 km/h at its closest point, so arrival
hung on coasting the last metre; and where the planner's snapped route end lay
at or past the goal the car drove through it at 12-14 km/h -- 5 of the 8 goal
failures. A clone copying that profile would lose arrivals to tiny speed
errors. So the target speed now falls with the straight-line distance to the
goal: cruise until APPROACH_SLOW_FROM_M, the final speed from
APPROACH_FINAL_AT_M in. Explore plans have no goal and keep the cruise speed.
"""
from types import SimpleNamespace

import pytest

from vae_nav.expert_oracle import BasicAgentExpert


def test_approach_speed_profile():
    f = BasicAgentExpert.approach_speed_kmh
    assert f(25.0, 15.0) == pytest.approx(15.0)
    assert f(10.0, 15.0) == pytest.approx(15.0)
    assert f(3.0, 15.0) == pytest.approx(6.0)
    assert f(0.5, 15.0) == pytest.approx(6.0)
    assert f(6.5, 15.0) == pytest.approx(10.5)
    speeds = [f(d / 10, 15.0) for d in range(0, 120)]
    assert all(a <= b for a, b in zip(speeds, speeds[1:]))     # monotone in distance


def test_a_slow_cruise_is_never_raised():
    assert BasicAgentExpert.approach_speed_kmh(20.0, 5.0) == pytest.approx(5.0)
    assert BasicAgentExpert.approach_speed_kmh(1.0, 5.0) == pytest.approx(5.0)


class _Agent:
    def __init__(self):
        self.speeds = []

    def done(self):
        return False

    def set_target_speed(self, v):
        self.speeds.append(v)

    def run_step(self):
        return (0.0, 0.5)


def _expert(goal, car=(0.0, 0.0)):
    e = BasicAgentExpert.__new__(BasicAgentExpert)
    e._planned = True
    e._goal_loc = goal
    e.target_speed = 15.0
    e._agent = _Agent()
    e._vehicle = SimpleNamespace(get_location=lambda: SimpleNamespace(x=car[0], y=car[1]))
    e._to_action = lambda control: control
    return e


def test_a_goal_episode_slows_by_distance_to_the_goal():
    e = _expert(SimpleNamespace(x=5.0, y=0.0))
    e.act()
    assert e._agent.speeds == [pytest.approx(BasicAgentExpert.approach_speed_kmh(5.0, 15.0))]
    assert e._agent.speeds[0] < 15.0


def test_an_explore_plan_drives_at_the_cruise_speed():
    """Also undoes a previous goal episode's slow target: the expert outlives episodes."""
    e = _expert(None)
    e._agent.speeds.append(6.0)
    e.act()
    assert e._agent.speeds[-1] == pytest.approx(15.0)
