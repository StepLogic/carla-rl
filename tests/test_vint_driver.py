"""Driving CARLA from ViNT's predictions (user 2026-11-07: evaluate ViNT like c0).

ViNT predicts where to go (5 waypoints) and how far the goal is; it does not
predict controls. The driver executes the waypoints with the EXPERT'S OWN
low-level controller -- the vendored CARLA LocalPlanner defaults the expert
drove the training data with -- so ViNT is judged on its predictions, not on a
controller we tuned for it:
  * lateral PID 1.95 / 0.05 / 0.2 at dt 0.05 on the signed angle to the target,
    steering moving at most 0.1 per step, |steer| <= 0.8;
  * target = the first predicted waypoint at least 3 m + 0.5 s x speed away;
  * longitudinal PID 1.0 / 0.05 / 0 toward the expert's approach profile
    (15 km/h, slowing from 10 m to 6 km/h at 3 m), fed with ViNT's PREDICTED
    goal distance instead of the true one;
  * throttle <= 0.75, brake <= 0.3 as a negative throttle, then the env's box.
ViNT frame: x forward, y LEFT. CARLA steer: + = RIGHT.
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from torch_rl.vint_driver import PID, ViNTDriver, pick_waypoint, signed_angle  # noqa: E402

LOW, HIGH = np.array([-1.0, -1.0], np.float32), np.array([1.0, 1.0], np.float32)
STRAIGHT = np.array([[1.1 * k, 0.0] for k in range(1, 6)])


class FakePolicy:
    def __init__(self, dist_m=50.0, wps=STRAIGHT):
        self.dist_m, self.wps, self.calls = dist_m, np.asarray(wps, float), []

    def __call__(self, ctx, goal):
        self.calls.append((list(ctx), goal))
        return self.dist_m, self.wps


def _frame(v):
    return np.full((4, 4, 3), v, np.uint8)


def test_the_pid_is_carlas_formula():
    pid = PID(1.95, 0.05, 0.2, dt=0.05)
    assert pid.step(0.1) == pytest.approx(1.95 * 0.1)              # no history: P only
    e = [0.1, 0.3]
    want = 1.95 * 0.3 + 0.2 * (0.3 - 0.1) / 0.05 + 0.05 * sum(e) * 0.05
    assert pid.step(0.3) == pytest.approx(np.clip(want, -1, 1))
    for _ in range(20):
        pid.step(5.0)
    assert pid.step(5.0) == 1.0 and len(pid.errors) == 10           # clipped, 10-error memory


def test_the_signed_angle_is_positive_to_the_right():
    assert signed_angle([5.0, -1.0]) > 0 and signed_angle([5.0, 1.0]) < 0
    assert signed_angle([5.0, 0.0]) == pytest.approx(0.0)
    assert abs(signed_angle([3.0, -3.0])) == pytest.approx(np.pi / 4)


def test_the_target_is_the_first_waypoint_past_the_planners_lookahead():
    assert pick_waypoint(STRAIGHT, 3.0)[0] == pytest.approx(3.3)
    assert pick_waypoint(STRAIGHT, 3.0 + 0.5 * 15 / 3.6)[0] == pytest.approx(5.5)
    assert pick_waypoint(STRAIGHT, 99.0)[0] == pytest.approx(5.5)    # none far enough: the last


def test_context_takes_every_stride_frame_newest_last_and_pads_the_start():
    pol = FakePolicy()
    d = ViNTDriver(pol, LOW, HIGH, context_size=5, stride=3)
    d.reset_hidden(1)
    d.act_env(_frame(0), _frame(99), speed_kmh=0.0)
    assert [int(f[0, 0, 0]) for f in pol.calls[-1][0]] == [0] * 6
    for v in range(1, 21):
        d.act_env(_frame(v), _frame(99), speed_kmh=0.0)
    assert [int(f[0, 0, 0]) for f in pol.calls[-1][0]] == [5, 8, 11, 14, 17, 20]


def test_a_waypoint_to_the_left_steers_left_and_the_rate_is_limited():
    left = STRAIGHT + np.array([0.0, 1.5])
    d = ViNTDriver(FakePolicy(wps=left), LOW, HIGH)
    d.reset_hidden(1)
    s1 = d.act_env(_frame(1), _frame(2), speed_kmh=10.0)[0]
    assert s1 == pytest.approx(-0.1)                                  # one step: at most 0.1
    for _ in range(20):
        s = d.act_env(_frame(1), _frame(2), speed_kmh=10.0)[0]
    assert -0.8 <= s < -0.1
    right = STRAIGHT - np.array([0.0, 1.5])
    d = ViNTDriver(FakePolicy(wps=right), LOW, HIGH)
    d.reset_hidden(1)
    assert d.act_env(_frame(1), _frame(2), speed_kmh=10.0)[0] > 0


def test_it_accelerates_from_rest_and_brakes_into_a_near_goal():
    d = ViNTDriver(FakePolicy(dist_m=60.0), LOW, HIGH, cruise_kmh=15.0)
    d.reset_hidden(1)
    assert d.act_env(_frame(1), _frame(2), speed_kmh=0.0)[1] == pytest.approx(0.75)
    d = ViNTDriver(FakePolicy(dist_m=2.0), LOW, HIGH, cruise_kmh=15.0)
    d.reset_hidden(1)
    a = d.act_env(_frame(1), _frame(2), speed_kmh=15.0)
    assert a[1] < 0 and d.last["target_kmh"] == pytest.approx(6.0)  # the expert's final approach


def test_actions_are_clipped_to_the_env_box():
    low, high = np.array([-0.5, -0.2], np.float32), np.array([0.5, 0.4], np.float32)
    d = ViNTDriver(FakePolicy(dist_m=60.0), low, high)
    d.reset_hidden(1)
    a = d.act_env(_frame(1), _frame(2), speed_kmh=0.0)
    assert a.dtype == np.float32 and a[1] == pytest.approx(0.4)


def test_reset_forgets_the_frames_and_the_controllers():
    pol = FakePolicy(wps=STRAIGHT + np.array([0.0, 1.5]))
    d = ViNTDriver(pol, LOW, HIGH)
    d.reset_hidden(1)
    for v in range(10):
        d.act_env(_frame(v), _frame(99), speed_kmh=10.0)
    d.reset_hidden(1)
    assert d.act_env(_frame(50), _frame(99), speed_kmh=10.0)[0] == pytest.approx(-0.1)
    assert [int(f[0, 0, 0]) for f in pol.calls[-1][0]] == [50] * 6
