"""Expert recovery for c0's UNBROKEN evaluation-map drive (user 2026-11-06: "i
want one unbroke drive" -> "Expert recovers"). At a fault the expert takes the
wheel where the car is -- it is never moved -- and hands back once the car is
settled in a lane; a car that cannot move at all ends the drive instead of being
teleported. These tests pin the hand-back rule without CARLA."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from torch_rl.recovery import RecoveryGate, best_lane, wrap_deg  # noqa: E402

OK = dict(fault=None, lat_m=0.2, yaw_err_deg=3.0, speed_kmh=12.0)


def test_c0_drives_until_a_fault():
    g = RecoveryGate()
    assert g.driver == "c0"
    assert g.update(t=0, **OK) is None and g.driver == "c0"
    assert g.update(t=1, **{**OK, "fault": "off_road"}) == "takeover"
    assert g.driver == "expert" and g.events[-1]["fault"] == "off_road"


def test_a_car_stopped_under_c0_is_a_fault_too():
    g = RecoveryGate(stall_steps=3)
    for t in range(2):
        assert g.update(t=t, **{**OK, "speed_kmh": 0.1}) is None
    assert g.update(t=2, **{**OK, "speed_kmh": 0.1}) == "takeover"
    assert g.events[-1]["fault"] == "stalled"


def test_hand_back_needs_min_steps_then_a_settled_run():
    g = RecoveryGate(min_steps=5, settle_steps=3)
    g.update(t=0, **{**OK, "fault": "collision"})
    out = [g.update(t=t, **OK) for t in range(1, 8)]
    # steps 1..5 are the expert's minimum; settled from step 1, so the run of 3
    # settled steps is complete at step 5 -- but not before min_steps elapsed
    assert out[:4] == [None] * 4 and out[4] == "handback"
    assert g.driver == "c0" and g.events[-1]["steps"] == 5


def test_any_unsettled_step_restarts_the_settle_count():
    g = RecoveryGate(min_steps=1, settle_steps=3)
    g.update(t=0, **{**OK, "fault": "crossing"})
    seq = [OK, OK, {**OK, "lat_m": 1.5}, OK, OK]
    assert [g.update(t=t, **s) for t, s in enumerate(seq, 1)] == [None] * 5
    assert g.update(t=6, **OK) == "handback"


def test_settled_means_in_lane_aligned_moving_and_fault_free():
    g = RecoveryGate(min_steps=1, settle_steps=1, max_lat_m=0.6, max_yaw_deg=15.0,
                     min_kmh=5.0)
    g.update(t=0, **{**OK, "fault": "off_road"})
    for bad in ({"lat_m": 0.7}, {"yaw_err_deg": -16.0}, {"speed_kmh": 4.0},
                {"fault": "crossing"}):
        assert g.update(t=1, **{**OK, **bad}) is None
    assert g.update(t=2, **OK) == "handback"


def test_a_car_that_cannot_move_ends_the_drive_rather_than_jump():
    g = RecoveryGate(wedged_steps=4, max_backups=0)
    g.update(t=0, **{**OK, "fault": "collision"})
    out = [g.update(t=t, **{**OK, "speed_kmh": 0.0}) for t in range(1, 5)]
    assert out == [None, None, None, "wedged"] and g.driver == "none"


def test_a_blocked_expert_reverses_then_replans():
    # job 464244: nose against a wall after leaving the road, full throttle, no
    # reverse in the action space -> 30 s of nothing. Now it backs out first.
    g = RecoveryGate(blocked_steps=2, backup_steps=3, max_backups=2, wedged_steps=50)
    g.update(t=0, **{**OK, "fault": "collision"})
    still = {**OK, "speed_kmh": 0.0}
    assert [g.update(t=t, **still) for t in (1, 2)] == [None, "backup"]
    assert g.backing
    # reversing: moving or not, no settle and no new block until it is over
    assert [g.update(t=t, **still) for t in (3, 4, 5)] == [None, None, "replan"]
    assert not g.backing and g.events[-1]["backups"] == 1
    assert [g.update(t=t, **still) for t in (6, 7)] == [None, "backup"]


def test_backups_are_limited_then_the_wedged_clock_decides():
    g = RecoveryGate(blocked_steps=1, backup_steps=1, max_backups=1, wedged_steps=3)
    g.update(t=0, **{**OK, "fault": "off_road"})
    still = {**OK, "speed_kmh": 0.0}
    out = [g.update(t=t, **still) for t in range(1, 6)]
    assert out == ["backup", "replan", None, None, "wedged"]
    assert g.summary()["backups"] == 1 and g.summary()["wedged"]


def test_summary_counts_takeovers_by_fault():
    g = RecoveryGate(min_steps=1, settle_steps=1)
    for f in ("off_road", "off_road", "collision"):
        g.update(t=0, **{**OK, "fault": f})
        g.update(t=1, **OK)
    s = g.summary()
    assert s["takeovers"] == 3 and s["by_fault"] == {"off_road": 2, "collision": 1}
    assert s["expert_steps"] == 3


def test_wrap_and_best_lane_pick_the_lane_the_car_points_along():
    assert wrap_deg(190.0) == -170.0 and wrap_deg(-190.0) == 170.0
    # candidates: (yaw, name); the car points at 178 deg -> the lane at -179
    assert best_lane(178.0, [(0.0, "a"), (-179.0, "b"), (90.0, "c")]) == "b"
    assert best_lane(10.0, []) is None
