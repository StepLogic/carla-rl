"""Junction hops for c0's evaluation-map trace (user 2026-11-06: "Can we explicitly
random actions at junctions" -> "Goal hop per junction").

c0 explores in explore mode; nearing a junction it is given a GOAL past a randomly
chosen exit (goal_chain.goal_past_next_junction) and drives there in goal mode,
then explores again. These tests pin the pure parts: when a hop starts, how a
hop ends, which exit it asked for, and that every mode switch is a soft reset
(the car stays, the GRU starts fresh).
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from torch_rl.junction_hops import Hop, HopSchedule, exit_class  # noqa: E402


def test_exit_class_reads_carla_yaw_right_handed_as_right():
    # CARLA is left-handed seen from above: a positive yaw change is a right turn
    assert exit_class(0.0, 5.0) == "straight"
    assert exit_class(0.0, 90.0) == "right"
    assert exit_class(0.0, -90.0) == "left"
    assert exit_class(170.0, -100.0) == "right"            # wraps through 180
    assert exit_class(0.0, 31.0) == "right" and exit_class(0.0, 29.0) == "straight"


def test_a_hop_starts_only_when_exploring_and_a_junction_is_close_ahead():
    s = HopSchedule(trigger_m=20.0)
    assert s.mode == "explore"
    assert not s.want_hop(ahead_m=None, in_junction=False)     # no junction ahead
    assert not s.want_hop(ahead_m=35.0, in_junction=False)     # too far
    assert not s.want_hop(ahead_m=0.0, in_junction=True)       # already inside one
    assert s.want_hop(ahead_m=12.0, in_junction=False)
    s.begin(Hop(0, goal_xy=(10.0, 0.0), goal_yaw=0.0, start_t=5, start_xy=(0.0, 0.0),
                start_yaw=0.0))
    assert s.mode == "pending" and not s.want_hop(ahead_m=12.0, in_junction=False)


def test_the_goal_mode_starts_once_the_goal_view_is_rendered_and_resets_the_gru():
    s = HopSchedule(trigger_m=20.0)
    s.begin(Hop(0, goal_xy=(30.0, 0.0), goal_yaw=0.0, start_t=0, start_xy=(0.0, 0.0),
                start_yaw=0.0))
    # the goal camera has not moved yet: still pending, nothing to reset
    assert s.goal_view_ready(goal_pos=(0.0, 50.0)) is False and s.mode == "pending"
    assert s.goal_view_ready(goal_pos=(30.4, 0.3)) is True
    assert s.mode == "goal" and s.mode_bit == 1 and s.take_reset() is True
    assert s.take_reset() is False                              # one reset per switch


def test_a_hop_that_never_gets_its_goal_view_is_abandoned():
    s = HopSchedule(trigger_m=20.0, max_pending_steps=2)
    s.begin(Hop(0, goal_xy=(30.0, 0.0), goal_yaw=0.0, start_t=0, start_xy=(0.0, 0.0),
                start_yaw=0.0))
    assert s.goal_view_ready(goal_pos=None) is False
    assert s.goal_view_ready(goal_pos=None) is False
    assert s.mode == "explore" and s.done[-1].end == "no_goal_view"


def test_arrival_within_the_radius_ends_the_hop_and_switches_back_to_explore():
    s = HopSchedule(trigger_m=20.0, arrival_m=2.0)
    s.begin(Hop(3, goal_xy=(30.0, 0.0), goal_yaw=90.0, start_t=0, start_xy=(0.0, 0.0),
                start_yaw=0.0))
    s.goal_view_ready(goal_pos=(30.0, 0.0))
    s.take_reset()
    assert s.step(t=10, pos=(20.0, 0.0), yaw=45.0) is None
    end = s.step(t=20, pos=(28.5, 0.5), yaw=88.0)
    assert end == "reached" and s.mode == "explore" and s.mode_bit == 0
    assert s.take_reset() is True
    hop = s.done[-1]
    assert (hop.id, hop.end, hop.end_t) == (3, "reached", 20)
    assert hop.planned_exit == "right" and hop.driven_exit == "right"


def test_driving_away_from_the_goal_ends_the_hop_as_missed():
    s = HopSchedule(trigger_m=20.0, arrival_m=2.0, miss_m=10.0)
    s.begin(Hop(0, goal_xy=(30.0, 0.0), goal_yaw=0.0, start_t=0, start_xy=(0.0, 0.0),
                start_yaw=0.0))
    s.goal_view_ready(goal_pos=(30.0, 0.0))
    for t, x in enumerate((5.0, 10.0, 15.0)):                 # closest approach 15 m
        assert s.step(t=t, pos=(x, 0.0), yaw=0.0) is None
    assert s.step(t=4, pos=(15.0, -20.0), yaw=-90.0) == "missed"   # 25 m: 10 past the closest
    assert s.done[-1].driven_exit == "left"


def test_a_hop_times_out():
    s = HopSchedule(trigger_m=20.0, max_hop_steps=50)
    s.begin(Hop(0, goal_xy=(300.0, 0.0), goal_yaw=0.0, start_t=100, start_xy=(0.0, 0.0),
                start_yaw=0.0))
    s.goal_view_ready(goal_pos=(300.0, 0.0))
    assert s.step(t=149, pos=(1.0, 0.0), yaw=0.0) is None
    assert s.step(t=150, pos=(1.0, 0.0), yaw=0.0) == "timeout"


def test_a_segment_end_closes_the_hop_with_its_fault_and_restarts_in_explore():
    s = HopSchedule(trigger_m=20.0)
    s.begin(Hop(0, goal_xy=(30.0, 0.0), goal_yaw=0.0, start_t=0, start_xy=(0.0, 0.0),
                start_yaw=0.0))
    s.goal_view_ready(goal_pos=(30.0, 0.0))
    s.segment_end(t=7, reason="collision")
    assert s.mode == "explore" and s.done[-1].end == "collision"
    s.segment_end(t=9, reason="off_road")                      # no hop open: nothing recorded
    assert len(s.done) == 1


def test_summary_counts_hops_by_end_and_exit():
    s = HopSchedule(trigger_m=20.0)
    for i, (gyaw, end) in enumerate(((0.0, "reached"), (90.0, "reached"), (-90.0, "collision"))):
        s.begin(Hop(i, goal_xy=(30.0, 0.0), goal_yaw=gyaw, start_t=10 * i, start_xy=(0.0, 0.0),
                    start_yaw=0.0))
        s.goal_view_ready(goal_pos=(30.0, 0.0))
        if end == "reached":
            s.step(t=10 * i + 5, pos=(30.0, 0.0), yaw=gyaw)
        else:
            s.segment_end(t=10 * i + 5, reason=end)
    out = s.summary()
    assert out["attempted"] == 3 and out["ends"] == {"reached": 2, "collision": 1}
    assert out["planned_exits"] == {"straight": 1, "right": 1, "left": 1}
    assert out["reached_by_exit"] == {"straight": 1, "right": 1}


# ---------------------------------------------------------------------------
# The wiring in run_trace. The loop itself needs CARLA, so these pin the parts
# that are easy to get wrong -- which GRU is reset, which mode bit is recorded,
# that the goal camera is moved the way the chain's soft reset moves it -- in the
# same source-reading style as tests/test_explore_trace.py.
# ---------------------------------------------------------------------------
def _trace_body():
    src = open(os.path.join(ROOT, "train_carla_sac.py")).read()
    return src.split("def run_trace(")[1].split("\ndef run_eval(")[0]


def test_the_hop_is_off_by_default_so_a_no_hop_trace_is_unchanged():
    src = open(os.path.join(ROOT, "train_carla_sac.py")).read()
    assert 'TRACE_HOPS = os.environ.get("TRACE_HOPS", "0") == "1"' in src
    assert 'TRACE_SAVE_BINOC = os.environ.get("TRACE_SAVE_BINOC", "0") == "1"' in src
    body = _trace_body()
    assert "if TRACE_HOPS else None" in body          # no schedule, no second GRU


def test_the_hop_moves_the_destination_and_the_goal_camera_like_the_chain():
    body = _trace_body()
    assert "chain_goal(core.map, loc, hop_rng)" in body     # the random exit
    assert "core.destination = dest" in body
    assert "sensor.update_location(dest)" in body
    assert "core.world.tick()" in body
    assert "pxr.goal_block(info[\"debit_goal\"])" in body    # re-latched goal view


def test_every_mode_switch_resets_the_driving_gru_and_only_it():
    body = _trace_body()
    i = body.index("hops.take_reset()")
    j = body.index("hidden = agent.reset_hidden(1)", i)
    assert body[i:j].count("h_exp") == 0            # the explore GRU is carried on
    assert "h_exp = hidden" in body                 # in explore mode they are one pass


def test_the_sample_records_the_mode_bit_the_network_was_fed():
    body = _trace_body()
    i = body.index("step_mode = 0 if hops is None else int(hops.mode_bit)")
    j = body.index("obs, r, term, trunc, info = env.step(a_env)")
    assert i < j                                     # captured BEFORE the step
    assert 'rec["mode"].append(step_mode)' in body
    assert 'rec["entropy_explore"].append(st_exp["entropy"])' in body


def test_a_segment_end_closes_an_open_hop_with_the_segments_reason():
    body = _trace_body()
    assert 'hops.segment_end(t=t, reason=end or "cap")' in body
    assert '**({"hops": hops.summary()} if hops is not None else {})' in body
