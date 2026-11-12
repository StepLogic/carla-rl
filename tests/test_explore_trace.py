"""Exploration-trace helpers (user 2026-11-01: "first drive policy in
exploration mode and store trace"), the pure parts of run_trace.

COVERAGE says when a trace has seen enough of a town to build a map from: the
fraction of the town's drivable waypoints within `radius` metres of a kept
sample. It is measured on a grid so it stays cheap as the trace grows.

STUCK DETECTION backs the continuation rule (the user's rule from the v13/v43
traces: restart at the nearest drivable waypoint to where the last episode
stopped). Sampled actions usually get past a hard corner on a later try, but a
policy that fails at the same spot every time would otherwise trace one corner
forever, so after `limit` restarts within `radius` of each other the restart
jumps further along the road.
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from vae_nav.explore_trace import CoverageGrid, StuckDetector  # noqa: E402


def test_coverage_starts_at_zero_and_counts_waypoints_near_samples():
    wps = np.array([[0.0, 0.0], [10.0, 0.0], [100.0, 0.0]])
    cov = CoverageGrid(wps, radius=5.0)
    assert cov.fraction() == 0.0
    cov.add((1.0, 0.0))
    assert cov.fraction() == pytest.approx(1 / 3)
    cov.add((10.0, 2.0))
    assert cov.fraction() == pytest.approx(2 / 3)


def test_coverage_respects_the_radius():
    cov = CoverageGrid(np.array([[0.0, 0.0]]), radius=5.0)
    cov.add((6.0, 0.0))
    assert cov.fraction() == 0.0
    cov.add((4.0, 0.0))
    assert cov.fraction() == 1.0


def test_coverage_is_idempotent():
    cov = CoverageGrid(np.array([[0.0, 0.0], [50.0, 0.0]]), radius=5.0)
    for _ in range(10):
        cov.add((0.0, 0.0))
    assert cov.fraction() == pytest.approx(0.5)


def test_coverage_handles_negative_coordinates():
    cov = CoverageGrid(np.array([[-100.0, -40.0]]), radius=5.0)
    cov.add((-98.0, -41.0))
    assert cov.fraction() == 1.0


def test_stuck_after_repeated_restarts_in_one_place():
    st = StuckDetector(radius=5.0, limit=3)
    assert not st.restart((0.0, 0.0))
    assert not st.restart((1.0, 0.0))
    assert st.restart((2.0, 1.0))           # third restart within 5 m


def test_a_restart_elsewhere_clears_the_count():
    st = StuckDetector(radius=5.0, limit=3)
    st.restart((0.0, 0.0))
    st.restart((1.0, 0.0))
    assert not st.restart((50.0, 0.0))      # moved on
    assert not st.restart((51.0, 0.0))


def test_after_firing_it_starts_over():
    st = StuckDetector(radius=5.0, limit=2)
    st.restart((0.0, 0.0))
    assert st.restart((0.0, 0.0))
    assert not st.restart((0.0, 0.0))       # counting again from this jump


# ---- wiring in train_carla_sac (needs CARLA to run; checked here by source) --

def _trainer_src():
    return open(os.path.join(ROOT, "train_carla_sac.py")).read()


def test_trace_mode_dispatches_before_eval():
    src = _trainer_src()
    i = src.index("if EVAL_TRACE:\n            return run_trace(")
    # The learner's eval comes after the trace dispatch. EVAL_POLICY=vint (a ViNT
    # checkpoint) returns earlier, so it must refuse every trace mode before its eval.
    j = src.rindex("return run_eval(env, agent, pxr, server, OUT)")
    assert i < j
    vint = src.split('if EVAL_POLICY == "vint":')[1].split("return run_eval(")[0]
    assert "EVAL_TRACE or EVAL_SGM_TRACE or EXPERT_TRACE" in vint and "raise ValueError" in vint


def test_the_trace_uses_the_same_action_mapping_and_fault_rule_as_eval():
    body = _trainer_src().split("def run_trace(")[1].split("\ndef run_eval(")[0]
    assert "to_env(" in body and "_fault(info)" in body
    assert "act_with_stats(" in body and "noise_scale=max(TRACE_NOISE_SCALE, 0.0)" in body


def test_continuation_and_resume_are_in_place():
    body = _trainer_src().split("def run_trace(")[1].split("\ndef run_eval(")[0]
    assert "set_start_transform" in body and "StuckDetector" in body
    assert 'os.replace(tmp, idx_path)' in body          # atomic index write
    assert 'index["segments"][-1]["stop_xy"]' in body    # resumes where it stopped


def test_unwrapping_reaches_the_innermost_env():
    sys.path.insert(0, ROOT)
    import importlib
    tcs = importlib.import_module("train_carla_sac")

    class Core:
        pass

    class W:
        def __init__(self, inner):
            self.env = inner

    core = Core()
    assert tcs._carla_env(W(W(W(core)))) is core


# ---- deterministic traces (user 2026-11-01: "redo and switch to deterministic
# actions") -------------------------------------------------------------------
# With training noise the Town02 trace faulted every ~11 m (148 segments in
# 2 km, 86 crossings / 49 off-road / 13 collisions, 7.5 km/h). The noise was
# there so a retried corner would not repeat; the stuck detector now handles
# that directly (and never fired in 148 segments). A deterministic car that
# never faults could instead circle one block, so a COVERAGE STALL restarts
# at the nearest waypoint not yet covered.

def test_nearest_uncovered_skips_covered_waypoints():
    wps = np.array([[0.0, 0.0], [10.0, 0.0], [30.0, 0.0]])
    cov = CoverageGrid(wps, radius=5.0)
    cov.add((0.0, 0.0))
    assert tuple(cov.nearest_uncovered((0.0, 0.0))) == (10.0, 0.0)
    cov.add((10.0, 0.0))
    assert tuple(cov.nearest_uncovered((0.0, 0.0))) == (30.0, 0.0)


def test_nearest_uncovered_is_none_when_all_covered():
    cov = CoverageGrid(np.array([[0.0, 0.0]]), radius=5.0)
    cov.add((0.0, 0.0))
    assert cov.nearest_uncovered((0.0, 0.0)) is None


def test_the_trace_reads_its_noise_scale_and_stall_settings():
    src = _trainer_src()
    assert 'TRACE_NOISE_SCALE = env_float("TRACE_NOISE_SCALE", 1.0)' in src
    body = src.split("def run_trace(")[1].split("\ndef run_eval(")[0]
    assert "deterministic=TRACE_NOISE_SCALE <= 0" in body
    assert "nearest_uncovered(" in body and "TRACE_STALL_M" in body
    assert "limit=TRACE_STUCK_LIMIT" in body


# ---- crossings do not end a trace segment (user 2026-11-01: "yes") ----------
# Lane crossings ended 102 of the deterministic Town02 trace's 134 segments,
# only because the trace inherited the eval's fault rule. Nothing is scored in
# a trace, so with TRACE_END_ON_CROSSING=0 a crossing is counted and the car
# drives on; off-road and collisions still end the segment. The environment
# must also stop terminating on crossings, which slurm/trace_nocross.env does.

def test_the_trace_can_drive_through_crossings():
    src = _trainer_src()
    assert 'TRACE_END_ON_CROSSING = os.environ.get("TRACE_END_ON_CROSSING", "1") == "1"' in src
    body = src.split("def run_trace(")[1].split("\ndef run_eval(")[0]
    assert 'if on_line and not TRACE_END_ON_CROSSING:' in body
    assert '"crossings": crossing.events' in body  # still counted per segment


def test_the_nocross_overlay_keeps_the_policy_inputs():
    """Only termination changes: the base overlay (the checkpoint's own) is
    sourced first, and nothing that shapes an observation is overridden."""
    ov = open(os.path.join(ROOT, "slurm", "trace_nocross.env")).read()
    assert 'slurm/v30_overlay.env' in ov
    assert "EXPLORE_LANE_MODE=penalise" in ov
    for shape_key in ("MONO_HEAD", "ENV_IMAGE_SIZE", "INCREMENTAL_ACTIONS", "MAX_STEER"):
        assert f"export {shape_key}" not in ov


# ---- crossing EVENTS, not steps (2026-11-01) ---------------------------------
# The first counter added one per step on a line, and the lane sensor keeps
# firing while the wheels are over a marking: Town02's segments came out at 4
# and 8 "crossings" (23 and 11 of 69 segments), about four steps per line, so
# 5.5 per 100 m overstated the rate ~4x.

def test_one_crossing_held_for_several_steps_counts_once():
    from vae_nav.explore_trace import CrossingCounter
    c = CrossingCounter()
    for on in (False, True, True, True, True, False, False):
        c.update(on)
    assert c.events == 1 and c.steps == 4


def test_separate_crossings_count_separately():
    from vae_nav.explore_trace import CrossingCounter
    c = CrossingCounter()
    for on in (True, True, False, True, False, False, True):
        c.update(on)
    assert c.events == 3 and c.steps == 4


def test_the_trace_uses_the_event_counter():
    body = _trainer_src().split("def run_trace(")[1].split("\ndef run_eval(")[0]
    assert "CrossingCounter()" in body
    assert '"crossings": crossing.events, "line_steps": crossing.steps' in body


def test_the_on_line_flag_is_computed_before_a_sample_records_it():
    """Regression: the flag was recorded in the keep-sample block BEFORE the
    step's fault check ran, so the first step of every segment raised
    UnboundLocalError (it killed the step_800000 exploration test)."""
    body = _trainer_src().split("def run_trace(")[1].split("\ndef run_eval(")[0]
    assert body.index('on_line = fault == "crossing"') < body.index('rec["on_line"].append(on_line)')
    assert body.index("crossing.update(on_line)") < body.index('rec["on_line"].append(on_line)')
