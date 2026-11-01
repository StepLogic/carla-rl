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
    j = src.index("return run_eval(env, agent, pxr, server, OUT)")
    assert i < j


def test_the_trace_uses_the_same_action_mapping_and_fault_rule_as_eval():
    body = _trainer_src().split("def run_trace(")[1].split("\ndef run_eval(")[0]
    assert "to_env(" in body and "_fault(info)" in body
    assert "act_with_stats(" in body and "noise_scale=1.0" in body


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
