"""Lane-centre potential shaping is REMOVED from the learner-C arm (user 2026-11-14: "Ok remove
lanshaping then").

Why it cannot help this arm: the shaping is potential-based, F = c(s) - gamma c(s'), so
Q_shaped(s, a) = Q(s, a) - Phi(s) -- a function of the state alone -- and dQ/da is unchanged
exactly. SAC's actor follows dQ/da, so the term cannot move the steering gradient however long it
runs. Measured as well: job 466411 ran it for 65k steps at v52's 2.0 / 1.0 and metres per fault
stayed at 56 / 55 (66 / 45 before) while strict success fell from 49/80 to 43/80.

Every learner-C overlay already inherits LANE_CENTRE_K=0 LANE_HEADING_K=0 from
slurm/v26_overlay.env (via il_expert.env -> v30 -> ... -> v26); the one overlay that turned it back
on is deleted. This refusal keeps it from coming back by accident.
"""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "src")]
os.environ.setdefault("MONO_CHW", "1")
os.environ.setdefault("SPEED_OBS", "1")
pytest.importorskip("torch")

from torch_rl.frl import carla as fc  # noqa: E402

BASE = {"MONO_CHW": "1", "SPEED_OBS": "1"}


def test_a_run_without_lane_shaping_is_not_refused():
    for off in ({}, {"LANE_CENTRE_K": "0", "LANE_HEADING_K": "0"},
                {"LANE_CENTRE_K": "0.0", "LANE_HEADING_K": ""}):
        assert fc.frl_refusals(["Town01"], False, {**BASE, **off}) == []


@pytest.mark.parametrize("knob", ["LANE_CENTRE_K", "LANE_HEADING_K"])
def test_a_run_with_lane_shaping_is_refused(knob):
    out = fc.frl_refusals(["Town01"], False, {**BASE, knob: "2.0"})
    assert len(out) == 1 and knob in out[0]


def test_the_overlay_that_turned_it_on_is_gone():
    assert not os.path.exists(os.path.join(ROOT, "slurm",
                                           "cluster_frl_c0emb_4towns_lane_resume.env"))
