"""The image-goal arm's coherence gate: refuse configurations that cannot learn."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))


def _set(monkeypatch, **kw):
    monkeypatch.delenv("IMAGE_GOAL_TASK", raising=False)
    monkeypatch.delenv("GATED", raising=False)
    monkeypatch.delenv("OBS_WITH_GOAL", raising=False)
    monkeypatch.delenv("OBS_WITH_COLLISION", raising=False)
    # the 2-mode pairing's flags must not leak in from the ambient env
    monkeypatch.delenv("OBS_WITH_STATE", raising=False)
    monkeypatch.delenv("OBS_WITH_MODE", raising=False)
    monkeypatch.delenv("VC1_READOUT", raising=False)
    monkeypatch.delenv("GATED_ONE_CRITIC", raising=False)
    monkeypatch.delenv("EPISODE_HER", raising=False)
    monkeypatch.delenv("TQC_TARGET_MIN", raising=False)
    monkeypatch.delenv("TQC_TARGET_MAX", raising=False)
    monkeypatch.delenv("NO_TIMELIMIT", raising=False)
    # ambient vars must not steer which coherence check a test passes through
    monkeypatch.delenv("PERCEPTION", raising=False)
    monkeypatch.delenv("LANE_FOLLOW", raising=False)
    monkeypatch.delenv("LANE_EXPLORE", raising=False)
    monkeypatch.delenv("TIME_TO_GOAL", raising=False)
    for k, v in kw.items():
        monkeypatch.setenv(k, v)
    import vae_nav.image_goal_reward as m
    import importlib
    importlib.reload(m)
    return m


def test_default_configuration_is_accepted(monkeypatch):
    m = _set(monkeypatch, IMAGE_GOAL_TASK="0")
    m.image_goal_coherence()          # no raise


def test_a_coherent_image_goal_run_is_accepted(monkeypatch):
    m = _set(monkeypatch, IMAGE_GOAL_TASK="1", GATED="0",
             OBS_WITH_COLLISION="1", PERCEPTION="vc1")
    m.image_goal_coherence()


def test_gated_pairing_requires_the_two_mode_flags(monkeypatch):
    # Since 2026-09-20 the GATED=1 pairing is REQUIRED, not refused -- it is
    # the 2-mode arm (TwoModeImageGoalReward). A bare GATED=1 with none of
    # the pairing's flags set is still refused, now because the 2-mode
    # requirements (OBS_WITH_STATE, OBS_WITH_MODE, one critic, HER, the
    # value bounds, ...) are unmet; the complete pairing is pinned in
    # tests/test_two_mode_image_goal.py.
    m = _set(monkeypatch, IMAGE_GOAL_TASK="1", GATED="1")
    with pytest.raises(SystemExit):
        m.image_goal_coherence()


def test_the_arm_requires_the_coll_key(monkeypatch):
    m = _set(monkeypatch, IMAGE_GOAL_TASK="1", OBS_WITH_COLLISION="0")
    with pytest.raises(SystemExit):
        m.image_goal_coherence()


def test_the_arm_requires_the_vc1_backend(monkeypatch):
    m = _set(monkeypatch, IMAGE_GOAL_TASK="1", PERCEPTION="mobilenet")
    with pytest.raises(SystemExit):
        m.image_goal_coherence()


def test_a_shaped_or_sparse_goal_reward_is_refused(monkeypatch):
    # IMAGE_GOAL_TASK's reward is ImageGoalReward; a leftover LANE_FOLLOW=1
    # would stack a second reward below it whose output is discarded --
    # exactly the LaneFollowReward-below-GatedReward pattern, which exists
    # there because GatedReward reads two of its info keys. Nothing here
    # reads them, so refuse the dead wrapper.
    m = _set(monkeypatch, IMAGE_GOAL_TASK="1", LANE_FOLLOW="1")
    with pytest.raises(SystemExit):
        m.image_goal_coherence()
