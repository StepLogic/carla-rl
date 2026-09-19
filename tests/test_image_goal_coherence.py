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


def test_gating_and_image_goal_are_refused(monkeypatch):
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