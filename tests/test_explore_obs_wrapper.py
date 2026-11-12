"""Explore observation: two memories, selectable pose, no goal."""
import gymnasium as gym
import numpy as np
import pytest

from vae_nav.explore_obs_wrapper import (EXPLORE_DIM, POS_SCALE_M,
                                         ExploreObsWrapper, explore_block)
from vae_nav.visit_ring import N_SLOTS


class StubSource:
    latent_dim = 8

    def latent(self, rgb):
        return np.full(8, float(np.asarray(rgb).mean()), np.float32)


class FakeEnv(gym.Env):
    """Drives a scripted (speed, steer) profile and reports true pose."""

    def __init__(self, profile, true_path=None):
        self.profile = profile
        self.true_path = true_path
        self.t = -1
        box = gym.spaces.Box(-1.0, 1.0, (2,), np.float32)
        self.observation_space = box
        self.action_space = box

    def _info(self):
        i = min(max(self.t, 0), len(self.profile) - 1)
        speed, steer = self.profile[i]
        pos = (self.true_path[i] if self.true_path is not None
               else np.zeros(2))
        return {"speed_kmh": speed, "applied_steer": steer,
                "hero_pos": np.asarray(pos, np.float64), "hero_yaw": 0.0,
                "rgb": np.full((4, 4, 3), 10 + i, np.uint8),
                "novelty_raw": 0.25}

    def reset(self, **kw):
        self.t = -1
        return np.zeros(2, np.float32), self._info()

    def step(self, a):
        self.t += 1
        return np.zeros(2, np.float32), 0.0, False, False, self._info()


def make(profile, pose="dead_reckoned", **kw):
    """Default to dead_reckoned here: most of these tests drive the estimator
    from a scripted (speed, steer) profile, and true pose would ignore it."""
    return ExploreObsWrapper(FakeEnv(profile, **kw), StubSource(), pose=pose)


# -- layout ---------------------------------------------------------------

def test_block_is_21_dimensional():
    assert EXPLORE_DIM == 2 + 2 + N_SLOTS + 1 == 21


def test_observation_has_both_keys_with_right_shapes():
    env = make([(10.0, 0.0)] * 3)
    obs, _i = env.reset()
    assert set(obs) == {"explore", "latent"}
    assert obs["explore"].shape == (EXPLORE_DIM,)
    assert obs["latent"].shape == (8,)
    assert obs["explore"].dtype == np.float32


def test_block_layout_is_dxy_heading_ring_novelty():
    ring = np.linspace(0, 1, N_SLOTS)
    b = explore_block({"displacement": np.array([POS_SCALE_M, 0.0]),
                       "yaw": 0.0, "visit_ring": ring, "novelty": 0.7})
    assert b[0] == pytest.approx(1.0), "dx scaled by POS_SCALE_M"
    assert b[1] == pytest.approx(0.0)
    assert b[2] == pytest.approx(0.0) and b[3] == pytest.approx(1.0), "sin,cos"
    np.testing.assert_allclose(b[4:4 + N_SLOTS], ring, atol=1e-6)
    assert b[-1] == pytest.approx(0.7)


def test_no_goal_and_no_mode_bit():
    """A permanently-zero goal pair and a constant mode bit would be three dead
    inputs the encoder must learn to ignore."""
    assert EXPLORE_DIM == 21, "23 would mean gx/gz or a mode bit crept back in"


def test_displacement_is_relative_not_absolute():
    """Absolute town coordinates are a memorisation crutch."""
    far = np.array([500.0, -300.0])
    env = make([(0.0, 0.0)] * 3, true_path=[far] * 3)
    obs, _i = env.reset()
    assert abs(obs["explore"][0]) < 1e-6 and abs(obs["explore"][1]) < 1e-6


# -- dead reckoning -------------------------------------------------------

def test_pose_advances_from_proprioception():
    env = make([(36.0, 0.0)] * 11, pose="dead_reckoned")   # 10 m/s
    env.reset()
    for _ in range(10):
        obs, *_ = env.step(np.zeros(2, np.float32))
    assert obs["explore"][0] * POS_SCALE_M == pytest.approx(10.0, rel=0.02)


def test_a_stationary_vehicle_does_not_accumulate_displacement():
    env = make([(0.0, 0.5)] * 20)
    env.reset()
    for _ in range(20):
        obs, *_ = env.step(np.zeros(2, np.float32))
    assert np.allclose(obs["explore"][:2], 0.0, atol=1e-9)


def test_drift_is_reported_every_step():
    """Drift is what dead reckoning costs; it must be measurable, not assumed."""
    true = [np.array([float(i), 0.0]) for i in range(12)]   # 1 m/step of truth
    env = make([(36.0, 0.0)] * 12, true_path=true)          # 1 m/step estimated
    env.reset()
    _o, _r, _t, _tr, info = env.step(np.zeros(2, np.float32))
    assert "dr_drift" in info
    assert set(info["dr_drift"]) == {"position_m", "heading_rad", "steps"}


def test_drift_grows_when_the_estimate_and_truth_disagree():
    true = [np.zeros(2) for _ in range(12)]        # truth: never moves
    env = make([(36.0, 0.0)] * 12, true_path=true)  # estimate: 1 m per step
    env.reset()
    drifts = []
    for _ in range(10):
        _o, _r, _t, _tr, info = env.step(np.zeros(2, np.float32))
        drifts.append(info["dr_drift"]["position_m"])
    assert drifts[-1] > drifts[0]
    assert drifts[-1] == pytest.approx(10.0, rel=0.05)


# -- the two memories -----------------------------------------------------

def test_visit_ring_fills_in_as_the_vehicle_moves():
    env = make([(36.0, 0.0)] * 40)
    env.reset()
    for _ in range(30):
        obs, *_ = env.step(np.zeros(2, np.float32))
    ring = obs["explore"][4:4 + N_SLOTS]
    assert ring.max() > 0.0, "cells behind the vehicle should be marked visited"


def test_ring_is_queried_at_the_estimated_pose_not_the_true_one():
    """Mixing estimated queries with true-pose cells would corrupt the ring as
    drift grows."""
    true = [np.array([0.0, 500.0]) for _ in range(40)]   # truth far away
    env = make([(36.0, 0.0)] * 40, true_path=true)
    env.reset()
    for _ in range(30):
        obs, *_ = env.step(np.zeros(2, np.float32))
    assert obs["explore"][4:4 + N_SLOTS].max() > 0.0, (
        "ring went blank, so it is being queried in a frame the cells are not in")


def test_novelty_is_read_from_info_not_recomputed():
    """The reward wrapper already admitted this frame; a second query would
    compare it against itself and read zero."""
    env = make([(10.0, 0.0)] * 3)
    obs, _i = env.reset()
    assert obs["explore"][-1] == pytest.approx(0.25), "the value info published"


def test_visited_set_resets_between_episodes():
    env = make([(36.0, 0.0)] * 40, pose="dead_reckoned")
    env.reset()
    for _ in range(20):
        env.step(np.zeros(2, np.float32))
    assert len(env.visited) > 1
    env.reset()
    assert len(env.visited) == 1


def test_missing_rgb_gives_a_zero_latent_rather_than_crashing():
    class NoRGB(FakeEnv):
        def _info(self):
            d = super()._info()
            d.pop("rgb")
            return d
    env = ExploreObsWrapper(NoRGB([(10.0, 0.0)] * 3), StubSource())
    obs, _i = env.reset()
    assert np.allclose(obs["latent"], 0.0)


# -- privileged (true) pose, the training default -------------------------

def test_true_pose_is_the_default():
    """Privileged during training is a deliberate choice, not an oversight."""
    from vae_nav.explore_obs_wrapper import EXPLORE_POSE
    assert EXPLORE_POSE == "true"


def test_true_pose_follows_the_simulator_not_the_estimator():
    """The estimator says 1 m/step; truth says the car never moves."""
    true = [np.zeros(2) for _ in range(20)]
    env = make([(36.0, 0.0)] * 20, pose="true", true_path=true)
    env.reset()
    for _ in range(10):
        obs, *_ = env.step(np.zeros(2, np.float32))
    assert np.allclose(obs["explore"][:2], 0.0, atol=1e-9), (
        "displacement tracked the estimator, not the simulator")


def test_true_pose_ring_is_immune_to_estimator_drift():
    """THE REASON FOR THE SWITCH.

    On the dead-reckoned run the estimate reached 36 m of error over 312 steps
    once the policy drove far enough to turn -- 72 cells at the 0.5 m grid,
    against a ring reaching 3 m. Under true pose that failure cannot occur.
    """
    true = [np.array([i * 1.0, 0.0]) for i in range(40)]
    env = make([(36.0, 0.6)] * 40, pose="true", true_path=true)  # estimator turns hard
    env.reset()
    for _ in range(30):
        obs, *_ = env.step(np.zeros(2, np.float32))
    assert obs["explore"][0] * POS_SCALE_M == pytest.approx(29.0, abs=1.5)
    assert obs["explore"][4:4 + N_SLOTS].max() > 0.0


def test_drift_is_still_reported_under_true_pose():
    """The non-privileged arm stays measurable and available."""
    true = [np.zeros(2) for _ in range(12)]
    env = make([(36.0, 0.0)] * 12, pose="true", true_path=true)
    env.reset()
    _o, _r, _t, _tr, info = env.step(np.zeros(2, np.float32))
    assert info["dr_drift"]["position_m"] > 0.0
