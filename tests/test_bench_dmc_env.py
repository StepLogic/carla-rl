"""scripts/bench/dmc_env.py (plan 2026-11-10, Task 2): a dm_control suite task as a gymnasium env,
in `states` or `pixels` mode, with action repeat and frame stacking (the PlaNet/DrQ protocol for
pixels: 84 px, 3 frames, action repeat 8). DMC episodes end on a time limit, so the end is a
TRUNCATION (the value keeps bootstrapping), never a termination. Tested against a stub dm_env."""
import collections
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts", "bench"))

gym = pytest.importorskip("gymnasium")
from dmc_env import DMCEnv  # noqa: E402

class Step(collections.namedtuple("Step", "step_type reward discount observation")):
    """Mirrors dm_env.TimeStep: step_type 0 FIRST, 1 MID, 2 LAST, with last()."""

    def last(self):
        return self.step_type == 2


class _Spec:
    def __init__(self, shape, lo=None, hi=None):
        self.shape, self.minimum, self.maximum = shape, lo, hi


class _StubPhysics:
    def __init__(self, env):
        self.env = env

    def render(self, height, width, camera_id=0):
        return np.full((height, width, 3), self.env.t % 256, dtype=np.uint8)


class _StubDMC:
    """Episode of `length` control steps; reward 1 per step; observation counts steps."""

    def __init__(self, length=5):
        self.length, self.t = length, 0
        self.physics = _StubPhysics(self)

    def action_spec(self):
        return _Spec((1,), np.array([-1.0]), np.array([1.0]))

    def observation_spec(self):
        return collections.OrderedDict(position=_Spec((3,)), velocity=_Spec((2,)))

    def _obs(self):
        return collections.OrderedDict(position=np.full(3, self.t, np.float64),
                                       velocity=np.full(2, -self.t, np.float64))

    def reset(self):
        self.t = 0
        return Step(0, None, None, self._obs())

    def step(self, action):
        self.t += 1
        last = self.t >= self.length
        return Step(2 if last else 1, 1.0, 1.0, self._obs())


def test_states_flatten_the_observation_dict_in_key_order():
    env = DMCEnv(env_factory=lambda seed: _StubDMC(), obs_type="states")
    obs, _ = env.reset(seed=0)
    assert obs.dtype == np.float32 and obs.shape == (5,) == env.observation_space.shape
    obs, *_ = env.step(np.zeros(1, np.float32))
    assert obs.tolist() == [1, 1, 1, -1, -1]
    assert env.action_space.shape == (1,) and env.action_space.low[0] == -1.0


def test_action_repeat_sums_rewards_and_stops_at_the_episode_end():
    env = DMCEnv(env_factory=lambda seed: _StubDMC(length=5), obs_type="states", action_repeat=2)
    env.reset(seed=0)
    _, r, term, trunc, _ = env.step(np.zeros(1, np.float32))
    assert r == 2.0 and not term and not trunc
    env.step(np.zeros(1, np.float32))
    _, r, term, trunc, _ = env.step(np.zeros(1, np.float32))   # only 1 step left of 5
    assert r == 1.0 and not term and trunc                     # a time limit: truncated only


def test_pixels_stack_frames_channel_first_oldest_first():
    env = DMCEnv(env_factory=lambda seed: _StubDMC(length=9), obs_type="pixels",
                 frame_stack=3, image_size=8)
    obs, _ = env.reset(seed=0)
    assert obs.dtype == np.uint8 and obs.shape == (9, 8, 8) == env.observation_space.shape
    assert np.all(obs == 0)                                    # reset fills the stack with frame 0
    obs, *_ = env.step(np.zeros(1, np.float32))
    assert np.all(obs[:6] == 0) and np.all(obs[6:] == 1)       # the newest frame is last
    obs, *_ = env.step(np.zeros(1, np.float32))
    assert np.all(obs[:3] == 0) and np.all(obs[3:6] == 1) and np.all(obs[6:] == 2)


def test_a_new_seed_rebuilds_the_task():
    seen = []
    env = DMCEnv(env_factory=lambda seed: (seen.append(seed), _StubDMC())[1], obs_type="states")
    env.reset(seed=3)
    env.reset()
    env.reset(seed=4)
    # built once to read the specs, rebuilt for each seed, kept on a seedless reset
    assert seen == [None, 3, 4]
