"""THE CAR DECLARES ARRIVAL (user 2026-10-28, "Yes, add a stop action").

The wrong-goal eval of v18@300k scored 0.600 against 0.725 with the true goal
image: driving down the lane and halting collected most successes without
reading the goal at all, because arrival fired automatically inside a 2 m
circle. DEBiT's task (Habitat imagenav) has a STOP action and scores success
only when `is_stop_called` AND the agent is within 1 m; wandering past the goal
earns nothing.

STOP_ACTION=1 adds a third action dimension:
  declared = a[2] > 0 AND speed <= the arrival speed (the car must halt to
             claim arrival, so a random early a[2] does not end episodes)
  declared inside  the radius -> "reached", the arrival bonus as before
  declared outside the radius -> the episode ENDS, no bonus ("false_stop")
  never declared                -> no arrival, whatever the distance
Off (default 0) every earlier run is unchanged: arrival stays automatic.
"""
import importlib
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))
pytest.importorskip("gymnasium")
import gymnasium as gym                       # noqa: E402
from gymnasium import spaces                  # noqa: E402


class StopEnv(gym.Env):
    """Scripted distance/speed; records the action it was given."""

    def __init__(self, dist=0.5, speed=0.0, stop_dim=True):
        self.observation_space = spaces.Dict({})
        n = 3 if stop_dim else 2
        self.action_space = spaces.Box(np.array([-1.0, 0.0] + [-1.0] * (n - 2), np.float32),
                                       np.array([1.0, 1.0] + [1.0] * (n - 2), np.float32))
        self.dist, self.speed = dist, speed
        self.last = None

    def _info(self):
        return {"arrived": self.dist <= 2.0,          # the env's own radius test
                "hero_pos": (0.0, 0.0), "goal_pos": (self.dist, 0.0),
                "hero_yaw": 0.0, "wp_forward": np.array([1.0, 0.0]),
                "speed_kmh": self.speed, "speed_factor": 0.3,
                "action_smoothness_penalty": 0.0,
                "goal_distance_m": self.dist, "route_remaining_m": self.dist}

    def reset(self, **kw):
        return np.zeros(4, np.float32), self._info()

    def step(self, action):
        self.last = np.asarray(action, np.float32)
        inf = self._info()
        inf["stop_action"] = float(self.last[2]) if self.last.size > 2 else 0.0
        return np.zeros(4, np.float32), 0.0, False, False, inf


def _gr(monkeypatch, stop=1, radius=2.0, max_speed=10.0):
    for k, v in dict(STOP_ACTION=stop, GOAL_ARRIVAL_RADIUS=radius,
                     GOAL_ARRIVAL_MAX_SPEED=max_speed, GATED_GOAL_REWARD="explore",
                     LANE_ANY_FATAL=0, LANE_INVASION_NOOP=0).items():
        monkeypatch.setenv(k, str(v))
    import vae_nav.lane_explore_reward as ler
    importlib.reload(ler)
    import vae_nav.gated_reward as gr
    return importlib.reload(gr)


def _run(gr, env, stop_val, steps=1):
    w = gr.GatedReward(env, force_mode=gr.GOAL, max_steps=50, seed=0)
    w.reset()
    out = None
    for _ in range(steps):
        a = np.array([0.0, 0.5, stop_val], np.float32)[: env.action_space.shape[0]]
        out = w.step(a)
        if out[2] or out[3]:
            break
    return out


def test_declaring_at_the_goal_is_an_arrival(monkeypatch):
    gr = _gr(monkeypatch)
    _o, r, term, _tr, info = _run(gr, StopEnv(dist=0.5, speed=2.0), +1.0)
    assert term and info.get("goal_end") == "reached" and info.get("is_success")
    assert r > 40                                    # the arrival bonus was paid


def test_declaring_away_from_the_goal_ends_the_episode_unpaid(monkeypatch):
    gr = _gr(monkeypatch)
    _o, r, term, _tr, info = _run(gr, StopEnv(dist=20.0, speed=2.0), +1.0)
    assert term and info.get("goal_end") == "false_stop"
    assert not info.get("is_success") and r < 40


def test_sitting_at_the_goal_without_declaring_is_not_an_arrival(monkeypatch):
    gr = _gr(monkeypatch)
    out = _run(gr, StopEnv(dist=0.5, speed=2.0), -1.0, steps=3)
    assert not out[2]
    assert out[4].get("goal_end") != "reached" and not out[4].get("is_success")


def test_declaring_at_speed_does_nothing(monkeypatch):
    """The car must halt to claim arrival, so a random a[2] early in an episode
    cannot end it at 40 km/h."""
    gr = _gr(monkeypatch)
    out = _run(gr, StopEnv(dist=0.5, speed=40.0), +1.0, steps=3)
    assert not out[2] and out[4].get("goal_end") != "reached"


def test_off_by_default_arrival_stays_automatic(monkeypatch):
    gr = _gr(monkeypatch, stop=0)
    _o, _r, term, _tr, info = _run(gr, StopEnv(dist=0.5, speed=2.0, stop_dim=False), 0.0)
    assert term and info.get("goal_end") == "reached"


def test_the_env_action_space_grows_a_third_dimension():
    """Source-level: the env config imports carla, which is not installed on
    this machine, so the box is pinned by reading the code."""
    src = open(os.path.join(ROOT, "src", "configs",
                            "train_env_config_v4_image_goal.py")).read()
    assert 'os.environ.get("STOP_ACTION", "0") == "1"' in src
    assert "[-self.max_steer, 0.0, -1.0]" in src
    assert "[self.max_steer, self.max_throttle, 1.0]" in src
    assert 'self.info["stop_action"] = float(action[2]) if len(action) > 2 else 0.0' in src


def test_her_declares_on_its_hindsight_arrival_row():
    """A leg's last row is relabelled as an arrival, so its stored action must
    declare -- otherwise HER teaches that arriving silently pays."""
    src = open(os.path.join(ROOT, "torch_rl", "image_her.py")).read()
    assert "stop_dim" in src and "STOP_ACTION" in src


def test_the_actor_is_sized_from_the_env_action_space():
    """With STOP_ACTION the box has 3 dimensions; a 2-output actor then dies on
    'operands could not be broadcast together with shapes (2,) (3,)' -- which a
    simulator probe hit before any run was launched."""
    src = open(os.path.join(ROOT, "train_carla_sac.py")).read()
    assert "ACT_DIM = int(env.action_space.shape[0])" in src
    assert "action_dim=ACT_DIM," in src
    i_env = min(src.index("venv = ThreadedVecEnv"), src.index("env = build_env"))
    assert src.index("ACT_DIM = int(env.action_space.shape[0])") > i_env


def test_the_learner_builds_as_many_outputs_as_asked():
    from tests.test_one_critic import _cfg
    from torch_rl.gated_recurrent_sac import GatedRecurrentSAC
    agent = GatedRecurrentSAC(_cfg(one=True, action_dim=3, prev_action=False))
    assert agent.actor.action_dim == 3
    assert agent.actor.trunk[-1].out_features == 6          # mean + log_std


def test_every_action_width_comes_from_the_env():
    """The replay buffer was built with a hardcoded 2 as well, which only
    showed up when an episode was stored: 'The expanded size of the tensor (2)
    must match the existing size (3)'. Nothing in the action path may assume 2."""
    src = open(os.path.join(ROOT, "train_carla_sac.py")).read()
    i = src.index("buf = DeviceEpisodeBuffer(")
    call = src[i:i + 200]
    assert "ACT_DIM" in call, call
    assert "ACT_DIM = int(env.action_space.shape[0])" in src
