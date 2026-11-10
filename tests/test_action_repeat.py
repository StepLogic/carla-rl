"""ACTION_REPEAT=k (plan 2026-11-09, C1): DrQ's action repeat for the RL arm.

Why: every SAC critic on c0's features ranked steering as if it barely mattered
(Q ~ V(s) + g(a), best steering at full lock). One hypothesis is the control step:
at 10 Hz one action barely changes the next frame, so Q(s, a) - V(s) shrinks with
the time step (Tallec et al., ICML 2019). Action repeat (standard in DrQ / DrQ-v2)
holds each action for k simulator ticks.

The wrapper sits right after GatedReward, so the DEBiT goal features are computed
once per decision and every inner rule (goal clock, explore cap, faults) still
counts ticks. Rewards are summed over the ticks. A tick that ends the episode, or
changes the goal (a chained goal's soft reset, an arrival, a declared stop), ends
the repeat early, so the goal wrapper above sees that tick's info. Fault and
crossing flags from earlier ticks are carried into the returned info.
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))
gym = pytest.importorskip("gymnasium")

from torch_rl.carla_env_build import ActionRepeat, action_repeat  # noqa: E402


class _Script(gym.Env):
    """Returns the scripted (reward, terminated, info) per tick; counts ticks."""
    observation_space = gym.spaces.Box(-1e9, 1e9, (1,), np.float32)
    action_space = gym.spaces.Box(-1.0, 1.0, (2,), np.float32)

    def __init__(self, script):
        self.script, self.ticks, self.actions = list(script), 0, []

    def reset(self, *, seed=None, options=None):
        return np.zeros(1, np.float32), {}

    def step(self, action):
        r, term, info = self.script[self.ticks]
        self.ticks += 1
        self.actions.append(np.asarray(action).copy())
        return np.full(1, self.ticks, np.float32), r, term, False, dict(info)


def test_each_action_is_held_for_k_ticks_and_rewards_are_summed():
    env = ActionRepeat(_Script([(1.0, False, {})] * 8), 4)
    obs, r, term, trunc, info = env.step(np.array([0.3, 0.5], np.float32))
    assert env.env.ticks == 4 and r == pytest.approx(4.0)
    assert obs[0] == 4                                # the LAST tick's observation
    assert all(np.allclose(a, [0.3, 0.5]) for a in env.env.actions)
    assert info["repeat_ticks"] == 4


def test_a_terminal_tick_ends_the_repeat():
    env = ActionRepeat(_Script([(1.0, False, {}), (-10.0, True, {"collision": True}),
                                (5.0, False, {})]), 4)
    _, r, term, _, info = env.step(np.zeros(2, np.float32))
    assert term and env.env.ticks == 2 and r == pytest.approx(-9.0)
    assert info["collision"] and info["repeat_ticks"] == 2


def test_a_goal_change_ends_the_repeat_so_the_goal_wrapper_sees_it():
    for key in ("chain_soft_reset", "is_success", "declared"):
        env = ActionRepeat(_Script([(0.0, False, {}), (50.0, False, {key: True}),
                                    (0.0, False, {})]), 4)
        _, r, term, _, info = env.step(np.zeros(2, np.float32))
        assert env.env.ticks == 2 and info[key], key


def test_a_crossing_on_an_earlier_tick_survives_into_the_returned_info():
    env = ActionRepeat(_Script([(0.0, False, {"lane_fined": True, "lane_invasion_illegal": True}),
                                (0.0, False, {"off_road": True}), (0.0, False, {}),
                                (0.0, False, {"lane_fined": False})]), 4)
    _, _, _, _, info = env.step(np.zeros(2, np.float32))
    assert info["lane_fined"] and info["lane_invasion_illegal"] and info["off_road"]


def test_the_setting(monkeypatch):
    monkeypatch.delenv("ACTION_REPEAT", raising=False)
    assert action_repeat() == 1
    monkeypatch.setenv("ACTION_REPEAT", "4")
    assert action_repeat() == 4
    for bad in ("0", "-2", "two"):
        monkeypatch.setenv("ACTION_REPEAT", bad)
        with pytest.raises(ValueError):
            action_repeat()


def test_recording_demonstrations_refuses_action_repeat(monkeypatch):
    """The recorder labels every tick with the expert's action; held actions would
    write demonstrations the expert never gave."""
    monkeypatch.setenv("ACTION_REPEAT", "2")
    monkeypatch.setenv("EXPERT_RECORD", "1")
    with pytest.raises(ValueError):
        action_repeat()


# ----------------------------------------------------- placement in the stack --
from tests.test_carla_env_build import BASE, _FakeCarlaEnv, _gated  # noqa: E402,F401


def _chain(env):
    out, node = [], env
    while node is not None:
        out.append(type(node).__name__)
        node = getattr(node, "env", None)
    return out


def test_it_sits_right_after_gated_reward_and_only_when_asked(monkeypatch):
    from torch_rl.carla_env_build import build_env
    monkeypatch.delenv("EXPERT_RECORD", raising=False)
    monkeypatch.delenv("ACTION_REPEAT", raising=False)
    try:
        plain = _chain(build_env("Town05", seed=0, port=2000, env_factory=_FakeCarlaEnv, base=BASE))
    except ImportError as e:
        pytest.skip(f"a wrapper dependency is missing here: {e}")
    assert "ActionRepeat" not in plain                  # every earlier run, unchanged
    monkeypatch.setenv("ACTION_REPEAT", "2")
    chain = _chain(build_env("Town05", seed=0, port=2000, env_factory=_FakeCarlaEnv, base=BASE))
    i = chain.index("ActionRepeat")
    assert chain[i - 1] == "DebitGoalWrapper" and chain[i + 1] == "GatedReward"
