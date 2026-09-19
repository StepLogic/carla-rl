"""The coll observation key: the critic's collision flag, z = has_collided."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

pytest.importorskip("gymnasium")

import gymnasium as gym  # noqa: E402

# A stand-in frame. The wrapper refuses rgb=None outright (it cannot tell a
# stub from an env that dropped the camera), and _FakeSource.latent ignores
# the frame anyway.
_FRAME = np.zeros((4, 4, 3), np.float32)


class _StubEnv(gym.Env):
    """Real gymnasium.Env: the Wrapper base asserts on the type."""
    observation_space = {}
    action_space = None

    def reset(self, **kw):
        return np.zeros(4, np.float32), {"rgb": _FRAME, "has_collided": False}

    def step(self, a):
        return (np.zeros(4, np.float32), 0.0, False, False,
                {"rgb": _FRAME, "has_collided": True})


class _FakeSource:
    name = "stub"
    latent_dim = 4

    def latent(self, rgb):
        return np.zeros(4, np.float32)

    def identity(self):
        return {"latent_backend": "stub"}


def _make(monkeypatch, coll="1"):
    monkeypatch.setenv("OBS_WITH_COLLISION", coll)
    monkeypatch.setenv("OBS_WITH_GOAL", "0")   # keep the stub small; goal block
    # needs info fields the stub does not carry
    import vae_nav.vae_goal_wrapper as m
    import importlib
    importlib.reload(m)
    return m.VAEGoalWrapper(_StubEnv(), source=_FakeSource())


def test_coll_key_off_by_default(monkeypatch):
    w = _make(monkeypatch, coll="0")
    obs, _ = w.reset()
    assert "coll" not in obs


def test_coll_key_present_when_requested_and_carries_the_flag(monkeypatch):
    w = _make(monkeypatch, coll="1")
    obs, _ = w.reset()
    assert obs["coll"].shape == (1,) and obs["coll"].dtype == np.float32
    obs2, _r, _t, _tr, _i = w.step(np.zeros(2, np.float32))
    assert float(obs2["coll"][0]) == 1.0    # the stub steps with has_collided=True


def test_coll_defaults_to_zero_when_info_has_no_flag(monkeypatch):
    monkeypatch.setenv("OBS_WITH_COLLISION", "1")
    monkeypatch.setenv("OBS_WITH_GOAL", "0")
    import vae_nav.vae_goal_wrapper as m
    import importlib
    importlib.reload(m)
    env = _StubEnv()
    env.reset = lambda **kw: (np.zeros(4, np.float32), {"rgb": _FRAME})
    w = m.VAEGoalWrapper(env, source=_FakeSource())
    obs, _ = w.reset()
    assert float(obs["coll"][0]) == 0.0


def test_identity_records_with_coll(monkeypatch):
    w = _make(monkeypatch, coll="1")
    assert w.identity()["with_coll"] is True