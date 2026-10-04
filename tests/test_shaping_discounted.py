"""Potential shaping must stay Ng-invariant when the goal critic is discounted.

GatedReward pays K*(d_prev - d), which is gamma*Phi(s') - Phi(s) for
Phi = -K*d ONLY at gamma = 1. At gamma < 1 the invariant form is
K*(d_prev - gamma*d); the difference (1-gamma)*K*d is an extra per-step
penalty proportional to distance-to-goal -- -4/step at 200 m with K=2 and
gamma=0.99, against a step cost of 1.5. v40 changes DISCOUNT 1.0 -> 0.99 to
test whether the gamma=1 goal critic is what diverges (v37-v39); without this
the gamma change would drag an unintended -d reward in with it.

GOAL_SHAPING_DISCOUNTED=1 uses the gamma-aware form. Default off: every
existing recipe is byte-identical, and at gamma=1 the two forms coincide.
"""
import importlib
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))
pytest.importorskip("gymnasium")
import gymnasium as gym                       # noqa: E402
from gymnasium import spaces                  # noqa: E402

ACT = np.zeros(2, np.float32)
BASE = {"GATED_GOAL_REWARD": "sorb", "GOAL_TRUNCATE_COUNTS": "1",
        "GOAL_ARRIVAL_BONUS": "100.0", "POTENTIAL_K": "2.0", "STEP_COST": "1.5",
        "EPISODE_HER": "0"}


def _module(extra):
    import vae_nav.gated_reward as mod
    keys = set(BASE) | set(extra) | {"DISCOUNT", "GOAL_SHAPING_DISCOUNTED"}
    saved = {k: os.environ.get(k) for k in keys}
    for k in ("DISCOUNT", "GOAL_SHAPING_DISCOUNTED"):
        os.environ.pop(k, None)
    os.environ.update(BASE); os.environ.update(extra)
    importlib.reload(mod)
    return mod, saved


def _restore(saved):
    import vae_nav.gated_reward as mod
    for k, v in saved.items():
        if v is None: os.environ.pop(k, None)
        else: os.environ[k] = v
    importlib.reload(mod)


class _RouteEnv(gym.Env):
    """Publishes route_remaining_m from a scripted sequence."""
    metadata = {}

    def __init__(self, dists):
        self.observation_space = spaces.Dict({})
        self.action_space = spaces.Box(np.array([-0.5, 0.0], np.float32),
                                       np.array([0.5, 0.6], np.float32))
        self.d = list(dists); self.i = 0

    def _info(self):
        return {"hero_pos": (0.0, 0.0), "speed_factor": 0.0,
                "route_remaining_m": float(self.d[min(self.i, len(self.d) - 1)])}

    def reset(self, **kw):
        self.i = 0; return np.zeros(4, np.float32), self._info()

    def step(self, a):
        self.i += 1; return np.zeros(4, np.float32), 0.0, False, False, self._info()


def _shaping(mod, dists):
    w = mod.GatedReward(_RouteEnv(dists), force_mode=mod.GOAL, max_steps=500, seed=0,
                        goal_kind="sorb", goal_max_steps=1000, step_cost=1.5)
    w.reset()
    return [w.step(ACT)[4]["gated_shaping"] for _ in range(len(dists) - 1)]


DISTS = [200.0, 199.7, 199.4, 199.1, 198.8]


@pytest.mark.parametrize("extra", [{"DISCOUNT": "1.0"}, {"DISCOUNT": "0.99"}])
def test_default_is_the_old_undiscounted_difference(extra):
    mod, saved = _module(extra)
    try:
        got = _shaping(mod, DISTS)
        # first step has nothing to difference
        want = [0.0] + [2.0 * (DISTS[i] - DISTS[i + 1]) for i in range(1, len(DISTS) - 1)]
        assert got == pytest.approx(want)
    finally:
        _restore(saved)


def test_discounted_form_at_gamma_099():
    mod, saved = _module({"DISCOUNT": "0.99", "GOAL_SHAPING_DISCOUNTED": "1"})
    try:
        got = _shaping(mod, DISTS)
        want = [0.0] + [2.0 * (DISTS[i] - 0.99 * DISTS[i + 1]) for i in range(1, len(DISTS) - 1)]
        assert got == pytest.approx(want)
        # it is the old form PLUS (1-g)*K*d -- here ~+4/step at ~199 m
        assert got[1] - 2.0 * (DISTS[1] - DISTS[2]) == pytest.approx(0.01 * 2.0 * DISTS[2])
    finally:
        _restore(saved)


def test_discounted_form_equals_old_form_at_gamma_one():
    mod, saved = _module({"DISCOUNT": "1.0", "GOAL_SHAPING_DISCOUNTED": "1"})
    try:
        want = [0.0] + [2.0 * (DISTS[i] - DISTS[i + 1]) for i in range(1, len(DISTS) - 1)]
        assert _shaping(mod, DISTS) == pytest.approx(want)
    finally:
        _restore(saved)


def test_refused_with_her_because_the_recost_does_not_mirror_it():
    with pytest.raises(ValueError, match="GOAL_SHAPING_DISCOUNTED"):
        mod, saved = _module({"DISCOUNT": "0.99", "GOAL_SHAPING_DISCOUNTED": "1",
                              "EPISODE_HER": "1"})
    os.environ["EPISODE_HER"] = "0"; os.environ.pop("GOAL_SHAPING_DISCOUNTED", None)
    import vae_nav.gated_reward as mod
    importlib.reload(mod)
