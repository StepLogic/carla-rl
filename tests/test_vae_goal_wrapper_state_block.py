"""The 4-d goal block and the mode-independent 10-d `state` block.

WHY THE ACTION HISTORY MOVED OUT OF THE GOAL BLOCK. The goal block is
zeroed in explore mode by apply_mode, so anything riding inside it -- the
applied-control history, which is exactly what a goal-free policy needs to
keep its Markov state -- was erased with it. The history therefore moves
into its own `state` block (OBS_WITH_STATE=1), which NO mode ever masks:
explore and goal rows carry identical state blocks by construction, pinned
below by test_apply_mode_zeroes_goal_not_state.

`coll` is NOT renamed. The OBS_WITH_COLLISION path stays byte-identical for
the single-mode arm (v1/v2); `state` is a new key behind a new flag, and the
coherence check (Task 4) refuses the 2-mode pairing with OBS_WITH_COLLISION=1
so the two can never silently share a run.
"""
import importlib
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

pytest.importorskip("gymnasium")

from gymnasium import spaces  # noqa: E402
import gymnasium as gym  # noqa: E402

_FRAME = np.zeros((4, 4, 3), np.float32)


class _StubSource:
    """Stands in for LatentSource so no VAE is needed."""
    name = "stub"
    latent_dim = 128

    def __init__(self, latent_dim=128):
        self.latent_dim = latent_dim

    def latent(self, rgb):
        return np.zeros(self.latent_dim, np.float32)

    def identity(self):
        return {"backend": "stub", "latent_dim": self.latent_dim}


class _StubEnv(gym.Env):
    """Real gymnasium.Env: the Wrapper base asserts on the type.

    `applied` is consumed one (steer, throttle) pair per step as
    info["applied_steer"/"applied_throttle"], which is how the wrapper's
    history deque is actually filled on a real run.
    """
    metadata = {}

    def __init__(self, applied=None, has_collided=False, speed_kmh=0.0,
                 mode=None, yaw=0.0, hero=(0.0, 0.0), goal=(10.0, 0.0),
                 vector=None):
        self._applied = list(applied or [])
        self.has_collided = bool(has_collided)
        self.speed_kmh = float(speed_kmh)
        self.mode = mode
        self.yaw = float(yaw)
        self.hero = tuple(hero)
        self.goal = tuple(goal)
        self.vector = vector
        self.t = 0
        self.observation_space = spaces.Dict({})
        self.action_space = spaces.Box(
            np.array([-0.5, 0.0], np.float32), np.array([0.5, 1.0], np.float32))

    def _raw_obs(self):
        # OBS_WITH_VECTOR reads the vector key from the env's raw observation,
        # not from info (mirroring train_env_config_v4_image_goal).
        return {} if self.vector is None else {"vector": self.vector}

    def _info(self):
        info = {
            "rgb": _FRAME,
            "hero_pos": np.array(self.hero, np.float64),
            "hero_yaw": self.yaw,
            "goal_pos": np.array(self.goal, np.float64),
            "has_collided": self.has_collided,
            "speed_kmh": self.speed_kmh,
        }
        if self.mode is not None:
            info["mode"] = self.mode
        i = self.t - 1
        if 0 <= i < len(self._applied):
            info["applied_steer"], info["applied_throttle"] = self._applied[i]
        return info

    def reset(self, **kw):
        self.t = 0
        return self._raw_obs(), self._info()

    def step(self, a):
        self.t += 1
        return self._raw_obs(), 0.0, False, False, self._info()


def _reload(**env):
    """Module-level flags are read at import, so a layout change needs a reload.

    Same pattern as test_gated_obs.py. Unlike that file (which deliberately
    restores nothing), the flags set here are RESTORED after every test: this
    file runs after test_gated_obs.py alphabetically and must not leak
    OBS_WITH_STATE / TARGET_SPEED into the files that follow it.
    """
    for k, v in env.items():
        os.environ[k] = v
    import vae_nav.vae_goal_wrapper as m
    return importlib.reload(m)


_LAYOUT_KEYS = ("OBS_WITH_STATE", "OBS_WITH_COLLISION", "OBS_WITH_GOAL",
                "OBS_WITH_VECTOR", "OBS_WITH_MODE", "OBS_STATE_COLLISION",
                "OBS_LATENT_AS_PIXELS", "TARGET_SPEED")


@pytest.fixture(autouse=True)
def _restore_layout_env():
    saved = {k: os.environ.get(k) for k in _LAYOUT_KEYS}
    yield
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def _driven_wrapper(w, hist):
    """A wrapper whose history deque was filled by REAL applied controls.

    hist is oldest-first. step() appends BEFORE building the observation, so
    after driving exactly len(hist) steps (deque width 4) the observation of
    the last step carries exactly `hist` -- no attribute poking.
    """
    w.reset()
    obs = None
    for _ in range(len(hist)):
        obs, _r, _t, _tr, _i = w.step(np.zeros(2, np.float32))
    return w, obs


# ------------------------------------------------------------ goal block ----

def test_goal_block_is_four_dim_and_goal_free_of_actions():
    m = _reload(OBS_WITH_STATE="0", OBS_WITH_GOAL="1", OBS_WITH_VECTOR="0",
                OBS_WITH_MODE="0")
    w = m.VAEGoalWrapper(_StubEnv(), source=_StubSource())
    gb = w.goal_block({"hero_pos": (0, 0), "hero_yaw": 0.0, "goal_pos": (10, 0)})
    assert gb.shape == (4,)
    np.testing.assert_allclose(gb, [10.0, 0.0, np.sin(0.0), np.cos(0.0)],
                               atol=1e-6)
    assert gb.dtype == np.float32


def test_goal_block_ignores_extra_fields_keys():
    # The trainer's record() call passes action_history alongside the pose
    # fields; the 4-d transform must ignore it, not shape from it. The values
    # below would appear in the block if the history concat survived.
    m = _reload(OBS_WITH_STATE="0", OBS_WITH_GOAL="1", OBS_WITH_VECTOR="0",
                OBS_WITH_MODE="0")
    w = m.VAEGoalWrapper(_StubEnv(), source=_StubSource())
    plain = w.goal_block({"hero_pos": (0, 0), "hero_yaw": 0.0, "goal_pos": (10, 0)})
    extra = w.goal_block({"hero_pos": (0, 0), "hero_yaw": 0.0,
                          "goal_pos": (10, 0),
                          "action_history": np.full((4, 2), 0.7, np.float32)})
    np.testing.assert_array_equal(plain, extra)
    assert extra.shape == (4,)


def test_goal_block_order_matches_the_old_first_four_entries():
    # NONZERO YAW, so a mirrored sin/cos or a stale history slot would show.
    # yaw = pi/2 faces +y; a goal at world +x is then 10 m to the LEFT under
    # CARLA's left-handed basis (the comment inside goal_block is the record).
    m = _reload(OBS_WITH_STATE="0", OBS_WITH_GOAL="1", OBS_WITH_VECTOR="0",
                OBS_WITH_MODE="0")
    w = m.VAEGoalWrapper(_StubEnv(), source=_StubSource())
    yaw = np.pi / 2
    gb = w.goal_block({"hero_pos": (0, 0), "hero_yaw": yaw, "goal_pos": (10, 0)})
    np.testing.assert_allclose(
        gb, [0.0, 10.0, np.sin(yaw), np.cos(yaw)], atol=1e-6)


# ------------------------------------------------------------ state block ----

def test_state_block_contents_and_order():
    m = _reload(OBS_WITH_STATE="1", OBS_WITH_GOAL="1", OBS_WITH_VECTOR="0",
                OBS_WITH_MODE="0", TARGET_SPEED="20.0")
    hist = [(0.1, 0.5), (0.2, 0.4), (0, 0), (0, 0)]
    w, obs = _driven_wrapper(
        m.VAEGoalWrapper(_StubEnv(applied=hist, speed_kmh=10.0),
                         source=_StubSource()), hist)
    s = obs["state"]
    assert s.shape == (10,)
    # [has_collided, speed/target, *history.flatten()] -- history oldest first.
    np.testing.assert_allclose(
        s, [0.0, 0.5, 0.1, 0.5, 0.2, 0.4, 0, 0, 0, 0], atol=1e-6)


def test_state_block_speed_clipped():
    m = _reload(OBS_WITH_STATE="1", OBS_WITH_GOAL="1", OBS_WITH_VECTOR="0",
                OBS_WITH_MODE="0", TARGET_SPEED="20.0")
    w = m.VAEGoalWrapper(_StubEnv(speed_kmh=99.0), source=_StubSource())
    _obs, _info = w.reset()
    obs, _r, _t, _tr, _i = w.step(np.zeros(2, np.float32))
    assert obs["state"][1] == pytest.approx(1.0)


def test_state_block_carries_collision_and_honours_target_speed():
    # has_collided is entry 0; TARGET_SPEED rescales entry 1 (10 km/h under a
    # 10 km/h target is the same ratio as 20 under 20).
    m = _reload(OBS_WITH_STATE="1", OBS_WITH_GOAL="1", OBS_WITH_VECTOR="0",
                OBS_WITH_MODE="0", TARGET_SPEED="10.0")
    w = m.VAEGoalWrapper(_StubEnv(speed_kmh=10.0, has_collided=True),
                         source=_StubSource())
    _obs, _info = w.reset()
    obs, _r, _t, _tr, _i = w.step(np.zeros(2, np.float32))
    assert obs["state"][0] == pytest.approx(1.0)
    assert obs["state"][1] == pytest.approx(1.0)
    assert obs["state"].dtype == np.float32


def test_state_key_absent_without_the_flag():
    m = _reload(OBS_WITH_STATE="0", OBS_WITH_GOAL="1", OBS_WITH_VECTOR="0",
                OBS_WITH_MODE="0")
    w = m.VAEGoalWrapper(_StubEnv(), source=_StubSource())
    obs, _info = w.reset()
    assert "state" not in obs
    assert "state" not in w.observation_space.spaces


# ------------------------------------------------------- mode independence ----

def test_apply_mode_zeroes_goal_not_state():
    # Mode 1 then mode 0 in info, identical everything else: the goal block is
    # the ONLY thing explore mode touches. Gating requires the vector block
    # (check_gated_coherence), so the stub carries one.
    m = _reload(OBS_WITH_STATE="1", OBS_WITH_GOAL="1", OBS_WITH_VECTOR="1",
                OBS_WITH_MODE="1")
    hist = [(0.1, 0.5), (0.2, 0.4), (0, 0), (0, 0)]
    env_g = m.VAEGoalWrapper(
        _StubEnv(applied=hist, speed_kmh=10.0, mode=1, vector=np.zeros(4, np.float32)),
        source=_StubSource())
    env_e = m.VAEGoalWrapper(
        _StubEnv(applied=hist, speed_kmh=10.0, mode=0, vector=np.zeros(4, np.float32)),
        source=_StubSource())
    _wg, obs_goal = _driven_wrapper(env_g, hist)
    _we, obs_explore = _driven_wrapper(env_e, hist)
    np.testing.assert_array_equal(obs_explore["state"], obs_goal["state"])
    assert np.allclose(obs_explore["goal"], 0.0)
    assert not np.allclose(obs_goal["goal"], 0.0)
    # And the state block of the goal-mode row is nonzero, i.e. it really
    # carries data rather than being zeroed alongside.
    assert not np.allclose(obs_goal["state"], 0.0)


def test_state_block_present_and_unmasked_in_explore_mode_space():
    # Space-level: with_mode=1 the state key is declared alongside mode.
    m = _reload(OBS_WITH_STATE="1", OBS_WITH_GOAL="1", OBS_WITH_VECTOR="1",
                OBS_WITH_MODE="1")
    w = m.VAEGoalWrapper(_StubEnv(vector=np.zeros(4, np.float32)),
                         source=_StubSource())
    assert w.observation_space["state"].shape == (10,)


# ------------------------------------------------------------- the space ----

def test_observation_space_matches():
    m = _reload(OBS_WITH_STATE="1", OBS_WITH_GOAL="1", OBS_WITH_VECTOR="0",
                OBS_WITH_MODE="0")
    space = m.VAEGoalWrapper(_StubEnv(), source=_StubSource()).observation_space
    assert space["goal"].shape == (4,) and space["state"].shape == (10,)


def test_vc1_grid_latent_dim_flows_through_unchanged():
    # 37632 flows from source.latent_dim with NO change in this task; a
    # regression here would quietly resize the agent's image projection.
    m = _reload(OBS_WITH_STATE="1", OBS_WITH_GOAL="1", OBS_WITH_VECTOR="0",
                OBS_WITH_MODE="0")
    w = m.VAEGoalWrapper(_StubEnv(), source=_StubSource(37632))
    assert w.latent_dim == 37632
    assert w.observation_space["latent"].shape == (37632,)


# ----------------------------------------------------- the coll path alone ----

def test_coll_path_is_unchanged_by_the_state_block():
    # The single-mode arm's `coll` key keeps its own flag and its own width.
    m = _reload(OBS_WITH_STATE="1", OBS_WITH_COLLISION="1", OBS_WITH_GOAL="1",
                OBS_WITH_VECTOR="0", OBS_WITH_MODE="0")
    w = m.VAEGoalWrapper(_StubEnv(), source=_StubSource())
    obs, _info = w.reset()
    assert "coll" in obs and obs["coll"].shape == (1,)
    assert "state" in obs and obs["state"].shape == (10,)


def test_the_banner_lists_the_state_key(capsys):
    m = _reload(OBS_WITH_STATE="1", OBS_WITH_GOAL="1", OBS_WITH_VECTOR="0",
                OBS_WITH_MODE="0")
    m.VAEGoalWrapper(_StubEnv(), source=_StubSource())
    out = capsys.readouterr().out
    assert "'state': (10,)" in out, out
    assert "'goal': (4,)" in out, out


def test_identity_records_the_state_flag():
    m = _reload(OBS_WITH_STATE="1", OBS_WITH_GOAL="1", OBS_WITH_VECTOR="0",
                OBS_WITH_MODE="0")
    w = m.VAEGoalWrapper(_StubEnv(), source=_StubSource())
    assert w.identity()["with_state"] is True