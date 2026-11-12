"""make_env's 2-mode branch and check_gated_run's acceptance of the pairing.

TWO THINGS ARE PINNED HERE. First, gated_config.check_gated_run must accept the
2-mode image-goal pairing -- PERCEPTION=vc1, and no OBS_WITH_VECTOR requirement
(the pairing's velocity lives in the mode-independent `state` block) -- while
refusing everything it refused before. Second, train_carla_tqc.make_env must
route IMAGE_GOAL_TASK=1 + GATED=1 through the TWO-MODE stack
(TwoModeImageGoalReward below VAEGoalWrapper) and leave the single-mode
IMAGE_GOAL_TASK arm and the GATED arm exactly as they were: the two
byte-identity constraints on this task are pinned by running those branches
against the same fake CARLA stack and comparing the wrapper classes.

The full make_env runs WITHOUT a simulator: CarlaEnv and the train-env config
module are replaced by fakes in sys.modules, and CARLA_START_SERVER=0 skips
ensure_carla. The branch ORDER (2-mode before the single-mode early return)
falls out of which wrapper classes the stack ends up carrying.
"""
import importlib
import os
import sys
import types

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

pytest.importorskip("gymnasium")
pytest.importorskip("jax")          # train_carla_tqc imports jax at module level
pytest.importorskip("jaxrl2")

from gymnasium import spaces  # noqa: E402
import gymnasium as gym  # noqa: E402

from vae_nav.gated_config import check_gated_run  # noqa: E402

# ------------------------------------------------ the check_gated_run part --
# Same base as tests/test_gated_config.py, plus the pairing flag (default OFF,
# so every legacy combo here is the old gated arm's, unchanged).

def _env(**kw):
    base = {"GATED": "1", "OBS_WITH_MODE": "1", "OBS_WITH_GOAL": "1",
            "OBS_WITH_VECTOR": "1", "PERCEPTION": "vae",
            "IMAGE_GOAL_TASK": "0",
            "GATED_GOAL_REWARD": "sorb", "STEP_COST": "1.0",
            "DISCOUNT": "1.0", "EXPLORE_DISCOUNT": "0.99",
            "LANE_EXPLORE": "0",
            "EXPLORE_PROB": "0.5", "GOAL_BATCH_FRAC": "0.5",
            "MAX_EPISODE_STEPS": "2000", "GATED_EXPLORE_MAX_STEPS": "500",
            "GATED_GOAL_MAX_STEPS": "1000"}
    base.update({k: str(v) for k, v in kw.items()})
    return base


def test_gated_accepts_vc1():
    # The 2-mode arm's backend, previously refused outright.
    check_gated_run(_env(PERCEPTION="vc1"))


def test_gated_accepts_vc1_vitb():
    check_gated_run(_env(PERCEPTION="vc1_vitb"))


def test_gated_still_refuses_other_backends():
    # vc1 is an ACCEPTANCE, not a widening of the list to everything.
    with pytest.raises(ValueError, match="PERCEPTION"):
        check_gated_run(_env(PERCEPTION="dino"))


def test_gated_two_mode_skips_vector_requirement():
    # OBS_WITH_VECTOR unset, pairing active: accepted. Velocity reaches the
    # critic through the 10-d `state` block instead (OBS_WITH_STATE=1).
    check_gated_run(_env(IMAGE_GOAL_TASK="1", OBS_WITH_VECTOR="0"))


def test_gated_without_the_pairing_still_requires_vector():
    # THE REGRESSION GUARD. Without IMAGE_GOAL_TASK the requirement is exactly
    # what it was: the frozen gated arms must see this refusal unchanged.
    with pytest.raises(ValueError, match="vector block"):
        check_gated_run(_env(OBS_WITH_VECTOR="0"))


def test_gated_still_refuses_lane_explore():
    # The skip must not smuggle in a general relaxation: the explorer arm is
    # still refused under the pairing.
    with pytest.raises(ValueError, match="LANE_EXPLORE"):
        check_gated_run(_env(IMAGE_GOAL_TASK="1", LANE_EXPLORE="1"))


# ------------------------------------------- the make_env branch + stack ----

class _Loc:
    def __init__(self, x, y):
        self.x, self.y, self.z = float(x), float(y), 0.0


class _Rot:
    def __init__(self, yaw):
        self.yaw = float(yaw)


class _TF:
    def __init__(self, x, y, yaw):
        self.location = _Loc(x, y)
        self.rotation = _Rot(yaw)


class _FakeCarlaEnv(gym.Env):
    """Stands in for rlib_integration.carla_env.CarlaEnv.

    Carries the `.core` / `.experiment` objects CoreStateWrapper reaches into,
    and an obs dict with the two keys VAEGoalWrapper reads (the image and,
    when OBS_WITH_VECTOR=1, the env's 4-vector).
    """

    metadata = {}

    def __init__(self, config=None):
        self.observation_space = spaces.Dict({
            "state": spaces.Box(-np.inf, np.inf, (84, 84, 1), np.float32),
            "vector": spaces.Box(-5.1, 5.1, (4,), np.float32)})
        self.action_space = spaces.Box(-1.0, 1.0, (2,), np.float32)
        self._tf = _TF(0.0, 0.0, 0.0)
        self.core = types.SimpleNamespace(
            hero=types.SimpleNamespace(get_transform=lambda: self._tf),
            destination=types.SimpleNamespace(location=_Loc(10.0, 0.0)))
        self.experiment = types.SimpleNamespace(
            obs_img=np.zeros((84, 84, 3), np.float32), velocity=20.0,
            target_speed=20.0)
        self._obs = {"state": np.zeros((84, 84, 1), np.float32),
                     "vector": np.zeros(4, np.float32)}

    def reset(self, **kwargs):
        return dict(self._obs), {}

    def step(self, action):
        return dict(self._obs), 0.0, False, False, {}

    def close(self):
        pass


class _FakeLatentSource:
    """The VC-1 source's small interface, with no weights loaded."""

    latent_dim = 768
    name = "vc1"

    def latent(self, rgb):
        return np.zeros(768, np.float32)

    def identity(self):
        return {"backend": "vc1", "latent_dim": 768}


# THE 2-MODE RUN'S FLAGS: exactly what Task 4's image_goal_coherence demands
# of the pairing, plus what check_gated_run needs around it.
TWO_MODE_ENV = dict(
    IMAGE_GOAL_TASK="1", GATED="1",
    OBS_WITH_STATE="1", OBS_WITH_MODE="1", OBS_WITH_GOAL="1",
    OBS_WITH_VECTOR="0", OBS_WITH_COLLISION="0",
    PERCEPTION="vc1", VC1_READOUT="cls",
    GATED_ONE_CRITIC="1", EPISODE_HER="1",
    TQC_TARGET_MIN="-1100", TQC_TARGET_MAX="0",
    LANE_FOLLOW="0", LANE_EXPLORE="0", TIME_TO_GOAL="0",
    NO_TIMELIMIT="0",
    GATED_GOAL_REWARD="sorb", STEP_COST="1.0",
    DISCOUNT="1.0", EXPLORE_DISCOUNT="0.99",
    EXPLORE_PROB="0.5", GOAL_BATCH_FRAC="0.5",
    MAX_EPISODE_STEPS="1000", GATED_EXPLORE_MAX_STEPS="500",
    CARLA_START_SERVER="0")

# The FROZEN single-mode image-goal arm's flags (v1/v2).
SINGLE_MODE_ENV = dict(
    IMAGE_GOAL_TASK="1", GATED="0",
    OBS_WITH_COLLISION="1", PERCEPTION="vc1",
    OBS_WITH_STATE="0", OBS_WITH_MODE="0", OBS_WITH_VECTOR="0",
    LANE_FOLLOW="0", LANE_EXPLORE="0", TIME_TO_GOAL="0",
    NO_TIMELIMIT="0",
    MAX_EPISODE_STEPS="1000", CARLA_START_SERVER="0")

# The FROZEN gated arm's flags (no image-goal task).
GATED_ONLY_ENV = dict(
    GATED="1", IMAGE_GOAL_TASK="0",
    OBS_WITH_MODE="1", OBS_WITH_GOAL="1", OBS_WITH_VECTOR="1",
    PERCEPTION="vae",
    GATED_GOAL_REWARD="sorb", STEP_COST="1.0",
    DISCOUNT="1.0", EXPLORE_DISCOUNT="0.99",
    EXPLORE_PROB="0.5", GOAL_BATCH_FRAC="0.5",
    MAX_EPISODE_STEPS="1000", GATED_EXPLORE_MAX_STEPS="500",
    LANE_EXPLORE="0",
    CARLA_START_SERVER="0")

_STALE = ("OBS_WITH_STATE", "OBS_WITH_MODE", "OBS_WITH_GOAL",
          "OBS_WITH_VECTOR", "OBS_WITH_COLLISION", "IMAGE_GOAL_TASK",
          "GATED", "GATED_ONE_CRITIC", "EPISODE_HER", "TQC_TARGET_MIN",
          "TQC_TARGET_MAX", "PERCEPTION", "VC1_READOUT", "LANE_FOLLOW",
          "LANE_EXPLORE", "TIME_TO_GOAL", "NO_TIMELIMIT",
          "GATED_GOAL_REWARD", "STEP_COST", "DISCOUNT", "EXPLORE_DISCOUNT",
          "EXPLORE_PROB", "GOAL_BATCH_FRAC", "MAX_EPISODE_STEPS",
          "GATED_EXPLORE_MAX_STEPS", "GATED_GOAL_MAX_STEPS",
          "EVAL_FORCE_MODE", "CARLA_START_SERVER", "TARGET_SPEED",
          "VAE_CKPT", "IG_MAX_STEPS")


def _reload_env_modules():
    """Bake the ambient env's import-time flags into the obs/reward modules.

    image_goal_reward.IMAGE_GOAL_TASK and vae_goal_wrapper's WITH_* flags are
    read at import, and other tests in this suite reload the same modules with
    their own values -- so a test here re-bakes them under ITS env before
    touching make_env, the same pattern tests/test_gated_obs.py uses.
    """
    import vae_nav.image_goal_reward as igm
    importlib.reload(igm)
    import vae_nav.vae_goal_wrapper as vgw
    importlib.reload(vgw)
    return igm, vgw


def _install_fakes(monkeypatch, trainer):
    fake_carla = types.ModuleType("rlib_integration.carla_env")
    fake_carla.CarlaEnv = _FakeCarlaEnv
    monkeypatch.setitem(sys.modules, "rlib_integration.carla_env", fake_carla)
    fake_cfg = types.ModuleType("src.configs.train_env_config_v4_image_goal")
    fake_cfg.config = {"town": "Town01", "experiment": {}, "env_config": {}}
    monkeypatch.setitem(
        sys.modules, "src.configs.train_env_config_v4_image_goal", fake_cfg)


def _make_trainer(monkeypatch, **env_overrides):
    """Set the run's env, re-bake the import-time flags, fake the CARLA side."""
    for k in _STALE:
        monkeypatch.delenv(k, raising=False)
    for k, v in env_overrides.items():
        monkeypatch.setenv(k, str(v))
    _reload_env_modules()
    import vae_nav.gated_config as gcfg
    importlib.reload(gcfg)          # no import-time flags; keeps the module fresh
    trainer = pytest.importorskip("train_carla_tqc")
    _, vgw = _reload_env_modules()  # AFTER train_carla_tqc may have imported
    # train_carla_tqc binds VAEGoalWrapper (and MAX_EPISODE_STEPS) at ITS import
    # time, possibly under another test's env. make_env reads the module
    # globals, so point them at the classes this test just baked.
    monkeypatch.setattr(trainer, "VAEGoalWrapper", vgw.VAEGoalWrapper)
    monkeypatch.setattr(trainer, "MAX_EPISODE_STEPS",
                        int(os.environ["MAX_EPISODE_STEPS"]))
    _install_fakes(monkeypatch, trainer)
    return trainer


def _wrapper_chain(env):
    names = []
    while env is not None:
        names.append(type(env).__name__)
        env = getattr(env, "env", None)
    return names


@pytest.fixture
def two_mode_trainer(monkeypatch):
    return _make_trainer(monkeypatch, **TWO_MODE_ENV)


def test_the_branch_predicate_is_the_pairing(two_mode_trainer, monkeypatch):
    # THE CONDITION, EXTRACTED: IMAGE_GOAL_TASK=1 AND GATED=1, and nothing
    # else -- in particular GATED=1 alone must NOT select the 2-mode arm, and
    # IMAGE_GOAL_TASK=1 alone must not select it either.
    f = two_mode_trainer._two_mode_image_goal
    assert f() is True
    monkeypatch.setenv("GATED", "0")
    assert f() is False
    monkeypatch.setenv("GATED", "1")
    monkeypatch.setenv("IMAGE_GOAL_TASK", "0")
    assert f() is False
    monkeypatch.delenv("IMAGE_GOAL_TASK", raising=False)
    monkeypatch.delenv("GATED", raising=False)
    assert f() is False


def test_the_pairing_selects_the_two_mode_stack(two_mode_trainer):
    env = two_mode_trainer.make_env("Town01", 0, _FakeLatentSource())
    chain = _wrapper_chain(env)
    # THE BRANCH ORDER, in classes: the reward sits BELOW the obs wrapper and
    # DIRECTLY above the core env, and the whole stack is TimeLimit-capped.
    assert chain[:5] == ["TimeLimit", "VAEGoalWrapper", "TwoModeImageGoalReward",
                         "CoreStateWrapper", "_FakeCarlaEnv"]
    reward = env.env.env
    assert reward.explore_prob == 0.5
    assert reward.force_mode is None            # a TRAINING env draws the mode
    assert reward.max_steps == 1000             # the run's MAX_EPISODE_STEPS


def test_eval_mode_forces_the_goal_mode(two_mode_trainer):
    env = two_mode_trainer.make_env("Town01", 0, _FakeLatentSource(),
                                    eval_mode=True)
    # EVAL_FORCE_MODE defaults to GOAL (1): evaluation must not average the two
    # tasks.
    assert env.env.env.force_mode == 1


def test_the_pairing_runs_both_guards(monkeypatch):
    # Prove the guards are LIVE calls in this branch, not vacuous: remove a
    # flag each one demands and the branch must refuse, before any wrapper is
    # built. (image_goal_coherence's module constant is re-baked by
    # _make_trainer, so the refusal can only come from this branch's call.)
    with pytest.raises(SystemExit, match="OBS_WITH_STATE"):
        _make_trainer(monkeypatch,
                      **{**TWO_MODE_ENV, "OBS_WITH_STATE": "0"}
                      ).make_env("Town01", 0, _FakeLatentSource())
    # check_gated_run, SECOND in the branch: a degenerate EXPLORE_PROB is
    # refused by it and by nothing else (image_goal_coherence does not read
    # the probability), so this refusal can only come from the branch's own
    # check_gated_run() call.
    with pytest.raises(ValueError, match="EXPLORE_PROB"):
        _make_trainer(monkeypatch,
                      **{**TWO_MODE_ENV, "EXPLORE_PROB": "1.0"}
                      ).make_env("Town01", 0, _FakeLatentSource())


def test_the_pairing_leaves_the_single_mode_arm_alone(monkeypatch):
    # IMAGE_GOAL_TASK=1 with GATED=0 must STILL build the single-mode stack:
    # the 2-mode branch may not capture the pairing's other half.
    trainer = _make_trainer(monkeypatch, **SINGLE_MODE_ENV)
    env = trainer.make_env("Town01", 0, _FakeLatentSource())
    chain = _wrapper_chain(env)
    assert chain[:5] == ["TimeLimit", "VAEGoalWrapper", "ImageGoalReward",
                         "CoreStateWrapper", "_FakeCarlaEnv"]


def test_gated_without_the_image_goal_task_is_unchanged(monkeypatch):
    # The frozen gated arm: GATED=1 alone still builds GatedReward over
    # LaneFollowReward, with NO TwoModeImageGoalReward in the stack.
    trainer = _make_trainer(monkeypatch, **GATED_ONLY_ENV)
    env = trainer.make_env("Town01", 0, _FakeLatentSource())
    chain = _wrapper_chain(env)
    assert chain[:6] == ["TimeLimit", "VAEGoalWrapper", "GatedReward",
                         "LaneFollowReward", "CoreStateWrapper",
                         "_FakeCarlaEnv"]
    assert "TwoModeImageGoalReward" not in chain


def test_the_wrapper_accepts_the_two_mode_observation(two_mode_trainer):
    # The stack must actually RUN under the 2-mode flags -- this is where a
    # stale coherence check (the wrapper's own vector requirement) would
    # surface. One reset, one step, through the whole chain.
    env = two_mode_trainer.make_env("Town01", 0, _FakeLatentSource())
    obs, info = env.reset(seed=0)
    assert set(obs) == {"goal", "latent", "mode", "state"}
    assert obs["goal"].shape == (4,)
    assert obs["state"].shape == (10,)
    obs, r, term, trunc, info = env.step(np.zeros(2, np.float32))
    assert set(obs) == {"goal", "latent", "mode", "state"}
    assert r < 0.0                              # the ReViND invariant, live
    assert not (term or trunc)