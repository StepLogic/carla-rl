"""build_agent and the replay choice, under GATED=1, without a simulator.

WHAT THIS CLOSES. Everything else tests the gated pieces in isolation. The
trainer is where they are actually selected, and a branch that silently fails to
fire would produce a perfectly healthy-looking single-critic run under a script
named run_gated.sh -- the most expensive way to learn that a flag did nothing.

It imports train_carla_tqc, which is heavy but does not touch CARLA at import.
make_env is NOT exercised here: it needs a live server. Its branch is covered by
the smoke run in the plan.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

pytest.importorskip("jax")
pytest.importorskip("jaxrl2")
pytest.importorskip("gymnasium")

from gymnasium import spaces  # noqa: E402

GATED_ENV = dict(
    GATED="1", OBS_WITH_MODE="1", OBS_WITH_GOAL="1", OBS_WITH_VECTOR="1",
    PERCEPTION="vae", DISCOUNT="0.99", EXPLORE_DISCOUNT="0.99",
    LANE_EXPLORE="0", EXPLORE_PROB="0.5", GOAL_BATCH_FRAC="0.5",
    EXPLORE_ACTOR_WARMUP="5000", TQC_KAPPA="1.0")

OBS_GATED = spaces.Dict({
    "goal": spaces.Box(-np.inf, np.inf, (12,), np.float32),
    "latent": spaces.Box(-np.inf, np.inf, (128,), np.float32),
    "vector": spaces.Box(-5.1, 5.1, (4,), np.float32),
    "mode": spaces.Box(0.0, 1.0, (1,), np.float32)})
OBS_PLAIN = spaces.Dict({
    "goal": spaces.Box(-np.inf, np.inf, (12,), np.float32),
    "latent": spaces.Box(-np.inf, np.inf, (128,), np.float32),
    "vector": spaces.Box(-5.1, 5.1, (4,), np.float32)})
ACT = spaces.Box(np.array([-0.5, 0.0], np.float32),
                 np.array([0.5, 0.6], np.float32))


@pytest.fixture
def trainer(monkeypatch):
    for k, v in GATED_ENV.items():
        monkeypatch.setenv(k, v)
    return pytest.importorskip("train_carla_tqc")


def test_build_agent_returns_the_gated_learner(trainer):
    agent = trainer.build_agent(OBS_GATED, ACT, 42)
    assert type(agent).__name__ == "GatedTQCLearner"
    assert hasattr(agent, "_critic_e") and hasattr(agent, "_temp_e")


def test_the_explore_discount_reaches_the_learner(trainer, monkeypatch):
    monkeypatch.setenv("EXPLORE_DISCOUNT", "0.95")
    agent = trainer.build_agent(OBS_GATED, ACT, 42)
    assert agent.discount_e == 0.95
    assert agent.discount == 0.99


def test_the_explore_discount_defaults_to_the_goal_discount(trainer, monkeypatch):
    # Both CARLA lane arms run at 0.99. The two-critic split here is justified
    # by return scale, not by a horizon mismatch, so equal discounts are the
    # expected configuration rather than a mistake.
    monkeypatch.delenv("EXPLORE_DISCOUNT", raising=False)
    agent = trainer.build_agent(OBS_GATED, ACT, 42)
    assert agent.discount_e == agent.discount


def test_the_actor_warmup_reaches_the_learner(trainer):
    assert trainer.build_agent(OBS_GATED, ACT, 42).explore_actor_warmup == 5000


def test_without_the_flag_the_plain_learner_is_built(trainer, monkeypatch):
    # THE REGRESSION GUARD. The two existing arms must be untouched.
    monkeypatch.setenv("GATED", "0")
    monkeypatch.setenv("OBS_WITH_MODE", "0")
    agent = trainer.build_agent(OBS_PLAIN, ACT, 42)
    assert type(agent).__name__ == "TQCLearner"
    assert not hasattr(agent, "_critic_e")


def test_an_incoherent_gated_config_is_refused_before_the_simulator(monkeypatch):
    # check_gated_run is called from make_env, which needs a server; call it
    # directly with the same environment to prove the refusal path works.
    #
    # PERCEPTION="mobilenet" was the example here until 2026-09-11, when the
    # user had that guard relaxed (it now warns; the checkpoint's encoder
    # identity catches a real mismatch at load time). The refusal path itself
    # is unchanged, so this uses a config that is still incoherent: the mode
    # bit removed from the observation, which leaves the actor unable to see
    # which of the two rewards it is being scored against.
    from vae_nav.gated_config import check_gated_run
    env = dict(GATED_ENV, OBS_WITH_MODE="0")
    with pytest.raises(ValueError, match="OBS_WITH_MODE"):
        check_gated_run(env)
