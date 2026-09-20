"""The GATED_ONE_CRITIC route: one plain TQC critic and a plain buffer.

WHAT THIS CLOSES. Task 5 left learner and buffer selection GATED-only, so the
2-mode image-goal pairing (IMAGE_GOAL_TASK=1 AND GATED=1, which Task 4's
coherence requires to run with GATED_ONE_CRITIC=1) would have built the
two-critic GatedTQCLearner over the ModeSplitBuffer. The unified-reward arm has
ONE critic: the second critic is never addressed by the reward, and the mode
stratified sampling splits a batch that one critic must see whole.

The predicates are pinned here WITHOUT building a learner: importing the
trainer needs jax, but the route itself is a pair of environment-variable
reads, and the kwargs builder is a pure dict. The learner-construction branch
in build_agent is pinned by the wiring test's shape (GatedTQCLearner under
GATED=1 alone) and by the smoke run.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

pytest.importorskip("jax")
pytest.importorskip("jaxrl2")

# The kwargs the plain TQCLearner must NEVER receive: none of them exist on
# jaxrl2's TQCLearner, which takes no **kwargs. On the one-critic route the
# bounds arrive through TQC_TARGET_MIN/MAX, read at import in critic_updater,
# not through value_floor/ceiling.
GATED_LEARNER_ONLY = ("value_floor", "value_ceiling", "discount_e",
                      "explore_actor_warmup", "actor_freeze_updates",
                      "actor_q_norm")


@pytest.fixture
def trainer(monkeypatch):
    # GATED=1 with NO GATED_ONE_CRITIC: the frozen two-critic route the
    # existing tests already pin. Each test adjusts from here.
    monkeypatch.setenv("GATED", "1")
    monkeypatch.delenv("GATED_ONE_CRITIC", raising=False)
    return pytest.importorskip("train_carla_tqc")


def test_wants_one_critic_truth_table(trainer, monkeypatch):
    # TRUE only when BOTH hold. GATED=1 alone is the frozen two-critic arm and
    # GATED_ONE_CRITIC=1 without GATED is meaningless.
    monkeypatch.setenv("GATED", "1")
    monkeypatch.setenv("GATED_ONE_CRITIC", "1")
    assert trainer.wants_one_critic() is True
    monkeypatch.setenv("GATED_ONE_CRITIC", "0")
    assert trainer.wants_one_critic() is False
    monkeypatch.setenv("GATED", "0")
    assert trainer.wants_one_critic() is False


def test_one_critic_kwargs_exclude_the_gated_learner_only_keys(
        trainer, monkeypatch):
    monkeypatch.setenv("GATED", "1")
    monkeypatch.setenv("GATED_ONE_CRITIC", "1")
    kw = trainer.agent_kwargs()
    for key in GATED_LEARNER_ONLY:
        assert key not in kw, (
            f"{key} would be rejected by the plain TQCLearner, which takes no "
            "**kwargs")


def test_one_critic_kwargs_still_carry_the_plain_tqc_config(
        trainer, monkeypatch):
    # Not an empty dict: the peer's config of record must survive the route.
    monkeypatch.setenv("GATED", "1")
    monkeypatch.setenv("GATED_ONE_CRITIC", "1")
    kw = trainer.agent_kwargs()
    assert kw["num_qs"] == 5
    assert kw["discount"] == 0.99
    assert "kappa" in kw and "action_space" in kw


def test_two_critic_kwargs_keep_the_gated_learner_only_keys(
        trainer, monkeypatch):
    # THE BYTE-IDENTICAL GUARD. GATED=1 without GATED_ONE_CRITIC is the frozen
    # two-critic arm; its build kwargs are exactly what they were.
    monkeypatch.setenv("GATED", "1")
    monkeypatch.delenv("GATED_ONE_CRITIC", raising=False)
    kw = trainer.agent_kwargs()
    for key in GATED_LEARNER_ONLY:
        assert key in kw


def test_two_critic_value_bounds_stay_derived_from_the_reward(
        trainer, monkeypatch):
    # The SORB derivation, unchanged: floor = -STEP_COST * GOAL_MAX_STEPS,
    # ceiling = 0, with the TQC_VALUE_FLOOR/CEILING overrides on top.
    monkeypatch.setenv("GATED", "1")
    monkeypatch.delenv("GATED_ONE_CRITIC", raising=False)
    monkeypatch.delenv("TQC_VALUE_FLOOR", raising=False)
    monkeypatch.delenv("TQC_VALUE_CEILING", raising=False)
    monkeypatch.delenv("STEP_COST", raising=False)
    monkeypatch.delenv("GATED_GOAL_MAX_STEPS", raising=False)
    monkeypatch.setenv("GATED_GOAL_REWARD", "sorb")
    kw = trainer.agent_kwargs()
    assert kw["value_floor"] == -1000.0
    assert kw["value_ceiling"] == 0.0


def test_plain_route_kwargs_exclude_the_gated_learner_only_keys(
        trainer, monkeypatch):
    # GATED=0 is the single-mode arm that has always built the plain learner.
    monkeypatch.setenv("GATED", "0")
    kw = trainer.agent_kwargs()
    for key in GATED_LEARNER_ONLY:
        assert key not in kw