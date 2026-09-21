"""Refuse gated runs that cannot learn, before the simulator starts.

The failures these catch are the expensive kind: they do not crash, they produce
a plausible loss curve and a meaningless result, and they cost a full training
budget to discover. This file follows tests/test_lane_explorer_config.py, which
exists for the same reason after a stale goal reward leaked into a goal-free run.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

from vae_nav.gated_config import check_gated_run  # noqa: E402


def _env(**kw):
    # DISCOUNT=1.0 because the default goal reward is now SORB minimum time.
    base = {"GATED": "1", "OBS_WITH_MODE": "1", "OBS_WITH_GOAL": "1",
            "OBS_WITH_VECTOR": "1", "PERCEPTION": "vae",
            "GATED_GOAL_REWARD": "sorb", "STEP_COST": "1.0",
            "DISCOUNT": "1.0", "EXPLORE_DISCOUNT": "0.99",
            "LANE_EXPLORE": "0",
            "EXPLORE_PROB": "0.5", "GOAL_BATCH_FRAC": "0.5",
            "MAX_EPISODE_STEPS": "2000", "GATED_EXPLORE_MAX_STEPS": "500",
            "GATED_GOAL_MAX_STEPS": "1000"}
    base.update({k: str(v) for k, v in kw.items()})
    return base


def test_a_coherent_gated_config_is_accepted():
    check_gated_run(_env())


def test_the_check_is_inert_when_not_gating():
    # It must not fire on the two existing arms, which set none of these.
    check_gated_run({})
    check_gated_run({"GATED": "0", "OBS_WITH_GOAL": "0"})


def test_rejects_a_gated_run_without_the_mode_key():
    with pytest.raises(ValueError, match="OBS_WITH_MODE"):
        check_gated_run(_env(OBS_WITH_MODE=0))


def test_rejects_a_gated_run_without_the_goal_block():
    with pytest.raises(ValueError, match="goal block"):
        check_gated_run(_env(OBS_WITH_GOAL=0))


def test_rejects_a_gated_run_without_the_vector_block():
    with pytest.raises(ValueError, match="vector block"):
        check_gated_run(_env(OBS_WITH_VECTOR=0))


def test_v8_state_block_replaces_the_vector_as_velocity_carrier():
    # THE v8 ESCAPE, same rule the wrapper's check_gated_coherence applies.
    # GATED with OBS_WITH_VECTOR=0 is permitted when the 9-d state block
    # carries the velocity (OBS_WITH_STATE=1, OBS_STATE_COLLISION=0): slot 0
    # IS the same clipped speed ratio the vector block carried, so the speed
    # the gated rewards' progress term assumes still reaches the observation.
    check_gated_run(_env(OBS_WITH_VECTOR=0, OBS_WITH_STATE=1,
                         OBS_STATE_COLLISION=0))


def test_v8_escape_requires_the_collision_flag_dropped():
    # A 10-d state block (collision flag present) is not the escape: the check
    # pins the 9-d block exactly, so this must fail with today's message.
    with pytest.raises(ValueError, match="vector block"):
        check_gated_run(_env(OBS_WITH_VECTOR=0, OBS_WITH_STATE=1,
                             OBS_STATE_COLLISION=1))


def test_vector_zero_alone_still_fails_with_todays_message():
    # The default (unset everything) behaviour is byte-identical to before the
    # v8 branch: no state block, no escape.
    with pytest.raises(ValueError, match="vector block"):
        check_gated_run(_env(OBS_WITH_VECTOR=0, OBS_WITH_STATE=0))


def test_the_image_goal_escape_still_passes():
    check_gated_run(_env(OBS_WITH_VECTOR=0, IMAGE_GOAL_TASK=1))


def test_accepts_the_mobilenet_backend_with_a_warning(capsys):
    # RELAXED 2026-09-11 (user): this used to raise. The width risk it guarded
    # against is already caught at load time by the checkpoint's encoder
    # identity (LatentSource.assert_matches), which names both sides, so the
    # guard was a blunter duplicate that also blocked a legitimate
    # from-scratch run. It must still WARN, loudly.
    check_gated_run(_env(PERCEPTION="mobilenet"))
    out = capsys.readouterr().out
    assert "384-d" in out and "NOT interchangeable" in out
    assert "from" in out and "scratch" in out


def test_still_rejects_an_unknown_backend():
    with pytest.raises(ValueError, match="PERCEPTION"):
        check_gated_run(_env(PERCEPTION="resnet50"))


def test_vae_backend_warns_about_nothing(capsys):
    check_gated_run(_env(PERCEPTION="vae"))
    assert "WARNING" not in capsys.readouterr().out


def test_rejects_gating_against_the_explorer_arm():
    with pytest.raises(ValueError, match="LANE_EXPLORE"):
        check_gated_run(_env(LANE_EXPLORE=1))


@pytest.mark.parametrize("p", ["0.0", "1.0", "-0.1", "1.5"])
def test_rejects_a_degenerate_explore_probability(p):
    # At 0 or 1 only one mode is ever drawn and one critic never trains.
    with pytest.raises(ValueError, match="EXPLORE_PROB"):
        check_gated_run(_env(EXPLORE_PROB=p))


@pytest.mark.parametrize("f", ["0.0", "1.0", "-0.2", "2.0"])
def test_rejects_a_degenerate_batch_fraction(f):
    # A degenerate split starves one critic entirely, which is the exact
    # failure the mode-split buffer exists to prevent.
    with pytest.raises(ValueError, match="GOAL_BATCH_FRAC"):
        check_gated_run(_env(GOAL_BATCH_FRAC=f))


def test_rejects_a_non_numeric_probability():
    with pytest.raises(ValueError, match="EXPLORE_PROB"):
        check_gated_run(_env(EXPLORE_PROB="half"))


def test_an_asymmetric_batch_split_is_allowed():
    # 0.5 is the default, not a requirement. A deliberate skew is a legitimate
    # experiment; only the degenerate endpoints are refused.
    check_gated_run(_env(GOAL_BATCH_FRAC="0.25"))


def test_differing_episode_caps_are_allowed_and_announced(capsys):
    # They differ on purpose (2000 against 500). The run should say so rather
    # than fail, because the fixed batch composition is what neutralises it.
    check_gated_run(_env())
    out = capsys.readouterr().out
    assert "episode caps differ on purpose" in out
    assert "skewed toward goal" in out


# ------------------------------------------------- the SORB goal reward ----

def test_sorb_requires_an_undiscounted_goal_critic():
    # THE SINGLE MOST EXPENSIVE MISCONFIGURATION HERE. At 0.99 the horizon is
    # 100 steps, Q saturates near -100, and a 400-step goal is 0.18 apart from
    # a 1000-step one. The run would train, converge, and produce a value
    # function useless as a distance -- the one thing SORB exists to provide.
    with pytest.raises(ValueError, match="requires DISCOUNT=1.0"):
        check_gated_run(_env(DISCOUNT="0.99"))


def test_sorb_refuses_an_undiscounted_explore_critic():
    # The explore arm has no terminal state, so gamma=1 there diverges.
    with pytest.raises(ValueError, match="EXPLORE_DISCOUNT"):
        check_gated_run(_env(EXPLORE_DISCOUNT="1.0"))


@pytest.mark.parametrize("c", ["0.0", "-1.0"])
def test_sorb_refuses_a_non_positive_step_cost(c):
    with pytest.raises(ValueError, match="STEP_COST"):
        check_gated_run(_env(STEP_COST=c))


def test_the_shaped_reward_still_allows_a_discounted_goal_critic():
    # The shaped arm is bounded by its own horizon and runs at 0.99, so the
    # DISCOUNT=1.0 requirement must NOT fire for it.
    check_gated_run(_env(GATED_GOAL_REWARD="shaped", DISCOUNT="0.99"))


def test_an_unknown_goal_reward_is_refused():
    with pytest.raises(ValueError, match="sorb|shaped"):
        check_gated_run(_env(GATED_GOAL_REWARD="dense"))


def test_the_run_announces_the_reward_and_its_return_range(capsys):
    check_gated_run(_env())
    out = capsys.readouterr().out
    assert "SORB minimum time" in out
    assert "[-1000, 0]" in out
    assert "steps to goal" in out


def test_discounted_sorb_is_still_refused_by_default():
    with pytest.raises(ValueError, match="DISCOUNT"):
        check_gated_run(_env(DISCOUNT="0.99"))


def test_discounted_sorb_is_allowed_with_the_explicit_opt_in(capsys):
    # User instruction 2026-09-11. Opt-in per run, and it must WARN with the
    # measured cost: Q stops reading as steps-to-goal, which is what the SGM
    # graph is built on.
    check_gated_run(_env(DISCOUNT="0.99", GATED_ALLOW_DISCOUNTED_SORB="1"))
    out = capsys.readouterr().out
    assert "NO LONGER READS AS" in out and "ORDERING" in out
