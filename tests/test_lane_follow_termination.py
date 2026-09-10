"""LaneFollowReward's termination rule, and the two ways of relaxing it.

WHY THIS EXISTS. The four-way ablation reports that the gated arms extrapolate
(0.90 / 0.82 beyond the training band) and the shaped baseline does not
(0.32 / 0.28), and records that number as CONFOUNDED: crossing a lane marking
ENDS an episode for the baseline and is a no-op for the gated arms, whose goal
mode discards the env's `done` entirely. The baseline failed almost entirely by
lane invasion -- 35 and 43 of 60 episodes -- while the gated arms only ever
timed out. Part of that gap is a termination rule rather than the gate.

The confound is removed by evaluating the SAME frozen baseline policy under the
gated arms' termination rule. That needs no retraining, but it does need this
wrapper to honour the rule, which it previously could not: LANE_INVASION_NOOP
was read only by gated_reward.py, a module the GATED=0 baseline never imports.

TWO RULES, DELIBERATELY SEPARATE, because they answer different questions.

    LANE_INVASION_NOOP=1          crossing a marking costs nothing and ends
                                  nothing. OFF-ROAD AND COLLISION STAY FATAL.
                                  Same meaning the flag already has in
                                  gated_reward.explore_reward, so one name does
                                  not mean two things.

    LF_ONLY_ARRIVAL_TERMINATES=1  only arrival ends the episode; everything
                                  else runs to the step cap. This is EXACTLY
                                  what the gated goal arm does under the SORB
                                  reward (goal_reward_sorb: "ONLY ARRIVAL
                                  TERMINATES"), so it is the rule that makes
                                  the comparison attributable to the gate.

SUPPRESSING THE WRAPPER'S TERMINAL IS NOT ENOUGH, and that is the trap this
file mostly guards. The env ORs its raw `diff_lane` into its own `done`, and
this wrapper forwards that as `truncated`. Relaxing only `task_terminal` would
leave the episode ending anyway, one field to the left and silently -- which is
the exact failure sparse_reward.py documents having hit. Every relaxation test
below therefore asserts on BOTH flags.
"""
import importlib
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

pytest.importorskip("gymnasium")

import gymnasium as gym  # noqa: E402
from gymnasium import spaces  # noqa: E402

ACT = np.zeros(2, np.float32)


def _reload(noop=False, only_arrival=False):
    os.environ["LANE_INVASION_NOOP"] = "1" if noop else "0"
    os.environ["LF_ONLY_ARRIVAL_TERMINATES"] = "1" if only_arrival else "0"
    import vae_nav.lane_follow_reward as m
    return importlib.reload(m)


class _StubEnv(gym.Env):
    """Publishes one step's info, and the env's own done, exactly as CoreState
    would: `done_reasons` is a dict of every condition the env ORs together."""

    metadata = {}

    def __init__(self, info, env_done):
        self.observation_space = spaces.Dict({})
        self.action_space = spaces.Box(
            np.array([-0.5, 0.0], np.float32), np.array([0.5, 0.6], np.float32))
        self.info = info
        self.env_done = bool(env_done)

    def reset(self, **kw):
        return np.zeros(4, np.float32), dict(self.info)

    def step(self, a):
        return np.zeros(4, np.float32), 0.0, self.env_done, False, dict(self.info)


def _info(**kw):
    """One step's info. Defaults are a clean step: on the road, under speed."""
    reasons = {"idle": False, "falling": False, "dist200": False,
               "lane_any": False, "collision": False, "speed": False,
               "wrong_dir": False, "off_road": False}
    reasons.update(kw.pop("reasons", {}))
    info = {"speed_kmh": 10.0, "target_speed_kmh": 20.0,
            "delta_distance": 0.5, "action_smoothness_penalty": 0.0,
            "collision": False, "lane_invasion": False, "off_road": False,
            "is_success": False, "arrived": False, "done_reasons": reasons}
    info.update(kw)
    return info


def _step(m, info, env_done=True):
    """Returns (reward, terminated, truncated, info) for one step."""
    env = m.LaneFollowReward(_StubEnv(info, env_done))
    _obs, reward, term, trunc, out = env.step(ACT)
    return float(reward), bool(term), bool(trunc), out


# A crossing of a painted marking, nothing else wrong. core_state ORs off_road
# into `lane_invasion`, so a pure crossing sets the flag with off_road False.
CROSS = dict(lane_invasion=True, reasons={"lane_any": True})
OFFROAD = dict(lane_invasion=True, off_road=True,
               reasons={"lane_any": True, "off_road": True})


# ------------------------------------------------ the default, pinned ----

def test_by_default_a_crossing_terminates_and_bills():
    # The behaviour the recorded ablation ran under. Pinned so the deconfounded
    # re-run is a CHANGE OF FLAG rather than a change of default.
    m = _reload()
    reward, term, _trunc, _ = _step(m, _info(**CROSS))
    clean, _, _, _ = _step(m, _info(), env_done=False)
    assert term is True
    assert clean - reward == pytest.approx(m.K_LANE)


def test_both_flags_default_to_off():
    for var in ("LANE_INVASION_NOOP", "LF_ONLY_ARRIVAL_TERMINATES"):
        os.environ.pop(var, None)
    import vae_nav.lane_follow_reward as m
    m = importlib.reload(m)
    assert m.LANE_INVASION_NOOP is False
    assert m.ONLY_ARRIVAL_TERMINATES is False


# --------------------------------------------- LANE_INVASION_NOOP=1 ----

def test_noop_makes_a_crossing_end_nothing_on_either_flag():
    # BOTH fields: the env's own done is forwarded as `truncated`, so an
    # episode suppressed on one flag and not the other still ends.
    m = _reload(noop=True)
    _reward, term, trunc, out = _step(m, _info(**CROSS))
    assert term is False
    assert trunc is False
    assert out.get("suppressed_lane_done") is True


def test_noop_makes_a_crossing_free():
    m = _reload(noop=True)
    crossed, _, _, _ = _step(m, _info(**CROSS))
    clean, _, _, _ = _step(m, _info(), env_done=False)
    assert crossed == pytest.approx(clean)


def test_noop_leaves_off_road_fatal_and_billed():
    # THE BACKSTOP. Off-road is not a lane crossing; it is the only remaining
    # term that makes staying on the drivable surface worth anything.
    m = _reload(noop=True)
    reward, term, _trunc, _ = _step(m, _info(**OFFROAD))
    clean, _, _, _ = _step(m, _info(), env_done=False)
    assert term is True
    assert clean - reward == pytest.approx(m.K_LANE)


def test_noop_leaves_a_collision_fatal_and_billed():
    m = _reload(noop=True)
    reward, term, _trunc, _ = _step(
        m, _info(collision=True, reasons={"collision": True}))
    clean, _, _, _ = _step(m, _info(), env_done=False)
    assert term is True
    assert clean - reward == pytest.approx(m.K_COLLISION)


def test_noop_does_not_suppress_a_done_that_had_another_reason():
    # A crossing that coincides with a real failure must still end the episode,
    # or the flag would launder every termination that happens to co-occur.
    m = _reload(noop=True)
    _reward, term, trunc, _ = _step(
        m, _info(lane_invasion=True, collision=True,
                 reasons={"lane_any": True, "collision": True}))
    assert (term or trunc) is True


def test_noop_does_not_suppress_an_idle_done():
    m = _reload(noop=True)
    _reward, term, trunc, _ = _step(m, _info(reasons={"idle": True}))
    assert (term or trunc) is True


# --------------------------------------- LF_ONLY_ARRIVAL_TERMINATES=1 ----

def test_only_arrival_lets_a_crossing_run_on():
    m = _reload(only_arrival=True)
    _reward, term, trunc, _ = _step(m, _info(**CROSS))
    assert term is False
    assert trunc is False


def test_only_arrival_lets_a_collision_run_on():
    # This is the gated goal arm's rule: with a per-step cost, a terminal
    # failure would stop the meter and crashing becomes an escape. The baseline
    # is not paid per step, but it must END episodes the same way or the
    # success rates are not comparable.
    m = _reload(only_arrival=True)
    _reward, term, trunc, _ = _step(
        m, _info(collision=True, reasons={"collision": True}))
    assert term is False
    assert trunc is False


def test_only_arrival_lets_off_road_run_on():
    m = _reload(only_arrival=True)
    _reward, term, trunc, _ = _step(m, _info(**OFFROAD))
    assert term is False
    assert trunc is False


def test_only_arrival_still_ends_on_arrival():
    m = _reload(only_arrival=True)
    _reward, term, _trunc, out = _step(m, _info(is_success=True))
    assert term is True
    assert out["is_success"] is True


def test_only_arrival_still_bills_every_failure():
    # The episode does not end, but the reward is untouched: this flag changes
    # WHEN an episode stops, never what a step is worth. Otherwise the arm
    # would be scored under a reward its policy was not trained on.
    m = _reload(only_arrival=True)
    clean, _, _, _ = _step(m, _info(), env_done=False)
    for case, k in ((CROSS, "K_LANE"), (OFFROAD, "K_LANE"),
                    (dict(collision=True, reasons={"collision": True}),
                     "K_COLLISION")):
        reward, _, _, _ = _step(m, _info(**case))
        assert clean - reward == pytest.approx(getattr(m, k)), case


def test_only_arrival_outranks_the_noop_flag():
    # Both set is not an error; the stricter relaxation simply wins, so a
    # driver script cannot produce a third, undocumented rule by accident.
    m = _reload(noop=True, only_arrival=True)
    _reward, term, trunc, _ = _step(m, _info(**OFFROAD))
    assert term is False
    assert trunc is False


# ------------------------------------------------------- the banner ----

def test_the_banner_states_which_termination_rule_is_live(capsys):
    # A setting that changes when episodes end must be visible on screen. The
    # only other live confirmation is /proc/<pid>/environ, which is the gap
    # that made an earlier observation banner misleading.
    _reload(noop=True).LaneFollowReward(_StubEnv(_info(), False))
    assert "LANE INVASION IS A NO-OP" in capsys.readouterr().out

    _reload(only_arrival=True).LaneFollowReward(_StubEnv(_info(), False))
    assert "ONLY ARRIVAL TERMINATES" in capsys.readouterr().out
