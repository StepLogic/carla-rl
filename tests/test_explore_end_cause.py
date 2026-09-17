"""Explore episodes must say WHY they ended, track HOW FAR they got, and end
on reaching a distance target.

MEASURED 2026-09-11: five explore-mode episodes from v6c@75k reported four
terminations whose cause was invisible -- the env published reward, shaping,
success and mode and nothing about the termination -- so "is explore ending on
lane crossings?" could not be answered from the logs.

Also measured: explore drives at 2.0-2.35 m/s, so a surviving 500-step (50 s)
episode covers 100-117 m. 100 m is therefore reachable at the cap, and the
user made it an ARRIVAL: reaching it ends the episode with a +2.0 bonus.

GatedReward.step now maintains info["explore_dist_m"] (cumulative PATH length,
not net displacement) and writes info["explore_end"] on the terminating step:
  "reached"   cumulative path >= EXPLORE_GOAL_DIST_M  (+EXPLORE_ARRIVAL_BONUS)
  "crossing"  painted marking on open road (junction-exempt)
  "off_road"  left the drivable surface
  "collision" hit something
  "cap"       hit the explore step cap
absent while the episode continues.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))
pytest.importorskip("gymnasium")
import gymnasium as gym                       # noqa: E402
from gymnasium import spaces                  # noqa: E402


class _StubEnv(gym.Env):
    """Scripted info; optionally advances hero_pos by `step_m` each step."""
    metadata = {}

    def __init__(self, info=None, step_m=0.0):
        self.observation_space = spaces.Dict({})
        self.action_space = spaces.Box(
            np.array([-0.5, 0.0], np.float32), np.array([0.5, 0.6], np.float32))
        self.info = dict(info or {})
        self.info.setdefault("speed_factor", 0.0)
        self.info.setdefault("hero_yaw", 0.0)
        self._x = 0.0
        self._step_m = float(step_m)

    def reset(self, **kw):
        self._x = 0.0
        out = dict(self.info); out["hero_pos"] = (0.0, 0.0)
        return np.zeros(4, np.float32), out

    def step(self, action):
        self._x += self._step_m
        out = dict(self.info); out["hero_pos"] = (self._x, 0.0)
        return np.zeros(4, np.float32), 0.0, False, False, out


def _wrap(info=None, max_steps=5, step_m=0.0):
    from vae_nav.gated_reward import GatedReward, EXPLORE
    w = GatedReward(_StubEnv(info, step_m), force_mode=EXPLORE,
                    max_steps=max_steps, seed=0)
    w.reset()
    return w


def test_crossing_is_named():
    w = _wrap(dict(lane_any=True, lane_marking_types=["Broken"],
                   in_junction=False))
    _o, _r, term, _tr, info = w.step(np.zeros(2))
    assert term and info.get("explore_end") == "crossing"


def test_off_road_is_named():
    w = _wrap(dict(off_road=True))
    _o, _r, term, _tr, info = w.step(np.zeros(2))
    assert term and info.get("explore_end") == "off_road"


def test_collision_is_named():
    w = _wrap(dict(collision=True))
    _o, _r, term, _tr, info = w.step(np.zeros(2))
    assert term and info.get("explore_end") == "collision"


def test_cap_is_named():
    w = _wrap(max_steps=2)
    w.step(np.zeros(2))
    _o, _r, term, trunc, info = w.step(np.zeros(2))
    assert not term and trunc and info.get("explore_end") == "cap"


def test_absent_while_running():
    w = _wrap(max_steps=9)
    _o, _r, term, trunc, info = w.step(np.zeros(2))
    assert not term and not trunc and "explore_end" not in info


def test_distance_is_cumulative_path():
    w = _wrap(max_steps=9, step_m=3.0)
    for _ in range(3):
        _o, _r, _t, _tr, info = w.step(np.zeros(2))
    assert info["explore_dist_m"] == pytest.approx(9.0)


def test_reaching_the_target_ends_the_episode_with_a_bonus(monkeypatch):
    monkeypatch.setenv("EXPLORE_GOAL_DIST_M", "10")
    monkeypatch.setenv("EXPLORE_ARRIVAL_BONUS", "2.0")
    import importlib
    import vae_nav.gated_reward as mod
    importlib.reload(mod)
    w = mod.GatedReward(_StubEnv(None, step_m=4.0), force_mode=mod.EXPLORE,
                        max_steps=99, seed=0)
    w.reset()
    r1 = w.step(np.zeros(2))          # 4 m
    r2 = w.step(np.zeros(2))          # 8 m
    assert not r1[2] and not r2[2]
    _o, r, term, trunc, info = w.step(np.zeros(2))   # 12 m >= 10
    assert term and not trunc
    assert info.get("explore_end") == "reached"
    assert info["explore_dist_m"] == pytest.approx(12.0)
    # the bonus is ON TOP of the step's own reward, so the step is a net gain
    assert r > 1.0, r
    importlib.reload(mod)


# --- per-step off-road penalty -------------------------------------------

def _reload(monkeypatch, **env):
    import importlib
    for k, v in env.items():
        monkeypatch.setenv(k, str(v))
    import vae_nav.gated_reward as mod
    return importlib.reload(mod)


def test_off_road_penalty_is_charged_per_step(monkeypatch):
    """With off-road NOT fatal, every off-road step costs the penalty."""
    mod = _reload(monkeypatch, EXPLORE_OFF_ROAD_PENALTY=1.5,
                  EXPLORE_OFF_ROAD_FATAL=0, EXPLORE_GOAL_DIST_M=0)
    w = mod.GatedReward(_StubEnv(dict(off_road=True)), force_mode=mod.EXPLORE,
                        max_steps=9, seed=0)
    w.reset()
    _o, r_off, term, _tr, _i = w.step(np.zeros(2))
    assert not term, "off-road must not end the episode when not fatal"

    mod2 = _reload(monkeypatch, EXPLORE_OFF_ROAD_PENALTY=0,
                   EXPLORE_OFF_ROAD_FATAL=0, EXPLORE_GOAL_DIST_M=0)
    w2 = mod2.GatedReward(_StubEnv(dict(off_road=True)),
                          force_mode=mod2.EXPLORE, max_steps=9, seed=0)
    w2.reset()
    _o, r_base, _t, _tr, _i = w2.step(np.zeros(2))
    assert r_off == pytest.approx(r_base - 1.5)
    _reload(monkeypatch, EXPLORE_OFF_ROAD_PENALTY=0, EXPLORE_OFF_ROAD_FATAL=1,
            EXPLORE_GOAL_DIST_M=100)


def test_on_road_steps_are_not_charged(monkeypatch):
    mod = _reload(monkeypatch, EXPLORE_OFF_ROAD_PENALTY=1.5,
                  EXPLORE_OFF_ROAD_FATAL=0, EXPLORE_GOAL_DIST_M=0)
    w = mod.GatedReward(_StubEnv(dict(off_road=False)), force_mode=mod.EXPLORE,
                        max_steps=9, seed=0)
    w.reset()
    _o, r_on, _t, _tr, _i = w.step(np.zeros(2))
    mod2 = _reload(monkeypatch, EXPLORE_OFF_ROAD_PENALTY=0,
                   EXPLORE_OFF_ROAD_FATAL=0, EXPLORE_GOAL_DIST_M=0)
    w2 = mod2.GatedReward(_StubEnv(dict(off_road=False)),
                          force_mode=mod2.EXPLORE, max_steps=9, seed=0)
    w2.reset()
    _o, r_base, _t, _tr, _i = w2.step(np.zeros(2))
    assert r_on == pytest.approx(r_base)
    _reload(monkeypatch, EXPLORE_OFF_ROAD_PENALTY=0, EXPLORE_OFF_ROAD_FATAL=1,
            EXPLORE_GOAL_DIST_M=100)


def test_default_is_off_so_existing_runs_are_unchanged():
    import importlib
    import vae_nav.gated_reward as mod
    importlib.reload(mod)
    assert mod.EXPLORE_OFF_ROAD_PENALTY == 0.0


def test_cause_is_not_blamed_on_off_road_when_off_road_is_not_fatal(monkeypatch):
    """With EXPLORE_OFF_ROAD_FATAL=0 a crossing is the cause, even off road.

    The cause branch tested info["off_road"] before the crossing, so an episode
    that ended on a painted line WHILE off the surface was labelled off_road --
    a cause that, with the terminal disabled, cannot end anything.
    """
    mod = _reload(monkeypatch, EXPLORE_OFF_ROAD_PENALTY=1.0,
                  EXPLORE_OFF_ROAD_FATAL=0, EXPLORE_GOAL_DIST_M=0)
    w = mod.GatedReward(
        _StubEnv(dict(off_road=True, lane_any=True,
                      lane_marking_types=["Broken"], in_junction=False)),
        force_mode=mod.EXPLORE, max_steps=9, seed=0)
    w.reset()
    _o, _r, term, _tr, info = w.step(np.zeros(2))
    assert term, "a painted crossing is still fatal"
    assert info.get("explore_end") == "crossing", info.get("explore_end")
    _reload(monkeypatch, EXPLORE_OFF_ROAD_PENALTY=0, EXPLORE_OFF_ROAD_FATAL=1,
            EXPLORE_GOAL_DIST_M=100)


# --- goal-mode per-step driving penalties ---------------------------------

def test_goal_off_road_and_lane_penalties_are_per_step(monkeypatch):
    """Both ride on top of the -1/step clock, and default to off."""
    from vae_nav.gated_reward import goal_reward_sorb
    base, _t, _tr, _a = goal_reward_sorb(dict(), t=1, max_steps=750)
    assert base == pytest.approx(-1.0)

    mod = _reload(monkeypatch, GOAL_OFF_ROAD_PENALTY=0.5, GOAL_LANE_PENALTY=0.25)
    off, _t, _tr, _a = mod.goal_reward_sorb(dict(off_road=True), t=1,
                                            max_steps=750)
    lane, _t, _tr, _a = mod.goal_reward_sorb(dict(lane_any=True), t=1,
                                             max_steps=750)
    both, _t, _tr, _a = mod.goal_reward_sorb(dict(off_road=True, lane_any=True),
                                             t=1, max_steps=750)
    assert off == pytest.approx(-1.5)
    assert lane == pytest.approx(-1.25)
    assert both == pytest.approx(-1.75)
    _reload(monkeypatch, GOAL_OFF_ROAD_PENALTY=0, GOAL_LANE_PENALTY=0)


def test_goal_penalties_never_touch_the_exact_terminal_bills(monkeypatch):
    """The crash/wedge bill pays the remaining clock EXACTLY, or ending early
    becomes an escape. A driving penalty must not change it."""
    mod = _reload(monkeypatch, GOAL_OFF_ROAD_PENALTY=5.0, GOAL_LANE_PENALTY=5.0)
    # collision_terminal passed explicitly: it is read from the environment at
    # import and the test environment does not set COLLISION_TERMINAL.
    r, term, _tr, _a = mod.goal_reward_sorb(
        dict(collision=True, off_road=True, lane_any=True), t=10,
        max_steps=750, collision_terminal=True)
    assert term
    assert r == pytest.approx(-(1 + (750 - 10)))
    _reload(monkeypatch, GOAL_OFF_ROAD_PENALTY=0, GOAL_LANE_PENALTY=0)


# --- option A: nothing but arrival terminates goal mode --------------------

def test_goal_mode_option_a_only_arrival_terminates(monkeypatch):
    """COLLISION_TERMINAL=0 WEDGE_TRUNCATE=0 GATED_GOAL_LANE_TERMINAL=0.

    User decision 2026-09-11: remove the crash bill, keep the step cap. With
    every failure terminal switched off there is no bill to pay, and crashing
    cannot 'stop the meter' -- the car keeps paying -1/step, which is the only
    thing that made billing necessary in the first place.
    """
    mod = _reload(monkeypatch, COLLISION_TERMINAL=0, WEDGE_TRUNCATE=0,
                  GATED_GOAL_LANE_TERMINAL=0, GOAL_OFF_ROAD_PENALTY=1.0,
                  GOAL_LANE_PENALTY=1.0)
    bad = dict(collision=True, wedged=True, off_road=True, lane_any=True,
               lane_marking_types=["Broken"], in_junction=False)
    r, term, trunc, arrived = mod.goal_reward_sorb(bad, t=10, max_steps=750)
    assert not term and not trunc and not arrived
    # -1 clock, -1 off road, -1 lane invasion. NO 740-step bill.
    assert r == pytest.approx(-3.0)

    # the cap still truncates
    _r, term, trunc, _a = mod.goal_reward_sorb(dict(), t=750, max_steps=750)
    assert trunc and not term

    # and arrival still ends it at zero
    _r, term, _tr, arrived = mod.goal_reward_sorb(dict(is_success=True), t=5,
                                                  max_steps=750)
    assert term and arrived and _r == 0.0
    _reload(monkeypatch, COLLISION_TERMINAL=1, WEDGE_TRUNCATE=1,
            GATED_GOAL_LANE_TERMINAL=1, GOAL_OFF_ROAD_PENALTY=0,
            GOAL_LANE_PENALTY=0)
