"""v26 (user 2026-10-30): one per-step price in both modes -- metres along the
road towards the episode's target -- with lane invasion ending the episode.

GOAL_ROUTE_ONLY=1. Goal episodes were paid TWICE for driving: explore's forward
progress (any direction the lane goes) plus the route potential towards the
goal (POTENTIAL_K). The first pays a car that drives away from the goal down
the wrong branch. With the switch on, goal mode keeps only the route term.

EXPLORE_JUNCTION_BONUS. Explore episodes have no goal image, so a specific
target junction would be invisible to the policy: at a fork it would be paid
for one branch and charged for the other with nothing in its view to tell
them apart. Instead the target is the next junction ahead, any exit counts:
forward road progress pays every metre towards it, clearing it pays the
bonus, and the next junction becomes the target. Clearing = leaving a junction
after at least EXPLORE_JUNCTION_MIN_STEPS steps inside it, at least
EXPLORE_JUNCTION_MIN_GAP_M of driving after the last paid one, so a waypoint
projection flickering at a junction edge cannot be farmed.

HER relabels explore episodes into goal rows, so the explore-only parts
(forward progress under GOAL_ROUTE_ONLY, the junction bonus) are published as
info["explore_only"] and removed from the relabelled row's base.
"""
import importlib
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))
pytest.importorskip("gymnasium")
import gymnasium as gym                       # noqa: E402
from gymnasium import spaces                  # noqa: E402

STEP_M = 0.3


class Road(gym.Env):
    """Drives STEP_M a step along +x. `junction[t]` says whether step t ends
    inside a junction; `route[t]` is the route distance left after step t."""

    def __init__(self, junction=None, route=None, collide_at=None):
        self.observation_space = spaces.Dict({})
        self.action_space = spaces.Box(np.array([-1.0, 0.0], np.float32),
                                       np.array([1.0, 1.0], np.float32))
        self.junction = list(junction or [])
        self.route = list(route or [])
        self.collide_at = collide_at
        self.t = 0

    def _info(self):
        t = self.t
        inf = {"hero_pos": (STEP_M * t, 0.0), "goal_pos": (500.0, 0.0),
               "hero_yaw": 0.0, "wp_forward": np.array([1.0, 0.0]),
               "speed_kmh": 20.0, "speed_factor": 0.5,
               "action_smoothness_penalty": 0.0, "arrived": False,
               "collision": t > 0 and t == self.collide_at,
               "off_road": False, "lane_any": False, "lane_marking_types": [],
               "in_junction": bool(self.junction[t - 1]) if 0 < t <= len(self.junction) else False}
        if self.route:
            inf["route_remaining_m"] = float(self.route[min(t, len(self.route) - 1)])
        return inf

    def reset(self, **kw):
        self.t = 0
        return np.zeros(4, np.float32), self._info()

    def step(self, action):
        self.t += 1
        return np.zeros(4, np.float32), 0.0, False, False, self._info()


@pytest.fixture(autouse=True)
def _clean_modules():
    """Every test here reloads the reward modules under its own environment;
    reload them again afterwards (monkeypatch has restored the environment by
    then) so the switches do not leak into tests that import them plainly."""
    yield
    import vae_nav.lane_explore_reward as ler
    import vae_nav.gated_reward as gr
    for k in ("GOAL_ROUTE_ONLY", "EXPLORE_JUNCTION_BONUS"):
        os.environ.pop(k, None)
    importlib.reload(ler)
    importlib.reload(gr)


def _gr(monkeypatch, **env):
    base = dict(GATED_GOAL_REWARD="explore", LANE_PROGRESS_MODE="projected",
                LANE_PROGRESS_SCALE=1, POTENTIAL_K=1, GOAL_ROUTE_ONLY=0,
                EXPLORE_JUNCTION_BONUS=0, EXPLORE_JUNCTION_MIN_STEPS=3,
                EXPLORE_JUNCTION_MIN_GAP_M=2, EXPLORE_GOAL_DIST_M=0,
                EXPLORE_IDLE_PENALTY=0, LANE_CENTRE_K=0, LANE_HEADING_K=0,
                STOP_ACTION=0, PROGRESS_SPEED_GATE=0, SHAPING_DISCOUNTED=0)
    base.update(env)
    for k, v in base.items():
        monkeypatch.setenv(k, str(v))
    import vae_nav.lane_explore_reward as ler
    importlib.reload(ler)
    import vae_nav.gated_reward as gr
    return importlib.reload(gr)


def _rewards(gr, env, mode, n):
    w = gr.GatedReward(env, force_mode=mode, max_steps=500, seed=0)
    w.reset()
    out = []
    for _ in range(n):
        o, r, term, trunc, info = w.step(np.zeros(2, np.float32))
        out.append((float(r), term, trunc, dict(info)))
        if term or trunc:
            break
    return out


# ---- goal episodes: route progress only -----------------------------------

def test_goal_mode_drops_the_forward_progress_term(monkeypatch):
    """Route distance flat (driving PAST the goal's branch): the old reward
    still paid 0.3 a step for moving along the lane; route-only pays nothing."""
    flat = [40.0] * 10
    gr = _gr(monkeypatch, GOAL_ROUTE_ONLY=0)
    old = _rewards(gr, Road(route=flat), gr.GOAL, 5)
    gr = _gr(monkeypatch, GOAL_ROUTE_ONLY=1)
    new = _rewards(gr, Road(route=flat), gr.GOAL, 5)
    for (ro, *_), (rn, *_) in zip(old[1:], new[1:]):
        assert ro - rn == pytest.approx(STEP_M)


def test_goal_mode_pays_each_metre_of_route_progress(monkeypatch):
    gr = _gr(monkeypatch, GOAL_ROUTE_ONLY=1)
    route = [40.0 - STEP_M * t for t in range(10)]
    closing = _rewards(gr, Road(route=route), gr.GOAL, 5)
    flat = _rewards(gr, Road(route=[40.0] * 10), gr.GOAL, 5)
    for (rc, *_), (rf, *_) in zip(closing[1:], flat[1:]):
        assert rc - rf == pytest.approx(STEP_M)


def test_route_only_without_a_route_term_is_refused(monkeypatch):
    """No forward progress AND no potential leaves goal mode no positive dense
    term at all -- a run that could only ever learn to stand still."""
    with pytest.raises(ValueError, match="GOAL_ROUTE_ONLY"):
        _gr(monkeypatch, GOAL_ROUTE_ONLY=1, POTENTIAL_K=0)


def test_explore_mode_keeps_forward_progress_under_route_only(monkeypatch):
    gr = _gr(monkeypatch, GOAL_ROUTE_ONLY=0)
    a = _rewards(gr, Road(), gr.EXPLORE, 4)
    gr = _gr(monkeypatch, GOAL_ROUTE_ONLY=1)
    b = _rewards(gr, Road(), gr.EXPLORE, 4)
    assert [r for r, *_ in a] == pytest.approx([r for r, *_ in b])


# ---- explore episodes: a bonus per junction cleared -----------------------

def _bonus_steps(out, base):
    return [i for i, (r, *_rest) in enumerate(out) if r - base[i][0] > 1.0]


def test_clearing_a_junction_pays_the_bonus_once_and_the_episode_goes_on(monkeypatch):
    j = [0, 0, 1, 1, 1, 1, 0, 0, 0, 0]
    gr = _gr(monkeypatch, EXPLORE_JUNCTION_BONUS=10)
    out = _rewards(gr, Road(junction=j), gr.EXPLORE, 10)
    gr0 = _gr(monkeypatch, EXPLORE_JUNCTION_BONUS=0)
    base = _rewards(gr0, Road(junction=j), gr0.EXPLORE, 10)
    assert _bonus_steps(out, base) == [6]                   # the exit step
    assert out[6][0] - base[6][0] == pytest.approx(10.0)
    assert out[6][3]["junction_cleared"] and not out[6][1]  # not terminal
    assert len(out) == 10


def test_a_brush_with_a_junction_edge_pays_nothing(monkeypatch):
    """Two steps inside (< MIN_STEPS 3) is the projection flickering, not a
    junction driven through."""
    j = [0, 1, 1, 0, 0, 0]
    gr = _gr(monkeypatch, EXPLORE_JUNCTION_BONUS=10)
    out = _rewards(gr, Road(junction=j), gr.EXPLORE, 6)
    assert not any(i["junction_cleared"] for *_x, i in out)


def test_back_to_back_exits_inside_the_gap_pay_once(monkeypatch):
    """Gap 2 m = 7 steps at 0.3 m: the second exit comes 4 steps after the
    first and is not paid; one well past the gap is."""
    j = [1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0]
    gr = _gr(monkeypatch, EXPLORE_JUNCTION_BONUS=10)
    out = _rewards(gr, Road(junction=j), gr.EXPLORE, len(j))
    paid = [t for t, (*_x, i) in enumerate(out) if i["junction_cleared"]]
    assert paid == [3, 15]


def test_crashing_inside_a_junction_pays_no_bonus(monkeypatch):
    j = [0, 1, 1, 1, 1, 0]
    gr = _gr(monkeypatch, EXPLORE_JUNCTION_BONUS=10, EXPLORE_COLLISION_FATAL=1)
    out = _rewards(gr, Road(junction=j, collide_at=4), gr.EXPLORE, 6)
    assert out[-1][1] and not any(i.get("junction_cleared") for *_x, i in out)


def test_goal_episodes_never_get_the_junction_bonus(monkeypatch):
    j = [0, 1, 1, 1, 1, 0, 0]
    gr = _gr(monkeypatch, EXPLORE_JUNCTION_BONUS=10, GOAL_ROUTE_ONLY=1)
    out = _rewards(gr, Road(junction=j, route=[40.0] * 10), gr.GOAL, 7)
    assert not any(i.get("junction_cleared") for *_x, i in out)


def test_the_count_resets_between_episodes(monkeypatch):
    gr = _gr(monkeypatch, EXPLORE_JUNCTION_BONUS=10)
    w = gr.GatedReward(Road(junction=[1, 1, 1, 0]), force_mode=gr.EXPLORE,
                       max_steps=500, seed=0)
    w.reset()
    for _ in range(4):
        w.step(np.zeros(2, np.float32))
    w.reset()
    infos = [w.step(np.zeros(2, np.float32))[4] for _ in range(4)]
    assert infos[-1]["junctions_cleared"] == 1


# ---- HER: explore-only parts are not carried into goal rows ---------------

def test_explore_only_is_published_for_her(monkeypatch):
    j = [0, 1, 1, 1, 0]
    gr = _gr(monkeypatch, EXPLORE_JUNCTION_BONUS=10, GOAL_ROUTE_ONLY=1)
    out = _rewards(gr, Road(junction=j), gr.EXPLORE, 5)
    eo = [i["explore_only"] for *_x, i in out]
    assert eo[0] == pytest.approx(STEP_M)                 # progress only
    assert eo[4] == pytest.approx(STEP_M + 10.0)          # + the bonus


def test_her_removes_explore_only_from_the_base(monkeypatch):
    from torch_rl.image_her import ImageGoalHER
    her = ImageGoalHER(None, None, None, goal_bonus=100.0, explore_bonus=10.0)
    her.begin(0, {}, mode=0)
    frame = np.zeros((2, 2, 3), np.uint8)
    her.record(0, np.zeros(3), np.zeros(2), 10.25, False, False, np.zeros(3),
               {"debit_cur": frame},
               {"debit_cur": frame, "explore_only": 10.3, "gated_shaping": 0.0})
    assert her._ep[0]["base"][0] == pytest.approx(-0.05)
