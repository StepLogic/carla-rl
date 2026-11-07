"""One benchmark episode on ViNT's topological map (EVAL_VINT_TOPOMAP): images-only
localisation of start and goal with ViNT's distance head, a planned route,
navigate.py along it, then the REAL goal image for the last leg; an episode with
no route (or start node == goal node) is driven mapless, as the benchmark's
EVAL_MAP_FALLBACK=mapless rule does."""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from torch_rl.vint_topomap import VintTopoEpisode  # noqa: E402


def _img(place):
    return np.full((2, 2, 3), place, np.uint8)


def _tmap(n=6, back=False):
    ctx = np.stack([np.stack([_img(i)] * 6) for i in range(n)])
    edges = [(i, i + 1, 1.0) for i in range(n - 1)]
    if back:
        edges.append((n - 1, 0, 1.0))
    return {"ctx85": ctx, "edges": np.asarray(edges, np.float32), "node_xy": np.zeros((n, 2))}


class Policy:
    """ViNT stand-in: distance = how many places ahead the goal is; behind = far."""

    def _d(self, here, there):
        a = there.astype(float) - here
        return np.where(a >= 0, a, 20.0)

    def dists_to_goals(self, ctx, goals):
        here = float(np.asarray(ctx[-1])[0, 0, 0])
        d = self._d(here, np.asarray(goals)[:, 0, 0, 0])
        return d, np.zeros((len(d), 5, 2))

    def dists_from_contexts(self, ctxs, goal):
        return self._d(np.asarray(ctxs)[:, -1, 0, 0, 0].astype(float),
                       np.full(len(ctxs), float(np.asarray(goal)[0, 0, 0])))


class Driver:
    def __init__(self):
        self.frames, self.calls, self.final_goal = [], [], True

    def push(self, f):
        self.frames.append(f)

    def context(self):
        return self.frames[-6:]

    def control(self, d, w, speed):
        self.calls.append(("control", float(d), self.final_goal))
        return np.zeros(2, np.float32)

    def act_env(self, cur, goal, speed):
        self.push(cur)
        self.calls.append(("goal", int(np.asarray(goal)[0, 0, 0]), self.final_goal))
        return np.zeros(2, np.float32)


def test_it_localises_plans_follows_the_route_then_drives_the_real_goal():
    ep = VintTopoEpisode(_tmap(), Policy())
    row = ep.start(_img(1), _img(4))
    assert row["start_node"] == 1 and row["goal_node"] == 4 and row["route_nodes"] == 4
    assert row["fallback"] is None
    drv = Driver()
    for place in (1, 2, 3, 4, 4):
        ep.act(drv, _img(place), _img(4), speed_kmh=10.0)
    kinds = [c[0] for c in drv.calls]
    assert kinds[0] == "control" and kinds[-1] == "goal"
    assert all(c[2] is False for c in drv.calls if c[0] == "control")   # cruise on the route
    assert drv.calls[-1][2] is True                                      # approach the goal
    assert ep.row()["final_at"] is not None


def test_no_route_is_driven_mapless_and_counted():
    ep = VintTopoEpisode(_tmap(back=False), Policy())
    row = ep.start(_img(4), _img(1))                     # the goal lies behind on a one-way chain
    assert row["fallback"] == "no route"
    drv = Driver()
    ep.act(drv, _img(4), _img(1), speed_kmh=5.0)
    assert drv.calls == [("goal", 1, True)]


def test_start_node_equal_to_goal_node_is_mapless():
    ep = VintTopoEpisode(_tmap(), Policy())
    assert ep.start(_img(3), _img(3))["fallback"] == "same node"


def test_a_loop_in_the_map_routes_around():
    ep = VintTopoEpisode(_tmap(back=True), Policy())
    row = ep.start(_img(4), _img(1))
    assert row["fallback"] is None and row["route_nodes"] == 4      # 4 -> 5 -> 0 -> 1


def test_privileged_diagnostics_are_separate_and_labelled():
    tm = _tmap()
    tm["node_xy"] = np.stack([np.arange(6) * 4.0, np.zeros(6)], 1)
    ep = VintTopoEpisode(tm, Policy())
    ep.start(_img(1), _img(4))
    diag = ep.diagnostics(start_xy=(4.5, 0.0), goal_xy=(15.0, 0.0))
    assert diag == {"privileged_start_node_err_m": pytest.approx(0.5),
                    "privileged_goal_node_err_m": pytest.approx(1.0)}
