"""ORIGIN_NOT_IN_JUNCTION: the junction-biased origin is never junction interior (user 2026-11-14).

pick_origin lives in rlib_integration, which is not tracked here; RLIB_CARLA_CORE points at a patched copy
(slurm/patch_rlib_origin_not_in_junction.py). The function is loaded from source without importing carla.
"""
import ast
import os
import random

import pytest

PATH = os.environ.get("RLIB_CARLA_CORE", "")
pytestmark = pytest.mark.skipif(not os.path.isfile(PATH), reason="RLIB_CARLA_CORE not set")


def _pick_origin():
    tree = ast.parse(open(PATH).read())
    fns = [n for n in tree.body if isinstance(n, ast.FunctionDef)
           and n.name in ("pick_origin", "straight_segment_origin")]
    ns = {"os": os, "random": random}
    exec(compile(ast.Module(body=fns, type_ignores=[]), PATH, "exec"), ns)
    return ns["pick_origin"]


class WP:
    """A waypoint on a line at 1 m spacing; indices in `junction` are junction interior."""

    def __init__(self, i, junction):
        self.i, self._j = i, junction

    @property
    def is_junction(self):
        return self.i in self._j

    def previous(self, d):
        k = self.i - int(round(d))
        return [WP(k, self._j)] if k >= 0 else []


JUNCTION = set(range(100, 130))          # a 30 m wide junction


def _draws(monkeypatch, on):
    monkeypatch.setenv("ORIGIN_NOT_IN_JUNCTION", "1" if on else "0")
    pick = _pick_origin()
    rng = random.Random(0)
    road = [WP(i, JUNCTION) for i in range(0, 100)]
    inside = [WP(i, JUNCTION) for i in JUNCTION]
    return [pick(road, inside, [5.0, 10.0, 15.0, 20.0, 25.0], 1.0, 0.0, rng=rng) for _ in range(300)]


def test_switch_off_can_spawn_inside_the_junction(monkeypatch):
    assert any(o.is_junction for o, _, kind in _draws(monkeypatch, False) if kind == "junction")


def test_switch_on_never_spawns_inside_the_junction(monkeypatch):
    out = _draws(monkeypatch, True)
    assert all(kind == "junction" for _, _, kind in out)       # still the junction-biased branch
    assert not any(o.is_junction for o, _, _ in out)
    assert all(o.i >= 100 - 25 - 2 for o, _, _ in out)         # and still just before that junction
