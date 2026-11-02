"""Topological image maps from an exploration trace (user 2026-11-02: map arms
for the v30 benchmark -- uniform, real-time entropy, random and SGM placement;
true-pose localisation; report all episodes and the routable subset).

A NODE is a trace sample kept as a waypoint: its true pose and the frame the
policy saw there, which becomes the policy's goal image when the node is a
subgoal. Nodes are DIRECTIONAL: the two sides of a CARLA road are different
lanes and a frame faces one way, so a sample joins a node only within the merge
radius AND within HEADING_TOL of its heading.

EDGES are where the traced car actually drove from one node to the next
(consecutive samples, carried across a segment boundary only when the restart
continues the drive -- not after a stuck/stall jump), so every edge is a
traversal the policy demonstrably made. Weighted by metres driven.
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from vae_nav import image_map as im  # noqa: E402


def _straight(n=60, step=2.0, y=0.0, yaw=0.0, seg=0, entropy=None):
    x = np.arange(n) * step
    return {"pos": np.stack([x, np.full(n, y)], 1),
            "yaw": np.full(n, yaw),
            "entropy": np.zeros(n) if entropy is None else np.asarray(entropy, float),
            "seg": np.full(n, seg)}


def _cat(*parts):
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


# ---- placement ---------------------------------------------------------------

def test_uniform_places_a_node_every_spacing_metres():
    tr = _straight(n=51)                         # 0..100 m
    nodes = im.place_uniform(tr, spacing=10.0)
    xs = sorted(tr["pos"][i, 0] for i in nodes)
    assert len(nodes) == 11 and xs[0] == 0.0 and xs[-1] == 100.0
    assert np.allclose(np.diff(xs), 10.0)


def test_uniform_does_not_duplicate_on_a_revisit_in_the_same_direction():
    tr = _cat(_straight(n=26, seg=0), _straight(n=26, seg=1))   # drive the road twice
    assert len(im.place_uniform(tr, spacing=10.0)) == 6


def test_the_opposite_direction_gets_its_own_nodes():
    back = _straight(n=26, seg=1, yaw=np.pi)
    back["pos"] = back["pos"][::-1].copy()
    tr = _cat(_straight(n=26, seg=0), back)
    assert len(im.place_uniform(tr, spacing=10.0)) == 12


def test_random_matches_the_requested_count_and_is_seeded():
    tr = _straight(n=100)
    a = im.place_random(tr, count=7, seed=3)
    b = im.place_random(tr, count=7, seed=3)
    assert len(a) == 7 and list(a) == list(b)


def test_entropy_places_nodes_where_entropy_spikes():
    e = np.zeros(100)
    e[[30, 60, 90]] = 5.0
    e += np.random.default_rng(0).normal(0, 0.05, 100)
    tr = _straight(n=100, entropy=e)
    nodes = im.place_entropy(tr, z=1.0, window_m=110.0, min_gap_m=4.0, max_gap_m=None)
    assert {30, 60, 90} <= set(nodes)


def test_entropy_gap_fill_keeps_the_map_traversable():
    """Subgoals farther apart than the policy can reach would make every route
    through that stretch unroutable in practice: max_gap_m adds spacing nodes."""
    tr = _straight(n=100, entropy=np.zeros(100))
    nodes = im.place_entropy(tr, z=1.0, window_m=110.0, min_gap_m=4.0, max_gap_m=20.0)
    xs = np.sort(tr["pos"][nodes, 0])
    assert np.max(np.diff(xs)) <= 20.0 + 1e-9


# ---- graph -------------------------------------------------------------------

def test_edges_follow_the_driving_direction():
    tr = _straight(n=26)                          # 0 -> 50 m
    g = im.build_graph(tr, im.place_uniform(tr, spacing=10.0))
    assert g.path(g.nearest((0.0, 0.0), 0.0), g.nearest((50.0, 0.0), 0.0)) is not None
    assert g.path(g.nearest((50.0, 0.0), 0.0), g.nearest((0.0, 0.0), 0.0)) is None


def test_a_jump_breaks_continuity_between_segments():
    a = _straight(n=11, seg=0)                    # 0 -> 20 m
    b = _straight(n=11, seg=1)
    b["pos"][:, 0] += 100.0                       # restarts 80 m on: a JUMP
    tr = _cat(a, b)
    g = im.build_graph(tr, im.place_uniform(tr, spacing=10.0), jumped_segs={1})
    assert g.path(g.nearest((0.0, 0.0), 0.0), g.nearest((120.0, 0.0), 0.0)) is None


def test_a_continuation_restart_keeps_continuity():
    a = _straight(n=11, seg=0)                    # stops at 20 m
    b = _straight(n=11, seg=1)
    b["pos"][:, 0] += 23.0                        # restarts 3 m on
    tr = _cat(a, b)
    g = im.build_graph(tr, im.place_uniform(tr, spacing=10.0))
    assert g.path(g.nearest((0.0, 0.0), 0.0), g.nearest((43.0, 0.0), 0.0)) is not None


def test_shortest_path_prefers_fewer_metres():
    # a square loop 0,0 -> 40,0 -> 40,40 driven, plus a direct 0,0 -> 40,40 run
    leg1 = _straight(n=21)                                         # (0..40, 0)
    leg2 = {"pos": np.stack([np.full(21, 40.0), np.arange(21) * 2.0], 1),
            "yaw": np.full(21, np.pi / 2), "entropy": np.zeros(21), "seg": np.full(21, 0)}
    tr = _cat(leg1, leg2)
    g = im.build_graph(tr, im.place_uniform(tr, spacing=10.0))
    p = g.path(g.nearest((0.0, 0.0), 0.0), g.nearest((40.0, 40.0), np.pi / 2))
    assert p is not None and g.length(p) == pytest.approx(80.0, abs=4.0)


def test_nearest_respects_heading_and_radius():
    tr = _straight(n=26)
    g = im.build_graph(tr, im.place_uniform(tr, spacing=10.0))
    assert g.nearest((10.5, 0.5), 0.0) is not None
    assert g.nearest((10.5, 0.5), np.pi) is None             # facing the wrong way
    assert g.nearest((10.5, 30.0), 0.0, radius=10.0) is None  # too far


def test_goal_node_is_the_reachable_one_nearest_the_goal():
    """The goal's heading is unknown, so among nodes near the goal position the
    one with the shortest route from the start is chosen."""
    fwd = _straight(n=26, seg=0)
    back = _straight(n=26, seg=1, yaw=np.pi)
    back["pos"] = back["pos"][::-1].copy() + np.array([0.0, 3.5])
    tr = _cat(fwd, back)
    g = im.build_graph(tr, im.place_uniform(tr, spacing=10.0))
    s = g.nearest((0.0, 0.0), 0.0)
    goal = g.goal_node(s, (30.0, 1.5), radius=6.0)
    assert goal is not None and g.nodes[goal]["yaw"] == pytest.approx(0.0)


# ---- navigation ----------------------------------------------------------------

def test_navigator_switches_subgoals_and_hands_over_the_goal():
    tr = _straight(n=26)
    g = im.build_graph(tr, im.place_uniform(tr, spacing=10.0))
    path = g.path(g.nearest((0.0, 0.0), 0.0), g.nearest((50.0, 0.0), 0.0))
    nav = im.Navigator(g, path, switch_m=4.0)
    assert nav.current() == path[1]               # first subgoal = the next node
    changed = nav.update((path and g.nodes[path[1]]["pos"]))
    assert changed and nav.current() == path[2]
    for n in path[2:]:
        nav.update(g.nodes[n]["pos"])
    assert nav.final                              # past the goal node: true goal image


def test_navigator_skips_ahead_when_a_later_node_is_reached_first():
    tr = _straight(n=26)
    g = im.build_graph(tr, im.place_uniform(tr, spacing=10.0))
    path = g.path(g.nearest((0.0, 0.0), 0.0), g.nearest((50.0, 0.0), 0.0))
    nav = im.Navigator(g, path, switch_m=4.0)
    nav.update(g.nodes[path[3]]["pos"])
    assert nav.current() == path[4]


def test_a_trivial_route_goes_straight_to_the_goal_image():
    tr = _straight(n=26)
    g = im.build_graph(tr, im.place_uniform(tr, spacing=10.0))
    s = g.nearest((0.0, 0.0), 0.0)
    nav = im.Navigator(g, [s], switch_m=4.0)
    assert nav.final


# ---- serialisation --------------------------------------------------------------

def test_a_map_round_trips_through_json(tmp_path):
    tr = _straight(n=26)
    g = im.build_graph(tr, im.place_uniform(tr, spacing=10.0))
    p = tmp_path / "m.json"
    g.save(str(p))
    h = im.Graph.load(str(p))
    assert len(h.nodes) == len(g.nodes) and h.edges == g.edges


# ---- SGM ---------------------------------------------------------------------------

def _sgm_line(n=10, step=2.0, behind=0.5):
    """Candidates along one road; d(i->j) = 0.5 * metres ahead, and `behind`
    per metre when j is behind. With behind=0.5 the critic sees neighbours as
    near-interchangeable; with a large value, s can never reach an earlier w,
    and the two-way test (k = w) refuses every merge."""
    xy = np.stack([np.arange(n) * step, np.zeros(n)], 1)
    pairs, q = [], []
    for a in range(n):
        for b in range(n):
            ahead = xy[b, 0] - xy[a, 0]
            pairs.append((a, b))
            q.append(-0.5 * ahead if ahead >= 0 else behind * ahead)
    return np.arange(n) * 3, np.array(pairs), np.array(q), xy, np.zeros(n)


def test_sgm_refuses_to_merge_states_that_cannot_reach_each_other():
    cand, pairs, q, xy, yaw = _sgm_line(behind=100.0)
    keep, _ = im.sgm_place(cand, pairs, q, xy, yaw, gate_m=10.0, cutoff=5.0)
    assert list(keep) == list(cand)


def test_sgm_merges_states_the_critic_cannot_tell_apart():
    cand, pairs, q, xy, yaw = _sgm_line()
    keep, st = im.sgm_place(cand, pairs, q, xy, yaw, gate_m=10.0, cutoff=5.0)
    assert 1 < len(keep) < len(cand) and st["members"] == len(keep)


def test_sgm_with_a_tiny_cutoff_keeps_everything():
    cand, pairs, q, xy, yaw = _sgm_line()
    keep, _ = im.sgm_place(cand, pairs, q, xy, yaw, gate_m=10.0, cutoff=1e-6)
    assert list(keep) == list(cand)


def test_sgm_never_merges_outside_the_gate():
    cand, pairs, q, xy, yaw = _sgm_line()
    keep, _ = im.sgm_place(cand, pairs, q, xy, yaw, gate_m=1.0, cutoff=1e9)
    assert list(keep) == list(cand)
