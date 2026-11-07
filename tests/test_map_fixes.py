"""The Habitat peer's map fixes for CARLA (user 2026-11-06, "proceed"): gap nodes
for every placement and no edge between opposite-heading nodes."""
import importlib.util
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, ROOT)
from vae_nav import image_map as im  # noqa: E402


def _trace(pos, yaw):
    pos = np.asarray(pos, float)
    return {"pos": pos, "yaw": np.asarray(yaw, float), "seg": np.zeros(len(pos), int),
            "jumped": set(), "driver": np.zeros(len(pos), int)}


def _out_and_back():
    # east along y=0, a gradual U-turn (radius 1.75 m), back west along y=3.5
    xs = np.arange(0, 40, 0.5)
    th = np.linspace(-np.pi / 2, np.pi / 2, 12)[1:-1]
    turn = [(40 + 1.75 * np.cos(t), 1.75 + 1.75 * np.sin(t)) for t in th]
    pos = [(x, 0.0) for x in xs] + turn + [(x, 3.5) for x in xs[::-1]]
    yaw = [0.0] * len(xs) + [t + np.pi / 2 for t in th] + [np.pi] * len(xs)
    return _trace(pos, yaw), len(xs), len(turn)


def test_the_opposite_lane_rule_drops_the_u_turn_edge_only():
    tr, n, m = _out_and_back()
    w = n + m                                  # first west-lane frame
    idx = [0, 20, 40, 79, w, w + 40, w + 79]
    loose = im.build_graph(tr, idx)
    strict = im.build_graph(tr, idx, max_turn_deg=90.0)
    turn = (3, 4)                              # last east node -> first west node
    assert turn in loose.edges and turn not in strict.edges
    assert (0, 1) in strict.edges and (4, 5) in strict.edges


def test_gap_nodes_leave_no_driven_frame_without_an_aligned_node_nearby():
    spec = importlib.util.spec_from_file_location(
        "b", os.path.join(ROOT, "scripts", "build_single_trace_maps.py"))
    b = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(b)
    tr, _n, _m = _out_and_back()
    key = [0]                                  # one key node, east lane only
    idx, added = b.with_gaps(key, tr, 8.0)
    assert 0 in idx and added == len(idx) - 1 > 0
    for i in range(len(tr["pos"])):            # every frame, both lanes
        d = np.hypot(*(tr["pos"][idx] - tr["pos"][i]).T)
        ok = (d <= 8.0) & im._aligned(tr["yaw"][idx], tr["yaw"][i])
        assert ok.any(), i
    assert b.with_gaps(key, tr, 0.0) == (np.asarray(key), 0)   # off by default


def test_gap_nodes_are_never_placed_on_expert_recovery_frames():
    spec = importlib.util.spec_from_file_location(
        "b2", os.path.join(ROOT, "scripts", "build_single_trace_maps.py"))
    b = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(b)
    tr, _n, _m = _out_and_back()
    tr["driver"][10:60] = 1
    idx, _ = b.with_gaps([0], tr, 4.0)
    assert not (tr["driver"][idx] == 1).any()


def _line_graph(n=30, step=3.0):
    nodes = [{"pos": (i * step, 0.0), "yaw": 0.0, "idx": i} for i in range(n)]
    edges = {(i, i + 1): step for i in range(n - 1)}
    return im.Graph(nodes, edges, {}), list(range(n))


def test_lookahead_off_keeps_the_old_navigator():
    g, path = _line_graph()
    nav = im.Navigator(g, path, switch_m=4.0)
    assert nav.current() == 1
    assert nav.update((3.0, 0.0), 0.0) and nav.current() == 3    # nodes 1, 2 within 4 m


def test_lookahead_aims_at_the_first_node_that_far_ahead():
    g, path = _line_graph()
    nav = im.Navigator(g, path, switch_m=4.0, lookahead_m=12.0)
    assert nav.look_ahead((0.0, 0.0)) and nav.current() == 4      # the first leg: 12 m
    assert not nav.update((5.0, 0.0), 0.0)                        # 7 m off: hold it
    assert not nav.update((7.9, 0.0), 0.0)                        # 4.1 m: still held
    assert nav.update((8.5, 0.0), 0.0) and nav.current() == 7     # reached: 21 - 8.5 >= 12
    assert abs(nav.subgoal_m((8.5, 0.0)) - 12.5) < 1e-9


def test_lookahead_hands_over_the_true_goal_near_the_end():
    g, path = _line_graph(n=5)                                    # last node at 12 m
    nav = im.Navigator(g, path, switch_m=4.0, lookahead_m=12.0)
    nav.look_ahead((1.0, 0.0))
    assert nav.final and nav.subgoal_m((1.0, 0.0)) is None


def test_off_road_distance_rule_is_off_by_default_and_junction_only():
    # user 2026-11-06 "use distance rule": inside a junction an off-road reading
    # counts only beyond OFF_ROAD_JUNCTION_M from the nearest Driving-lane centre
    src = open(os.path.join(ROOT, "src", "configs", "train_env_config_v4_image_goal.py")).read()
    assert 'OFF_ROAD_JUNCTION_M = float(os.environ.get("OFF_ROAD_JUNCTION_M", 0.0) or 0.0)' in src
    body = src.split("self.off_road = wp is None")[1].split("self.lane_marking_types")[0]
    assert "self.off_road_raw = self.off_road" in body
    assert "if self.off_road and OFF_ROAD_JUNCTION_M > 0:" in body
    assert "_pwp.is_junction" in body and "<= OFF_ROAD_JUNCTION_M" in body
    assert "lane_type=carla.LaneType.Driving" in body
    cs = open(os.path.join(ROOT, "src", "vae_nav", "core_state.py")).read()
    assert 'info["off_road_raw"]' in cs


def test_replan_is_off_by_default_and_relocalises_to_the_same_goal_node():
    src = open(os.path.join(ROOT, "train_carla_sac.py")).read()
    assert 'EVAL_MAP_REPLAN = int(os.environ.get("EVAL_MAP_REPLAN", 0) or 0)' in src
    body = src.split("def run_eval(")[1].split("\ndef ")[0]
    assert "elif nav is not None and EVAL_MAP_REPLAN > 0 and not nav.final:" in body
    assert 'gmap.path(s2, map_row["map_goal_node"])' in body       # same goal node
    assert "nav.k = len(nav.path)" in body                          # else the real goal
    assert "leg_steps, replans, replan_log = 0, 0, []" in body      # reset per episode


def test_lane_graph_can_require_every_junction_turn():
    from vae_nav.coverage_tour import lane_graph

    class W:
        def __init__(self, x):
            class L: pass
            self.transform = type("T", (), {"location": L()})()
            self.transform.location.x, self.transform.location.y = x, 0.0

    class G:
        def __init__(self):
            import networkx as nx
            self._graph = nx.DiGraph()
            T = type("K", (), {"name": "LANEFOLLOW"})
            self._graph.add_edge(0, 1, type=T, entry_waypoint=W(0), exit_waypoint=W(10), path=[], intersection=False)
            self._graph.add_edge(1, 2, type=T, entry_waypoint=W(10), exit_waypoint=W(15), path=[], intersection=True)
    _e, lanes_only, _w = lane_graph(G())
    _e, every_turn, _w = lane_graph(G(), junctions=True)
    assert lanes_only == [(0, 1)] and sorted(every_turn) == [(0, 1), (1, 2)]
