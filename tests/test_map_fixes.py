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
