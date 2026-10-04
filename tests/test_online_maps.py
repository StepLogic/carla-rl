"""The ported placers: online MAD, temporal control, uniform control, edges.

Every rule here came from the Habitat peer 2026-10-03 and each test pins one of
them, because the rules are cheap to get subtly wrong and the resulting map still
looks plausible. The two that cost them real results are the dedupe scope
(against ANY node, not just the last -- 3.6x the nodes otherwise) and matched
node counts (a placer comparison must be about WHERE, never how many).
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

from vae_nav.online_maps import (build_edges, online_mad,  # noqa: E402
                                 temporal_nodes, uniform_nodes)


def _line(n, spacing=2.2, seg=None):
    """n samples along +x at a fixed spacing, all one segment by default."""
    pos = np.c_[np.arange(n) * spacing, np.zeros(n)]
    return pos, (np.zeros(n, int) if seg is None else np.asarray(seg))


def test_window_is_a_travel_distance_not_a_sample_count():
    """The peer's window means ~metres of local context; at our spacing a
    150-SAMPLE port would be a 330 m estimator, a different thing entirely."""
    pos, seg = _line(400)
    h = np.zeros(400)
    _, st = online_mad(h, pos, seg, window_m=110.0)
    assert st["sample_spacing_m"] == pytest.approx(2.2)
    assert st["window_samples"] == 50               # 110 / 2.2
    assert st["window_m"] == pytest.approx(110.0)


def test_a_spike_above_the_rolling_threshold_is_admitted():
    pos, seg = _line(200)
    h = np.zeros(200)
    h[::7] = 0.4                       # texture, so MAD is not degenerate
    h[120] = 50.0                      # the spike
    idx, _ = online_mad(h, pos, seg, window_m=110.0, z=1.0)
    assert 120 in idx


def test_flat_entropy_admits_nothing_and_is_reported_as_degenerate():
    """A TQC actor pinned at its entropy ceiling gives MAD=0. The placer must
    emit nothing and SAY so, not scatter nodes by candidate density."""
    pos, seg = _line(300)
    idx, st = online_mad(np.full(300, 2.838), pos, seg, guarantee=False)
    assert len(idx) == 0
    assert st["degenerate_window_frac"] == pytest.approx(1.0)


def test_nothing_is_admitted_before_the_warmup():
    pos, seg = _line(200)
    h = np.zeros(200)
    h[:5] = 99.0                       # huge, but inside the warmup
    h[9] = 0.3
    idx, _ = online_mad(h, pos, seg, warmup=30, guarantee=False)
    assert not any(i < 30 for i in idx)


def test_the_buffer_resets_at_a_segment_boundary():
    """carry_buffer=0: a new segment starts with no history, so its first
    `warmup` samples cannot admit however extreme they are."""
    pos, _ = _line(120)
    seg = np.r_[np.zeros(60, int), np.ones(60, int)]
    h = np.zeros(120)
    h[::5] = 0.2
    h[62] = 99.0                       # 2 samples into segment 1
    idx, _ = online_mad(h, pos, seg, warmup=30, guarantee=False)
    assert 62 not in idx


def test_a_segment_that_admits_nothing_still_contributes_its_best():
    """guarantee=1. Without it, every segment shorter than the warmup vanishes
    from the map -- 35% of their segments."""
    pos, _ = _line(40)
    seg = np.r_[np.zeros(20, int), np.ones(20, int)]
    h = np.zeros(40)
    h[25] = 7.0                        # best of segment 1, which is too short
    idx, _ = online_mad(h, pos, seg, warmup=30, guarantee=True)
    assert 25 in idx


def test_dedupe_rejects_against_ANY_node_not_just_the_previous_one():
    """THE 3.6x BUG. A revisit passes close to an early node, not the last one."""
    # Out and back along the same line: sample 0 and sample 20 are co-located.
    out = np.c_[np.arange(11) * 2.2, np.zeros(11)]
    back = out[::-1]
    pos = np.r_[out, back]
    seg = np.zeros(len(pos), int)
    h = np.zeros(len(pos))
    h[0] = h[len(pos) - 1] = 9.0       # both ends of the loop are the same place
    idx, _ = online_mad(h, pos, seg, warmup=0, window_m=22.0, dedupe_m=0.8,
                        guarantee=False)
    kept = pos[idx]
    for i in range(len(kept)):
        for j in range(i + 1, len(kept)):
            assert np.linalg.norm(kept[i] - kept[j]) >= 0.8


def test_temporal_matches_the_target_count():
    pos, seg = _line(500)
    for target in (7, 23, 50):
        got = temporal_nodes(pos, seg, n_target=target)
        assert abs(len(got) - target) <= 1, f"target {target}, got {len(got)}"


def test_uniform_matches_the_target_count():
    pos, seg = _line(500)
    for target in (7, 23, 50):
        got = uniform_nodes(pos, seg, n_target=target)
        assert abs(len(got) - target) <= 1, f"target {target}, got {len(got)}"


def test_temporal_is_evenly_spread_rather_than_clustered():
    pos, seg = _line(500)
    got = temporal_nodes(pos, seg, n_target=10)
    gaps = np.diff(got)
    assert gaps.std() < 0.25 * gaps.mean()


def test_seglinks_are_OFF_by_default():
    """A seglink spans a respawn, so it is not a drivable connection. Default
    off (user 2026-10-03); the graph is then one component per segment."""
    pos, _ = _line(30)
    seg = np.r_[np.zeros(10, int), np.ones(10, int), np.full(10, 2)]
    idx = np.array([1, 3, 12, 15, 22])
    edges, kinds = build_edges(idx, pos, seg)
    assert "seglink" not in kinds
    for (a, b) in edges:
        assert seg[idx[a]] == seg[idx[b]], "an edge crossed a respawn boundary"


def test_edges_link_within_a_segment_and_seglink_across():
    pos, _ = _line(30)
    seg = np.r_[np.zeros(10, int), np.ones(10, int), np.full(10, 2)]
    idx = np.array([1, 3, 12, 15, 22])
    edges, kinds = build_edges(idx, pos, seg, seglinks=True)
    # Positions are indices INTO idx, so (0,1) is 1->3.
    assert ("trace", (0, 1)) == (kinds[0], edges[0])
    assert kinds.count("seglink") == 2
    for (a, b), k in zip(edges, kinds):
        if k == "seglink":
            assert seg[idx[a]] != seg[idx[b]]


def test_a_segment_with_no_nodes_does_not_break_the_seglink_chain():
    """Segment 1 contributes nothing; 0 must link straight to 2, not vanish."""
    pos, _ = _line(30)
    seg = np.r_[np.zeros(10, int), np.ones(10, int), np.full(10, 2)]
    idx = np.array([2, 5, 21, 25])
    edges, kinds = build_edges(idx, pos, seg, seglinks=True)
    links = [(a, b) for (a, b), k in zip(edges, kinds) if k == "seglink"]
    assert links == [(1, 2)]           # last of seg 0 -> first of seg 2


def test_ragged_input_is_refused():
    pos, seg = _line(10)
    with pytest.raises(ValueError, match="ragged"):
        online_mad(np.zeros(9), pos, seg)


# ------------------------------------------------------- shortcut candidates --
from vae_nav.online_maps import (shortcut_candidates,  # noqa: E402
                                 trace_edge_cost_m)


def test_opposite_carriageways_are_gated_out_by_heading():
    """THE ROAD CORRECTION. Two nodes 1.5 m apart facing opposite ways are a
    median apart, not a shortcut. Indoors this gate is unnecessary; here it is
    the whole point, and it runs BEFORE the critic is asked."""
    xy = np.array([[0.0, 0.0], [0.0, 1.5]])
    yaw = np.array([0.0, np.pi])                 # 180 degrees apart
    assert shortcut_candidates(xy, yaw) == []


def test_same_heading_within_radius_is_a_candidate_both_ways():
    xy = np.array([[0.0, 0.0], [0.0, 1.5]])
    yaw = np.array([0.0, np.deg2rad(10)])
    assert sorted(shortcut_candidates(xy, yaw)) == [(0, 1), (1, 0)]


def test_pairs_beyond_the_radius_are_not_candidates():
    xy = np.array([[0.0, 0.0], [0.0, 9.0]])
    assert shortcut_candidates(xy, np.zeros(2), radius=3.0) == []


def test_already_connected_pairs_are_skipped_in_both_directions():
    xy = np.array([[0.0, 0.0], [0.0, 1.0], [0.0, 2.0]])
    yaw = np.zeros(3)
    got = shortcut_candidates(xy, yaw, existing_edges=[(0, 1)])
    assert (0, 1) not in got and (1, 0) not in got
    assert (0, 2) in got


def test_heading_gate_wraps_around_the_circle():
    """359 degrees and 1 degree differ by 2, not 358."""
    xy = np.array([[0.0, 0.0], [0.0, 1.0]])
    yaw = np.array([np.deg2rad(359), np.deg2rad(1)])
    assert sorted(shortcut_candidates(xy, yaw)) == [(0, 1), (1, 0)]


def test_trace_edges_cost_driven_metres_not_straight_line():
    """A node pair joined around a corner must cost the way round. Costing it
    straight-line is what makes a wormhole look cheap."""
    # Right-angle path: (0,0) -> (0,10) -> (10,10); nodes at the two ends.
    pos = np.array([[0.0, 0.0], [0.0, 10.0], [10.0, 10.0]])
    seg = np.zeros(3, int)
    costs = trace_edge_cost_m([0, 2], pos, seg)
    assert len(costs) == 1
    (_, c), = costs
    assert c == pytest.approx(20.0)              # driven, not 14.1 straight
