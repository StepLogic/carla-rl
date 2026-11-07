"""scripts/vint/vint_bench_report.py: ViNT on the frozen map benchmark, its own
block (ViNT-native topomap vs ViNT mapless on the same episodes), per town x band
cell: paired SR with exact McNemar on the discordant pairs, the topomap's
fallbacks and localisation diagnostics, and both arms' speed and end mix."""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts", "vint"))

import vint_bench_report as rep  # noqa: E402


def _row(seed, reached, end=None, steps=100, path_m=30.0, shortest=25.0, topo=None):
    r = {"seed": seed, "town": "Town02", "band": "easy", "reached": reached,
         "reached_2m": reached, "end": end or ("reached" if reached else "crossing"),
         "steps": steps, "path_m": path_m, "shortest_m": shortest}
    if topo is not None:
        r["vint_topo"] = topo
    return r


def _topo(fallback=None, s_err=1.0, g_err=1.0, final_at=None, end_idx=None, true_idx=None,
          off_m=0.5):
    t = {"fallback": fallback, "final_at": final_at, "route_idx_end": end_idx,
         "privileged_start_node_err_m": s_err, "privileged_goal_node_err_m": g_err}
    if true_idx is not None:
        t.update({"privileged_end_route_idx": true_idx, "privileged_end_route_err_m": off_m})
    return t


def test_paired_stats_count_discordant_pairs_on_common_seeds_only():
    topo = {1: _row(1, True), 2: _row(2, True), 3: _row(3, False), 4: _row(4, True)}
    mapless = {1: _row(1, True), 2: _row(2, False), 3: _row(3, True), 5: _row(5, True)}
    s = rep.pair_stats(topo, mapless)
    assert s["n"] == 3                                   # seeds 1, 2, 3
    assert s["sr_topo"] == pytest.approx(2 / 3) and s["sr_mapless"] == pytest.approx(2 / 3)
    assert (s["only_topo"], s["only_mapless"]) == (1, 1)
    assert s["p"] == pytest.approx(1.0)


def test_fallbacks_and_localisation_diagnostics():
    rows = {1: _row(1, False, topo=_topo(fallback="no route", s_err=40.0, g_err=2.0)),
            2: _row(2, False, topo=_topo(fallback="same node", s_err=1.0, g_err=1.0)),
            3: _row(3, True, topo=_topo(s_err=2.0, g_err=100.0, final_at=30)),
            4: _row(4, False, topo=_topo(s_err=3.0, g_err=4.0, end_idx=5, true_idx=9,
                                         off_m=12.0))}
    d = rep.topo_diag(rows)
    assert (d["no_route"], d["same_node"]) == (1, 1)
    assert d["start_err_med_m"] == pytest.approx(2.5)
    assert d["start_within_8m"] == pytest.approx(0.75)
    assert d["goal_within_8m"] == pytest.approx(0.75)
    assert d["final_leg"] == pytest.approx(0.25)        # final_at set in 1 of 4
    # failed routed episodes: seed 4 only; it ended 12 m off its route
    assert d["routed_fails"] == 1 and d["off_route_fails"] == 1
    assert d["belief_gap_med_nodes"] == pytest.approx(4.0)


def test_speed_is_pooled_path_over_pooled_time_at_ten_hertz():
    rows = {1: _row(1, True, steps=100, path_m=30.0), 2: _row(2, False, steps=50, path_m=15.0)}
    assert rep.speed_kmh(rows) == pytest.approx(45.0 / 15.0 * 3.6)


def test_end_mix_groups_the_clock_and_other_ends():
    rows = {1: _row(1, True), 2: _row(2, False, end="crossing"),
            3: _row(3, False, end="off_road"), 4: _row(4, False, end="timeout"),
            5: _row(5, False, end="collision")}
    assert rep.end_mix(rows) == {"reached": 1, "crossing": 1, "off_road": 1,
                                 "collision": 1, "clock/other": 1}


def test_near_pass_share_counts_closest_approach_within_2m_at_any_speed():
    rows = {1: dict(_row(1, True), closest_m=0.4), 2: dict(_row(2, False), closest_m=1.6),
            3: dict(_row(3, False), closest_m=7.0), 4: dict(_row(4, False), closest_m=None)}
    assert rep.near_pass_share(rows) == pytest.approx(0.5)


def test_approach_stats_separate_getting_near_from_arriving():
    rows = {1: dict(_row(1, True), closest_m=0.5, speed_at_closest_kmh=6.0),
            2: dict(_row(2, False), closest_m=1.4, speed_at_closest_kmh=12.0),
            3: dict(_row(3, False), closest_m=1.8, speed_at_closest_kmh=10.0),
            4: dict(_row(4, False), closest_m=9.0, speed_at_closest_kmh=15.0)}
    a = rep.approach_stats(rows)
    assert a["near"] == 3 and a["converted"] == 1
    assert a["speed_near_med_kmh"] == pytest.approx(10.0)
    assert a["miss_m_med"] == pytest.approx(1.6)          # near passes that did not arrive
