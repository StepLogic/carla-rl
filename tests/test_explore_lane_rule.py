"""Explore mode's lane rule: off-road fatal, NONE crossings exempt, painted fatal.

User decision 2026-09-11, after v4's explore half learned to end its episodes
by crossing a line at step ~37 (median 1.7 m driven at evaluation).
"""
import os

import numpy as np
import pytest

from vae_nav.gated_reward import explore_lane_fatal, explore_reward


def test_off_road_is_fatal_regardless_of_markings():
    assert explore_lane_fatal(False, [], True) is True
    assert explore_lane_fatal(True, ["NONE"], True) is True


def test_no_crossing_is_not_fatal():
    assert explore_lane_fatal(False, [], False) is False


def test_none_crossing_is_exempt():
    assert explore_lane_fatal(True, ["NONE"], False) is False
    assert explore_lane_fatal(True, ["NONE", "NONE"], False) is False
    # the sensor's enum spelling survives the exemption
    assert explore_lane_fatal(True, ["LaneMarkingType.NONE"], False) is False


def test_painted_crossing_on_open_road_is_fatal():
    assert explore_lane_fatal(True, ["Broken"], False) is True
    assert explore_lane_fatal(True, ["NONE", "Broken"], False) is True   # judged as the Broken
    assert explore_lane_fatal(True, ["Solid"], False) is True


def test_painted_crossing_inside_a_junction_is_the_turn():
    assert explore_lane_fatal(True, ["Broken"], False, in_junction=True) is False
    assert explore_lane_fatal(True, ["Solid"], False, in_junction=True) is True


def test_empty_marking_list_fails_closed():
    assert explore_lane_fatal(True, [], False) is True
    assert explore_lane_fatal(True, None, False) is True


def test_goal_mode_rule_is_untouched():
    # goal mode goes through goal_reward_sorb / lane_departure_is_fatal, not
    # this function; pin that the explore helper is not imported there.
    import vae_nav.gated_reward as G
    src = open(G.__file__).read()
    goal_fn = src[src.index("def goal_reward_sorb"):src.index("def goal_reward(")]
    assert "explore_lane_fatal" not in goal_fn


def test_explore_reward_uses_the_rule():
    base = {"wp_forward": (1.0, 0.0), "hero_yaw": 0.0, "speed_factor": 0.5,
            "action_smoothness_penalty": 0.0, "collision": False,
            "in_junction": False, "off_road": False}
    r_none, fatal_none = explore_reward({**base, "lane_any": True,
                                         "lane_marking_types": ["NONE"]},
                                        np.array([0.2, 0.0]))
    r_brk, fatal_brk = explore_reward({**base, "lane_any": True,
                                       "lane_marking_types": ["Broken"]},
                                      np.array([0.2, 0.0]))
    assert fatal_none is False and fatal_brk is True
    assert r_brk < r_none                      # the departure penalty applied
    _r, fatal_off = explore_reward({**base, "off_road": True}, np.array([0.2, 0.0]))
    assert fatal_off is True


def test_penalise_mode_charges_a_crossing_but_does_not_end_the_episode(monkeypatch):
    import vae_nav.gated_reward as G
    monkeypatch.setattr(G, "EXPLORE_LANE_MODE", "penalise")
    monkeypatch.setattr(G, "LANE_INVASION_NOOP", False)
    base = {"wp_forward": (1.0, 0.0), "hero_yaw": 0.0, "speed_factor": 0.5,
            "action_smoothness_penalty": 0.0, "collision": False,
            "in_junction": False, "off_road": False}
    r_ok, f_ok = G.explore_reward({**base, "lane_any": False}, np.array([0.2, 0.0]))
    r_x, f_x = G.explore_reward({**base, "lane_any": True,
                                 "lane_marking_types": ["Broken"]}, np.array([0.2, 0.0]))
    assert f_x is False and f_ok is False                # continues
    assert abs((r_ok - r_x) - G.LANE_DEPARTURE_PENALTY) < 1e-9   # charged
    _r, f_off = G.explore_reward({**base, "off_road": True}, np.array([0.2, 0.0]))
    assert f_off is True                                  # off-road still ends it


def test_terminal_mode_is_v4s_rule(monkeypatch):
    import vae_nav.gated_reward as G
    monkeypatch.setattr(G, "EXPLORE_LANE_MODE", "terminal")
    monkeypatch.setattr(G, "LANE_INVASION_NOOP", False)
    base = {"wp_forward": (1.0, 0.0), "hero_yaw": 0.0, "speed_factor": 0.5,
            "action_smoothness_penalty": 0.0, "collision": False,
            "in_junction": False, "off_road": False}
    _r, f_x = G.explore_reward({**base, "lane_any": True,
                                "lane_marking_types": ["Broken"]}, np.array([0.2, 0.0]))
    assert f_x is True


def test_heading_progress_ignores_the_road_direction(monkeypatch):
    """LANE_PROGRESS_MODE=heading: progress is displacement along the car's own
    yaw; wp_forward (the road) must have no effect (user: remove projection)."""
    import vae_nav.gated_reward as G
    monkeypatch.setattr(G, "PROGRESS_MODE", "heading")
    monkeypatch.setattr(G, "LANE_INVASION_NOOP", False)
    base = {"hero_yaw": 0.0, "speed_factor": 0.5, "action_smoothness_penalty": 0.0,
            "collision": False, "in_junction": False, "off_road": False,
            "lane_any": False}
    disp = np.array([0.3, 0.0])                      # 0.3 m along +x = heading
    r_a, _ = G.explore_reward({**base, "wp_forward": (1.0, 0.0)}, disp)
    r_b, _ = G.explore_reward({**base, "wp_forward": (-1.0, 0.0)}, disp)   # road reversed
    assert abs(r_a - r_b) < 1e-12                    # road direction irrelevant
    assert abs((r_a + G.BASE_PENALTY) - 0.3) < 1e-9  # paid the heading progress
    r_back, _ = G.explore_reward({**base, "wp_forward": (1.0, 0.0)}, -disp)
    assert r_back < r_a                              # reversing is negative progress


def test_progress_scale_multiplies_only_the_progress_term(monkeypatch):
    import vae_nav.gated_reward as G
    monkeypatch.setattr(G, "PROGRESS_MODE", "heading")
    monkeypatch.setattr(G, "LANE_INVASION_NOOP", False)
    base = {"hero_yaw": 0.0, "speed_factor": 0.5, "action_smoothness_penalty": 0.0,
            "collision": False, "in_junction": False, "off_road": False,
            "lane_any": False, "wp_forward": (1.0, 0.0)}
    disp = np.array([0.3, 0.0])
    monkeypatch.setattr(G, "PROGRESS_SCALE", 1.0)
    r1, _ = G.explore_reward(base, disp)
    monkeypatch.setattr(G, "PROGRESS_SCALE", 10.0)
    r10, _ = G.explore_reward(base, disp)
    assert abs((r10 + G.BASE_PENALTY) - 3.0) < 1e-9        # 0.3 m x 10
    assert abs((r1 + G.BASE_PENALTY) - 0.3) < 1e-9
    # penalties are NOT scaled: a terminal crossing still costs exactly 5.0 more
    rx, _ = G.explore_reward({**base, "lane_any": True,
                              "lane_marking_types": ["Broken"]}, disp)
    assert abs((r10 - rx) - G.LANE_DEPARTURE_PENALTY) < 1e-9
