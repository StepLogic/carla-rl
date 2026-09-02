"""Acceptance gates for V as a distance metric.

THE PRIMARY METRIC IS RANK CORRELATION, NOT ABSOLUTE ERROR. Dijkstra's path
selection depends only on the RELATIVE ordering of edge costs, so a V that is
uniformly 30% too long still routes correctly. Absolute calibration is
secondary and affects only whether reported costs are meaningful.
"""
import numpy as np
import pytest

from vae_nav.value_validation import (DISTANCE_BINS_M, gate_report,
                                      median_abs_rel_error, per_bin_spearman,
                                      spearman)


def test_spearman_is_one_for_a_monotone_relationship():
    true = np.arange(1, 21, dtype=float)
    assert spearman(true * 3.0 + 7.0, true) == pytest.approx(1.0)


def test_spearman_ignores_a_constant_scale_error():
    """A V uniformly 30% too long still routes correctly."""
    true = np.arange(1, 21, dtype=float)
    assert spearman(true * 1.3, true) == pytest.approx(1.0)


def test_spearman_is_near_zero_for_noise():
    rng = np.random.default_rng(0)
    true = np.arange(1, 201, dtype=float)
    assert abs(spearman(rng.permutation(true), true)) < 0.3


def test_median_abs_rel_error_of_a_perfect_predictor_is_zero():
    true = np.array([10.0, 50.0, 100.0])
    assert median_abs_rel_error(true, true) == pytest.approx(0.0)


def test_median_abs_rel_error_of_a_uniform_overestimate():
    true = np.array([10.0, 50.0, 100.0])
    assert median_abs_rel_error(true * 1.25, true) == pytest.approx(0.25)


def test_per_bin_reveals_a_predictor_that_fails_only_at_short_range():
    """THE CASE THIS EXISTS FOR.

    A V accurate at 25 m and meaningless at 3 m still scores well pooled,
    while every short edge in the graph is garbage. Only the bins show it.
    """
    rng = np.random.default_rng(1)
    short_d = rng.uniform(2.0, 5.0, 200)
    long_d = rng.uniform(20.0, 30.0, 200)
    dist = np.concatenate([short_d, long_d])
    true = dist / 0.283
    pred = np.concatenate([rng.permutation(short_d / 0.283),   # garbage
                           long_d / 0.283])                     # perfect
    assert spearman(pred, true) > 0.8, "pooled score hides the failure"
    bins = per_bin_spearman(pred, true, dist)
    assert bins["2.0-5.0m"]["rho"] < 0.5
    assert bins["20.0-30.0m"]["rho"] > 0.9


def test_per_bin_reports_counts_and_skips_empty_bins():
    dist = np.array([3.0, 3.5, 4.0, 25.0, 26.0, 27.0])
    true = dist / 0.283
    out = per_bin_spearman(true, true, dist)
    assert out["2.0-5.0m"]["n"] == 3
    assert "5.0-10.0m" not in out, "empty bins must not report a rho"


def test_gate_report_passes_a_good_predictor():
    rng = np.random.default_rng(2)
    dist = rng.uniform(2.0, 30.0, 400)
    true = dist / 0.283
    pred = true * rng.uniform(0.95, 1.05, dist.size)
    v = -0.1 * pred
    rep = gate_report(pred, true, dist, v)
    assert rep["passed"] is True
    assert rep["gates"]["v_in_bounds"]["passed"] is True


def test_gate_report_fails_when_v_leaves_the_achievable_range():
    """This is the failure that actually happened: q = -280 against a -200 floor."""
    dist = np.array([10.0, 20.0])
    true = dist / 0.283
    v = np.array([-5.0, -45.0])           # -45 is outside [-30, 0]
    rep = gate_report(true, true, dist, v)
    assert rep["gates"]["v_in_bounds"]["passed"] is False
    assert rep["passed"] is False


def test_gate_report_fails_on_poor_rank_correlation():
    rng = np.random.default_rng(3)
    dist = rng.uniform(2.0, 30.0, 300)
    true = dist / 0.283
    pred = rng.permutation(true)
    rep = gate_report(pred, true, dist, -0.1 * pred)
    assert rep["gates"]["spearman"]["passed"] is False
    assert rep["passed"] is False


def test_gate_report_is_json_serialisable():
    import json
    dist = np.array([3.0, 10.0, 25.0])
    true = dist / 0.283
    json.dumps(gate_report(true, true, dist, -0.1 * true))


def test_collect_pairs_keeps_only_successful_episodes():
    """Realised steps exist only where the goal was reached.

    An episode that timed out has no ground-truth distance -- counting its
    length as the true distance would teach the gate that failures are simply
    long journeys.
    """
    from validate_value_distance import pairs_from_episodes

    episodes = [
        {"arrived": True, "steps": 80, "goal_dist_m": 22.0, "v0": -8.0},
        {"arrived": False, "steps": 300, "goal_dist_m": 28.0, "v0": -30.0},
        {"arrived": True, "steps": 30, "goal_dist_m": 8.0, "v0": -3.0},
    ]
    pred, true, dist, v = pairs_from_episodes(episodes, step_cost=0.1)
    assert list(true) == [80.0, 30.0]
    assert list(dist) == [22.0, 8.0]
    np.testing.assert_allclose(pred, [80.0, 30.0], atol=1e-6)


def test_collect_pairs_returns_empty_arrays_when_nothing_arrived():
    from validate_value_distance import pairs_from_episodes

    pred, true, dist, v = pairs_from_episodes(
        [{"arrived": False, "steps": 300, "goal_dist_m": 20.0, "v0": -30.0}],
        step_cost=0.1)
    assert len(pred) == len(true) == len(dist) == len(v) == 0
