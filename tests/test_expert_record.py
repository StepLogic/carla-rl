"""Pure recording logic (imitation-first plan, Task 4): the held-out guard, the
per-episode mode/noise schedule, recovery noise, which action is executed and
which is the label, and the M0 expert gate."""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from torch_rl.expert_record import (assert_recordable_town, dart_action,  # noqa: E402
                                    episode_plan, gate_report, keep_episode,
                                    record_step)

LOW = np.array([-1.0, 0.0], np.float32)
HIGH = np.array([1.0, 1.0], np.float32)


def test_held_out_towns_refused():
    # Town07 joined them 2026-11-12 (user: it replaces Town03 in the main tables, so it is an
    # evaluation town); Town03 stays held out for the appendix.
    for t in ("Town02", "Town03", "Town07"):
        with pytest.raises(ValueError, match="held out"):
            assert_recordable_town(t)
    assert_recordable_town("Town05")


def test_episode_plan_is_balanced():
    plans = [episode_plan(s) for s in range(600)]
    assert sum(p["mode"] for p in plans) == 300
    assert {p["sigma"] for p in plans} == {0.0, 0.1, 0.2}


def test_dart_noise_touches_steer_only_and_clips():
    rng = np.random.default_rng(0)
    a = np.array([0.95, 0.4], np.float32)
    outs = np.stack([dart_action(a, 0.5, rng, LOW, HIGH) for _ in range(200)])
    assert np.all(outs[:, 1] == np.float32(0.4))            # throttle untouched
    assert outs[:, 0].std() > 0.0                           # steer perturbed
    assert outs[:, 0].max() <= 1.0 and outs[:, 0].min() >= -1.0


def test_dart_zero_sigma_is_identity():
    rng = np.random.default_rng(0)
    a = np.array([0.3, 0.7], np.float32)
    out = dart_action(a, 0.0, rng, LOW, HIGH)
    assert np.array_equal(out, a) and out is not a


def test_record_step_labels_are_always_the_expert():
    rng = np.random.default_rng(0)
    a_exec, a_label = record_step("policy", np.array([0.3, 0.4]), np.array([-0.2, 0.6]),
                                  0.2, rng, LOW, HIGH)
    assert np.allclose(a_exec, [0.3, 0.4]) and np.allclose(a_label, [-0.2, 0.6])
    a_exec, a_label = record_step("expert", None, np.array([-0.2, 0.6]), 0.0, rng, LOW, HIGH)
    assert np.allclose(a_exec, [-0.2, 0.6]) and np.allclose(a_label, [-0.2, 0.6])


def test_keep_episode_drops_only_empty():
    assert not keep_episode(0)
    assert keep_episode(1) and keep_episode(500)


def test_gate_counts_only_fault_ends():
    rep = gate_report([{"end": "reached", "path_m": 20}] * 20,
                      [{"end": "cap", "path_m": 150}] * 14 + [{"end": "crossing", "path_m": 100}])
    assert rep["faults"] == 1 and rep["m_between"] == pytest.approx(2200.0) and rep["pass"]


def test_gate_fails_on_low_arrivals():
    goal = [{"end": "reached", "path_m": 20}] * 18 + [{"end": "off_road", "path_m": 8}] * 2
    rep = gate_report(goal, [{"end": "cap", "path_m": 150}] * 20)
    assert not rep["pass"] and "arrival" in rep["reason"]


def test_gate_needs_enough_explore_driving():
    rep = gate_report([{"end": "reached", "path_m": 20}] * 20,
                      [{"end": "cap", "path_m": 150}] * 6)          # 900 m, no faults
    assert rep["m_between"] == float("inf")
    assert not rep["pass"] and "insufficient" in rep["reason"]


def test_dry_benchmark_runs_the_expert_without_noise():
    """The M0 gate measures the expert itself: in the first diagnostic every
    fault came from a sigma=0.2 noise episode (job 463139)."""
    from torch_rl.expert_record import effective_sigma
    assert effective_sigma(0.2, dry=True) == 0.0
    assert effective_sigma(0.2, dry=False) == 0.2
    assert effective_sigma(0.0, dry=False) == 0.0
