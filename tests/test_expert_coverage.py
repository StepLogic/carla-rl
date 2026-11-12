"""Coverage recording (user 2026-11-06: "PERTURB EXPERT MORE SO WE HAVE COVERAGE
DATA").

The recorded expert data only ever showed the critic small i.i.d. steering
noise (DART sigma 0 / 0.1 / 0.2, steering only), so it had no evidence about
the sustained hard steering the collapsed fine-tunes drifted into. Coverage
recording adds, all off by default:
  EXPERT_SIGMAS            the steering noise levels cycled per seed
  EXPERT_THROTTLE_SIGMA    i.i.d. noise on the throttle too
  EXPERT_BURST_*           temporally correlated bursts: with probability p per
                           step, one constant (steer, throttle) offset held for
                           k steps while the expert keeps driving
The LABEL is always the expert's clean action (DART); what is executed -- and
stored as act_exec for the critic -- carries the perturbation.
"""
import importlib
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

LOW = np.array([-1.0, 0.0], np.float32)
HIGH = np.array([1.0, 1.0], np.float32)


def test_the_noise_levels_come_from_the_setting(monkeypatch):
    import torch_rl.expert_record as er
    monkeypatch.setenv("EXPERT_SIGMAS", "0.2,0.4")
    er = importlib.reload(er)
    try:
        assert er.SIGMAS == (0.2, 0.4)
        assert {er.episode_plan(s)["sigma"] for s in range(8)} == {0.2, 0.4}
    finally:
        monkeypatch.delenv("EXPERT_SIGMAS")
        importlib.reload(er)
    assert importlib.import_module("torch_rl.expert_record").SIGMAS == (0.0, 0.1, 0.2)


def test_throttle_noise_only_when_asked_and_inside_the_box():
    from torch_rl.expert_record import dart_action
    a = np.array([0.1, 0.5], np.float32)
    out = dart_action(a, 0.0, np.random.default_rng(0), LOW, HIGH)
    assert out[1] == 0.5                                     # unchanged by default
    outs = [dart_action(a, 0.0, np.random.default_rng(s), LOW, HIGH, throttle_sigma=0.5)
            for s in range(50)]
    assert any(o[1] != 0.5 for o in outs)
    assert all(LOW[1] <= o[1] <= HIGH[1] for o in outs)


def test_bursts_are_off_by_default():
    from torch_rl.expert_record import BurstPerturber
    b = BurstPerturber(prob=0.0)
    a = np.array([0.1, 0.5], np.float32)
    assert np.array_equal(b.apply(a, np.random.default_rng(0), LOW, HIGH), a)


def test_a_burst_holds_one_offset_for_its_whole_length():
    from torch_rl.expert_record import BurstPerturber
    b = BurstPerturber(prob=1.0, steps=(4, 4), steer=(0.5, 0.5), throttle=(0.2, 0.2))
    rng = np.random.default_rng(3)
    a = np.array([0.0, 0.5], np.float32)
    outs = [b.apply(a, rng, LOW, HIGH) for _ in range(4)]
    assert all(np.array_equal(o, outs[0]) for o in outs)     # one offset, 4 steps
    assert abs(abs(float(outs[0][0])) - 0.5) < 1e-6          # |steer offset| 0.5
    assert abs(abs(float(outs[0][1]) - 0.5) - 0.2) < 1e-6    # |throttle offset| 0.2


def test_a_burst_is_clipped_to_the_action_box():
    from torch_rl.expert_record import BurstPerturber
    b = BurstPerturber(prob=1.0, steps=(2, 2), steer=(1.0, 1.0), throttle=(1.0, 1.0))
    out = b.apply(np.array([0.9, 0.9], np.float32), np.random.default_rng(0), LOW, HIGH)
    assert np.all(out >= LOW) and np.all(out <= HIGH)


def test_reset_ends_a_running_burst():
    from torch_rl.expert_record import BurstPerturber
    b = BurstPerturber(prob=1.0, steps=(10, 10), steer=(0.5, 0.5), throttle=(0.0, 0.0))
    b.apply(np.zeros(2, np.float32), np.random.default_rng(0), LOW, HIGH)
    b.reset()
    assert b.left == 0


def test_the_label_stays_clean_under_every_perturbation():
    from torch_rl.expert_record import BurstPerturber, record_step
    a_exp = np.array([0.1, 0.5], np.float32)
    b = BurstPerturber(prob=1.0, steps=(3, 3), steer=(0.8, 0.8), throttle=(0.3, 0.3))
    a_exec, a_label = record_step("expert", None, a_exp, 0.4, np.random.default_rng(1),
                                  LOW, HIGH, throttle_sigma=0.3, burst=b)
    assert np.array_equal(a_label, a_exp)
    assert not np.array_equal(a_exec, a_exp)


def test_the_recorder_wires_coverage_and_a_dry_run_never_perturbs():
    s = open(os.path.join(ROOT, "train_carla_sac.py")).read()
    rec = s[s.index("def run_record("):]
    rec = rec[:rec.index("\ndef ", 10)]                      # the whole function
    assert "burst.reset()" in rec                            # per episode
    assert "throttle_sigma=throttle_sigma, burst=burst" in rec
    from train_carla_sac import coverage_perturbation
    ts, burst = coverage_perturbation(dry=True)
    assert ts == 0.0 and burst.prob == 0.0                   # the M0 gate measures the expert
