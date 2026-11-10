"""EXPERT_BIAS and episode returns (user 2026-11-06: "run debugging and see if
the expert behavior scores the best reward").

Both fine-tune critics (SAC run F, IQL run E2) rate "steer more left, less
throttle" above the expert's own action in every recorded expert state. To
test whether the TRAINING REWARD really prefers that, the expert drives the
same seeded episodes with a constant bias added to every EXECUTED action (the
label stays the expert's clean action) and the recorder logs each episode's
return under the env's reward, so biased and unbiased returns can be compared.
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

LOW = np.array([-1.0, 0.0], np.float32)
HIGH = np.array([1.0, 1.0], np.float32)


def _step(a_exp, bias=None, driver="expert", a_pol=None):
    from torch_rl.expert_record import record_step
    return record_step(driver, a_pol, np.asarray(a_exp, np.float32), 0.0,
                       np.random.default_rng(0), LOW, HIGH, bias=bias)


def test_a_bias_shifts_the_executed_action_and_leaves_the_label():
    a_exec, a_label = _step([0.1, 0.5], bias=np.array([-0.1, -0.2], np.float32))
    assert np.allclose(a_exec, [0.0, 0.3])
    assert np.array_equal(a_label, np.array([0.1, 0.5], np.float32))


def test_a_bias_is_clipped_to_the_action_box():
    a_exec, _ = _step([-0.95, 0.1], bias=np.array([-0.1, -0.2], np.float32))
    assert np.allclose(a_exec, [-1.0, 0.0])


def test_no_bias_by_default():
    a_exec, _ = _step([0.1, 0.5])
    assert np.allclose(a_exec, [0.1, 0.5])


def test_the_policy_driver_is_never_biased():
    a_exec, _ = _step([0.1, 0.5], bias=np.array([-0.1, -0.2], np.float32), driver="policy",
                      a_pol=np.array([0.2, 0.4], np.float32))
    assert np.allclose(a_exec, [0.2, 0.4])


def test_the_bias_setting(monkeypatch):
    from torch_rl.expert_record import expert_bias
    monkeypatch.delenv("EXPERT_BIAS", raising=False)
    assert expert_bias() is None
    monkeypatch.setenv("EXPERT_BIAS", "-0.1,0")
    assert np.allclose(expert_bias(), [-0.1, 0.0])
    monkeypatch.setenv("EXPERT_BIAS", "-0.1")
    with pytest.raises(ValueError):
        expert_bias()


def test_episode_returns_are_the_sum_and_the_discounted_sum():
    from torch_rl.expert_record import episode_returns
    total, disc = episode_returns([1.0, 2.0, 3.0], 0.5)
    assert total == 6.0 and disc == pytest.approx(1.0 + 0.5 * 2.0 + 0.25 * 3.0)
    assert episode_returns([], 0.99) == (0.0, 0.0)


def test_the_recorder_wires_the_bias_and_logs_returns():
    s = open(os.path.join(ROOT, "train_carla_sac.py")).read()
    rec = s[s.index("def run_record("):]
    rec = rec[:rec.index("\ndef ", 10)]                      # the whole function
    assert "bias = expert_bias()" in rec
    assert "burst=burst, bias=bias" in rec
    assert 'episode_returns(rec["reward"], RECORD_GAMMA)' in rec
    assert "return={" in rec and '"return": ' in rec
