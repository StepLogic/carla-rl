"""TARGET_ENTROPY_UNIT: the target is restated in the UNIT action space.

The policy's log_prob carries the tanh-rescale Jacobian, so a target compared
against it lives in the RESCALED space. SAC's -dim(A) heuristic is defined in
the unit space, and the two differ by J = sum log(0.5 * (high - low)) -- zero
only for the [-1, 1]^d box this stack does not use.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class _Box:
    """Minimal stand-in: _target_entropy reads only .low and .high."""

    def __init__(self, low, high):
        self.low = np.asarray(low, np.float32)
        self.high = np.asarray(high, np.float32)


@pytest.fixture()
def clean_env(monkeypatch):
    for k in ("TARGET_ENTROPY", "TARGET_ENTROPY_UNIT"):
        monkeypatch.delenv(k, raising=False)
    return monkeypatch


def _fn():
    from train_carla_tqc import _target_entropy
    return _target_entropy


def test_default_is_byte_identical(clean_env):
    """Flag off: the shipped -2.0 is returned untouched, whatever the box."""
    box = _Box([-0.5, 0.0], [0.5, 0.4])
    assert _fn()(box) == -2.0
    clean_env.setenv("TARGET_ENTROPY", "-3.5")
    assert _fn()(box) == -3.5


def test_flag_off_ignores_the_box(clean_env):
    """The Jacobian is added ONLY under the flag -- no silent behaviour change."""
    assert _fn()(_Box([-0.5, 0.0], [0.5, 0.4])) == -2.0
    assert _fn()(_Box([-1.0, -1.0], [1.0, 1.0])) == -2.0


def test_this_stacks_box(clean_env):
    """[-0.5, 0.5] x [0, 0.4]: J = log(0.5) + log(0.2) = -2.303."""
    clean_env.setenv("TARGET_ENTROPY_UNIT", "1")
    got = _fn()(_Box([-0.5, 0.0], [0.5, 0.4]))
    assert got == pytest.approx(-2.0 + np.log(0.5) + np.log(0.2), abs=1e-6)
    assert got == pytest.approx(-4.3026, abs=1e-3)


def test_unit_box_is_a_no_op(clean_env):
    """On [-1, 1]^d the Jacobian is 0, so the flag changes nothing. This is why
    the bug was invisible until the action box was tightened."""
    clean_env.setenv("TARGET_ENTROPY_UNIT", "1")
    assert _fn()(_Box([-1.0, -1.0], [1.0, 1.0])) == pytest.approx(-2.0, abs=1e-9)


def test_asymmetric_and_wider_boxes(clean_env):
    """A box WIDER than unit raises the target; the sign must follow the box."""
    clean_env.setenv("TARGET_ENTROPY_UNIT", "1")
    wide = _fn()(_Box([-2.0, -2.0], [2.0, 2.0]))
    assert wide == pytest.approx(-2.0 + 2 * np.log(2.0), abs=1e-6)
    assert wide > -2.0


def test_respects_an_explicit_target(clean_env):
    """TARGET_ENTROPY is the UNIT-space value the flag corrects."""
    clean_env.setenv("TARGET_ENTROPY_UNIT", "1")
    clean_env.setenv("TARGET_ENTROPY", "-1.0")
    assert _fn()(_Box([-0.5, 0.0], [0.5, 0.4])) == pytest.approx(
        -1.0 + np.log(0.5) + np.log(0.2), abs=1e-6)


def test_no_action_space_is_a_no_op(clean_env):
    """The eval path passes no action space and never tunes a temperature."""
    clean_env.setenv("TARGET_ENTROPY_UNIT", "1")
    assert _fn()(None) == -2.0
