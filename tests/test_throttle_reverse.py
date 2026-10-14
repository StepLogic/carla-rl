"""Throttle spans [-max, +max] and negative means REVERSE, not brake.

User 2026-10-14, after watching a debugging video: "change action space of
throttle to -1,1 and -1 maps to reverse".

WHY IT MATTERED. The space was [0, max_throttle] while compute_action read a
negative command as BRAKE. The actor is a tanh and emits [-1, 1] on both
dimensions, so about half its probability mass was braking -- and a live run
showed exactly that: across a 1000-step episode the camera barely changed (mean
frame-to-frame difference 1.6, against 4-8 in episodes where the car moved), the
throttle command read -0.76, +0.09, -0.32, and the episode ran to the step cap
accumulating -0.63 a step. The policy was not failing to learn; it was holding
the brake.

The tests use a stub for carla.VehicleControl so they need no simulator.
"""
import os
import sys
import types

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class _Control:
    def __init__(self):
        self.steer = 0.0
        self.throttle = 0.0
        self.brake = 0.0
        self.reverse = False
        self.hand_brake = False


class _Exp:
    """The two methods under test, lifted onto a stub with the same attributes
    the real experiment sets. Reading them from the source keeps the test honest:
    it exercises the shipped code, not a paraphrase."""

    max_steer = 1.0
    max_throttle = 1.0
    incremental_actions = False
    steer = 0.0
    throttle = 0.0
    last_action = None


def _load():
    """Pull get_action_space and compute_action off the real class."""
    sys.path.insert(0, ROOT)
    sys.path.insert(0, os.path.join(ROOT, "src"))
    # Use the REAL carla if it is importable -- a hand-made stub shadowed it and
    # then the env config failed on carla.Location, which is a test artefact
    # rather than a finding. Only fall back to a stub when there is no client.
    for extra in ("/home/robotlab/Apps/CARLA_0.9.15/PythonAPI/carla",):
        if os.path.isdir(extra) and extra not in sys.path:
            sys.path.append(extra)
    try:
        import carla  # noqa: F401
    except Exception:                                    # noqa: BLE001
        carla = types.ModuleType("carla")
        carla.VehicleControl = _Control
        sys.modules["carla"] = carla
    try:
        from src.configs.train_env_config_v4_image_goal import STBL3Experiment
    except Exception as e:                               # noqa: BLE001
        pytest.skip(f"env config not importable here: {e}")
    return STBL3Experiment


@pytest.fixture(scope="module")
def cls():
    return _load()


def _space(cls):
    return cls.get_action_space(_Exp())


def _control(cls, steer, throttle):
    return cls.compute_action(_Exp(), np.array([steer, throttle], np.float32), None)


# ----------------------------------------------------------------- the space --
def test_throttle_spans_negative_to_positive(cls):
    sp = _space(cls)
    assert float(sp.low[1]) == pytest.approx(-1.0)
    assert float(sp.high[1]) == pytest.approx(1.0)


def test_the_space_is_symmetric_like_the_actor_that_feeds_it(cls):
    """A tanh actor is symmetric about zero; an asymmetric space wastes half of
    it or, as here, gives that half a different meaning."""
    sp = _space(cls)
    assert float(sp.low[1]) == pytest.approx(-float(sp.high[1]))


# --------------------------------------------------------------- the mapping --
def test_minus_one_is_full_reverse(cls):
    c = _control(cls, 0.0, -1.0)
    assert c.reverse is True
    assert c.throttle == pytest.approx(1.0)
    assert c.brake == pytest.approx(0.0)


def test_plus_one_is_full_forward(cls):
    c = _control(cls, 0.0, 1.0)
    assert c.reverse is False
    assert c.throttle == pytest.approx(1.0)


def test_zero_is_neither(cls):
    c = _control(cls, 0.0, 0.0)
    assert c.throttle == pytest.approx(0.0) and c.reverse is False


def test_nothing_brakes_any_more(cls):
    """The old mapping turned the actor's negative half into braking, which is
    what parked the car. There is no brake under this mapping at all."""
    for a in (-1.0, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0):
        assert _control(cls, 0.0, a).brake == pytest.approx(0.0), a


def test_the_magnitude_is_the_throttle_in_both_directions(cls):
    """Symmetry in the thing that actually reaches the car."""
    for m in (0.25, 0.5, 0.75, 1.0):
        fwd, rev = _control(cls, 0.0, m), _control(cls, 0.0, -m)
        assert fwd.throttle == pytest.approx(rev.throttle) == pytest.approx(m)
        assert fwd.reverse is False and rev.reverse is True


def test_the_throttle_is_capped_by_max_throttle(cls):
    class _Half(_Exp):
        max_throttle = 0.4
    c = cls.compute_action(_Half(), np.array([0.0, -1.0], np.float32), None)
    assert c.throttle == pytest.approx(0.4) and c.reverse is True


def test_the_recorded_throttle_keeps_its_SIGN(cls):
    """Anything reading it back -- a log, a video overlay -- should see the
    commanded direction, not just how hard."""
    e = _Exp()
    cls.compute_action(e, np.array([0.0, -0.6], np.float32), None)
    assert e.throttle == pytest.approx(-0.6)
    cls.compute_action(e, np.array([0.0, 0.6], np.float32), None)
    assert e.throttle == pytest.approx(0.6)


def test_steer_is_untouched_by_this_change(cls):
    for s in (-1.0, -0.3, 0.0, 0.3, 1.0):
        assert _control(cls, s, 0.5).steer == pytest.approx(s)
