"""Throttle is FORWARD-ONLY: the car's throttle is [0, max_throttle].

User 2026-10-19: "cap throttle between 0 and 1". This replaces the 2026-10-14
mapping, where negative throttle meant reverse.

The actor is a tanh and its raw output reaches compute_action unscaled, so the
env's action space is [0, max] on throttle and the action IS the throttle.
The trainer rescales the actor's [-1, 1] onto that box (to_env) and stores the
actor's own value in replay; warm-up samples drawn from the box are mapped back
(to_actor) so replay holds one distribution.

The tests use a stub for carla.VehicleControl when no client is importable.
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
def test_the_env_throttle_box_is_zero_to_max(cls):
    sp = _space(cls)
    assert float(sp.low[1]) == pytest.approx(0.0)
    assert float(sp.high[1]) == pytest.approx(1.0)


# --------------------------------------------------------------- the mapping --
def test_the_action_is_the_throttle(cls):
    for a in (0.0, 0.25, 0.5, 1.0):
        c = _control(cls, 0.0, a)
        assert c.throttle == pytest.approx(a) and c.reverse is False


def test_never_reverse_never_brake_and_always_in_range(cls):
    for a in np.linspace(-1.5, 1.5, 31):
        c = _control(cls, 0.0, float(a))
        assert c.reverse is False and c.brake == pytest.approx(0.0), a
        assert 0.0 <= c.throttle <= 1.0, a


def test_steer_is_untouched_by_this_change(cls):
    for s in (-1.0, -0.3, 0.0, 0.3, 1.0):
        assert _control(cls, s, 0.5).steer == pytest.approx(s)


# ------------------------------------------------- the trainer's rescaling --
class _Box:
    low = np.array([-1.0, 0.0], np.float32)
    high = np.array([1.0, 1.0], np.float32)


def _trainer():
    sys.path.insert(0, ROOT)
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "train_carla_sac", os.path.join(ROOT, "train_carla_sac.py"))
    m = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(m)
    except Exception as e:                               # noqa: BLE001
        pytest.skip(f"trainer not importable here: {e}")
    return m


def test_tanh_extremes_land_on_the_box_edges():
    t = _trainer()
    np.testing.assert_allclose(t.to_env(np.array([-1, -1], np.float32), _Box), [-1, 0])
    np.testing.assert_allclose(t.to_env(np.array([1, 1], np.float32), _Box), [1, 1])
    np.testing.assert_allclose(t.to_env(np.array([0, 0], np.float32), _Box), [0, 0.5])


def test_warmup_samples_round_trip_into_actor_space():
    t = _trainer()
    rng = np.random.default_rng(0)
    e = np.c_[rng.uniform(-1, 1, 50), rng.uniform(0, 1, 50)].astype(np.float32)
    a = t.to_actor(e, _Box)
    assert a.min() >= -1 - 1e-6 and a.max() <= 1 + 1e-6
    np.testing.assert_allclose(t.to_env(a, _Box), e, atol=1e-6)
