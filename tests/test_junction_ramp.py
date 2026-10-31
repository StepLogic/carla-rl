"""JUNCTION RAMP (v27, user 2026-10-30): goals start easy and harden.

v18 reached 0.725 on Town02 with half its goals past a junction; at 0.85 no run
has beaten 0.40. The ramp starts JUNCTION_GOAL_FRAC low and steps it up as
TRAINING goal success rises, and back down when it falls -- v44 lost its run to
a curriculum that could not demote. The goal sampler (carla_core) reads the
variable at every reset, in this process, so changing os.environ is the whole
mechanism.
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from torch_rl.junction_ramp import JunctionRamp  # noqa: E402


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("JUNCTION_GOAL_FRAC", "0.85")


def _ramp(**kw):
    kw = dict(dict(start=0.5, target=0.85, step=0.05, up=0.3, down=0.1, window=10), **kw)
    return JunctionRamp(**kw)


def _feed(r, n, arrived):
    msgs = [r.record(arrived) for _ in range(n)]
    return [m for m in msgs if m]


def test_it_starts_at_start_and_sets_the_sampler():
    r = _ramp()
    assert r.value == pytest.approx(0.5)
    assert os.environ["JUNCTION_GOAL_FRAC"] == "0.50"


def test_promotes_after_a_full_window_at_or_above_up():
    r = _ramp()
    msgs = _feed(r, 7, False) + _feed(r, 3, True)        # 0.30 over 10
    assert r.value == pytest.approx(0.55) and len(msgs) == 1
    assert os.environ["JUNCTION_GOAL_FRAC"] == "0.55"
    assert "0.50 -> 0.55" in msgs[0]


def test_nothing_changes_before_the_window_is_full():
    r = _ramp()
    _feed(r, 9, True)
    assert r.value == pytest.approx(0.5)


def test_the_window_restarts_after_a_change():
    """The episodes that earned a promotion were sampled at the OLD fraction
    and say nothing about the new one."""
    r = _ramp()
    _feed(r, 10, True)
    assert r.value == pytest.approx(0.55)
    _feed(r, 9, True)
    assert r.value == pytest.approx(0.55)
    _feed(r, 1, True)
    assert r.value == pytest.approx(0.60)


def test_demotes_below_down_but_not_under_start():
    r = _ramp()
    _feed(r, 10, True)
    assert r.value == pytest.approx(0.55)
    msgs = _feed(r, 10, False)
    assert r.value == pytest.approx(0.50) and "0.55 -> 0.50" in msgs[0]
    _feed(r, 10, False)
    assert r.value == pytest.approx(0.50)


def test_never_passes_the_target():
    r = _ramp(start=0.80)
    _feed(r, 10, True)
    assert r.value == pytest.approx(0.85)
    _feed(r, 10, True)
    assert r.value == pytest.approx(0.85)


def test_holds_between_down_and_up():
    r = _ramp()
    _feed(r, 8, False)
    _feed(r, 2, True)                                       # 0.20
    assert r.value == pytest.approx(0.5)


def test_target_must_not_be_below_start():
    with pytest.raises(ValueError):
        _ramp(start=0.9, target=0.85)


# ---- wiring into the training loop ---------------------------------------

def test_the_loop_feeds_goal_episodes_to_the_ramp(tmp_path):
    pytest.importorskip("gymnasium")
    pytest.importorskip("torch")
    from torch_rl.episode_buffer import DeviceEpisodeBuffer
    from torch_rl.parallel_loop import run_parallel
    from vae_nav.vec_env import ThreadedVecEnv
    from tests.test_parallel_loop import BINOC, FakeAgent, FakeTown, _row

    class Spy:
        value = 0.5

        def __init__(self):
            self.seen = []

        def record(self, arrived):
            self.seen.append(bool(arrived))
            return "[Ramp] spy" if len(self.seen) == 2 else None

    spy = Spy()
    venv = ThreadedVecEnv([lambda: FakeTown(0, 4, True)], names=["TownA"])
    buf = DeviceEpisodeBuffer(1 + BINOC, 2, 1000, "cpu")
    logs = []
    run_parallel(venv, FakeAgent(), buf, _row, venv.action_space,
                 to_env=lambda a, b: a, to_actor=lambda e, b: e,
                 goal_block=lambda g: np.asarray(g, np.float32).reshape(-1),
                 max_steps=40, learning_starts=0, update_every=100, updates_per=1,
                 chunk_b=1, chunk_t=2, burn_in=1, ckpt_every=100,
                 out_dir=str(tmp_path), log_every=2, log=logs.append,
                 junction_ramp=spy)
    assert len(spy.seen) >= 5                     # FakeTown runs goal mode
    assert "[Ramp] spy" in logs
    ep = [s for s in logs if s.startswith("[Ep")]
    assert ep and "jfrac=0.50" in ep[-1]
