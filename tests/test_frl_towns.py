"""Learner C in several towns (user 2026-11-11: "start town05 and town01 train"). One env (and one CARLA
server) per town; the loop drives ONE env at a time and moves to the next town at every episode end, so it
stays FastRLAP's loop exactly -- one transition and one update per step, the same replay, the same settings
-- and only the towns change. A failed simulator step abandons the episode and restarts it in the SAME town;
each town's env has its own rebuild. The [FRL] line adds a per-town goal success at its END (log parsers
read the fields before it unchanged)."""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "src")]
os.environ.setdefault("MONO_CHW", "1")
os.environ.setdefault("SPEED_OBS", "1")
torch = pytest.importorskip("torch")

from torch_rl.frl import carla as fc  # noqa: E402
from tests.test_frl_carla import _FakeEnv, _row_of, c0_dir  # noqa: E402,F401  (the tiny c0 fixture)


def _settings(steps, log_every=1000):
    return fc.LoopSettings(max_steps=steps, start_training=100, utd=1, batch=2, replay=100,
                           ckpt_every=steps, log_every=log_every, device="cpu", seed=0)


def test_episodes_alternate_between_the_towns(c0_dir, tmp_path):  # noqa: F811
    a, b = _FakeEnv(), _FakeEnv()
    obs, info = a.reset()                       # main's first reset, town A
    msgs = []
    _, rb, live = fc.train(a, obs, info, str(tmp_path / "run"), trunk=fc.C0Trunk(c0_dir, "cpu"),
                           row_of=_row_of, settings=_settings(20, log_every=20), c0_dir=c0_dir,
                           log=msgs.append, extra_envs=[(b, None)], towns=("Town05", "Town01"))
    # _FakeEnv: odd resets end on step 3 (termination), even resets on step 5 (time-out).
    # A 3 | B 3 | A 5 | B 5 | A 3 | B 1 (the run ends at step 20)
    assert len(a.actions) == 3 + 5 + 3 and len(b.actions) == 3 + 5 + 1
    assert a.n_reset == 3 and b.n_reset == 3
    assert len(rb) == 20
    assert live == [a, b]                       # the caller closes every live env
    line = next(m for m in msgs if m.startswith("[FRL] step=20 "))
    assert " | towns Town05 " in line and " Town01 " in line
    assert line.index("ends=") < line.index(" | towns ")       # the per-town block comes last


def test_one_town_keeps_the_log_line_and_returns_its_env(c0_dir, tmp_path):  # noqa: F811
    a = _FakeEnv()
    obs, info = a.reset()
    msgs = []
    _, _, live = fc.train(a, obs, info, str(tmp_path / "run"), trunk=fc.C0Trunk(c0_dir, "cpu"),
                          row_of=_row_of, settings=_settings(8, log_every=8), c0_dir=c0_dir,
                          log=msgs.append)
    assert live == [a]
    assert not any(" | towns " in m for m in msgs)


class _FailsOnce(_FakeEnv):
    """Its 4th step raises once, as CARLA does on a sensor timeout."""

    def step(self, act):
        if len(self.actions) == 3 and not getattr(self, "failed", False):
            self.failed = True
            raise RuntimeError("A sensor took too long to send their data")
        return super().step(act)


def test_a_failed_step_restarts_the_episode_in_the_same_town(c0_dir, tmp_path):  # noqa: F811
    a, b = _FakeEnv(), _FailsOnce()
    obs, info = a.reset()
    msgs = []
    _, rb, _ = fc.train(a, obs, info, str(tmp_path / "run"), trunk=fc.C0Trunk(c0_dir, "cpu"),
                        row_of=_row_of, settings=_settings(15), c0_dir=c0_dir, log=msgs.append,
                        extra_envs=[(b, None)], towns=("Town05", "Town01"))
    # steps 1-3 A (term) | 4-6 B (term) | 7-11 A (time-out) | 12 B FAILS: nothing stored, B reset
    # again and B drives on -- 13-15 B (its 3rd reset: term) -- then A is reset for the next one
    assert len(b.actions) == 3 + 3 and b.n_reset == 3
    assert len(a.actions) == 3 + 5 and a.n_reset == 3
    assert len(rb) == 14 and any("[Env] step 12 failed" in m for m in msgs)


def test_each_town_rebuilds_its_own_env(c0_dir, tmp_path):  # noqa: F811
    class _ResetDies(_FakeEnv):
        def reset(self, seed=None):
            if self.n_reset >= 1:
                raise RuntimeError("actor registry destroyed")
            return super().reset(seed)

    a, b = _FakeEnv(), _ResetDies()
    b.reset()                                   # its construction-time reset is fine
    built = []

    def rebuild_b():
        built.append(_FakeEnv())
        return built[-1]

    obs, info = a.reset()
    _, _, live = fc.train(a, obs, info, str(tmp_path / "run"), trunk=fc.C0Trunk(c0_dir, "cpu"),
                          row_of=_row_of, settings=_settings(6), c0_dir=c0_dir, log=lambda m: None,
                          rebuild_env=None, extra_envs=[(b, rebuild_b)], towns=("Town05", "Town01"))
    assert len(built) == 1 and live == [a, built[0]]


def test_several_towns_are_accepted_but_the_parallel_loop_is_not():
    ok = {"MONO_CHW": "1", "SPEED_OBS": "1"}
    assert fc.frl_refusals(["Town05", "Town01"], False, ok) == []
    assert any("PARALLEL" in m for m in fc.frl_refusals(["Town05", "Town01"], True, ok))


def test_run_carla_closes_every_env_and_stops_every_server(c0_dir, tmp_path, monkeypatch):  # noqa: F811
    for k, v in {"FRL_C0": c0_dir, "FRL_DEVICE": "cpu", "FRL_MAX_STEPS": "8",
                 "FRL_START_TRAINING": "100", "FRL_CKPT_EVERY": "8", "FRL_LOG_EVERY": "100",
                 "FRL_PROBE_CACHE": "", "MONO_CHW": "1", "SPEED_OBS": "1"}.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("JUNCTION_RAMP_START", raising=False)
    monkeypatch.delenv("FRL_EXPERT", raising=False)

    class _Env(_FakeEnv):
        def close(self):
            self.closed = True

    class _Server:
        def stop(self):
            self.stopped = True

    a, b = _Env(), _Env()
    s0, s1 = _Server(), _Server()
    obs, info = a.reset()
    msgs = []
    fc.run_carla(a, obs, info, s0, str(tmp_path / "run"), row_of=_row_of, to_env=fc.box_map,
                 log=msgs.append, extra_envs=[(b, None)], extra_servers=[s1],
                 towns=("Town05", "Town01"))
    assert a.closed and b.closed and s0.stopped and s1.stopped
    assert any("TOWNS: Town05, Town01" in m for m in msgs)
    assert b.n_reset >= 1                       # town B was driven
