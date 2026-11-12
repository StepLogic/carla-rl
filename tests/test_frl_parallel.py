"""Learner C with its towns stepped IN PARALLEL (user 2026-11-12: "build it", "can we add more towns", "then
deploy on another l40"). One env (and CARLA server) per town, all stepped at once on a thread pool, as
src/vae_nav/vec_env.ThreadedVecEnv does for the gated loop. The learning stays FastRLAP's per transition:
every env step is one transition, and every env step at or past start_training is followed by one update
(UTD 8 on a 256 batch); the entropy target anneals per transition; max_steps, log_every and ckpt_every
count transitions, and checkpoints are named by the multiple they pass (step_25000, ...). Each env keeps
its own GRU state and episode; a failed step abandons that env's episode only."""
import json
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


def _settings(steps, start=100, log_every=1000, ckpt_every=None, **kw):
    return fc.LoopSettings(max_steps=steps, start_training=start, utd=1, batch=2, replay=100,
                           ckpt_every=ckpt_every or steps, log_every=log_every, device="cpu",
                           seed=0, parallel=1, **kw)


def _run(c0_dir, tmp_path, envs, settings, msgs=None, rebuilds=None):
    a = envs[0]
    obs, info = a.reset()                         # main's first reset of env 0
    rebuilds = rebuilds or [None] * len(envs)
    return fc.train_parallel(a, obs, info, str(tmp_path / "run"), trunk=fc.C0Trunk(c0_dir, "cpu"),
                             row_of=_row_of, settings=settings, c0_dir=c0_dir,
                             log=(msgs.append if msgs is not None else (lambda m: None)),
                             rebuild_env=rebuilds[0],
                             extra_envs=list(zip(envs[1:], rebuilds[1:])),
                             towns=[f"T{i}" for i in range(len(envs))])


def test_every_env_steps_every_iteration_and_resets_on_its_own(c0_dir, tmp_path):  # noqa: F811
    a, b = _FakeEnv(), _FakeEnv()
    _, rb, live = _run(c0_dir, tmp_path, [a, b], _settings(20))
    # each env: 3 steps (term), reset, 5 steps (time-out), reset, 2 steps -> 10 iterations = 20 steps
    assert len(a.actions) == 10 and len(b.actions) == 10
    assert a.n_reset == 3 and b.n_reset == 3
    assert len(rb) == 20 and live == [a, b]


def test_one_update_per_transition_once_training_starts(c0_dir, tmp_path, monkeypatch):  # noqa: F811
    from torch_rl.frl.learner import SACLearner
    calls = []
    real = SACLearner.update
    monkeypatch.setattr(SACLearner, "update",
                        lambda self, *a, **kw: calls.append(1) or real(self, *a, **kw))
    a, b, c = _FakeEnv(), _FakeEnv(), _FakeEnv()
    learner, _, _ = _run(c0_dir, tmp_path, [a, b, c], _settings(12, start=5))
    assert len(calls) == 12 - 5 + 1                          # steps 5..12, one update each
    assert learner.target_entropy == pytest.approx(-1.0 - 12 * 25e-7)   # annealed per transition


class _FailsOnce(_FakeEnv):
    def step(self, act):
        if len(self.actions) == 1 and not getattr(self, "failed", False):
            self.failed = True
            raise RuntimeError("A sensor took too long to send their data")
        return super().step(act)


def test_a_failed_step_abandons_only_that_envs_episode(c0_dir, tmp_path):  # noqa: F811
    a, b = _FakeEnv(), _FailsOnce()
    msgs = []
    _, rb, _ = _run(c0_dir, tmp_path, [a, b], _settings(6), msgs=msgs)
    # 3 iterations: b's 2nd step fails (nothing stored, b reset), a is untouched
    assert len(a.actions) == 3 and a.n_reset == 1 + 1        # a: 3 steps -> term -> reset
    assert len(b.actions) == 2 and b.n_reset == 1 + 1        # b: start reset + reset after the fault
    assert len(rb) == 6 - 1
    assert any("[Env] step" in m and "(T1)" in m for m in msgs)


def test_checkpoints_are_named_by_the_multiple_they_pass(c0_dir, tmp_path):  # noqa: F811
    envs = [_FakeEnv() for _ in range(3)]
    msgs = []
    _run(c0_dir, tmp_path, envs, _settings(20, log_every=10, ckpt_every=10), msgs=msgs)
    out = tmp_path / "run"
    for step in (10, 20):                                    # passed at 12 and 21 transitions
        meta = json.load(open(out / f"step_{step}" / "config.json"))
        assert meta["step"] == step
    assert not (out / "step_12").exists()
    assert sum(m.startswith("[FRL] step=10 ") for m in msgs) == 1
    assert any(" | towns T0 " in m for m in msgs)


def test_each_env_rebuilds_on_its_own(c0_dir, tmp_path):  # noqa: F811
    class _ResetDies(_FakeEnv):
        def reset(self, seed=None):
            if self.n_reset >= 1:
                raise RuntimeError("actor registry destroyed")
            return super().reset(seed)

    a, b = _FakeEnv(), _ResetDies()
    b.reset()
    built = []

    def rebuild_b():
        built.append(_FakeEnv())
        return built[-1]

    _, _, live = _run(c0_dir, tmp_path, [a, b], _settings(6), rebuilds=[None, rebuild_b])
    assert len(built) == 1 and live == [a, built[0]]


def test_settings_read_the_parallel_switch():
    assert fc.LoopSettings.from_env({"FRL_PARALLEL": "1"}).parallel == 1
    assert fc.LoopSettings.from_env({}).parallel == 0


def test_run_carla_uses_the_parallel_loop_when_asked(c0_dir, tmp_path, monkeypatch):  # noqa: F811
    for k, v in {"FRL_C0": c0_dir, "FRL_DEVICE": "cpu", "FRL_MAX_STEPS": "8", "FRL_PARALLEL": "1",
                 "FRL_START_TRAINING": "100", "FRL_CKPT_EVERY": "8", "FRL_LOG_EVERY": "100",
                 "FRL_PROBE_CACHE": "", "MONO_CHW": "1", "SPEED_OBS": "1"}.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("JUNCTION_RAMP_START", raising=False)
    monkeypatch.delenv("FRL_EXPERT", raising=False)

    class _Env(_FakeEnv):
        def close(self):
            self.closed = True

    a, b = _Env(), _Env()
    obs, info = a.reset()
    msgs = []
    fc.run_carla(a, obs, info, None, str(tmp_path / "run"), row_of=_row_of, to_env=fc.box_map,
                 log=msgs.append, extra_envs=[(b, None)], towns=("Town05", "Town01"))
    assert any("IN PARALLEL" in m for m in msgs)
    assert a.closed and b.closed and len(a.actions) == 4 and len(b.actions) == 4
