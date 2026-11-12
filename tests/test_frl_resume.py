"""Resuming a learner-C run from one of its checkpoints in the parallel loop (user 2026-11-12: "if its good -
then resume current run with new code"). A checkpoint holds the whole learner: actor, critic, target critic,
temperature, entropy target and the three optimizers; it does NOT hold the replay. So a resumed run restores
all of that and the step count, starts a fresh replay (the pinned expert rows plus an empty ring for its own),
drives with the restored policy (no random start), and resumes updates once its own rows refill to
start_training -- the size the original run first updated at. Checkpoints continue at the multiples past the
resumed step. A checkpoint made with other learning settings is refused."""
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


def _settings(steps, ckpt_every, resume="", **kw):
    base = dict(max_steps=steps, start_training=5, utd=1, batch=2, replay=100, ckpt_every=ckpt_every,
                log_every=1000, device="cpu", seed=0, parallel=1, resume=resume)
    base.update(kw)
    return fc.LoopSettings(**base)


def _run(c0_dir, out, s, n=2):
    envs = [_FakeEnv() for _ in range(n)]
    obs, info = envs[0].reset()
    return fc.train_parallel(envs[0], obs, info, str(out), trunk=fc.C0Trunk(c0_dir, "cpu"),
                             row_of=_row_of, settings=s, c0_dir=c0_dir, log=lambda m: None,
                             extra_envs=[(e, None) for e in envs[1:]],
                             towns=[f"T{i}" for i in range(n)])


def _first(tmp_path, c0_dir):
    _run(c0_dir, tmp_path / "a", _settings(12, 12))      # updates from step 5; checkpoint step_12
    return str(tmp_path / "a" / "step_12")


def test_a_resumed_run_restores_the_whole_learner_and_the_step_count(c0_dir, tmp_path):  # noqa: F811
    ck = _first(tmp_path, c0_dir)
    saved = torch.load(os.path.join(ck, "params.pt"))
    # one iteration of 2 envs: 2 own rows < start_training (5) -> no update yet
    learner, rb, _ = _run(c0_dir, tmp_path / "b", _settings(14, 4, resume=ck))
    for key, mod in (("actor", learner.actor), ("critic", learner.critic),
                     ("target_critic", learner.target_critic)):
        for name, t in mod.state_dict().items():
            assert torch.equal(t, saved[key][name]), (key, name)
    assert float(learner.log_temp) == pytest.approx(float(saved["log_temp"]))
    assert learner.target_entropy == pytest.approx(saved["target_entropy"] - 2 * 25e-7)
    assert json.load(open(tmp_path / "b" / "step_14" / "config.json"))["step"] == 14
    assert rb.n == 2                                         # a fresh replay: only its own 2 rows


def test_updates_resume_once_the_own_rows_refill(c0_dir, tmp_path, monkeypatch):  # noqa: F811
    from torch_rl.frl.learner import SACLearner
    ck = _first(tmp_path, c0_dir)
    calls = []
    real = SACLearner.update
    monkeypatch.setattr(SACLearner, "update",
                        lambda self, *a, **kw: calls.append(1) or real(self, *a, **kw))
    _run(c0_dir, tmp_path / "c", _settings(22, 4, resume=ck))
    # resumed at 12; own rows 2, 4, 6 (>= 5 from the 3rd iteration, steps 17-18), ... -> steps 17..22
    assert len(calls) == 22 - 17 + 1
    for step in (16, 20, 22):
        assert (tmp_path / "c" / f"step_{step}" / "config.json").exists()


def test_a_checkpoint_with_other_learning_settings_is_refused(c0_dir, tmp_path):  # noqa: F811
    ck = _first(tmp_path, c0_dir)
    with pytest.raises(ValueError, match="utd"):
        _run(c0_dir, tmp_path / "d", _settings(20, 4, resume=ck, utd=2, batch=2))


def test_settings_read_the_resume_path():
    assert fc.LoopSettings.from_env({"FRL_RESUME": "/x/step_150000"}).resume == "/x/step_150000"
    assert fc.LoopSettings.from_env({}).resume == ""
