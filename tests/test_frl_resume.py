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


# RESUME WITH A NEW ACTOR (user 2026-11-13: "use saved checkpoint. and replace actor with c0 actor").
# The trained critic is what the run spent 1.3M steps building; the actor is the weak part (34-66 m
# between fault ends against c0's 325-886). FRL_RESUME_NEW_ACTOR=1 restores the critic, the target
# critic, the temperature, the entropy target and their optimizers from the checkpoint and KEEPS the
# actor this run built -- with FRL_ACTOR=c0, a trainable copy of c0's head, so the policy starts as c0
# while the critic starts where the run left it. The actor's optimizer stays fresh: the new actor's
# parameters are a different shape, so the checkpoint's Adam state does not apply to them.

def test_resume_with_a_new_actor_keeps_the_critic_and_replaces_the_actor(c0_dir, tmp_path):  # noqa: F811
    ck = _first(tmp_path, c0_dir)
    saved = torch.load(os.path.join(ck, "params.pt"))
    trunk = fc.C0Trunk(c0_dir, "cpu")
    learner = fc.make_learner(trunk.emb_dim, trunk.state_dim, 2, "cpu", 0, actor="c0", trunk=trunk)
    s = _settings(99, 99, resume=ck, actor="c0", resume_new_actor=1)
    step = fc.restore_learner(learner, ck, s, c0_dir)
    assert step == 12                                        # the checkpoint's step, as usual
    for key, mod in (("critic", learner.critic), ("target_critic", learner.target_critic)):
        for name, t in mod.state_dict().items():
            assert torch.equal(t, saved[key][name]), (key, name)   # the critic IS the checkpoint's
    assert float(learner.log_temp) == pytest.approx(float(saved["log_temp"]))
    assert learner.target_entropy == pytest.approx(float(saved["target_entropy"]))
    # the actor is c0's head, NOT the checkpoint's actor
    assert isinstance(learner.actor, fc.C0HeadActor)
    for name, t in trunk.agent.actor.state_dict().items():
        assert torch.equal(learner.actor.net.state_dict()[name], t), name
    assert learner.actor_opt.state_dict()["state"] == {}      # fresh Adam for the new parameters


def test_resume_into_a_different_actor_is_refused_without_the_flag(c0_dir, tmp_path):  # noqa: F811
    ck = _first(tmp_path, c0_dir)                             # made with actor="frl"
    trunk = fc.C0Trunk(c0_dir, "cpu")
    learner = fc.make_learner(trunk.emb_dim, trunk.state_dim, 2, "cpu", 0, actor="c0", trunk=trunk)
    with pytest.raises(ValueError, match="actor"):
        fc.restore_learner(learner, ck, _settings(99, 99, resume=ck, actor="c0"), c0_dir)


def test_the_loop_resumes_a_checkpoint_into_a_c0_actor(c0_dir, tmp_path):  # noqa: F811
    ck = _first(tmp_path, c0_dir)
    s = _settings(16, 4, resume=ck, actor="c0", resume_new_actor=1)
    learner, _, _ = _run(c0_dir, tmp_path / "e", s)
    assert isinstance(learner.actor, fc.C0HeadActor)
    cfg = json.load(open(tmp_path / "e" / "step_16" / "config.json"))
    assert cfg["step"] == 16 and cfg["settings"]["actor"] == "c0"
    assert cfg["settings"]["resume_new_actor"] == 1            # provenance: it was an actor swap


def test_settings_read_the_new_actor_flag():
    assert fc.LoopSettings.from_env({"FRL_RESUME_NEW_ACTOR": "1"}).resume_new_actor == 1
    assert fc.LoopSettings.from_env({}).resume_new_actor == 0
