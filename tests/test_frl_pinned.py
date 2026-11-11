"""Expert rows pinned in learner C's own replay (user 2026-11-11: "ok add expert data"; earlier "restart run
with expert pinned in buffer"). Unlike FastRLAP's separate expert update (FRL_EXPERT_MODE=update, the run
already made), the expert transitions sit in the MAIN replay: never overwritten, sampled uniformly together
with the RL's own transitions (no forced share, the plain-SAC rule), and -- the prerecorded-data rule --
never moving the temperature: the entropy that tunes it is measured on the RL's own rows only."""
import copy
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
from torch_rl.frl.learner import SACLearner  # noqa: E402
from tests.test_frl_expert import _replay, trunk  # noqa: E402,F401  (the tiny c0 fixture)

E, S = 3, 2


def _pinned(n):
    return {"emb": torch.full((n, E), -1.0, dtype=torch.float16),
            "next_emb": torch.full((n, E), -2.0, dtype=torch.float16),
            "states": torch.full((n, S), -1.0), "next_states": torch.full((n, S), -2.0),
            "actions": torch.zeros(n, 2), "rewards": torch.full((n,), 7.0),
            "masks": torch.ones(n)}


def _obs(v):
    return {"embeddings": torch.full((1, E), float(v)), "states": torch.full((1, S), float(v))}


def test_pinned_rows_are_never_overwritten_and_the_ring_wraps_after_them():
    rb = fc.FeatureReplay(4, E, S, 2, "cpu", seed=0, pinned=_pinned(3))
    for i in range(10):
        rb.add(_obs(i), np.zeros(2, np.float32), float(i), 1.0, _obs(i + 0.5))
    assert len(rb) == 3 + 4
    assert torch.all(rb.rew[:3] == 7.0)                                   # the expert rows intact
    assert sorted(rb.rew[3:7].tolist()) == [6.0, 7.0, 8.0, 9.0]           # the last 4 online rows


def test_sampling_is_uniform_over_expert_and_own_rows_and_marks_the_own_ones():
    rb = fc.FeatureReplay(10, E, S, 2, "cpu", seed=0, pinned=_pinned(3))
    for i in range(3):
        rb.add(_obs(i), np.zeros(2, np.float32), float(i), 1.0, _obs(i))
    b = rb.sample(4000)
    online = b["online"]
    assert online.dtype == torch.bool
    assert torch.equal(online, b["rewards"] != 7.0)                       # own rows marked
    assert 0.45 < float(online.float().mean()) < 0.55                     # 3 of 6 rows: uniform
    rb0 = fc.FeatureReplay(10, E, S, 2, "cpu", seed=0)                    # no pinned rows: all own
    rb0.add(_obs(0), np.zeros(2, np.float32), 0.0, 1.0, _obs(0))
    assert bool(rb0.sample(5)["online"].all())


def _learner():
    return SACLearner(action_dim=2, embedding_dim=E, state_dim=S, latent_dim=8, hidden_dims=(16, 16),
                      num_qs=4, num_min_qs=2, critic_layer_norm=True, backup_entropy=False,
                      init_temperature=0.1, seed=0)


def _batch(n, online):
    g = torch.Generator().manual_seed(1)
    obs = {"embeddings": torch.randn(n, E, generator=g), "states": torch.randn(n, S, generator=g)}
    return {"observations": obs, "next_observations": copy.deepcopy(obs),
            "actions": torch.rand(n, 2, generator=g) * 2 - 1, "rewards": torch.ones(n),
            "masks": torch.ones(n), "online": torch.as_tensor(online)}


def test_the_entropy_that_tunes_the_temperature_comes_from_own_rows_only():
    learner = _learner()
    mb = _batch(4, [True, False, False, True])
    before = copy.deepcopy(learner.actor)
    torch.manual_seed(5)
    info = learner.update_actor(mb)
    torch.manual_seed(5)
    _, logp, _ = before.sample(mb["observations"])
    assert info["entropy"] == pytest.approx(float(-logp.detach()[[0, 3]].mean()), rel=1e-5)


def test_a_minibatch_of_expert_rows_only_leaves_the_temperature_alone():
    learner = _learner()
    t0 = float(learner.log_temp)
    learner.update(_batch(4, [False] * 4), utd_ratio=2)
    assert float(learner.log_temp) == t0
    learner.update(_batch(4, [True] * 4), utd_ratio=2)                    # own rows move it
    assert float(learner.log_temp) != t0


def test_settings_read_the_expert_mode():
    assert fc.LoopSettings.from_env({"FRL_EXPERT_MODE": "pinned"}).expert_mode == "pinned"
    assert fc.LoopSettings.from_env({}).expert_mode == "update"         # the expert run's mode


def test_an_unknown_expert_mode_is_refused():
    with pytest.raises(ValueError):
        fc.LoopSettings(expert_mode="mixed")


def test_pinned_mode_without_an_expert_replay_is_refused():
    ok = {"MONO_CHW": "1", "SPEED_OBS": "1"}
    assert fc.frl_refusals(["Town01"], False,
                           {**ok, "FRL_EXPERT_MODE": "pinned", "FRL_EXPERT": "/x.npz"}) == []
    assert any("FRL_EXPERT" in m
               for m in fc.frl_refusals(["Town01"], False, {**ok, "FRL_EXPERT_MODE": "pinned"}))


def test_in_pinned_mode_the_expert_sits_in_the_replay_and_there_is_no_second_update(
        trunk, tmp_path, monkeypatch):  # noqa: F811
    from tests.test_frl_carla import _FakeEnv, _row_of
    from torch_rl.frl import expert as fx
    xr = fx.ExpertReplay.load(_replay(trunk, tmp_path), seed=0)          # 9 expert transitions
    calls = []
    monkeypatch.setattr(fx, "expert_update", lambda *a, **kw: calls.append(1))
    env = _FakeEnv()
    obs, info = env.reset()
    s = fc.LoopSettings(max_steps=8, start_training=5, utd=2, batch=4, replay=100, ckpt_every=8,
                        log_every=8, device="cpu", seed=0, expert_mode="pinned")
    msgs = []
    _, rb, _ = fc.train(env, obs, info, str(tmp_path / "run"), trunk=trunk, row_of=_row_of,
                        settings=s, c0_dir=trunk.c0_dir, log=msgs.append, expert=xr)
    assert calls == []                                                  # no FastRLAP second update
    assert rb.n_pinned == 9 and len(rb) == 9 + 8
    assert torch.equal(rb.rew[:9], xr.t["rewards"].float())             # the expert rows, first
    assert torch.equal(rb.act[:9], xr.t["actions"].float())
    assert not any("x_critic_loss=" in m for m in msgs)
