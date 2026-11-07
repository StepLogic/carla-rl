"""The RL fine-tune of a clone (imitation-first plan, Task 10): a critic warm-up
with the encoder, GRU and actor frozen, then SAC with a cloning loss that fades.

Why a warm-up: the critic of a freshly loaded clone knows nothing, and the
first actor steps against an untrained critic would undo the cloned policy (the
earlier warm starts lost goal steering within 50k steps). While the critic
learns the clone's values, nothing the clone learned may move.
"""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))
torch = pytest.importorskip("torch")

from tests.test_one_critic import _cfg, _chunk  # noqa: E402
from torch_rl.gated_recurrent_sac import GatedRecurrentSAC  # noqa: E402

MODULES = ("compress", "core", "actor", "critics_g")


def _snap(agent):
    return {m: [p.detach().clone() for p in getattr(agent, m).parameters()] for m in MODULES}


def _moved(before, agent, m):
    return any(not torch.equal(a, b) for a, b in zip(before[m], getattr(agent, m).parameters()))


def test_warmup_freezes_encoder_core_and_actor():
    agent = GatedRecurrentSAC(_cfg())
    agent.set_warmup(True)
    before = _snap(agent)
    metrics = agent.update(_chunk(agent.cfg, [1, 0]))
    assert not _moved(before, agent, "compress")
    assert not _moved(before, agent, "core")
    assert not _moved(before, agent, "actor")
    assert _moved(before, agent, "critics_g")
    assert metrics["actor_loss"] == 0.0


def test_warmup_leaves_the_temperature_alone():
    agent = GatedRecurrentSAC(_cfg())
    agent.set_warmup(True)
    a0 = float(agent.log_alpha_g.detach())
    agent.update(_chunk(agent.cfg, [1, 0]))
    assert float(agent.log_alpha_g.detach()) == a0


def test_warmup_off_trains_everything_again():
    agent = GatedRecurrentSAC(_cfg())
    agent.set_warmup(True)
    agent.set_warmup(False)
    assert all(p.requires_grad for p in agent.compress.parameters())
    before = _snap(agent)
    agent.update(_chunk(agent.cfg, [1, 0]))
    for m in MODULES:
        assert _moved(before, agent, m), m


def test_bc_weight_schedule():
    from torch_rl.bc_trainer import bc_weight
    assert bc_weight(0, 1.0, 10000) == 1.0
    assert bc_weight(5000, 1.0, 10000) == 0.5
    assert bc_weight(20000, 1.0, 10000) == 0.0
    assert bc_weight(100, 0.0, 10000) == 0.0


def test_checkpoint_format_unchanged(tmp_path):
    """The cloning optimiser lives in the trainer, never in agent.save."""
    from torch_rl.bc_trainer import bc_step, make_bc_optimizer
    agent = GatedRecurrentSAC(_cfg())
    opt = make_bc_optimizer(agent)
    agent.update(_chunk(agent.cfg, [1, 0]))
    bc_step(agent, opt, _chunk(agent.cfg, [1, 0]), weight=0.5)
    agent.save(str(tmp_path / "ck"))
    GatedRecurrentSAC(_cfg()).load(str(tmp_path / "ck"))       # strict


def test_run_parallel_calls_after_update_once_per_update(tmp_path):
    import numpy as np
    from tests.test_parallel_loop import BINOC, FakeAgent, FakeTown, _row
    from torch_rl.episode_buffer import DeviceEpisodeBuffer
    from torch_rl.parallel_loop import run_parallel
    from vae_nav.vec_env import ThreadedVecEnv

    class Counting(FakeAgent):
        n_upd = 0

        def update(self, chunk):
            Counting.n_upd += 1
            return {"q_g": 0.0}

    hooks = []
    venv = ThreadedVecEnv([lambda: FakeTown(0, 7, True), lambda: FakeTown(1, 11, False)],
                          names=["TownA", "TownB"])
    run_parallel(venv, Counting(), DeviceEpisodeBuffer(1 + BINOC, 2, 1000, "cpu"), _row,
                 venv.action_space, to_env=lambda a, b: a, to_actor=lambda e, b: e,
                 goal_block=lambda g: np.asarray(g, np.float32).reshape(-1),
                 max_steps=60, learning_starts=0, update_every=10, updates_per=2,
                 chunk_b=2, chunk_t=3, burn_in=1, ckpt_every=20, out_dir=str(tmp_path),
                 log_every=2, log=lambda m: None, after_update=lambda: hooks.append(1))
    assert Counting.n_upd > 0 and len(hooks) == Counting.n_upd


def test_il_finetune_prepare_preloads_warms_up_and_returns_the_hook(tmp_path, monkeypatch):
    import numpy as np
    from tests.test_bc_trainer import _shards, _small_pixel_agent
    from torch_rl import pixel_rows as pxr
    from torch_rl.bc_trainer import il_finetune_prepare
    from torch_rl.episode_buffer import DeviceEpisodeBuffer
    from torch_rl.expert_data import episode_paths, load_episode
    _shards(tmp_path / "data")                                 # 3 episodes x 10 steps, Town05
    monkeypatch.setenv("REPLAY_DEVICE", "cpu")
    agent = _small_pixel_agent(pxr.BINOC_DIM, 2)
    buf = DeviceEpisodeBuffer(pxr.mono_row_dim(), 2, 200, "cpu", goal_dim=pxr.PIX)
    actor0 = [p.detach().clone() for p in agent.actor.parameters()]
    critic0 = [p.detach().clone() for p in agent.critics_g.parameters()]
    hook = il_finetune_prepare(agent, buf, str(tmp_path / "data"), ["Town05"],
                               warmup_updates=3, chunk_b=2, chunk_t=4, w0=1.0,
                               decay_updates=10, bc_chunks=2, bc_len=4, aug=None,
                               log=lambda m: None)
    # the replay holds the EXECUTED actions of every recorded step
    assert buf.size == 30
    execs = np.concatenate([load_episode(p)["act_exec"]
                            for p in episode_paths(str(tmp_path / "data"), ["Town05"])])
    # an episode of n steps holds n+1 rows (the last observation has no action)
    held = np.concatenate([buf.actions[s:s + n].numpy() for s, n, _ in buf._episodes])
    assert sorted(map(tuple, np.round(held, 5))) == sorted(map(tuple, np.round(execs, 5)))
    # the warm-up moved the critic, not the actor, and is over
    assert all(torch.equal(a, b) for a, b in zip(actor0, agent.actor.parameters()))
    assert any(not torch.equal(a, b) for a, b in zip(critic0, agent.critics_g.parameters()))
    assert agent.warmup is False and all(p.requires_grad for p in agent.core.parameters())
    # the hook clones on the expert labels
    hook()
    assert any(not torch.equal(a, b) for a, b in zip(actor0, agent.actor.parameters()))


def test_preload_transform_filters_the_replay_preload(tmp_path, monkeypatch):
    """IL_PRELOAD_TRANSFORM names a torch_rl.slope_diag variant applied to the
    recorded episodes on their way into the REPLAY (the critic's data; user
    2026-11-07 after the critic-slope diagnosis): here one that leaves out the
    episode with seed 100001."""
    from tests.test_bc_trainer import _shards, _small_pixel_agent
    from torch_rl import pixel_rows as pxr
    from torch_rl import slope_diag
    from torch_rl.bc_trainer import il_finetune_prepare
    from torch_rl.episode_buffer import DeviceEpisodeBuffer
    _shards(tmp_path / "data")                                 # 3 episodes x 10 steps
    monkeypatch.setenv("REPLAY_DEVICE", "cpu")
    monkeypatch.setitem(slope_diag.VARIANTS, "drop_100001",
                        lambda ep: None if int(ep["seed"]) == 100001 else ep)
    monkeypatch.setenv("IL_PRELOAD_TRANSFORM", "drop_100001")
    agent = _small_pixel_agent(pxr.BINOC_DIM, 2)
    buf = DeviceEpisodeBuffer(pxr.mono_row_dim(), 2, 200, "cpu", goal_dim=pxr.PIX)
    logs = []
    il_finetune_prepare(agent, buf, str(tmp_path / "data"), ["Town05"],
                        warmup_updates=1, chunk_b=2, chunk_t=4, w0=0.0,
                        decay_updates=10, bc_chunks=2, bc_len=4, aug=None, log=logs.append)
    assert buf.size == 20 and buf.episodes == 2
    assert any("drop_100001" in m for m in logs)


def test_an_unknown_preload_transform_is_refused(tmp_path, monkeypatch):
    from tests.test_bc_trainer import _shards, _small_pixel_agent
    from torch_rl import pixel_rows as pxr
    from torch_rl.bc_trainer import il_finetune_prepare
    from torch_rl.episode_buffer import DeviceEpisodeBuffer
    _shards(tmp_path / "data")
    monkeypatch.setenv("REPLAY_DEVICE", "cpu")
    monkeypatch.setenv("IL_PRELOAD_TRANSFORM", "no_such_variant")
    agent = _small_pixel_agent(pxr.BINOC_DIM, 2)
    buf = DeviceEpisodeBuffer(pxr.mono_row_dim(), 2, 200, "cpu", goal_dim=pxr.PIX)
    with pytest.raises(SystemExit, match="IL_PRELOAD_TRANSFORM"):
        il_finetune_prepare(agent, buf, str(tmp_path / "data"), ["Town05"],
                            warmup_updates=1, chunk_b=2, chunk_t=4, w0=0.0,
                            decay_updates=10, bc_chunks=2, bc_len=4, aug=None,
                            log=lambda m: None)


def test_trainer_wires_the_preload_warmup_and_hook():
    """main(): prepare before either loop, hook after every update in both."""
    s = open(os.path.join(ROOT, "train_carla_sac.py")).read()
    assert "il_hook = il_finetune_prepare(" in s
    assert "random_warmup=not INIT_FROM, after_update=il_hook)" in s
    assert "metrics = agent.update(chunk)\n                if il_hook is not None:\n                    il_hook()" in s
    # a preloaded replay is not empty, so the warm start's refill guard stands down
    assert "if LEARNING_STARTS < INIT_MIN_LEARNING_STARTS and not IL_PRELOAD_ROOT:" in s


def test_a_frozen_trunk_keeps_encoder_and_gru_but_trains_the_heads():
    """TRUNK FREEZE (user 2026-11-05, after two collapsed fine-tunes): the
    encoder, action embedding and GRU take no gradient from any loss, while
    the actor head, the critic heads and the temperature keep learning."""
    agent = GatedRecurrentSAC(_cfg())
    agent.set_trunk_frozen(True)
    before = _snap(agent)
    a0 = float(agent.log_alpha_g.detach())
    agent.update(_chunk(agent.cfg, [1, 0]))
    assert not _moved(before, agent, "compress")
    assert not _moved(before, agent, "core")
    assert _moved(before, agent, "actor")
    assert _moved(before, agent, "critics_g")
    assert float(agent.log_alpha_g.detach()) != a0


def test_ending_the_warmup_leaves_a_frozen_trunk_frozen():
    """The online freeze ends with set_warmup(False), which used to switch every
    trunk gradient back on."""
    agent = GatedRecurrentSAC(_cfg())
    agent.set_trunk_frozen(True)
    agent.set_warmup(True)
    agent.set_warmup(False)
    assert not any(p.requires_grad for p in agent.compress.parameters())
    assert not any(p.requires_grad for p in agent.core.parameters())


def test_cloning_with_a_frozen_trunk_moves_only_the_actor():
    from torch_rl.bc_trainer import bc_step, make_bc_optimizer
    agent = GatedRecurrentSAC(_cfg())
    agent.set_trunk_frozen(True)
    opt = make_bc_optimizer(agent)
    before = _snap(agent)
    bc_step(agent, opt, _chunk(agent.cfg, [1, 0]), weight=1.0)
    assert not _moved(before, agent, "compress")
    assert not _moved(before, agent, "core")
    assert _moved(before, agent, "actor")


def test_trainer_freezes_the_trunk_on_request():
    s = open(os.path.join(ROOT, "train_carla_sac.py")).read()
    assert 'IL_FREEZE_TRUNK = os.environ.get("IL_FREEZE_TRUNK", "0") == "1"' in s
    assert "agent.set_trunk_frozen(True)" in s
    # after the warm start's load, before the preload's warm-up
    assert s.index("agent.set_trunk_frozen(True)") > s.index("agent.load(INIT_FROM")
    assert s.index("agent.set_trunk_frozen(True)") < s.index("il_hook = il_finetune_prepare(")


def test_set_temperature_overrides_the_loaded_temperatures():
    """IL_ALPHA0 (user 2026-11-05): a clone's checkpoint carries the from-scratch
    temperature 0.1, whose entropy bonus inflated c0's steering noise ~8x
    (log-std -2.23 -> ~-0.2) within ~90 updates of the unfreeze while the
    temperature itself drifted ~1% per 100 updates. The fine-tune starts low."""
    import math
    agent = GatedRecurrentSAC(_cfg())
    agent.set_temperature(1e-3)
    assert math.isclose(agent.alpha_g, 1e-3, rel_tol=1e-6)
    assert math.isclose(agent.alpha_e, 1e-3, rel_tol=1e-6)
    assert agent.log_alpha_g.requires_grad              # still learned from there


def test_trainer_sets_the_starting_temperature_after_the_warm_start():
    s = open(os.path.join(ROOT, "train_carla_sac.py")).read()
    assert 'IL_ALPHA0 = env_float("IL_ALPHA0", 0.0)' in s
    assert "agent.set_temperature(IL_ALPHA0)" in s
    assert s.index("agent.set_temperature(IL_ALPHA0)") > s.index("agent.load(INIT_FROM")


class _ChunkBuf:
    def __init__(self, agent):
        self.agent = agent

    def sample_chunks(self, n, length, burn_in):
        return _chunk(self.agent.cfg, [1, 0])


def test_the_clone_stays_frozen_for_the_first_online_updates():
    """ONLINE FREEZE (2026-11-05). Unfreezing the clone on the first live update
    collapsed the fine-tune (job 463440: policy= 0.76 -> 0.10 within ~190
    updates, cloning loss at full weight). With freeze_updates=N the first N
    online updates train the critic only, on the clone's own rollouts, with no
    cloning step; after the N-th everything trains, and the cloning weight
    starts from w0 then."""
    from torch_rl.bc_trainer import il_bc_hook, make_bc_optimizer
    agent = GatedRecurrentSAC(_cfg())
    weights = []
    hook = il_bc_hook(agent, make_bc_optimizer(agent), _ChunkBuf(agent), n_chunks=2,
                      length=8, w0=1.0, decay_updates=4, on_weight=weights.append,
                      freeze_updates=2)
    before = _snap(agent)
    for _ in range(2):
        agent.update(_chunk(agent.cfg, [1, 0]))
        hook()
    for m in ("compress", "core", "actor"):
        assert not _moved(before, agent, m), m
    assert _moved(before, agent, "critics_g")
    assert weights == [] and agent.warmup is False   # the next update trains everything
    agent.update(_chunk(agent.cfg, [1, 0]))
    hook()
    assert _moved(before, agent, "actor")
    assert weights == [1.0]                           # the schedule starts after the freeze


def test_the_freeze_end_is_logged():
    from torch_rl.bc_trainer import il_bc_hook, make_bc_optimizer
    agent = GatedRecurrentSAC(_cfg())
    lines = []
    hook = il_bc_hook(agent, make_bc_optimizer(agent), _ChunkBuf(agent), n_chunks=2,
                      length=8, w0=1.0, decay_updates=4, freeze_updates=1,
                      log=lines.append)
    agent.update(_chunk(agent.cfg, [1, 0]))
    hook()
    assert any("online freeze over" in ln for ln in lines)


def test_prepare_leaves_the_clone_frozen_for_an_online_freeze(tmp_path, monkeypatch):
    from tests.test_bc_trainer import _shards, _small_pixel_agent
    from torch_rl import pixel_rows as pxr
    from torch_rl.bc_trainer import il_finetune_prepare
    from torch_rl.episode_buffer import DeviceEpisodeBuffer
    _shards(tmp_path / "data")
    monkeypatch.setenv("REPLAY_DEVICE", "cpu")
    agent = _small_pixel_agent(pxr.BINOC_DIM, 2)
    buf = DeviceEpisodeBuffer(pxr.mono_row_dim(), 2, 200, "cpu", goal_dim=pxr.PIX)
    il_finetune_prepare(agent, buf, str(tmp_path / "data"), ["Town05"],
                        warmup_updates=2, chunk_b=2, chunk_t=4, w0=1.0, decay_updates=10,
                        bc_chunks=2, bc_len=4, aug=None, log=lambda m: None,
                        online_freeze_updates=5)
    assert agent.warmup is True                       # the first online update is critic-only


def test_trainer_passes_the_online_freeze():
    s = open(os.path.join(ROOT, "train_carla_sac.py")).read()
    assert 'IL_ONLINE_FREEZE_UPDATES = env_int("IL_ONLINE_FREEZE_UPDATES", 0)' in s
    assert "online_freeze_updates=IL_ONLINE_FREEZE_UPDATES" in s


def test_after_update_hook_takes_a_weighted_cloning_step():
    """il_bc_hook: the online loop's per-update cloning step, weight by schedule."""
    from torch_rl.bc_trainer import il_bc_hook, make_bc_optimizer
    agent = GatedRecurrentSAC(_cfg())

    class _Buf:
        def sample_chunks(self, n, length, burn_in):
            return _chunk(agent.cfg, [1, 0])

    weights = []
    hook = il_bc_hook(agent, make_bc_optimizer(agent), _Buf(), n_chunks=2, length=8,
                      w0=1.0, decay_updates=4, on_weight=weights.append)
    before = _snap(agent)
    for _ in range(6):
        hook()
    assert weights == [1.0, 0.75, 0.5, 0.25, 0.0, 0.0]
    assert _moved(before, agent, "actor")
