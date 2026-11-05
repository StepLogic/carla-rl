"""Behaviour cloning of the recurrent SAC learner (imitation-first plan, Task 6).

The cloning loss is the pre-tanh Gaussian NLL of the expert's (clipped,
atanh'd) action under the actor, on exactly the row alignment the SAC update
uses: core states s_t of the trained window, paired with action t. Gradients
must reach the trainable visual encoder and the GRU as well as the actor --
cloning is how the encoder learns to see lanes at all -- and rows the buffer
marks invalid (padding) must not count.
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))
torch = pytest.importorskip("torch")

from tests.test_one_critic import _cfg, _chunk  # noqa: E402
from torch_rl.bc_trainer import (bc_loss, bc_step, make_bc_optimizer,  # noqa: E402
                                 sync_targets)
from torch_rl.gated_recurrent_sac import GatedRecurrentSAC  # noqa: E402


def _agent():
    return GatedRecurrentSAC(_cfg())


def test_bc_loss_decreases_on_a_fixed_chunk():
    agent = _agent()
    chunk = _chunk(agent.cfg, [1, 0])
    opt = make_bc_optimizer(agent, lr=3e-3)
    first, _ = bc_loss(agent, chunk)
    for _ in range(30):
        stats = bc_step(agent, opt, chunk)
    final, _ = bc_loss(agent, chunk)
    first, final = float(first.detach()), float(final.detach())
    assert first > 0.0
    assert final < 0.5 * first
    assert set(stats) >= {"bc_nll", "bc_mae", "bc_log_std"}


def test_bc_gradients_reach_encoder_core_and_actor():
    agent = _agent()
    loss, _ = bc_loss(agent, _chunk(agent.cfg, [1, 0]))
    loss.backward()
    for mod in (agent.compress, agent.core, agent.actor):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in mod.parameters())


def test_bc_loss_ignores_invalid_rows():
    agent = _agent()
    chunk = _chunk(agent.cfg, [1, 0])
    chunk["valid"][0, 3:] = 0.0
    chunk["valid"][1, :2] = 0.0
    before, _ = bc_loss(agent, chunk)
    changed = {k: v.clone() for k, v in chunk.items()}
    bad = changed["valid"] == 0
    changed["actions"][bad] = -changed["actions"][bad] * 0.5 + 0.3
    after, _ = bc_loss(agent, changed)
    assert torch.allclose(before, after)


def test_sync_targets_copies_online_weights():
    agent = _agent()
    opt = make_bc_optimizer(agent, lr=1e-2)
    bc_step(agent, opt, _chunk(agent.cfg, [1, 0]))
    pairs = [(agent.compress, agent.compress_targ), (agent.core, agent.core_targ)]
    if agent.act_embed is not None:
        pairs.append((agent.act_embed, agent.act_embed_targ))
    # cloning moved the online branch; the targets are still the init copy
    assert any(not torch.equal(p, q) for on, tg in pairs
               for p, q in zip(on.state_dict().values(), tg.state_dict().values()))
    sync_targets(agent)
    for on, tg in pairs:
        for k, v in on.state_dict().items():
            assert torch.equal(v, tg.state_dict()[k]), k


def test_bc_optimizer_covers_exactly_the_cloned_modules():
    agent = _agent()
    opt = make_bc_optimizer(agent)
    in_opt = {id(p) for g in opt.param_groups for p in g["params"]}
    mods = [agent.compress, agent.core, agent.actor]
    if agent.act_embed is not None:
        mods.append(agent.act_embed)
    want = {id(p) for m in mods for p in m.parameters() if p.requires_grad}
    assert in_opt == want                       # no critic, no target, no temperature


def test_bc_checkpoint_reloads(tmp_path):
    agent = _agent()
    opt = make_bc_optimizer(agent, lr=1e-2)
    for _ in range(3):
        bc_step(agent, opt, _chunk(agent.cfg, [1, 0]))
    sync_targets(agent)
    agent.save(str(tmp_path / "step_3"))
    fresh = _agent()
    fresh.load(str(tmp_path / "step_3"))
    row = np.asarray(_chunk(agent.cfg, [1])["obs"][0, -1].numpy(), np.float32)
    a1, _ = agent.act(row, agent.reset_hidden(1), deterministic=True)
    a2, _ = fresh.act(row, fresh.reset_hidden(1), deterministic=True)
    assert np.array_equal(a1, a2)


# ------------------------------------------------- the BC_TRAIN=1 run itself --
def _shards(root, n=3, T=10):
    from torch_rl import pixel_rows as pxr
    from torch_rl.expert_data import save_episode
    for i in range(n):
        rng = np.random.default_rng(i)
        save_episode(os.path.join(root, "Town05", f"ep_{100000 + i}.npz"), {
            "frames": rng.integers(0, 256, (T + 1, 128, 128, 3), dtype=np.uint8),
            "binoc": rng.normal(size=(T + 1, pxr.BINOC_DIM)).astype(np.float16),
            "act_exec": rng.uniform(-1, 1, (T, 2)).astype(np.float32),
            "act_label": rng.uniform(-1, 1, (T, 2)).astype(np.float32),
            "reward": np.zeros(T, np.float32), "speed": np.full(T, 12.0, np.float32),
            "pos": np.zeros((T, 2), np.float32), "mode": np.int8(i % 2),
            "terminal": False, "truncated": True,
            "goal": pxr.goal_block(np.zeros((112, 112, 3), np.uint8)),
            "sigma": np.float32(0.0), "driver": "expert", "end": "route_done",
            "town": "Town05", "seed": np.int64(100000 + i)})


def _small_pixel_agent(binoc_dim, act_dim):
    """A CPU learner on the real pixel-row layout (128 px frame | mode | binoc),
    shrunk so a few updates take seconds."""
    from torch_rl import pixel_rows as pxr
    from torch_rl.gated_recurrent_sac import GatedRecurrentSACConfig
    sd = pxr.mono_state_dim(binoc_dim)
    return GatedRecurrentSAC(GatedRecurrentSACConfig(
        mono_encoder="resnet", embed_dim=3, grid=pxr.MONO_IMG, action_dim=act_dim,
        goal_tokens=False, prev_action=False, mono_baseplanes=8, mono_ngroups=4,
        core_state_cols=pxr.mono_core_state_cols(binoc_dim), state_dim=sd, mode_index=0,
        one_critic=True, goal_cols=tuple(range(sd)), explore_cols=(), hidden=16,
        visual_dim=16, mlp=(8, 8), burn_in=2, device="cpu", compile=False, seed=0,
        n_critics=2, explore_actor_warmup=0))


def _bc_env(monkeypatch, root, **kw):
    env = {"BC_DATA_ROOT": str(root), "BC_TOWNS": "Town05", "BC_UPDATES": "3",
           "BC_CHUNKS": "2", "BC_LEN": "4", "BC_SAVE_EVERY": "2", "REPLAY_DEVICE": "cpu"}
    env.update(kw)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("BC_INIT_FROM", raising=False)


def test_run_bc_train_checkpoints_on_schedule_and_at_the_end(tmp_path, monkeypatch, capsys):
    import json
    from torch_rl.bc_trainer import run_bc_train
    _shards(tmp_path / "data")
    _bc_env(monkeypatch, tmp_path / "data")
    out = tmp_path / "out"
    run_bc_train(str(out), make_agent=_small_pixel_agent)
    assert (out / "step_2" / "params.pt").exists()      # BC_SAVE_EVERY
    assert (out / "step_3" / "params.pt").exists()      # the final update
    split = json.loads((out / "bc_split.json").read_text())
    assert len(split["train"]) == 2 and len(split["val"]) == 1
    assert not set(split["train"]) & set(split["val"])
    assert "[BC] step=3 train_nll=" in capsys.readouterr().out
    # the saved clone reloads with its targets synced: a DAgger round starts from it
    again = _small_pixel_agent(3136, 2)
    again.load(str(out / "step_3"))
    for k, v in again.core.state_dict().items():
        assert torch.equal(v, again.core_targ.state_dict()[k])


def test_run_bc_train_refuses_a_held_out_town(tmp_path, monkeypatch):
    from torch_rl.bc_trainer import run_bc_train
    _bc_env(monkeypatch, tmp_path, BC_TOWNS="Town05 Town02")
    with pytest.raises(ValueError, match="held out"):
        run_bc_train(str(tmp_path / "out"), make_agent=_small_pixel_agent)


def test_bc_max_seed_keeps_only_the_expert_episodes(tmp_path, monkeypatch):
    """The DAgger rounds write into the same data root (seeds 300000+); a
    round-0 clone must train on the expert's own episodes (seeds 1xxxxx) only."""
    import json
    from torch_rl.bc_trainer import run_bc_train
    _shards(tmp_path / "data", n=3)                       # seeds 100000-100002
    import shutil
    for i in range(2):                                    # two 'DAgger' episodes
        shutil.copy(tmp_path / "data" / "Town05" / f"ep_{100000 + i}.npz",
                    tmp_path / "data" / "Town05" / f"ep_{300000 + i}.npz")
    _bc_env(monkeypatch, tmp_path / "data", BC_MAX_SEED="199999")
    run_bc_train(str(tmp_path / "out"), make_agent=_small_pixel_agent)
    split = json.loads((tmp_path / "out" / "bc_split.json").read_text())
    used = split["train"] + split["val"]
    assert len(used) == 3 and not any("ep_3" in os.path.basename(p) for p in used)
