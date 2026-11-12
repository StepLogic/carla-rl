"""Two replays, one batch (user 2026-11-05: "keep expert buffer and rl buffer and
sample from both"). RLPD's symmetric sampling: the recorded expert transitions
live in a replay of their own that the agent's data never overwrites, and every
update draws a fixed share of its chunks from it and the rest from the agent's
own replay. With one shared replay the batches at the unfreeze of fine-tune job
463472 were ~88% expert rows, so the critic barely saw the clone's own states.
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))
torch = pytest.importorskip("torch")

from torch_rl.episode_buffer import DeviceEpisodeBuffer  # noqa: E402
from torch_rl.mixed_replay import MixedReplay  # noqa: E402

OBS, ACT = 3, 2


def _buf(cap=400):
    return DeviceEpisodeBuffer(OBS, ACT, cap, "cpu", seed=0)


def _episodes(buf, reward, n=12):
    """One goal-mode and one explore-mode episode, every reward `reward`."""
    for mode in (1, 0):
        for t in range(n):
            o = np.full(OBS, t, np.float32)
            buf.add(o, np.zeros(ACT, np.float32), reward, t == n - 1, False, o + 1, mode)
        buf.end_episode()


def _source(chunk):
    """Per chunk: 1.0 if it came from the expert replay, 2.0 if online."""
    r, v = chunk["rewards"], chunk["valid"]
    return ((r * v).sum(1) / v.sum(1)).tolist()


def test_half_the_chunks_come_from_each_replay():
    expert, online = _buf(), _buf()
    _episodes(expert, 1.0)
    _episodes(online, 2.0)
    chunk = MixedReplay(online, expert, expert_frac=0.5).sample_chunks(8, 4, 1)
    src = _source(chunk)
    assert sorted(src) == [1.0] * 4 + [2.0] * 4
    assert all(v.shape[0] == 8 for v in chunk.values())


def test_the_expert_share_is_set_by_expert_frac():
    expert, online = _buf(), _buf()
    _episodes(expert, 1.0)
    _episodes(online, 2.0)
    src = _source(MixedReplay(online, expert, expert_frac=0.25).sample_chunks(8, 4, 1))
    assert src.count(1.0) == 2 and src.count(2.0) == 6


def test_every_chunk_is_expert_until_the_online_replay_has_an_episode():
    """The offline critic warm-up runs before the first env step."""
    expert, online = _buf(), _buf()
    _episodes(expert, 1.0)
    src = _source(MixedReplay(online, expert, expert_frac=0.5).sample_chunks(6, 4, 1))
    assert src == [1.0] * 6


def test_writes_and_sizes_are_the_online_replays():
    """add, end_episode, size and sampleable are the agent's replay: the loops
    use the mixed replay exactly as they used one buffer."""
    expert, online = _buf(), _buf()
    _episodes(expert, 1.0)
    mix = MixedReplay(online, expert, expert_frac=0.5)
    _episodes(mix, 2.0)
    assert online.size == 24 and expert.size == 24
    assert mix.size == online.size and mix.sampleable == online.sampleable


def test_prepare_can_fill_a_separate_expert_replay(tmp_path, monkeypatch):
    from tests.test_bc_trainer import _shards, _small_pixel_agent
    from torch_rl import pixel_rows as pxr
    from torch_rl.bc_trainer import il_finetune_prepare
    _shards(tmp_path / "data")                                 # 3 episodes x 10 steps
    monkeypatch.setenv("REPLAY_DEVICE", "cpu")
    agent = _small_pixel_agent(pxr.BINOC_DIM, 2)
    buf = DeviceEpisodeBuffer(pxr.mono_row_dim(), 2, 200, "cpu", goal_dim=pxr.PIX)
    xbuf = DeviceEpisodeBuffer(pxr.mono_row_dim(), 2, 200, "cpu", goal_dim=pxr.PIX)
    critic0 = [p.detach().clone() for p in agent.critics_g.parameters()]
    il_finetune_prepare(agent, buf, str(tmp_path / "data"), ["Town05"],
                        warmup_updates=2, chunk_b=2, chunk_t=4, w0=1.0, decay_updates=10,
                        bc_chunks=2, bc_len=4, aug=None, log=lambda m: None,
                        expert_replay=xbuf)
    assert buf.size == 0 and xbuf.size == 30                   # the agent's replay starts empty
    assert any(not torch.equal(a, b) for a, b in zip(critic0, agent.critics_g.parameters()))


def test_trainer_wires_the_mixed_replay():
    s = open(os.path.join(ROOT, "train_carla_sac.py")).read()
    assert 'IL_EXPERT_FRAC = env_float("IL_EXPERT_FRAC", 0.0)' in s
    assert "expert_replay=xbuf" in s
    assert "buf = MixedReplay(buf, xbuf, IL_EXPERT_FRAC)" in s
    # wrapped after the preload, before either loop samples it
    assert s.index("buf = MixedReplay(buf, xbuf, IL_EXPERT_FRAC)") > s.index("il_hook = il_finetune_prepare(")
    assert s.index("buf = MixedReplay(buf, xbuf, IL_EXPERT_FRAC)") < s.index("steps, episodes = run_parallel(")
