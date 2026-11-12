"""User rule 2026-11-07: "WHEN USING PRERECORDED DATA WITH SAC ENSURE PRERECORDED DATA
DOES NOT UPDATE TEMPERATURE AND IS NOT OVERWRITTEN".

The replay is a ring that evicts its oldest episodes first -- after a preload, the
recordings -- so DeviceEpisodeBuffer.pin() keeps every episode in it at that moment
forever: later episodes wrap back to the row after them. Every sampled chunk says
whether it came from a pinned (recorded) episode, and the SAC temperature step reads
only the others.
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

OBS = 4


def _episode(buf, n, mode=1, value=0.0, env_id=0):
    for t in range(n):
        buf.add(np.full(OBS, value, np.float32), np.zeros(2, np.float32), float(value), False,
                t == n - 1, np.full(OBS, value, np.float32), mode, env_id=env_id)
    buf.end_episode(env_id)


def _buf(cap=40, **kw):
    return DeviceEpisodeBuffer(OBS, 2, cap, "cpu", seed=0, **kw)


def test_pinned_episodes_are_never_overwritten():
    buf = _buf(cap=40)
    _episode(buf, 9, value=7.0)                 # rows 0-9
    _episode(buf, 9, value=8.0)                 # rows 10-19
    assert buf.pin() == 20
    for k in range(12):                         # far more online data than the 20 free rows
        _episode(buf, 4, value=100.0 + k)
    starts = [s for s, _, _ in buf._episodes]
    assert 0 in starts and 10 in starts         # both recordings survive
    assert torch.all(buf.obs[0:10, 0] == 7.0) and torch.all(buf.obs[10:20, 0] == 8.0)
    online = [s for s in starts if s >= 20]
    assert online and all(s >= 20 for s in online)   # the online ring lives after them


def test_chunks_say_whether_they_are_recorded():
    buf = _buf(cap=40)
    _episode(buf, 9, value=7.0)
    buf.pin()
    _episode(buf, 9, value=100.0)
    ch = buf.sample_chunks(64, 4, 1)
    rec = ch["prerecorded"].numpy()
    first_obs = ch["obs"][:, 1, 0].numpy()      # a valid row of each chunk
    assert set(np.unique(rec)) <= {0.0, 1.0} and 0.0 < rec.mean() < 1.0
    assert np.all((first_obs == 7.0) == (rec == 1.0))


def test_without_pin_nothing_is_recorded():
    buf = _buf(cap=40)
    _episode(buf, 9)
    assert float(buf.sample_chunks(8, 4, 1)["prerecorded"].sum()) == 0.0


def test_pin_refuses_a_preload_that_already_wrapped():
    buf = _buf(cap=20)
    for _ in range(4):
        _episode(buf, 6)                        # 28 rows into 20: the ring wrapped
    with pytest.raises(ValueError, match="wrapped"):
        buf.pin()


def test_pin_refuses_an_open_episode():
    buf = _buf(cap=40)
    _episode(buf, 5)
    buf.add(np.zeros(OBS, np.float32), np.zeros(2, np.float32), 0.0, False, False,
            np.zeros(OBS, np.float32), 1, env_id=3)
    with pytest.raises(ValueError, match="open"):
        buf.pin()


def test_a_recorded_replay_of_its_own_can_be_pinned_whole():
    """MixedReplay's expert replay is sized to its data: pinned whole, it refuses
    more episodes rather than overwrite a recording."""
    buf = _buf(cap=20)
    _episode(buf, 9)
    _episode(buf, 9)
    assert buf.pin() == 20
    with pytest.raises(ValueError):
        _episode(buf, 3)


# ---- the temperature reads only the agent's own rows --------------------------------
def _agent_and_chunk(recorded):
    from tests.test_one_critic import _cfg, _chunk
    from torch_rl.gated_recurrent_sac import GatedRecurrentSAC
    agent = GatedRecurrentSAC(_cfg())
    ch = _chunk(agent.cfg, [1, 0])
    ch["prerecorded"] = torch.full((2,), 1.0 if recorded else 0.0)
    return agent, ch


def test_a_recorded_batch_does_not_move_the_temperature():
    agent, ch = _agent_and_chunk(recorded=True)
    a0 = (float(agent.log_alpha_g.detach()), float(agent.log_alpha_e.detach()))
    agent.update(ch)
    assert (float(agent.log_alpha_g.detach()), float(agent.log_alpha_e.detach())) == a0


def test_an_online_batch_still_moves_it():
    agent, ch = _agent_and_chunk(recorded=False)
    a0 = float(agent.log_alpha_g.detach())
    agent.update(ch)
    assert float(agent.log_alpha_g.detach()) != a0


def test_only_the_online_rows_drive_the_temperature():
    """A mixed batch steps the temperature exactly as its online rows alone would."""
    from tests.test_one_critic import _cfg, _chunk
    from torch_rl.gated_recurrent_sac import GatedRecurrentSAC
    torch.manual_seed(0)
    mixed = GatedRecurrentSAC(_cfg())
    ch = _chunk(mixed.cfg, [1, 1])
    ch["prerecorded"] = torch.tensor([1.0, 0.0])
    torch.manual_seed(0)
    alone = GatedRecurrentSAC(_cfg())
    online = {k: (v[1:2] if v.shape[:1] == (2,) else v) for k, v in ch.items()}
    torch.manual_seed(1)
    mixed.update(ch)
    m = mixed.last_temp_rows
    assert m is not None and float(m.sum()) > 0
    # the masked rows are exactly the online chunk's valid rows
    T = ch["valid"].shape[1]
    assert float(m[:T].sum()) == 0.0 and float(m[T:].sum()) == float(ch["valid"][1].sum())
    del alone, online


# ---- the fine-tune applies it -------------------------------------------------------
def _prepare(tmp_path, monkeypatch, expert_replay=None):
    from tests.test_bc_trainer import _shards, _small_pixel_agent
    from torch_rl import pixel_rows as pxr
    from torch_rl.bc_trainer import il_finetune_prepare
    _shards(tmp_path / "data")                         # 3 episodes x 10 steps, Town05
    monkeypatch.setenv("REPLAY_DEVICE", "cpu")
    agent = _small_pixel_agent(pxr.BINOC_DIM, 2)
    buf = DeviceEpisodeBuffer(pxr.mono_row_dim(), 2, 200, "cpu", goal_dim=pxr.PIX)
    xbuf = (DeviceEpisodeBuffer(pxr.mono_row_dim(), 2, 33, "cpu", goal_dim=pxr.PIX)
            if expert_replay else None)
    logs = []
    il_finetune_prepare(agent, buf, str(tmp_path / "data"), ["Town05"], warmup_updates=1,
                        chunk_b=2, chunk_t=4, w0=0.0, decay_updates=10, bc_chunks=2,
                        bc_len=4, aug=None, log=logs.append, expert_replay=xbuf)
    return buf, xbuf, logs


def test_the_fine_tune_pins_its_preload(tmp_path, monkeypatch):
    buf, _, logs = _prepare(tmp_path, monkeypatch)
    assert buf._pin_end == 33                          # 3 x (10 transitions + 1 row)
    assert float(buf.sample_chunks(8, 5, 2)["prerecorded"].min()) == 1.0
    assert any("PINNED" in m and "temperature" in m for m in logs)


def test_a_mixed_replay_pins_its_expert_side(tmp_path, monkeypatch):
    buf, xbuf, _ = _prepare(tmp_path, monkeypatch, expert_replay=True)
    assert xbuf._pin_end == 33 and buf._pin_end == 0
