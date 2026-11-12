"""PER (user 2026-11-06: "use PER"): prioritized experience replay for the
recurrent chunk replay, for a pure-SAC fine-tune of the clone (no noise pin,
no cloning).

Every row of an episode starts exactly one trained window, so priorities live
per row: new rows (expert preload, HER legs, fresh driving) enter at the
current maximum priority, evicted rows drop out, sampling keeps the goal /
explore strata and draws window starts with probability p^alpha, each chunk
carries an importance weight (N P)^-beta normalised to max 1 (beta rising from
beta0 to 1), the SAC critic loss is weighted by it, and the update returns each
chunk's TD priority (0.9 max + 0.1 mean |delta| over its valid transitions, as
in R2D2) for the buffer. PER_ALPHA = 0 (the default) samples exactly as before.
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


def _buf(capacity=200, **kw):
    return DeviceEpisodeBuffer(3, 2, capacity, "cpu", seed=0, **kw)


def _episode(buf, n, mode, value):
    for t in range(n):
        o = np.full(3, value, np.float32)
        buf.add(o, np.zeros(2, np.float32), 0.0, t == n - 1, False, o, mode)


def test_per_is_off_by_default_and_the_chunk_keys_are_unchanged():
    b = _buf()
    _episode(b, 10, 1, 1.0)
    out = b.sample_chunks(4, 5, 2)
    assert "is_weight" not in out and "per_rows" not in out


def test_high_priority_windows_are_sampled_more():
    b = _buf(per_alpha=1.0, goal_batch_frac=1.0)
    _episode(b, 10, 1, 1.0)                       # window-start rows 0..9
    _episode(b, 10, 1, 2.0)                       # window-start rows 11..20
    b.update_priorities(np.arange(0, 10), np.full(10, 1.0))
    b.update_priorities(np.arange(11, 21), np.full(10, 9.0))
    rows = np.concatenate([b.sample_chunks(8, 5, 2)["per_rows"].numpy() for _ in range(300)])
    assert 0.85 < np.mean(rows >= 11) < 0.95      # about 9 / 10


def test_new_rows_enter_at_the_current_max_priority():
    b = _buf(per_alpha=1.0, goal_batch_frac=1.0)
    _episode(b, 10, 1, 1.0)
    b.update_priorities(np.arange(0, 10), np.full(10, 0.01))   # low, max stays 1
    _episode(b, 10, 1, 2.0)                                    # enters at max (1.0)
    rows = np.concatenate([b.sample_chunks(8, 5, 2)["per_rows"].numpy() for _ in range(100)])
    assert np.mean(rows >= 11) > 0.9


def test_importance_weights_offset_the_priorities():
    b = _buf(per_alpha=1.0, per_beta0=1.0, goal_batch_frac=1.0)
    _episode(b, 10, 1, 1.0)
    _episode(b, 10, 1, 2.0)
    b.update_priorities(np.arange(0, 10), np.full(10, 1.0))
    b.update_priorities(np.arange(11, 21), np.full(10, 9.0))
    seen_lo, seen_hi = [], []
    for _ in range(100):
        out = b.sample_chunks(8, 5, 2)
        w, rows = out["is_weight"].numpy(), out["per_rows"].numpy()
        assert w.max() == pytest.approx(1.0) and (w > 0).all()
        seen_lo += list(w[rows < 11])
        seen_hi += list(w[rows >= 11])
    assert np.mean(seen_hi) < np.mean(seen_lo)


def test_beta_rises_from_beta0_to_one_and_stays():
    b = _buf(per_alpha=1.0, per_beta0=0.4, per_beta_updates=10)
    _episode(b, 10, 1, 1.0)
    betas = []
    for _ in range(15):
        b.sample_chunks(2, 5, 2)
        betas.append(b.per_beta)
    assert betas[0] == pytest.approx(0.4)
    assert all(x <= y + 1e-12 for x, y in zip(betas, betas[1:]))
    assert betas[-1] == pytest.approx(1.0)


def test_evicted_rows_are_never_sampled():
    b = _buf(capacity=25, per_alpha=1.0, goal_batch_frac=1.0)
    _episode(b, 10, 1, 1.0)                       # rows 0..10
    _episode(b, 10, 1, 2.0)                       # rows 11..21
    _episode(b, 10, 1, 3.0)                       # wraps to row 0: evicts episode 1
    K = 2
    for _ in range(50):
        out = b.sample_chunks(8, 5, K)
        first = out["obs"][:, K, 0].numpy()       # the trained window's first row
        assert set(np.round(first, 3)) <= {2.0, 3.0}


def test_the_mode_strata_are_kept():
    b = _buf(per_alpha=1.0, goal_batch_frac=0.5)
    _episode(b, 10, 1, 1.0)
    _episode(b, 10, 0, 2.0)
    out = b.sample_chunks(8, 5, 2)
    assert int(out["mode"].sum()) == 4


def test_is_weights_scale_the_sac_critic_loss_and_the_update_returns_td_priorities():
    from tests.test_one_critic import _cfg, _chunk
    from torch_rl.gated_recurrent_sac import GatedRecurrentSAC
    agent = GatedRecurrentSAC(_cfg())
    ch = _chunk(agent.cfg, [1, 0], seed=0)
    ch["is_weight"] = torch.zeros(2)
    c0 = [p.detach().clone() for p in agent.critics_g.parameters()]
    agent.update(ch)
    assert all(torch.equal(a, b) for a, b in zip(c0, agent.critics_g.parameters()))
    td = agent.last_td_prio
    assert td.shape == (2,) and (td >= 0).all()
    ch["is_weight"] = torch.ones(2)
    agent.update(ch)
    assert any(not torch.equal(a, b) for a, b in zip(c0, agent.critics_g.parameters()))


def test_without_per_the_update_reports_no_td_priorities():
    from tests.test_one_critic import _cfg, _chunk
    from torch_rl.gated_recurrent_sac import GatedRecurrentSAC
    agent = GatedRecurrentSAC(_cfg())
    agent.update(_chunk(agent.cfg, [1, 0], seed=0))
    assert agent.last_td_prio is None


def test_the_loops_feed_the_td_priorities_back():
    loop = open(os.path.join(ROOT, "torch_rl", "parallel_loop.py")).read()
    prep = open(os.path.join(ROOT, "torch_rl", "bc_trainer.py")).read()
    call = 'update_priorities(chunk["per_rows"], agent.last_td_prio)'
    assert call in loop and call in prep


def test_the_trainer_reads_per_and_refuses_it_with_iql():
    s = open(os.path.join(ROOT, "train_carla_sac.py")).read()
    assert 'per_alpha=env_float("PER_ALPHA", 0.0)' in s
    assert "PER is implemented for ALGO=sac" in s
