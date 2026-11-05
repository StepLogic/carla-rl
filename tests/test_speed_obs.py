"""Speed in the policy's observation (user 2026-11-05: "add speed").

Without it the clone cannot tell a standstill from cruising in one frame, and
DAgger's speed-dependent labels (brake just above 15 km/h, full throttle just
below) contradict each other on identical-looking frames: the cloning loss
averaged them into ~0.08 throttle and the round-1 clone crawled at 2-5 km/h.

SPEED_OBS=1 puts speed/20 between the mode bit and the binocular features,
[frame | mode | speed | binoc], so it reaches the actor directly and the GRU
through core_state_cols. Off by default: every existing checkpoint keeps its
layout.
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from torch_rl import pixel_rows as pxr  # noqa: E402

D, P = pxr.BINOC_DIM, pxr.MONO_PIX


@pytest.fixture
def speed_on(monkeypatch):
    monkeypatch.setattr(pxr, "SPEED_OBS", True)


def _frame():
    return np.zeros((128, 128, 3), np.uint8)


def test_speed_sits_between_the_mode_bit_and_the_binoc(speed_on):
    b = np.arange(D, dtype=np.float32)
    row = pxr.mono_row(_frame(), 1.0, b, speed_kmh=15.0)
    assert pxr.mono_row_dim() == P + 2 + D and row.shape == (P + 2 + D,)
    assert row[P] == 1.0
    assert row[P + 1] == pytest.approx(15.0 / pxr.SPEED_SCALE_KMH)
    assert np.array_equal(row[P + 2:], b)


def test_speed_reaches_the_gru_as_part_of_the_state_block(speed_on):
    assert pxr.mono_state_dim() == 2 + D
    assert pxr.mono_core_state_cols() == tuple(range(1, 2 + D))


def test_a_row_without_speed_is_refused_when_speed_is_on(speed_on):
    with pytest.raises(ValueError, match="speed"):
        pxr.mono_row(_frame(), 1.0, np.zeros(D, np.float32))


def test_off_by_default_the_layout_is_unchanged():
    assert not pxr.SPEED_OBS
    row = pxr.mono_row(_frame(), 1.0, np.zeros(D, np.float32), speed_kmh=15.0)
    assert row.shape == (P + 1 + D,) and pxr.mono_state_dim() == 1 + D


def test_cloning_rows_carry_the_speed_each_frame_was_taken_at(speed_on, tmp_path):
    """Recording stores frame t+1 with the speed from the same step, speed[t];
    frame 0 is the reset frame, taken at rest. Live rows read the same pairing
    from info["speed_kmh"]."""
    from tests.test_expert_data import _ep
    from torch_rl.episode_buffer import DeviceEpisodeBuffer
    from torch_rl.expert_data import fill_buffer, save_episode
    ep = _ep(T=4)
    ep["speed"] = np.array([3.0, 6.0, 9.0, 12.0], np.float32)
    p = str(tmp_path / "ep_100000.npz")
    save_episode(p, ep)
    buf = DeviceEpisodeBuffer(pxr.mono_row_dim(), 2, 50, "cpu", goal_dim=pxr.PIX)
    assert fill_buffer(buf, [p]) == 4
    start, n, _ = buf._episodes[0]
    got = buf.obs[start:start + n + 1, P + 1].float().numpy() * pxr.SPEED_SCALE_KMH
    assert np.allclose(got, [0.0, 3.0, 6.0, 9.0, 12.0], atol=0.02)


def test_hindsight_rows_keep_their_speed(speed_on):
    """HER rewrites the mode bit and the binoc for the hindsight goal; the speed
    is the car's, not the goal's, and must survive the rewrite."""
    from tests.test_image_her import FakeEnc
    from tests.test_image_her import _frame as hframe
    from torch_rl.episode_buffer import DeviceEpisodeBuffer
    from torch_rl.image_her import ImageGoalHER
    buf = DeviceEpisodeBuffer(pxr.mono_row_dim(), 2, 5000, "cpu", goal_dim=pxr.PIX)
    her = ImageGoalHER(buf, FakeEnc(), pxr, seed=0, n_relabel=1)
    speeds = [30.0] * 60
    speeds[40] = 5.0
    her.begin(0, {}, 0)
    for t in range(60):
        info_t = {"debit_cur": hframe(t)}
        info_n = {"debit_cur": hframe(t + 1), "hero_pos": np.array([0.5 * (t + 1), 0.0]),
                  "speed_kmh": speeds[t], "gated_shaping": 0.3}
        row = pxr.mono_row(hframe(t), 0.0, np.zeros(D, np.float32),
                           speed_kmh=0.0 if t == 0 else speeds[t - 1])
        nrow = pxr.mono_row(hframe(t + 1), 0.0, np.zeros(D, np.float32), speed_kmh=speeds[t])
        her.record(0, row, np.zeros(2, np.float32), 0.2, False, False, nrow, info_t, info_n)
    her.flush(0, terminated=False, truncated=False)
    (start, n, mode), = buf._episodes
    rows = buf.obs[start:start + n].float().numpy()
    assert mode == 1 and np.all(rows[:, P] == 1.0)
    sp = rows[:, P + 1] * pxr.SPEED_SCALE_KMH
    assert np.all(np.isclose(sp[:, None], [0.0, 30.0, 5.0], atol=0.05).any(axis=1))
    assert np.allclose(rows[:, P + 2], rows[:, -1])          # binoc tail rewritten whole
    assert not np.allclose(rows[:, -1], 0.0)                  # ... with the re-encoded values


def test_a_cloning_run_with_speed_trains_and_saves_the_wider_learner(speed_on, tmp_path, monkeypatch):
    from tests.test_bc_trainer import _bc_env, _shards, _small_pixel_agent
    from torch_rl.bc_trainer import run_bc_train
    _shards(tmp_path / "data")
    _bc_env(monkeypatch, tmp_path / "data")
    run_bc_train(str(tmp_path / "out"), make_agent=_small_pixel_agent)
    again = _small_pixel_agent(D, 2)
    assert again.cfg.state_dim == 2 + D
    again.load(str(tmp_path / "out" / "step_3"))      # strict: widths match
