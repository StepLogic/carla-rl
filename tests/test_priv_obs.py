"""PRIV_OBS=1 (privileged critic, user 2026-11-07): the replay row gains the
route distance to the goal -- [frame | mode | speed | dist/25 | known | binoc]
-- for the critic alone. The GRU and the actor never read the two columns
(mono_core_state_cols / mono_actor_state_cols leave them out), and the binoc
stays the row's tail, where hindsight relabelling rewrites it."""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from torch_rl import pixel_rows as pxr  # noqa: E402


@pytest.fixture
def priv(monkeypatch):
    monkeypatch.setattr(pxr, "SPEED_OBS", True)
    monkeypatch.setattr(pxr, "PRIV_OBS", True)


def _row(**kw):
    frame = np.zeros((128, 128, 3), np.uint8)
    binoc = np.arange(pxr.BINOC_DIM, dtype=np.float32) / pxr.BINOC_DIM
    return pxr.mono_row(frame, 1.0, binoc, speed_kmh=12.0, **kw), binoc


def test_the_row_carries_distance_and_a_known_flag(priv):
    row, binoc = _row(priv_m=10.0)
    st = row[pxr.MONO_PIX:]
    assert st[0] == 1.0 and st[1] == pytest.approx(12.0 / pxr.SPEED_SCALE_KMH)
    assert st[2] == pytest.approx(10.0 / pxr.PRIV_SCALE_M) and st[3] == 1.0
    assert np.allclose(st[4:], binoc)                    # the binoc is still the tail
    assert len(row) == pxr.mono_row_dim()


def test_an_unknown_distance_is_zero_with_the_flag_down(priv):
    st = _row(priv_m=None)[0][pxr.MONO_PIX:]
    assert st[2] == 0.0 and st[3] == 0.0


def test_a_far_goal_is_clipped(priv):
    st = _row(priv_m=1e4)[0][pxr.MONO_PIX:]
    assert st[2] == pytest.approx(pxr.PRIV_CLIP)


def test_the_gru_and_the_actor_skip_the_privileged_columns(priv):
    S = pxr.mono_state_dim()
    assert S == 1 + 1 + 2 + pxr.BINOC_DIM
    assert pxr.mono_priv_cols() == (2, 3)
    core = pxr.mono_core_state_cols()
    actor = pxr.mono_actor_state_cols()
    assert 2 not in core and 3 not in core and 1 in core and S - 1 in core
    assert actor == tuple(c for c in range(S) if c not in (2, 3))
    assert set(core) <= set(actor) and 0 in actor


def test_off_by_default_nothing_changes(monkeypatch):
    monkeypatch.setattr(pxr, "SPEED_OBS", True)
    monkeypatch.setattr(pxr, "PRIV_OBS", False)
    assert pxr.mono_actor_state_cols() is None and pxr.mono_priv_cols() == ()
    assert pxr.mono_state_dim() == 1 + 1 + pxr.BINOC_DIM
    row, _ = _row()
    assert len(row) == pxr.mono_row_dim()


@pytest.mark.parametrize("speed_obs", [False, True])
def test_off_the_row_is_byte_identical_to_the_earlier_layout(monkeypatch, speed_obs):
    """The mapping session's trace and eval rows (and every earlier checkpoint)
    must not move: with PRIV_OBS off, [frame | mode | (speed) | binoc] exactly,
    whatever priv_m is passed."""
    monkeypatch.setattr(pxr, "SPEED_OBS", speed_obs)
    monkeypatch.setattr(pxr, "PRIV_OBS", False)
    rng = np.random.default_rng(3)
    frame = rng.integers(0, 256, (128, 128, 3), dtype=np.uint8)
    binoc = rng.normal(size=pxr.BINOC_DIM).astype(np.float32)
    img = pxr.to_frame(frame, pxr.MONO_IMG) / 255.0
    pix = (img.transpose(2, 0, 1) if pxr.MONO_CHW else img).reshape(-1)
    parts = [pix, np.asarray([1.0], np.float32)]
    if speed_obs:
        parts.append(np.asarray([13.0 / pxr.SPEED_SCALE_KMH], np.float32))
    parts.append(binoc)
    want = np.concatenate(parts).astype(np.float32)
    for priv_m in (None, 7.5):
        got = pxr.mono_row(frame, 1.0, binoc, speed_kmh=13.0, priv_m=priv_m)
        assert got.dtype == want.dtype and got.tobytes() == want.tobytes()


# ---- recorded episodes (expert_data.fill_buffer) --------------------------------
def _ep(T=6, end="reached", terminal=True, mode=1):
    rng = np.random.default_rng(0)
    return {
        "frames": rng.integers(0, 256, (T + 1, 128, 128, 3), dtype=np.uint8),
        "binoc": rng.normal(size=(T + 1, pxr.BINOC_DIM)).astype(np.float16),
        "act_exec": np.zeros((T, 2), np.float32), "act_label": np.zeros((T, 2), np.float32),
        "reward": np.zeros(T, np.float32), "speed": np.full(T, 12.0, np.float32),
        # one metre a step along x: pos[t] is where the car is AFTER step t
        "pos": np.stack([np.arange(1, T + 1, dtype=np.float32), np.zeros(T, np.float32)], 1),
        "mode": np.int8(mode), "terminal": bool(terminal), "truncated": not terminal,
        "goal": pxr.goal_block(np.zeros((112, 112, 3), np.uint8)),
        "sigma": np.float32(0.0), "driver": "expert", "end": end, "town": "Town05",
        "seed": np.int64(100000)}


def _priv_rows(tmp_path, ep, transform=None):
    pytest.importorskip("torch")
    from torch_rl.episode_buffer import DeviceEpisodeBuffer
    from torch_rl.expert_data import fill_buffer, save_episode
    p = str(tmp_path / "ep_100000.npz")
    save_episode(p, ep)
    buf = DeviceEpisodeBuffer(pxr.mono_row_dim(), 2, 400, "cpu", goal_dim=pxr.PIX)
    fill_buffer(buf, [p], which="exec", transform=transform)
    start, n, _ = buf._episodes[0]
    c = pxr.MONO_PIX + pxr.mono_priv_cols()[0]
    return buf.obs[start:start + n + 1, c:c + 2].float().numpy()


def test_a_reached_episode_gets_the_path_distance_to_where_it_stopped(priv, tmp_path):
    T = 6
    rows = _priv_rows(tmp_path, _ep(T=T))
    dist = rows[:, 0] * pxr.PRIV_SCALE_M
    # frame k (k >= 1) was taken at pos[k-1], T - k metres of path from the stop;
    # the reset frame (k = 0) is read at pos[0]
    assert np.allclose(dist, [T - 1] + [T - k for k in range(1, T + 1)], atol=0.02)
    assert np.all(rows[:, 1] == 1.0)


@pytest.mark.parametrize("end,terminal,mode", [("crossing", True, 1), ("clock", False, 1),
                                               ("route_done", False, 0)])
def test_episodes_without_a_known_goal_are_unknown(priv, tmp_path, end, terminal, mode):
    rows = _priv_rows(tmp_path, _ep(end=end, terminal=terminal, mode=mode))
    assert np.all(rows == 0.0)


def test_hindsight_rows_carry_the_path_distance_to_the_hindsight_goal(priv):
    """HER's goal is a place the car passed: the critic's distance column is the
    path metres to it, the same metres the relabeller pays progress on."""
    pytest.importorskip("torch")
    from tests.test_image_her import FakeEnc
    from tests.test_image_her import _frame as hframe
    from torch_rl.episode_buffer import DeviceEpisodeBuffer
    from torch_rl.image_her import ImageGoalHER
    buf = DeviceEpisodeBuffer(pxr.mono_row_dim(), 2, 5000, "cpu", goal_dim=pxr.PIX)
    her = ImageGoalHER(buf, FakeEnc(), pxr, seed=0, n_relabel=1, radius=2.0)
    speeds = [30.0] * 60
    speeds[40] = 5.0
    her.begin(0, {}, 0)                                # explore, relabelled to goal mode
    for t in range(60):
        info_t = {"debit_cur": hframe(t)}
        info_n = {"debit_cur": hframe(t + 1), "hero_pos": np.array([0.5 * (t + 1), 0.0]),
                  "speed_kmh": speeds[t], "gated_shaping": 0.3}
        row = pxr.mono_row(hframe(t), 0.0, np.zeros(pxr.BINOC_DIM, np.float32),
                           speed_kmh=0.0 if t == 0 else speeds[t - 1])
        nrow = pxr.mono_row(hframe(t + 1), 0.0, np.zeros(pxr.BINOC_DIM, np.float32),
                            speed_kmh=speeds[t])
        her.record(0, row, np.zeros(2, np.float32), 0.2, False, False, nrow, info_t, info_n)
    her.flush(0, terminated=False, truncated=False)
    (start, n, _), = buf._episodes
    c = pxr.MONO_PIX + pxr.mono_priv_cols()[0]
    dist = buf.obs[start:start + n, c].float().numpy() * pxr.PRIV_SCALE_M
    known = buf.obs[start:start + n, c + 1].float().numpy()
    assert np.all(known == 1.0)
    assert np.allclose(np.diff(dist), -0.5, atol=0.05)          # 0.5 m of path a step
    assert 5.0 - 0.5 <= dist[0] <= 25.0 + 0.5                   # the leg's band
    assert dist[-1] <= 2.0 + 0.5 + 0.05                         # the arrival step


def test_the_trainer_builds_the_actor_view_and_live_rows_carry_the_distance():
    s = open(os.path.join(ROOT, "train_carla_sac.py")).read()
    assert "actor_state_cols=pxr.mono_actor_state_cols(binoc_dim)" in s
    assert "priv_m=_priv_m(o, i)" in s


def test_the_banner_states_both_views():
    """Run K's log said the critic's 3140 columns were 'same as the actor' --
    false under PRIV_OBS. The banner must name the actor's view separately."""
    s = open(os.path.join(ROOT, "train_carla_sac.py")).read()
    assert "same as the actor" not in s
    assert "actor view = " in s and "critic view = " in s
    assert "critic only" in s


def test_live_rows_read_the_route_distance_in_goal_mode_only(priv):
    import train_carla_sac as T
    assert T._priv_m({"mode": np.array([1.0], np.float32)}, {"route_remaining_m": 12.5}) == 12.5
    assert T._priv_m({"mode": np.array([0.0], np.float32)}, {"route_remaining_m": 12.5}) is None
    assert T._priv_m({"mode": np.array([1.0], np.float32)}, {"route_remaining_m": None}) is None
    assert T._priv_m({"mode": 1.0}, {}) is None


def test_live_rows_are_unknown_without_priv_obs(monkeypatch):
    import train_carla_sac as T
    monkeypatch.setattr(pxr, "PRIV_OBS", False)
    assert T._priv_m({"mode": np.array([1.0], np.float32)}, {"route_remaining_m": 12.5}) is None


def test_an_arrival_cut_short_is_unknown(priv, tmp_path):
    """slope_diag.cut_arrival leaves end='reached' but the stop is gone: the last
    kept position is not the goal."""
    from torch_rl.slope_diag import cut_arrival
    rows = _priv_rows(tmp_path, _ep(T=40), transform=lambda e: cut_arrival(e, 30))
    assert np.all(rows == 0.0)
