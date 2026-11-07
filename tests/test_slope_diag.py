"""Offline critic-slope diagnosis (user 2026-11-07, "Diagnose the slope first").

Every SAC fine-tune of c0 collapsed once its actor trained, and the warm-up
critic already scored "steer left + less throttle" above the expert's own action
before the actor moved. scripts/offline_critic_slope.py re-trains that critic
offline with ONE candidate cause removed per variant; these are the episode
transforms that remove them, applied while the recorded episodes fill the
replay (expert_data.fill_buffer's `transform` hook)."""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))
torch = pytest.importorskip("torch")

from torch_rl import pixel_rows as pxr  # noqa: E402
from torch_rl.episode_buffer import DeviceEpisodeBuffer  # noqa: E402
from torch_rl.expert_data import fill_buffer, save_episode  # noqa: E402
from torch_rl.slope_diag import VARIANTS, cut_arrival, expert_driven  # noqa: E402


def _ep(T=6, end="reached", terminal=True, driver="expert", seed=100000):
    rng = np.random.default_rng(seed)
    return {
        "frames": rng.integers(0, 256, (T + 1, 128, 128, 3), dtype=np.uint8),
        "binoc": rng.normal(size=(T + 1, pxr.BINOC_DIM)).astype(np.float16),
        "act_exec": np.full((T, 2), -0.5, np.float32),
        "act_label": np.full((T, 2), 0.5, np.float32),
        "reward": np.arange(T, dtype=np.float32),
        "speed": np.full(T, 12.0, np.float32),
        "pos": np.zeros((T, 2), np.float32),
        "mode": np.int8(1),
        "terminal": bool(terminal),
        "truncated": not terminal,
        "goal": pxr.goal_block(np.zeros((112, 112, 3), np.uint8)),
        "sigma": np.float32(0.1),
        "driver": driver,
        "end": end,
        "town": "Town05",
        "seed": np.int64(seed),
    }


def _buf():
    return DeviceEpisodeBuffer(pxr.mono_row_dim(), 2, 400, "cpu", goal_dim=pxr.PIX)


def _rewards(buf):
    ch = buf.sample_chunks(16, 5, 2)
    return set(np.round(ch["rewards"].numpy()[ch["valid"].numpy() > 0], 3).tolist())


def test_fill_buffer_applies_a_transform(tmp_path):
    p = str(tmp_path / "ep_1.npz")
    save_episode(p, _ep(T=6))

    def double(ep):
        ep = dict(ep)
        ep["reward"] = ep["reward"] * 2.0
        return ep

    buf = _buf()
    assert fill_buffer(buf, [p], which="exec", transform=double) == 6
    assert _rewards(buf) <= {0.0, 2.0, 4.0, 6.0, 8.0, 10.0}
    assert _rewards(buf) & {6.0, 8.0, 10.0}          # doubled values, not the originals


def test_fill_buffer_skips_an_episode_the_transform_drops(tmp_path):
    keep, drop = str(tmp_path / "ep_1.npz"), str(tmp_path / "ep_2.npz")
    save_episode(keep, _ep(T=5, driver="expert"))
    save_episode(drop, _ep(T=7, driver="policy"))
    buf = _buf()
    assert fill_buffer(buf, [keep, drop], which="exec", transform=expert_driven) == 5
    assert buf.episodes == 1


def test_cut_arrival_removes_the_approach_and_the_bonus():
    ep = _ep(T=40)
    ep["reward"][-1] = 100.0
    out = cut_arrival(ep, 30)
    assert len(out["act_exec"]) == len(out["act_label"]) == len(out["reward"]) == 10
    assert len(out["speed"]) == len(out["pos"]) == 10
    assert len(out["frames"]) == len(out["binoc"]) == 11
    assert out["reward"].max() < 100.0
    # the shortened episode ends TRUNCATED: its last state is not a terminal one
    assert out["terminal"] is False and out["truncated"] is True
    # the loaded episode is not modified in place
    assert ep["terminal"] is True and len(ep["act_exec"]) == 40 and ep["reward"][-1] == 100.0


def test_cut_arrival_leaves_episodes_that_did_not_arrive():
    ep = _ep(T=40, end="crossing")
    assert cut_arrival(ep, 30) is ep


def test_cut_arrival_drops_an_arrival_shorter_than_the_cut():
    assert cut_arrival(_ep(T=20), 30) is None
    assert cut_arrival(_ep(T=30), 30) is None


def test_expert_driven_keeps_only_the_experts_episodes():
    ep = _ep(driver="expert")
    assert expert_driven(ep) is ep
    assert expert_driven(_ep(driver="policy")) is None


def test_variants_name_one_change_each():
    assert VARIANTS["baseline"] is None
    assert set(VARIANTS) >= {"baseline", "no_arrival", "expert_only"}
    ep = _ep(T=40)
    assert len(VARIANTS["no_arrival"](ep)["act_exec"]) < 40
