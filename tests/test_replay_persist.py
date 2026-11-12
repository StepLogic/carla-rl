"""Replay buffer survives save -> resume (user 2026-09-28)."""
import numpy as np
import pytest

pytest.importorskip("jaxrl2")
spaces = pytest.importorskip("gymnasium").spaces

from vae_nav.mode_split_buffer import ModeSplitBuffer  # noqa: E402
from vae_nav import replay_persist as rp  # noqa: E402

OBS = spaces.Dict({"latent": spaces.Box(-np.inf, np.inf, (8,), np.float32),
                   "mode": spaces.Box(0.0, 1.0, (1,), np.float32)})
ACT = spaces.Box(-1.0, 1.0, (2,), np.float32)


def _filled(n=40):
    b = ModeSplitBuffer(OBS, ACT, 200)
    rng = np.random.default_rng(0)
    for i in range(n):
        m = i % 2
        o = {"latent": rng.normal(size=8).astype(np.float32),
             "mode": np.array([float(m)], np.float32)}
        b.insert(dict(observations=o, next_observations=o,
                      actions=rng.uniform(-1, 1, 2).astype(np.float32),
                      rewards=float(i), masks=1.0, dones=0.0), mode=m)
    return b


def test_round_trip(tmp_path):
    b = _filled()
    assert rp.save_replay(b, str(tmp_path), 450000, reserve_bytes=0)
    fresh = ModeSplitBuffer(OBS, ACT, 200)
    keep = fresh                                   # a reference held elsewhere
    assert rp.load_replay_into(fresh, str(tmp_path), 450000)
    assert len(keep) == len(b) == 40 and keep.counts() == b.counts()
    s = keep.sample(16)
    assert s["actions"].shape == (16, 2)


def test_refuses_when_disk_is_short(tmp_path, monkeypatch):
    b = _filled()
    monkeypatch.setattr(rp.shutil, "disk_usage",
                        lambda p: type("U", (), {"free": 10})())
    assert rp.save_replay(b, str(tmp_path), 1) is None


def test_missing_file_is_empty_start(tmp_path):
    assert not rp.load_replay_into(ModeSplitBuffer(OBS, ACT, 200), str(tmp_path), 5)
