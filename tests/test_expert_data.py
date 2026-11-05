"""Expert episode shards (imitation-first plan, Task 3): one .npz per episode,
validated on save, loadable into the learner's DeviceEpisodeBuffer with either
the expert's clean label actions (cloning) or the executed actions (replay)."""
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
from torch_rl.expert_data import (episode_paths, fill_buffer,  # noqa: E402
                                  load_episode, save_episode, split_by_episode)


def _ep(T=3, mode=1, seed=100000, terminal=True, truncated=False):
    rng = np.random.default_rng(seed)
    return {
        "frames": rng.integers(0, 256, (T + 1, 128, 128, 3), dtype=np.uint8),
        "binoc": rng.normal(size=(T + 1, pxr.BINOC_DIM)).astype(np.float16),
        "act_exec": np.full((T, 2), -0.5, np.float32),
        "act_label": np.full((T, 2), 0.5, np.float32),
        "reward": np.arange(T, dtype=np.float32),
        "speed": np.full(T, 12.0, np.float32),
        "pos": np.zeros((T, 2), np.float32),
        "mode": np.int8(mode),
        "terminal": bool(terminal),
        "truncated": bool(truncated),
        "goal": pxr.goal_block(np.zeros((112, 112, 3), np.uint8)),
        "sigma": np.float32(0.1),
        "driver": "expert",
        "end": "reached",
        "town": "Town05",
        "seed": np.int64(seed),
    }


def _buf():
    return DeviceEpisodeBuffer(pxr.mono_row_dim(), 2, 400, "cpu", goal_dim=pxr.PIX)


def test_round_trip(tmp_path):
    ep = _ep(T=4)
    p = str(tmp_path / "ep_100000.npz")
    save_episode(p, ep)
    back = load_episode(p)
    for k in ("frames", "binoc", "act_exec", "act_label", "reward", "speed", "pos", "goal"):
        assert np.array_equal(back[k], ep[k]) and back[k].dtype == ep[k].dtype, k
    for k in ("mode", "terminal", "truncated", "sigma", "seed"):
        assert back[k] == ep[k], k
    for k in ("driver", "end", "town"):
        assert back[k] == ep[k] and isinstance(back[k], str), k


def test_save_rejects_wrong_frame_shape(tmp_path):
    ep = _ep(T=3)
    ep["frames"] = np.zeros((4, 120, 120, 3), np.uint8)
    with pytest.raises(ValueError, match="frames"):
        save_episode(str(tmp_path / "e.npz"), ep)


def test_save_rejects_empty_episode(tmp_path):
    ep = _ep(T=3)
    ep.update(frames=ep["frames"][:1], binoc=ep["binoc"][:1], act_exec=ep["act_exec"][:0],
              act_label=ep["act_label"][:0], reward=ep["reward"][:0],
              speed=ep["speed"][:0], pos=ep["pos"][:0])
    with pytest.raises(ValueError, match="empty"):
        save_episode(str(tmp_path / "e.npz"), ep)


@pytest.mark.parametrize("which,value", [("label", 0.5), ("exec", -0.5)])
def test_fill_buffer_uses_the_requested_actions(tmp_path, which, value):
    p = str(tmp_path / "ep_1.npz")
    save_episode(p, _ep(T=6))
    buf = _buf()
    assert fill_buffer(buf, [p], which=which) == 6
    ch = buf.sample_chunks(4, 5, 2)
    a, v = ch["actions"].numpy(), ch["valid"].numpy() > 0
    assert v.any() and np.allclose(a[v], value)


def test_fill_buffer_skips_empty_episodes(tmp_path):
    good = str(tmp_path / "ep_1.npz")
    save_episode(good, _ep(T=5))
    bad = str(tmp_path / "ep_2.npz")
    e = _ep(T=1)
    np.savez(bad, **{**e, "frames": e["frames"][:1], "binoc": e["binoc"][:1],
                     "act_exec": e["act_exec"][:0], "act_label": e["act_label"][:0],
                     "reward": e["reward"][:0], "speed": e["speed"][:0], "pos": e["pos"][:0]})
    buf = _buf()
    assert fill_buffer(buf, [good, bad]) == 5
    assert buf.episodes == 1


def _stop_tail_ep(T=30, cut=10, kmh=12.0):
    """A goal episode whose expert label turns into the final-approach stop
    (steer 0, throttle 0 env = actor [0, -1]) at row `cut` while the car keeps
    driving -- what DAgger recorded once the clone had passed the goal."""
    ep = _ep(T=T)
    ep["act_label"][cut:] = np.array([0.0, -1.0], np.float32)
    ep["speed"][:] = np.float32(kmh)
    ep["terminal"], ep["truncated"] = False, True
    return ep


def test_label_fill_cuts_a_stop_forever_tail(tmp_path):
    """DAgger r1: 9 episodes held 98% of the stop labels, ~950 rows each, all
    given while the clone drove on past the goal -- they taught it to crawl."""
    p = str(tmp_path / "ep_300000.npz")
    save_episode(p, _stop_tail_ep(T=30, cut=10))
    buf = _buf()
    assert fill_buffer(buf, [p], which="label") == 10
    ch = buf.sample_chunks(8, 6, 2)
    a, v = ch["actions"].numpy(), ch["valid"].numpy() > 0
    assert v.any() and not np.any(np.all(a[v] == [0.0, -1.0], axis=-1))


def test_isolated_stop_labels_are_kept(tmp_path):
    """A single stop at the goal or at a route's end is a real label."""
    ep = _ep(T=12)
    ep["act_label"][5] = np.array([0.0, -1.0], np.float32)
    ep["act_label"][11] = np.array([0.0, -1.0], np.float32)
    p = str(tmp_path / "ep_300001.npz")
    save_episode(p, ep)
    assert fill_buffer(_buf(), [p], which="label") == 12


def test_a_stopped_car_keeps_its_stop_labels(tmp_path):
    """The cut needs the car MOVING: at a standstill a stop label is no lie."""
    p = str(tmp_path / "ep_300002.npz")
    save_episode(p, _stop_tail_ep(T=20, cut=5, kmh=0.0))
    assert fill_buffer(_buf(), [p], which="label") == 20


def test_replay_fill_keeps_every_executed_step(tmp_path):
    """The cut is a LABEL fix: the transitions themselves stay valid for replay."""
    p = str(tmp_path / "ep_300003.npz")
    save_episode(p, _stop_tail_ep(T=30, cut=10))
    assert fill_buffer(_buf(), [p], which="exec") == 30


def test_episode_paths_are_sorted_per_town(tmp_path):
    for town, seeds in (("Town05", (3, 1)), ("Town01", (2,))):
        os.makedirs(tmp_path / town, exist_ok=True)
        for s in seeds:
            save_episode(str(tmp_path / town / f"ep_{s}.npz"), _ep(T=2, seed=s))
    got = episode_paths(str(tmp_path), ["Town05", "Town01"])
    assert got == sorted(got) and len(got) == 3


def test_split_is_deterministic_and_by_episode():
    paths = [f"/x/ep_{i}.npz" for i in range(40)]
    a = split_by_episode(paths, 0.05, seed=0)
    b = split_by_episode(paths, 0.05, seed=0)
    assert a == b
    train, val = a
    assert not set(train) & set(val) and len(train) + len(val) == 40 and len(val) >= 1
    assert len(split_by_episode(paths[:2], 0.05, seed=0)[1]) == 1
