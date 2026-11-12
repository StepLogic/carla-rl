"""Replay in host RAM, with the camera frame stored as uint8.

User 2026-10-19: "move to main memory". The L40 holds ~350k replay rows in
fp16 next to CARLA; the node has 500 GB of RAM. The /255 camera frame is 49,152
of each row's 52,289 values and is k/255 for integer k (pixel_rows.mono_row), so
storing it as uint8 levels is EXACT and halves the row.

These tests run on CPU only (subprocess tests must stay off the GPU): the store
device and the delivery device are both "cpu" here, which exercises the same
gather-then-move code path the cluster runs with delivery on "cuda".
"""
import numpy as np
import pytest
import torch

from torch_rl.episode_buffer import DeviceEpisodeBuffer

PIXELS, REST = 12, 5           # a toy row: 12 pixel columns then 5 state columns
D = PIXELS + REST


def _row(rng):
    pix = rng.integers(0, 256, PIXELS) / 255.0          # exact k/255, as mono_row
    rest = rng.normal(size=REST)
    return np.concatenate([pix, rest]).astype(np.float32)


def _fill(buf, seed=0, episodes=6):
    rng = np.random.default_rng(seed)
    for e in range(episodes):
        n = int(rng.integers(20, 40))
        mode = e % 2
        obs = _row(rng)
        for t in range(n):
            nxt = _row(rng)
            buf.add(obs, rng.uniform(-1, 1, 2), float(rng.normal()),
                    terminal=(t == n - 1), truncated=False, next_obs=nxt, mode=mode)
            obs = nxt


def _pair(**kw):
    a = DeviceEpisodeBuffer(D, 2, 500, "cpu", seed=3, **kw)
    b = DeviceEpisodeBuffer(D, 2, 500, "cpu", seed=3, store_device="cpu",
                            u8_cols=PIXELS, **kw)
    _fill(a)
    _fill(b)
    return a, b


def test_u8_samples_equal_fp16_samples():
    """Same data, same seed: the uint8 store returns what the fp16 store does
    (to fp16 precision, which is what the fp16 store itself holds)."""
    a, b = _pair()
    sa = a.sample_chunks(8, 12, 4)
    sb = b.sample_chunks(8, 12, 4)
    for k in ("actions", "rewards", "masks", "valid", "keep", "mode"):
        assert torch.equal(sa[k], sb[k]), k
    assert torch.allclose(sa["obs"], sb["obs"], atol=1e-3)


def test_the_pixel_round_trip_is_exact():
    """k/255 -> uint8 -> /255 gives back k/255 exactly in float32 -- tighter
    than the fp16 store, which rounds it."""
    _, b = _pair()
    s = b.sample_chunks(8, 12, 4)
    pix = s["obs"][..., :PIXELS]
    levels = pix * 255.0
    assert torch.allclose(levels, torch.round(levels), atol=1e-4)


def test_non_pixel_columns_are_refused_not_corrupted():
    buf = DeviceEpisodeBuffer(D, 2, 200, "cpu", u8_cols=PIXELS)
    bad = np.full(D, 0.3, np.float32)                  # 0.3*255 = 76.5: half-way between levels
    with pytest.raises(ValueError, match="pixel levels"):
        for t in range(3):
            buf.add(bad, np.zeros(2), 0.0, terminal=(t == 2), truncated=False,
                    next_obs=bad, mode=1)


def test_the_store_is_smaller():
    a = DeviceEpisodeBuffer(D, 2, 1000, "cpu")
    b = DeviceEpisodeBuffer(D, 2, 1000, "cpu", u8_cols=PIXELS)
    assert b.nbytes < a.nbytes
    # pixel columns go from 2 bytes to 1
    assert a.nbytes - b.nbytes == 1000 * PIXELS


def test_goal_rows_follow_the_store_and_arrive_on_the_learner_device():
    buf = DeviceEpisodeBuffer(D, 2, 500, "cpu", store_device="cpu",
                              u8_cols=PIXELS, goal_dim=7)
    rng = np.random.default_rng(0)
    for e in range(4):
        obs = _row(rng)
        for t in range(25):
            nxt = _row(rng)
            buf.add(obs, np.zeros(2), 0.0, terminal=(t == 24), truncated=False,
                    next_obs=nxt, mode=e % 2, goal=np.arange(7, dtype=np.float32))
            obs = nxt
    s = buf.sample_chunks(4, 10, 2)
    assert s["goal"].shape == (4, 7) and s["goal"].dtype == torch.float32
    assert s["goal"].device == buf.device


def test_defaults_are_the_old_buffer():
    buf = DeviceEpisodeBuffer(D, 2, 100, "cpu")
    assert buf.u8_cols == 0 and buf.store_device == buf.device


def test_full_size_row_bytes():
    """The arithmetic quoted to the user: 52,289-value rows, 49,152 of them
    pixels, ~55 KB a row with uint8 frames vs ~105 KB in fp16."""
    from torch_rl import pixel_rows as pxr
    d = pxr.mono_row_dim()
    fp16 = d * 2
    u8 = pxr.MONO_PIX + (d - pxr.MONO_PIX) * 2
    assert d == 52289 and pxr.MONO_PIX == 49152
    assert 104_000 < fp16 < 105_000 and 55_000 < u8 < 56_000
