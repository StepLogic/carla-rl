"""fp16 replay-buffer storage for the pixels key (v8 buffer-cost mitigation).

WHAT THIS CLOSES. The v8 buffer stores the (7, 7, 768) VC-1 token grid per
step: 37632 floats ~= 150 KB in float32, ~68 GB at the 450k-step run (and the
jaxrl2 ReplayBuffer preallocates a next_observations mirror besides). The
mitigation is a dtype knob: BUFFER_PIXELS_FP16=1 stores ONLY the `pixels`
obs key as float16 and restores float32 on sample. The goal/mode/state keys
and every HER-relabelled field stay float32 -- the goal block is what relabel
arithmetic touches, and it must not lose precision.

WHERE THE CONVERSION LIVES. EpisodeRelabelBuffer is the shared record path
for both it and ImageGoalRelabelBuffer: every row -- on-policy, non-HER, and
hindsight -- reaches storage through buffer.insert, and every batch leaves
through buffer.sample. Fp16PixelsBuffer therefore wraps the replay buffer
itself (not the relabel buffer), so both subclasses and the non-HER path are
covered by one mechanism keyed on the obs key name "pixels" only.

WHY THE TEST ALSO PINS THE OBS SPACE. Wrapping alone is not enough: jaxrl2's
ReplayBuffer preallocates its storage arrays at the observation space's
per-key dtype (`_init_replay_dict`, replay_buffer.py:14-25), so float16
STORAGE requires the buffer to be built with a float16 `pixels` space. The
trainer does that via train_carla_tqc.fp16_pixel_obs_space; these tests
mirror that wiring with a real ReplayBuffer.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

pytest.importorskip("jaxrl2")

from jaxrl2.data import ReplayBuffer  # noqa: E402

from vae_nav.episode_her import EpisodeRelabelBuffer, Fp16PixelsBuffer  # noqa: E402
from vae_nav.image_goal_her import ImageGoalRelabelBuffer  # noqa: E402

# Pixel values whose fp16 round-trip error is far inside the 1e-2 tolerance.
_PIXEL = 0.75


def _obs_space(pixels_dtype=np.float32):
    from gymnasium import spaces
    return spaces.Dict({
        "goal": spaces.Box(-np.inf, np.inf, (4,), np.float32),
        "pixels": spaces.Box(-np.inf, np.inf, (7, 7, 768), pixels_dtype),
        "mode": spaces.Box(0.0, 1.0, (1,), np.float32),
        "state": spaces.Box(-np.inf, np.inf, (9,), np.float32)})


_ACT = None


def _act_space():
    global _ACT
    if _ACT is None:
        from gymnasium import spaces
        _ACT = spaces.Box(np.array([0.0, 0.0], np.float32),
                          np.array([1.0, 1.0], np.float32))
    return _ACT


def _pixels(fill):
    px = np.full((7, 7, 768), fill, np.float32)
    px[0, 0, 0] = fill + 0.001  # not constant, so a mix-up cannot alias
    return px


def _tr(fill):
    return dict(
        observations={"goal": np.zeros((4,), np.float32),
                      "pixels": _pixels(fill),
                      "mode": np.array([1.0], np.float32),
                      "state": np.zeros((9,), np.float32)},
        actions=np.zeros((2,), np.float32),
        rewards=np.float32(-0.1), masks=np.float32(1.0),
        dones=np.float32(0.0),
        next_observations={"goal": np.zeros((4,), np.float32),
                           "pixels": _pixels(fill),
                           "mode": np.array([1.0], np.float32),
                           "state": np.zeros((9,), np.float32)})


def _goal_block(fields):
    return np.full(4, float(fields["goal_pos"][0]), np.float32)


def _stored(buf):
    """The raw storage arrays behind a wrapped (or bare) ReplayBuffer."""
    inner = buf._inner if isinstance(buf, Fp16PixelsBuffer) else buf
    return inner.dataset_dict


@pytest.fixture
def clean_flag(monkeypatch):
    monkeypatch.delenv("BUFFER_PIXELS_FP16", raising=False)
    return monkeypatch


# -- flag on ---------------------------------------------------------------

def test_flag_on_stores_pixels_fp16_and_goals_f32(clean_flag, monkeypatch):
    monkeypatch.setenv("BUFFER_PIXELS_FP16", "1")
    inner = ReplayBuffer(_obs_space(np.float16), _act_space(), 16)
    buf = Fp16PixelsBuffer(inner)
    her = EpisodeRelabelBuffer(
        buf, _goal_block, goal_radius=2.0, k_goal=0.0, k_collision=0.0,
        time_to_goal=True, step_cost=0.1, n_relabel=1, min_disp=5.0,
        max_steps=100)
    for step, x in enumerate((0.0, 12.0, 24.0)):
        her.record(_tr(x), fields={"goal_pos": np.array([x, 0.0])},
                   pos=np.array([x, 0.0]), hit=False)
    her.flush_episode()
    assert len(buf) >= 3  # the hindsight relabel landed too

    stored = _stored(buf)
    assert stored["observations"]["pixels"].dtype == np.float16
    assert stored["next_observations"]["pixels"].dtype == np.float16
    for field in ("observations", "next_observations"):
        for key in ("goal", "mode", "state"):
            assert stored[field][key].dtype == np.float32, (field, key)


def test_flag_on_samples_pixels_f32_within_fp16_tolerance(
        clean_flag, monkeypatch):
    monkeypatch.setenv("BUFFER_PIXELS_FP16", "1")
    inner = ReplayBuffer(_obs_space(np.float16), _act_space(), 16)
    buf = Fp16PixelsBuffer(inner)
    originals = []
    for step, x in enumerate((0.0, 12.0, 24.0)):
        tr = _tr(x)
        originals.append(tr["observations"]["pixels"])
        her = EpisodeRelabelBuffer(
            buf, _goal_block, goal_radius=2.0, k_goal=0.0, k_collision=0.0,
            time_to_goal=True, step_cost=0.1, n_relabel=1, min_disp=5.0,
            max_steps=100)
        her.record(tr, fields={"goal_pos": np.array([x, 0.0])},
                   pos=np.array([x, 0.0]), hit=False)
        her.flush_episode()

    batch = buf.sample(8)
    px = np.asarray(batch["observations"]["pixels"])
    assert px.dtype == np.float32
    assert np.asarray(batch["next_observations"]["pixels"]).dtype == np.float32
    # Every sampled row must round-trip to one of the recorded originals
    # within the fp16 tolerance the brief sets.
    worst = min(
        np.max(np.abs(px[i] - orig))
        for i in range(px.shape[0]) for orig in originals)
    assert worst <= 1e-2


# -- flag off: byte-identical to today -------------------------------------

def test_flag_off_stores_float32_byte_identical(clean_flag):
    inner = ReplayBuffer(_obs_space(np.float32), _act_space(), 16)
    buf = Fp16PixelsBuffer(inner)  # wrapped, but the flag is unset: inert
    tr = _tr(0.0)
    buf.insert(tr)
    stored = _stored(buf)
    assert stored["observations"]["pixels"].dtype == np.float32
    assert np.array_equal(stored["observations"]["pixels"][0],
                          tr["observations"]["pixels"])


def test_flag_off_her_path_unchanged(clean_flag):
    """Today's path, no wrapper: the HER record/flush stores float32 and the
    sampled values are exact, not merely close."""
    inner = ReplayBuffer(_obs_space(np.float32), _act_space(), 16)
    her = EpisodeRelabelBuffer(
        inner, _goal_block, goal_radius=2.0, k_goal=0.0, k_collision=0.0,
        time_to_goal=True, step_cost=0.1, n_relabel=1, min_disp=5.0,
        max_steps=100)
    tr = _tr(0.0)
    her.record(tr, fields={"goal_pos": np.array([0.0, 0.0])},
               pos=np.array([0.0, 0.0]), hit=False)
    stored = _stored(inner)
    assert stored["observations"]["pixels"].dtype == np.float32
    assert np.array_equal(stored["observations"]["pixels"][0],
                          tr["observations"]["pixels"])
    batch = inner.sample(4)
    assert np.asarray(batch["observations"]["pixels"]).dtype == np.float32
    assert np.array_equal(np.asarray(batch["observations"]["pixels"])[0],
                          tr["observations"]["pixels"])


# -- the image-goal subclass shares the same record path --------------------

def test_image_goal_subclass_stores_fp16(clean_flag, monkeypatch):
    monkeypatch.setenv("BUFFER_PIXELS_FP16", "1")
    inner = ReplayBuffer(_obs_space(np.float16), _act_space(), 16)
    buf = Fp16PixelsBuffer(inner)
    her = ImageGoalRelabelBuffer(
        buf, _goal_block, goal_radius=2.0, min_disp=5.0, max_steps=100)
    # A clean goal-mode episode that travels past min_disp.
    for step, x in enumerate((0.0, 12.0, 24.0)):
        her.record(_tr(x), fields={"goal_pos": np.array([24.0, 0.0])},
                   pos=np.array([x, 0.0]), hit=False, mode=1)
    her.flush_episode()
    stored = _stored(buf)
    assert stored["observations"]["pixels"].dtype == np.float16
    batch = buf.sample(4)
    assert np.asarray(batch["observations"]["pixels"]).dtype == np.float32


# -- the trainer's buffer obs space -----------------------------------------

def test_trainer_buffer_space_pixels_fp16(clean_flag, monkeypatch):
    """fp16_pixel_obs_space recasts ONLY the pixels key's dtype."""
    pytest.importorskip("jax")
    import train_carla_tqc

    monkeypatch.setenv("BUFFER_PIXELS_FP16", "1")
    out = train_carla_tqc.fp16_pixel_obs_space(_obs_space(np.float32))
    assert out["pixels"].dtype == np.float16
    assert out["pixels"].shape == (7, 7, 768)
    assert out["goal"].dtype == np.float32
    assert out["mode"].dtype == np.float32
    assert out["state"].dtype == np.float32


def test_trainer_buffer_space_untouched_without_the_flag(clean_flag):
    pytest.importorskip("jax")
    import train_carla_tqc

    space = _obs_space(np.float32)
    assert train_carla_tqc.fp16_pixel_obs_space(space) is space