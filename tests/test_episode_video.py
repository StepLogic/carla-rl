"""Episode videos: the pair, the channel order, and the bounds.

User 2026-10-14: "Store some Videos for debugging". Each test below is a way a
debugging aid turns into a debugging problem:
  * the channels swapped, so every video looks like a perception bug
  * the goal half missing or silently the same as the camera half, so the video
    cannot answer the one question it exists for
  * recording every episode, or without a frame cap, so a long run fills a quota
  * an encoder failure taking the training run down with it
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

cv2 = pytest.importorskip("cv2")

from torch_rl.episode_video import EpisodeVideo, _to_panel  # noqa: E402


def _frame(fill, size=120, channel=None):
    a = np.full((size, size, 3), int(fill), np.uint8)
    if channel is not None:                     # a pure single-channel frame
        a[:] = 0
        a[:, :, channel] = 255
    return a


def _rec(tmp_path, **kw):
    kw.setdefault("every", 1)
    return EpisodeVideo(str(tmp_path), **kw)


# ----------------------------------------------------------------- the pair --
def test_the_frame_is_the_camera_BESIDE_the_goal():
    """The question this video answers is whether the car is driving towards the
    view it was given, which a single panel cannot show."""
    left = _to_panel(_frame(10, 120))
    right = _to_panel(_frame(200, 112))
    assert left.shape[0] == right.shape[0], "panels must share a height to join"


def test_both_halves_are_present_and_different(tmp_path):
    v = _rec(tmp_path)
    v.start(0)
    v.add(_frame(10), _frame(200), step=0, mode=1, reward=0.0,
          action=[0.0, 0.0], ret=0.0)
    pair = v.frames[0]
    h, w = pair.shape[:2]
    lo = pair[h // 2, w // 4]                   # mid-left: the camera
    hi = pair[h // 2, 3 * w // 4]               # mid-right: the goal
    assert not np.array_equal(lo, hi), \
        "the two halves are identical: the goal is not being drawn"


def test_a_divider_separates_the_halves(tmp_path):
    v = _rec(tmp_path)
    v.start(0)
    v.add(_frame(10), _frame(10), 0, 1, 0.0, [0, 0], 0.0)
    pair = v.frames[0]
    assert (pair == 255).any(), "no divider drawn"


# --------------------------------------------------------- the channel order --
def test_the_written_video_is_BGR_not_RGB(tmp_path):
    """cv2 writes BGR and these frames are RGB (ENV_RGB_CHANNEL_FIX=1). Get it
    wrong and every video has an orange sky, which reads as a perception bug.

    Checked by writing a pure-RED pair and reading the file back: a correct
    writer gives red back, a missing conversion gives blue.
    """
    v = _rec(tmp_path, max_frames=4)
    v.start(0)
    red = _frame(0, 64, channel=0)              # RGB red
    for _ in range(3):
        v.add(red, red, 0, 1, 0.0, [0, 0], 0.0)
    path = v.save("test")
    if path is None:
        pytest.skip("no mp4 encoder available in this build of cv2")
    cap = cv2.VideoCapture(path)
    ok, got = cap.read()
    cap.release()
    assert ok, "wrote a file that cannot be read back"
    h, w = got.shape[:2]
    px = got[h // 2, w // 4]                    # cv2 hands back BGR
    assert px[2] > 150 and px[0] < 100, f"expected red, read BGR={px}"


# -------------------------------------------------------------- the bounds --
def test_only_one_episode_in_EVERY_is_recorded(tmp_path):
    v = _rec(tmp_path, every=10)
    assert v.start(0) is True
    assert v.start(5) is False
    assert v.start(10) is True


def test_a_non_recorded_episode_collects_nothing(tmp_path):
    v = _rec(tmp_path, every=10)
    v.start(3)
    for _ in range(50):
        v.add(_frame(1), _frame(2), 0, 1, 0.0, [0, 0], 0.0)
    assert v.frames == []
    assert v.save("x") is None


def test_the_frame_cap_holds(tmp_path):
    v = _rec(tmp_path, max_frames=5)
    v.start(0)
    for _ in range(50):
        v.add(_frame(1), _frame(2), 0, 1, 0.0, [0, 0], 0.0)
    assert len(v.frames) == 5


def test_old_videos_are_pruned(tmp_path):
    v = _rec(tmp_path, max_frames=2, keep=3)
    made = 0
    for ep in range(6):
        v.start(ep)
        v.add(_frame(ep + 1, 64), _frame(9, 64), 0, 1, 0.0, [0, 0], 0.0)
        if v.save(f"e{ep}") is not None:
            made += 1
    if made == 0:
        pytest.skip("no mp4 encoder available in this build of cv2")
    left = [f for f in os.listdir(tmp_path) if f.endswith(".mp4")]
    assert len(left) <= 3, left


def test_keep_zero_prunes_nothing(tmp_path):
    v = _rec(tmp_path, max_frames=2, keep=0)
    for ep in range(3):
        v.start(ep)
        v.add(_frame(1, 64), _frame(2, 64), 0, 1, 0.0, [0, 0], 0.0)
        v.save(f"e{ep}")
    left = [f for f in os.listdir(tmp_path) if f.endswith(".mp4")]
    if not left:
        pytest.skip("no mp4 encoder available in this build of cv2")
    assert len(left) == 3


# ------------------------------------------------------------- robustness --
def test_the_filename_carries_the_episode_and_the_end_reason(tmp_path):
    """So the interesting videos can be found without opening them: a run's
    crossings and its arrivals are different debugging sessions."""
    v = _rec(tmp_path, max_frames=2)
    v.start(42)
    v.add(_frame(1, 64), _frame(2, 64), 0, 1, 0.0, [0, 0], 0.0)
    path = v.save("crossing")
    if path is None:
        pytest.skip("no mp4 encoder available in this build of cv2")
    assert "ep000042" in os.path.basename(path)
    assert "crossing" in os.path.basename(path)


def test_saving_with_no_frames_is_harmless(tmp_path):
    v = _rec(tmp_path)
    v.start(0)
    assert v.save("x") is None


def test_a_float_frame_is_accepted_rather_than_crashing_the_run(tmp_path):
    """Everything here is a debugging aid; it must never be the thing that ends
    a twelve-hour job."""
    v = _rec(tmp_path)
    v.start(0)
    v.add(np.zeros((64, 64, 3), np.float32), _frame(2, 64), 0, 1, 0.0,
          [0, 0], 0.0)
    assert len(v.frames) == 1
