"""Hindsight relabelling for the DEBiT image-goal SAC arm (user 2026-10-21).

Pinned here, because two earlier relabellers in this repo got them wrong:
  * every leg ends in an arrival the env itself would have paid -- +bonus and
    TERMINAL on the arrival step (an earlier HER never paid arrival rows);
  * every other leg step carries the route-progress term for the HINDSIGHT
    goal (an earlier HER dropped the shaping from 83% of its rows);
plus the dirty cut, the timeout rule, the mode bit and the re-encoded binoc.
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from torch_rl import pixel_rows as pxr  # noqa: E402
from torch_rl.episode_buffer import DeviceEpisodeBuffer  # noqa: E402
from torch_rl.image_her import ImageGoalHER  # noqa: E402

D = pxr.BINOC_DIM
P = pxr.MONO_PIX


class FakeEnc:
    """binoc = (goal brightness - view brightness), so it is checkable."""

    def encode_batch(self, cur, goal):
        v = goal.reshape(len(goal), -1).mean(1) - cur.reshape(len(cur), -1).mean(1)
        return np.repeat(v[:, None], D, axis=1).astype(np.float32)


def _frame(t):
    return np.full((112, 112, 3), t % 250, np.uint8)


def _drive(her, speeds, *, dirty_at=None, mode=1, shaping=0.3, step_m=0.5,
           end_reached_at=None, truncated=False):
    """Feed one straight-line episode: step t moves step_m metres."""
    her.begin(0, {}, mode)
    n = len(speeds)
    for t in range(n):
        info_t = {"debit_cur": _frame(t)}
        info_n = {"debit_cur": _frame(t + 1), "hero_pos": np.array([step_m * (t + 1), 0.0]),
                  "speed_kmh": speeds[t], "gated_shaping": shaping}
        if dirty_at is not None and t == dirty_at:
            info_n["collision"] = True
        r = -0.1 + shaping
        if end_reached_at == t:
            info_n["goal_end"] = "reached"
            r += 50.0
        row = pxr.mono_row(_frame(t), np.array([float(mode)], np.float32), np.zeros(D, np.float32))
        nrow = pxr.mono_row(_frame(t + 1), np.array([float(mode)], np.float32), np.zeros(D, np.float32))
        her.record(0, row, np.zeros(2, np.float32), r, False, False, nrow, info_t, info_n)
    return her.flush(0, terminated=end_reached_at is not None, truncated=truncated)


def _her(**kw):
    buf = DeviceEpisodeBuffer(pxr.mono_row_dim(), 2, 5000, "cpu", goal_dim=pxr.PIX)
    kw.setdefault("n_relabel", 1)
    return buf, ImageGoalHER(buf, FakeEnc(), pxr, seed=0, **kw)


def _legs(buf):
    out = []
    for start, n, mode in buf._episodes:
        out.append((start, n, mode))
    return out


def test_every_leg_ends_in_a_paid_terminal_arrival():
    buf, her = _her()
    speeds = [30.0] * 60
    speeds[40] = 5.0                         # the only slow moment
    assert _drive(her, speeds) > 0
    (start, n, mode), = _legs(buf)
    assert mode == 1
    assert float(buf.rewards[start + n - 1]) == pytest.approx(-0.1 + 50.0)
    assert float(buf.masks[start + n - 1]) == 0.0           # terminal
    assert all(float(buf.masks[start + k]) == 1.0 for k in range(n - 1))


def test_non_terminal_leg_steps_carry_progress_to_the_HINDSIGHT_goal():
    """0.5 m a step towards the goal at K=1 pays +0.5; the original shaping
    (0.3, for the episode's real goal) is removed, not kept."""
    buf, her = _her(potential_k=1.0)
    speeds = [30.0] * 60
    speeds[40] = 5.0
    _drive(her, speeds, shaping=0.3, step_m=0.5)
    (start, n, _), = _legs(buf)
    for k in range(n - 1):
        assert float(buf.rewards[start + k]) == pytest.approx(-0.1 + 0.5)


def test_the_leg_length_is_in_the_training_band():
    buf, her = _her(n_relabel=5)
    speeds = [30.0] * 80
    speeds[70] = 5.0
    _drive(her, speeds, step_m=0.5)
    for start, n, _ in _legs(buf):
        assert 5.0 <= n * 0.5 <= 25.0 + 0.5


def test_rows_are_goal_mode_with_the_re_encoded_binoc():
    buf, her = _her()
    speeds = [30.0] * 60
    speeds[40] = 5.0
    _drive(her, speeds, mode=0)              # an EXPLORE episode, relabelled
    (start, n, mode), = _legs(buf)
    row = buf.obs[start].float()
    assert float(row[P]) == 1.0              # mode bit -> goal
    # FakeEnc: goal frame (t=41) brightness minus view brightness
    first_view = float(row[0]) * 255.0
    assert float(row[P + 1]) == pytest.approx(41 - first_view, abs=0.6)


def test_legs_come_only_from_the_clean_prefix():
    buf, her = _her(n_relabel=5)
    speeds = [30.0] * 80
    speeds[30] = 5.0                         # before the crash: usable
    speeds[60] = 5.0                         # after the crash: must not be used
    _drive(her, speeds, dirty_at=45, step_m=0.5)
    assert _legs(buf)
    for start, n, _ in _legs(buf):
        assert n <= 31


def test_no_slow_moment_means_no_leg_rather_than_an_unpaid_one():
    buf, her = _her()
    assert _drive(her, [30.0] * 60) == 0
    assert her.counts["no_slow_goal"] == 1
    assert not _legs(buf)


def test_the_last_frame_of_a_timed_out_episode_is_not_a_goal():
    buf, her = _her(n_relabel=5)
    speeds = [30.0] * 40 + [5.0]             # slow ONLY at the clock-out frame
    assert _drive(her, speeds, truncated=True) == 0


def test_a_real_arrival_bonus_is_removed_before_recomputing():
    """The env paid +50 on the real arrival; a leg ending elsewhere must not
    inherit it on a non-terminal step."""
    buf, her = _her(n_relabel=1)
    speeds = [30.0] * 60
    speeds[50] = 5.0
    _drive(her, speeds, end_reached_at=None)
    (start, n, _), = _legs(buf)
    assert max(float(buf.rewards[start + k]) for k in range(n - 1)) < 1.0
