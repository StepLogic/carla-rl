"""A FAILED SIMULATOR STEP MUST NOT END THE RUN (2026-11-09).

vae_nav.vec_env resets an env whose step raised (CARLA: "time-out of 50000ms while
waiting for the simulator") and reports the step as done with info {"env_error":
True}, "so the caller DISCARDS it". run_parallel did not: it built the next row from
that synthetic info, the trainer's row builder refused it (no DEBiT encoding,
"info['binoc_encoded'] is False") and the local projection run died at ~387k steps.
The fix stores nothing for that step, closes the replay episode at its last real
step (a cut, so it still bootstraps), flushes HER, counts the end as env_error, and
starts the next episode as after any other end.
"""
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from tests.test_parallel_loop import BINOC, FakeAgent, FakeTown  # noqa: E402
from torch_rl.episode_buffer import DeviceEpisodeBuffer  # noqa: E402
from torch_rl.parallel_loop import run_parallel  # noqa: E402
from vae_nav.vec_env import ThreadedVecEnv  # noqa: E402


class FaultyTown(FakeTown):
    """Raises once, on its `fail_at`-th step overall (a simulator time-out)."""

    def __init__(self, *a, fail_at=4, **k):
        super().__init__(*a, **k)
        self.calls, self.fail_at = 0, fail_at

    def _info(self):
        inf = super()._info()
        inf["binoc_encoded"] = True
        return inf

    def step(self, action):
        self.calls += 1
        if self.calls == self.fail_at:
            raise RuntimeError("time-out of 50000ms while waiting for the simulator")
        return super().step(action)


def _strict_row(o, inf):
    """As train_carla_sac.row_of: refuses a row whose binoc was never encoded."""
    if not inf.get("binoc_encoded", False):
        raise RuntimeError("info['binoc_encoded'] is False")
    return np.concatenate([np.asarray(o["mode"], np.float32), np.asarray(o["binoc"], np.float32)])


class _Her:
    def __init__(self):
        self.flushed, self.begun = [], 0

    def begin(self, i, info, mode):
        self.begun += 1

    def record(self, *a, **k):
        pass

    def flush(self, i, terminated=False, truncated=False):
        self.flushed.append((i, terminated, truncated))

    def describe_counts(self):
        return ""


def _run(tmp_path, her=None):
    venv = ThreadedVecEnv([lambda: FaultyTown(0, 7, True, fail_at=4)], names=["TownA"])
    buf = DeviceEpisodeBuffer(1 + BINOC, 2, 1000, "cpu")
    logs = []
    steps, eps = run_parallel(
        venv, FakeAgent(), buf, _strict_row, venv.action_space, to_env=lambda a, b: a,
        to_actor=lambda e, b: e, goal_block=lambda g: np.asarray(g, np.float32).reshape(-1),
        max_steps=30, learning_starts=0, update_every=10, updates_per=1, chunk_b=1,
        chunk_t=2, burn_in=1, ckpt_every=1000, out_dir=str(tmp_path), log_every=1,
        log=logs.append, her=her)
    return buf, steps, eps, logs


def test_the_run_survives_a_failed_step_and_keeps_training(tmp_path):
    buf, steps, eps, logs = _run(tmp_path)
    assert steps == 30


def test_no_synthetic_transition_enters_the_replay(tmp_path):
    buf, *_ = _run(tmp_path)
    # the faulted episode kept its 3 real steps (t = 1, 2, 3 after reset), closed there
    first_start, first_n, _ = buf._episodes[0]
    assert first_n == 3
    assert [float(buf.obs[first_start + j, 1]) for j in range(4)] == [0.0, 1.0, 2.0, 3.0]
    assert float(buf.masks[first_start + 2]) == 1.0          # a cut, not a terminal
    # every stored reward is a real step's (the synthetic step paid 0.0)
    for s, n, _ in buf._episodes:
        assert torch.all(buf.rewards[s:s + n] == 1.0)


def test_the_end_is_counted_and_logged_as_env_error(tmp_path):
    buf, steps, eps, logs = _run(tmp_path)
    assert any("env_error" in m or "abandoned" in m for m in logs)


def test_her_is_flushed_for_the_abandoned_episode(tmp_path):
    her = _Her()
    _run(tmp_path, her=her)
    assert her.flushed[0] == (0, False, True)
