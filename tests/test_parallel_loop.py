"""The SAC trainer's parallel-env loop, with fake envs and a fake agent.

User 2026-10-21: Town05 and Town01 at once, no switching, using the existing
vae_nav.vec_env.ThreadedVecEnv. These tests drive torch_rl.parallel_loop through
the real ThreadedVecEnv and the real replay buffer, on CPU, without CARLA.
"""
import os
import sys

import numpy as np
import pytest
import torch
from gymnasium import Env, spaces

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from torch_rl.episode_buffer import DeviceEpisodeBuffer  # noqa: E402
from torch_rl.parallel_loop import run_parallel  # noqa: E402
from vae_nav.vec_env import ThreadedVecEnv  # noqa: E402

BINOC = 3


class FakeTown(Env):
    """Episodes of fixed length; the last step is terminal or truncated."""

    def __init__(self, town_id, ep_len, terminal):
        self.town_id, self.ep_len, self.terminal = town_id, ep_len, terminal
        self.observation_space = spaces.Dict({
            "mode": spaces.Box(0, 1, (1,), np.float32),
            "binoc": spaces.Box(-1, 1, (BINOC,), np.float32)})
        self.action_space = spaces.Box(np.array([-1, 0], np.float32),
                                       np.array([1, 1], np.float32))
        self.t = 0
        self.resets = 0

    def _obs(self):
        return {"mode": np.array([1.0], np.float32),
                "binoc": np.full(BINOC, self.town_id * 100 + self.t, np.float32)}

    def _info(self):
        return {"debit_cur": np.zeros((4, 4, 3), np.uint8) + self.t,
                "debit_goal": np.zeros((4, 4, 3), np.uint8)}

    def reset(self, seed=None, options=None):
        self.t = 0
        self.resets += 1
        return self._obs(), self._info()

    def step(self, action):
        self.t += 1
        last = self.t >= self.ep_len
        inf = self._info()
        if last:
            inf["goal_end"] = "reached" if self.terminal else "clock"
        return self._obs(), 1.0, last and self.terminal, last and not self.terminal, inf


class FakeAgent:
    def __init__(self):
        self.calls, self.saved = 0, []

    def reset_hidden(self, n):
        return torch.ones(1, n, 2)

    @staticmethod
    def zero_hidden_columns(hidden, ids):
        hidden[:, ids, :] = 0.0
        return hidden

    def act_batch(self, rows, hidden, goal=None):
        assert rows.shape[0] == hidden.shape[1] == goal.shape[0]
        self.calls += 1
        return np.zeros((rows.shape[0], 2), np.float32), hidden

    def update(self, chunk):
        return {"q_g": 0.0}

    def save(self, d):
        self.saved.append(os.path.basename(d))


def _row(o, i):
    return np.concatenate([np.asarray(o["mode"], np.float32),
                           np.asarray(o["binoc"], np.float32)])


def _run(tmp_path, max_steps=60, learning_starts=0):
    venv = ThreadedVecEnv([lambda: FakeTown(0, 7, True), lambda: FakeTown(1, 11, False)],
                          names=["TownA", "TownB"])
    agent = FakeAgent()
    buf = DeviceEpisodeBuffer(1 + BINOC, 2, 1000, "cpu")
    box = venv.action_space
    to_env = lambda a, b: a
    to_actor = lambda e, b: e
    logs = []
    steps, eps = run_parallel(
        venv, agent, buf, _row, box, to_env=to_env, to_actor=to_actor,
        goal_block=lambda g: np.asarray(g, np.float32).reshape(-1),
        max_steps=max_steps, learning_starts=learning_starts, update_every=10,
        updates_per=1, chunk_b=2, chunk_t=3, burn_in=1, ckpt_every=20,
        out_dir=str(tmp_path), log_every=2, log=logs.append)
    return venv, agent, buf, steps, eps, logs


def test_it_runs_counts_transitions_and_checkpoints(tmp_path):
    venv, agent, buf, steps, eps, logs = _run(tmp_path)
    assert steps == 60                       # 30 vec steps x 2 envs
    assert agent.saved[:3] == ["step_20", "step_40", "step_60"]
    assert eps == 60 // 7 // 2 * 0 + (30 // 7) + (30 // 11)


def test_each_town_keeps_its_own_episodes(tmp_path):
    """No episode in the replay mixes the two towns' rows."""
    venv, agent, buf, *_ = _run(tmp_path)
    for start, n, _mode in buf._episodes:
        town = buf.obs[start:start + n + 1, 1:2].float() // 100
        assert torch.all(town == town[0]), (start, n)


def test_the_next_row_of_a_finished_episode_is_the_terminal_observation(tmp_path):
    """Not the auto-reset observation: town B's episodes end by time limit and
    must bootstrap from their own last frame (t = 11), not the next start."""
    venv, agent, buf, *_ = _run(tmp_path)
    ends = [(s, n) for s, n, _ in buf._episodes
            if float(buf.obs[s, 1]) // 100 == 1]
    assert ends
    for s, n in ends:
        assert float(buf.obs[s + n, 1]) == 100 + 11


def test_the_hidden_column_of_a_finished_env_is_reset(tmp_path):
    venv = ThreadedVecEnv([lambda: FakeTown(0, 3, True), lambda: FakeTown(1, 100, False)])
    agent = FakeAgent()
    seen = []
    orig = agent.act_batch

    def spy(rows, hidden, goal=None):
        seen.append(hidden.clone())
        return orig(rows, hidden, goal=goal)
    agent.act_batch = spy
    buf = DeviceEpisodeBuffer(1 + BINOC, 2, 1000, "cpu")
    run_parallel(venv, agent, buf, _row, venv.action_space,
                 to_env=lambda a, b: a, to_actor=lambda e, b: e,
                 goal_block=lambda g: np.asarray(g, np.float32).reshape(-1),
                 max_steps=12, learning_starts=0, update_every=100, updates_per=1,
                 chunk_b=1, chunk_t=2, burn_in=1, ckpt_every=100,
                 out_dir=str(tmp_path), log_every=100, log=lambda s: None)
    # after env 0's first episode (3 steps) its column is zero, env 1's is not
    assert float(seen[3][0, 0].abs().sum()) == 0.0
    assert float(seen[3][0, 1].abs().sum()) > 0.0


def test_log_lines_name_the_town(tmp_path):
    *_, logs = _run(tmp_path)
    ep_lines = [s for s in logs if s.startswith("[Ep")]
    assert ep_lines and any("TownA:" in s or "TownB:" in s for s in ep_lines)


def test_one_town_still_uses_the_parallel_loop():
    """HER, noise-free episodes and enc_alive live only in the parallel loop."""
    from train_carla_sac import use_parallel
    assert use_parallel(True, "", 1) and use_parallel(True, "", 2)
    assert not use_parallel(False, "", 2)
    assert not use_parallel(True, "/ckpt/step_1", 2)       # eval never


def test_the_loop_runs_with_a_single_env(tmp_path):
    venv = ThreadedVecEnv([lambda: FakeTown(0, 5, True)], names=["TownA"])
    buf = DeviceEpisodeBuffer(1 + BINOC, 2, 1000, "cpu")
    steps, eps = run_parallel(
        venv, FakeAgent(), buf, _row, venv.action_space,
        to_env=lambda a, b: a, to_actor=lambda e, b: e,
        goal_block=lambda g: np.asarray(g, np.float32).reshape(-1),
        max_steps=20, learning_starts=0, update_every=100, updates_per=1,
        chunk_b=1, chunk_t=2, burn_in=1, ckpt_every=100, out_dir=str(tmp_path),
        log_every=100, log=lambda s: None)
    assert steps == 20 and eps == 4
