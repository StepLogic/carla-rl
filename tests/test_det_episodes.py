"""NOISE-FREE TRAINING EPISODES (user 2026-10-25, "restart now as v15").

The eval drives the mean action; training drove samples, so states only the
mean action reaches -- v14's frozen full-lock park at 7 of 21 Town02 failures
-- were almost absent from replay and the critic never priced them.
DET_EPISODE_FRAC of training episodes, drawn per env per episode, now act with
the noise-free action (noise scale 0), so those states enter replay with their
real cost. The draw is per EPISODE: one episode is all-noise-free or all-sampled.
"""
import os
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))
pytest.importorskip("gymnasium")

from torch_rl.nets import Actor  # noqa: E402
from torch_rl.episode_buffer import DeviceEpisodeBuffer  # noqa: E402
from torch_rl.parallel_loop import run_parallel  # noqa: E402
from vae_nav.vec_env import ThreadedVecEnv  # noqa: E402

from tests.test_parallel_loop import BINOC, FakeTown, _row  # noqa: E402


# ---- the actor takes a per-row noise scale --------------------------------

def test_per_row_scale_mixes_mean_and_sampled_rows():
    torch.manual_seed(0)
    a, s = Actor(6, 2), torch.randn(4, 6)
    scale = torch.tensor([[0.0], [1.0], [0.0], [1.0]])
    torch.manual_seed(5)
    full, _ = a.sample(s)
    torch.manual_seed(5)
    mixed, _ = a.sample(s, noise_scale=scale)
    det = a.deterministic(s)
    assert torch.allclose(mixed[[0, 2]], det[[0, 2]], atol=0, rtol=0)
    assert torch.allclose(mixed[[1, 3]], full[[1, 3]])


def test_negative_per_row_scale_is_refused():
    with pytest.raises(ValueError):
        Actor(6, 2).sample(torch.randn(2, 6), noise_scale=torch.tensor([[1.0], [-1.0]]))


# ---- the loop draws it per env per episode --------------------------------

class ScaleSpyAgent:
    """Records the noise scale each env acted with, call by call."""

    def __init__(self):
        self.scales = []

    def reset_hidden(self, n):
        return torch.ones(1, n, 2)

    @staticmethod
    def zero_hidden_columns(hidden, ids):
        hidden[:, ids, :] = 0.0
        return hidden

    def act_batch(self, rows, hidden, goal=None, noise_scale=None):
        self.scales.append(None if noise_scale is None
                           else np.asarray(noise_scale, np.float64).ravel().copy())
        return np.zeros((rows.shape[0], 2), np.float32), hidden

    def update(self, chunk):
        return {"q_g": 0.0}

    def save(self, d):
        pass


def _loop(tmp_path, det_frac, ep_len=4, max_steps=400, logs=None):
    venv = ThreadedVecEnv([lambda: FakeTown(0, ep_len, True),
                           lambda: FakeTown(1, ep_len, True)],
                          names=["TownA", "TownB"])
    agent = ScaleSpyAgent()
    buf = DeviceEpisodeBuffer(1 + BINOC, 2, 5000, "cpu")
    run_parallel(venv, agent, buf, _row, venv.action_space,
                 to_env=lambda a, b: a, to_actor=lambda e, b: e,
                 goal_block=lambda g: np.asarray(g, np.float32).reshape(-1),
                 max_steps=max_steps, learning_starts=0, update_every=10_000,
                 updates_per=1, chunk_b=1, chunk_t=2, burn_in=1,
                 ckpt_every=10_000, out_dir=str(tmp_path), log_every=10,
                 log=(logs.append if logs is not None else (lambda s: None)),
                 det_frac=det_frac, det_seed=0)
    return agent


def test_off_passes_no_scale(tmp_path):
    agent = _loop(tmp_path, 0.0, max_steps=40)
    assert agent.scales and all(s is None for s in agent.scales)


def test_all_on_acts_noise_free_everywhere(tmp_path):
    agent = _loop(tmp_path, 1.0, max_steps=40)
    assert all(s is not None and np.all(s == 0.0) for s in agent.scales)


def test_scale_is_constant_within_an_episode_and_the_share_is_right(tmp_path):
    ep_len = 4
    agent = _loop(tmp_path, 0.25, ep_len=ep_len, max_steps=4000)
    sc = np.stack(agent.scales)                     # (vec steps, 2 envs)
    assert set(np.unique(sc)) <= {0.0, 1.0}
    n_ep = sc.shape[0] // ep_len
    for env in range(2):
        per_ep = sc[: n_ep * ep_len, env].reshape(n_ep, ep_len)
        assert np.all(per_ep == per_ep[:, :1])      # one value per episode
        share = float(np.mean(per_ep[:, 0] == 0.0))
        assert 0.18 < share < 0.32, share           # 500 episodes per env


def test_log_line_reports_the_noise_free_episodes(tmp_path):
    logs = []
    _loop(tmp_path, 0.5, max_steps=200, logs=logs)
    ep = [s for s in logs if s.startswith("[Ep")]
    assert ep and all("det=" in s for s in ep)


def test_real_agent_noise_free_row_matches_its_deterministic_action():
    from tests.test_one_critic import EMBED, GRID, STATE, _cfg
    from torch_rl.gated_recurrent_sac import GatedRecurrentSAC
    agent = GatedRecurrentSAC(_cfg(one=True))
    rows = np.random.default_rng(0).normal(size=(2, agent.cfg.obs_dim)).astype(np.float32)
    det, _ = agent.act_batch(rows, agent.reset_hidden(2), deterministic=True)
    mixed, _ = agent.act_batch(rows, agent.reset_hidden(2),
                               noise_scale=np.array([0.0, 1.0], np.float32))
    assert np.allclose(mixed[0], det[0], atol=1e-6)
    assert not np.allclose(mixed[1], det[1], atol=1e-6)
