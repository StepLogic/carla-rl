"""act_with_stats: the action plus the two per-step signals an exploration
trace records for map building (user 2026-11-01, "first drive policy in
exploration mode and store trace").

  * policy entropy -- the Gaussian's entropy before the tanh squash, the same
    quantity the training [Diag] line reports. The peer's best Gibson map
    ("real-time entropy", 92.3 vs 90.8 mapless, Holm p=.004) places nodes on it.
  * critic disagreement -- the spread of the REDQ ensemble's Q at the action
    taken: an epistemic signal that costs nothing, since the 10 critics run
    anyway.

The action must be the SAME draw act() would make, or a trace would describe a
policy other than the one that drove it.
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))
torch = pytest.importorskip("torch")

from tests.test_one_critic import _cfg, _chunk  # noqa: E402
from torch_rl.gated_recurrent_sac import GatedRecurrentSAC  # noqa: E402


def _row(agent, seed=0):
    ch = _chunk(agent.cfg, [0], seed=seed)
    obs = ch["obs"] if isinstance(ch, dict) else ch[0]
    return np.asarray(obs[0, -1].cpu().numpy() if hasattr(obs, "cpu") else obs[0, -1],
                      np.float32)


def test_same_action_as_act_for_the_same_rng():
    agent = GatedRecurrentSAC(_cfg())
    row = _row(agent)
    torch.manual_seed(3)
    a1, _ = agent.act(row, agent.reset_hidden(1))
    torch.manual_seed(3)
    a2, _, stats = agent.act_with_stats(row, agent.reset_hidden(1))
    assert np.allclose(a1, a2)


def test_deterministic_matches_too():
    agent = GatedRecurrentSAC(_cfg())
    row = _row(agent)
    a1, _ = agent.act(row, agent.reset_hidden(1), deterministic=True)
    a2, _, _ = agent.act_with_stats(row, agent.reset_hidden(1), deterministic=True)
    assert np.allclose(a1, a2)


def test_entropy_is_the_pre_tanh_gaussian_entropy():
    agent = GatedRecurrentSAC(_cfg())
    row = _row(agent)
    _, _, st = agent.act_with_stats(row, agent.reset_hidden(1))
    d = agent.cfg.action_dim
    # recompute from the actor's own (mean, log_std) at the same input
    x = torch.as_tensor(row[None], device=agent.device)
    s, _ = agent.core.step(agent._embed(agent.compress, agent.act_embed, x, None),
                           agent.reset_hidden(1))
    _mean, log_std = agent._actor_forward(torch.cat([s, agent._state(x)], dim=-1))
    want = float(log_std.sum() + d * 0.5 * np.log(2 * np.pi * np.e))
    assert st["entropy"] == pytest.approx(want, rel=1e-5)


def test_critic_spread_is_over_the_ensemble():
    agent = GatedRecurrentSAC(_cfg())
    _, _, st = agent.act_with_stats(_row(agent), agent.reset_hidden(1))
    assert st["q_std"] >= 0.0 and np.isfinite(st["q_mean"])
    assert st["n_critics"] == agent.cfg.n_critics


def test_hidden_state_advances_like_act():
    agent = GatedRecurrentSAC(_cfg())
    row = _row(agent)
    _, h1 = agent.act(row, agent.reset_hidden(1), deterministic=True)
    _, h2, _ = agent.act_with_stats(row, agent.reset_hidden(1), deterministic=True)
    assert torch.allclose(h1, h2)
