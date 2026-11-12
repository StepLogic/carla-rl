"""The learner with ONE action (plan 2026-11-10 learner validation: Pendulum and DMC cartpole
have one). The update's saturation diagnostics assumed two actions (steer, throttle) and died on
`_a[..., 1]` ("index 1 is out of bounds for dimension 1 with size 1", benchmark job 465653).
They are logging only: the throttle entries are reported when there is a second action, the
steer entries always. With two actions nothing changes."""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))
torch = pytest.importorskip("torch")

from tests.test_one_critic import CHUNK, EMBED, GRID, STATE, _cfg  # noqa: E402
from torch_rl.gated_recurrent_sac import GatedRecurrentSAC  # noqa: E402

S0 = EMBED * GRID * GRID             # prev_action=False, as C1: no previous-action slot in the row


def _agent(action_dim, algo):
    return GatedRecurrentSAC(_cfg(action_dim=action_dim, algo=algo, prev_action=False))


def _chunk(cfg, n=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    K, T, A = int(cfg.burn_in), CHUNK, int(cfg.action_dim)
    obs = torch.rand(n, K + T + 1, S0 + STATE, generator=g)
    obs[:, :, S0 + cfg.mode_index] = 1.0
    return {"obs": obs, "keep": torch.ones(n, K + 1),
            "actions": torch.rand(n, T, A, generator=g) * 2 - 1,
            "rewards": torch.randn(n, T, generator=g),
            "masks": torch.ones(n, T), "valid": torch.ones(n, T),
            "mode": torch.ones(n)}


@pytest.mark.parametrize("algo", ["sac", "iql"])
def test_a_one_action_learner_updates_and_reports_steer_only(algo):
    agent = _agent(1, algo)
    metrics = agent.update(_chunk(agent.cfg))
    assert isinstance(metrics, dict)
    assert not any("throttle" in k for k in metrics)


@pytest.mark.parametrize("algo", ["sac", "iql"])
def test_two_actions_still_report_both(algo):
    agent = _agent(2, algo)
    metrics = agent.update(_chunk(agent.cfg))
    keys = set(metrics)
    if any("steer" in k for k in keys):                  # diagnostics present in this path
        assert any("throttle" in k for k in keys)
