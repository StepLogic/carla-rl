"""ACTOR OUTPUT PENALTY (user 2026-10-26, v16).

v14/v15 evals parked: steering at +-0.97 and throttle ~0.02 for 1000 steps, the
car never moving. Those actions sit where tanh is flat (slope 0.06-0.08), so the
actor's gradient out of them is cut ~15x. The penalty
    c * sum_dims relu(|u| - free)^2,   u = the actor's PRE-tanh mean
keeps the mean out of that region; |u| <= free (1.5: steer 0.905, throttle
0.05-0.95 after the box map) is untouched.
"""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
torch = pytest.importorskip("torch")

from torch_rl.gated_recurrent_tqc import pretanh_penalty  # noqa: E402


def test_zero_inside_the_free_band():
    u = torch.tensor([[0.0, 1.0], [-1.5, 1.5]])
    assert torch.all(pretanh_penalty(u, 1.5) == 0)


def test_quadratic_beyond_it_summed_over_dims():
    u = torch.tensor([[2.0, -2.5]])
    assert torch.allclose(pretanh_penalty(u, 1.5), torch.tensor([0.25 + 1.0]))


def test_pulls_a_saturated_actor_back():
    """End to end: an actor pushed to u = +3 on both dims shrinks FASTER under the
    penalty and does not without it, over the same updates."""
    from tests.test_one_critic import _cfg, _chunk
    from torch_rl.gated_recurrent_sac import GatedRecurrentSAC

    def run(c):
        agent = GatedRecurrentSAC(_cfg(one=True, actor_pretanh_l2=c,
                                       actor_pretanh_free=1.5))
        last = agent.actor.trunk[-1]
        with torch.no_grad():
            last.bias[: agent.cfg.action_dim] = 3.0
        out = {}
        for i in range(80):
            out = agent.update(_chunk(agent.cfg, [1, 1], seed=i))
        return float(out["mean_abs_u_steer"]), out

    u_on, out_on = run(1.0)
    u_off, _ = run(0.0)
    assert float(out_on["pretanh_pen"]) >= 0.0
    assert u_on < u_off - 0.1, (u_on, u_off)
