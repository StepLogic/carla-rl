"""A dense goal term for goal mode, potential-based.

My decision 2026-10-14 under the user's delegation ("monitor and make decisions
about reward"), after v6 finished 1M steps.

WHY. The explore reward is IDENTICAL in both modes -- that is v52's design --
so goal mode had nothing to make it behave differently from explore mode; its
only distinction was an arrival bonus it almost never collected. v6 showed the
consequence: the goal arm decayed 0.100 -> 0.010 while the explore arm improved
to 0.20. The policy optimised the objective that pays on every step. The peer's
Habitat arm, which works, has this term (d_prev - d each step); ours had nothing
until arrival, with an image goal at 5-25 m.

POTENTIAL-BASED so the optimal policy is unchanged (Ng et al. 1999):
Phi(s) = -K * route_remaining, and a step pays gamma*Phi(s') - Phi(s). The
telescoping is what makes it safe -- it cannot invent a better policy than the
unshaped reward has, it only makes the existing one easier to find.
"""
import importlib
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

FWD = np.array([0.4, 0.0])


def _wrapper(k, **env):
    """A GatedReward whose step() we can drive, with POTENTIAL_K set."""
    os.environ["POTENTIAL_K"] = str(k)
    os.environ.setdefault("GATED_GOAL_REWARD", "explore")
    for a, b in env.items():
        os.environ[a] = str(b)
    import vae_nav.gated_reward as g
    importlib.reload(g)
    g.PROGRESS_MODE, g.PROGRESS_SCALE = "heading", 1.0
    return g


@pytest.fixture(autouse=True)
def _restore():
    yield
    for k in ("POTENTIAL_K", "GATED_GOAL_REWARD", "SHAPING_DISCOUNTED"):
        os.environ.pop(k, None)
    importlib.reload(importlib.import_module("vae_nav.gated_reward"))


def _info(remaining, **kw):
    d = {"hero_yaw": 0.0, "speed_factor": 0.5, "action_smoothness_penalty": 0.0,
         "collision": False, "off_road": False, "lane_any": False,
         "lane_marking_types": [], "in_junction": False, "speed_kmh": 10.0,
         "route_remaining_m": remaining}
    d.update(kw)
    return d


def test_the_term_is_off_by_default_which_is_v52():
    g = _wrapper(0)
    assert g.POTENTIAL_K == 0.0


def test_closing_on_the_goal_pays_and_retreating_costs():
    """The whole point: goal mode now has a signal that distinguishes driving
    TOWARDS the goal from merely driving."""
    g = _wrapper(1.0)
    # Phi = -K*remaining, so a step paying gamma*Phi(s') - Phi(s) is positive
    # when remaining shrinks.
    closer = 1.0 * (20.0 - 1.0 * 18.0)        # prev 20 -> now 18
    further = 1.0 * (20.0 - 1.0 * 22.0)
    assert closer > 0 > further


def test_it_telescopes_so_a_round_trip_pays_nothing():
    """Potential-based shaping cannot be farmed: going out and coming back
    sums to zero, which is why it cannot invent a worse policy that scores
    better -- the failure mode of a naive distance bonus."""
    k, gamma = 1.0, 1.0
    legs = [(20.0, 18.0), (18.0, 15.0), (15.0, 18.0), (18.0, 20.0)]
    total = sum(k * (a - gamma * b) for a, b in legs)
    assert total == pytest.approx(0.0)


def test_the_shaping_is_recorded_in_info():
    """So a run can be audited afterwards: how much of the return was shaping
    rather than the task."""
    import vae_nav.gated_reward as g
    src = open(g.__file__).read()
    branch = src[src.index('elif self.mode == GOAL and self.goal_kind == "explore":'):]
    branch = branch[:branch.index("goal_explore_end")]
    assert 'info["gated_shaping"] = float(_shaped)' in branch
    assert 'info["gated_shaping"] = 0.0' not in branch


def test_no_shaping_is_paid_on_a_terminal_step():
    """Phi(terminal) must be 0 for the invariance to hold, and paying into an
    arrival or a crash would make those bills inexact."""
    import vae_nav.gated_reward as g
    src = open(g.__file__).read()
    branch = src[src.index('elif self.mode == GOAL and self.goal_kind == "explore":'):]
    branch = branch[:branch.index("goal_explore_end")]
    assert "if POTENTIAL_K and not (terminated or truncated):" in branch


def test_explore_mode_gets_no_goal_shaping():
    """Explore mode has no goal, so route_remaining is meaningless there and a
    goal-shaped signal would corrupt the half that currently works."""
    import vae_nav.gated_reward as g
    src = open(g.__file__).read()
    # The shaping block sits inside the GOAL branch only.
    goal_branch = src.index('elif self.mode == GOAL and self.goal_kind == "explore":')
    assert src.index("_shaped = POTENTIAL_K * (self._prev_remaining", goal_branch) > goal_branch
