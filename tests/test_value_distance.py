"""Turning critic quantiles into a distance.

THE DIRECTION MATTERS. Distance is -V, so UNDERestimating distance means V is
too HIGH. That is the error mode that wrecks a graph: Dijkstra preferentially
selects edges it believes are cheap, so optimistic errors get chosen while
pessimistic ones are ignored. Search concentrates the error rather than
averaging it away.
"""
import numpy as np
import pytest

from vae_nav.value_distance import (ensemble_spread, pessimistic_value,
                                    pooled_quantiles, value_to_steps)

NUM_QS, N_Q = 5, 25


def make_q(values):
    """(B,) per-sample constant -> (num_qs, B, n_q)."""
    v = np.asarray(values, np.float32)
    return np.broadcast_to(v[None, :, None], (NUM_QS, v.size, N_Q)).copy()


def test_pooled_quantiles_shape():
    q = make_q([-1.0, -2.0])
    assert pooled_quantiles(q).shape == (2, NUM_QS * N_Q)


def test_pessimistic_value_on_constant_quantiles_is_that_constant():
    q = make_q([-5.0, -12.0])
    np.testing.assert_allclose(pessimistic_value(q), [-5.0, -12.0], atol=1e-5)


def test_pessimistic_value_picks_the_low_tail_not_the_mean():
    """A low percentile of V = a LARGE distance = the conservative choice."""
    q = np.zeros((NUM_QS, 1, N_Q), np.float32)
    q[..., :] = np.linspace(-20.0, 0.0, N_Q)      # spread from -20 to 0
    v = pessimistic_value(q, percentile=10.0)[0]
    assert v < np.mean(q), "must be below the mean, i.e. a longer distance"
    assert -20.0 <= v <= -14.0, f"10th percentile of [-20,0] should be ~-18, got {v}"


def test_higher_percentile_is_less_conservative():
    q = np.zeros((NUM_QS, 1, N_Q), np.float32)
    q[..., :] = np.linspace(-20.0, 0.0, N_Q)
    assert pessimistic_value(q, 10.0)[0] < pessimistic_value(q, 50.0)[0]


def test_value_to_steps_inverts_the_reward_scale():
    v = np.array([-10.6, -30.0, 0.0])
    np.testing.assert_allclose(value_to_steps(v, step_cost=0.1),
                               [106.0, 300.0, 0.0], atol=1e-4)


def test_value_to_steps_is_never_negative():
    """A critic may predict a small positive V; distance cannot be negative."""
    assert np.all(value_to_steps(np.array([0.5, -1.0]), 0.1) >= 0.0)


def test_ensemble_spread_is_zero_when_critics_agree():
    q = make_q([-7.0])
    assert ensemble_spread(q)[0] == pytest.approx(0.0, abs=1e-6)


def test_ensemble_spread_grows_when_critics_disagree():
    q = make_q([-7.0])
    q[0] -= 5.0                                   # one critic dissents
    assert ensemble_spread(q)[0] > 1.0


def test_batch_entries_are_independent():
    q = make_q([-1.0, -20.0])
    v = pessimistic_value(q)
    assert v[0] == pytest.approx(-1.0, abs=1e-5)
    assert v[1] == pytest.approx(-20.0, abs=1e-5)


from vae_nav.value_distance import (H_STEPS, edge_reject_mask,
                                    goal_relabelled_obs)


def fake_goal_block(fields):
    """Stand-in for VAEGoalWrapper.goal_block: only the goal offset matters."""
    d = np.asarray(fields["goal_pos"], np.float64) - np.asarray(
        fields["hero_pos"], np.float64)
    return np.concatenate([d, np.zeros(10)]).astype(np.float32)


def test_goal_relabelled_obs_rewrites_only_the_goal_key():
    base = {"latent": np.arange(4, dtype=np.float32),
            "goal": np.zeros(12, np.float32)}
    fields = {"hero_pos": (0.0, 0.0), "hero_yaw": 0.0,
              "action_history": np.zeros((4, 2))}
    out = goal_relabelled_obs(base, fields, (3.0, 4.0), fake_goal_block)
    np.testing.assert_allclose(out["goal"][:2], [3.0, 4.0])
    # The latent is what the camera SAW and must not change with the goal.
    np.testing.assert_array_equal(out["latent"], base["latent"])


def test_goal_relabelled_obs_does_not_mutate_the_input():
    base = {"latent": np.zeros(4, np.float32), "goal": np.zeros(12, np.float32)}
    fields = {"hero_pos": (0.0, 0.0), "hero_yaw": 0.0,
              "action_history": np.zeros((4, 2))}
    goal_relabelled_obs(base, fields, (1.0, 1.0), fake_goal_block)
    assert base["goal"].sum() == 0.0, "caller's dict was mutated"


def test_h_steps_matches_the_measured_traversal_rate():
    """30 m at the measured 0.283 m/step."""
    assert H_STEPS == pytest.approx(30.0 / 0.283, rel=0.01)


def test_edges_beyond_the_horizon_are_rejected():
    cost = np.array([50.0, 105.0, 107.0, 300.0])
    spread = np.zeros(4)
    rej = edge_reject_mask(cost, spread, max_steps=H_STEPS, tau=1.0)
    assert list(rej) == [False, False, True, True], (
        "V is only trained on goals within H; beyond it every query is "
        "extrapolation, which is how q reached -280 against a -200 floor")


def test_edges_with_disagreeing_critics_are_rejected():
    cost = np.array([10.0, 10.0])
    spread = np.array([0.1, 9.0])
    assert list(edge_reject_mask(cost, spread, H_STEPS, tau=1.0)) == [False, True]


def test_rejection_reasons_compose():
    cost = np.array([500.0])
    spread = np.array([99.0])
    assert edge_reject_mask(cost, spread, H_STEPS, 1.0)[0]


from vae_nav.value_distance import critic_quantiles, edge_cost_steps


class StubCritic:
    """Mimics a Flax TrainState carrying the quantile ensemble."""

    def __init__(self, q):
        self._q = q
        self.params = {"dummy": 0}
        self.seen_actions = None

    def apply_fn(self, variables, observations, actions):
        self.seen_actions = np.asarray(actions)
        return self._q


class StubAgent:
    """Enough of a TQC agent for the query path, with no JAX involved."""

    def __init__(self, q, action_dim=2):
        self._critic = StubCritic(q)
        self._action_dim = action_dim
        self.seen_actions = None

    def eval_actions(self, observations):
        n = len(next(iter(observations.values())))
        # Non-zero, batch-index-dependent values to catch bugs in action passing
        actions = np.arange(n * self._action_dim, dtype=np.float32).reshape(n, self._action_dim) + 1.0
        self.seen_actions = actions.copy()
        return actions


def test_critic_quantiles_returns_the_ensemble_shape():
    q = make_q([-4.0, -8.0])
    agent = StubAgent(q)
    obs = {"latent": np.zeros((2, 4), np.float32),
           "goal": np.zeros((2, 12), np.float32)}
    out = critic_quantiles(agent, obs, agent.eval_actions(obs))
    assert out.shape == (NUM_QS, 2, N_Q)


def test_edge_cost_steps_converts_to_steps():
    """V = -10.6 must come back as ~106 steps, i.e. exactly H."""
    q = make_q([-10.6])
    agent = StubAgent(q)
    obs = {"latent": np.zeros((1, 4), np.float32),
           "goal": np.zeros((1, 12), np.float32)}
    cost, spread = edge_cost_steps(agent, obs, step_cost=0.1)
    assert cost[0] == pytest.approx(106.0, abs=0.5)
    assert spread[0] == pytest.approx(0.0, abs=1e-6)


def test_edge_cost_steps_is_pessimistic_not_mean():
    q = np.zeros((NUM_QS, 1, N_Q), np.float32)
    q[..., :] = np.linspace(-20.0, 0.0, N_Q)
    agent = StubAgent(q)
    obs = {"latent": np.zeros((1, 4), np.float32),
           "goal": np.zeros((1, 12), np.float32)}
    cost, _ = edge_cost_steps(agent, obs, step_cost=0.1)
    mean_cost = -np.mean(q) / 0.1
    assert cost[0] > mean_cost, "must over-estimate distance, never under"


def test_edge_cost_steps_passes_policy_actions_to_critic():
    """V(s) = Q(s, pi(s)): critic must receive the policy's actions, not zeros.

    Verifies that edge_cost_steps obtains actions from agent.eval_actions()
    and passes those exact actions to the critic. A bug using wrong actions,
    zeros, None, or batch-mismatched shapes would not fail without this test.
    Uses batch size > 1 to catch batch-broadcasting bugs.
    """
    q = make_q([-5.0, -8.0, -10.0])  # batch size 3
    agent = StubAgent(q, action_dim=2)
    obs = {"latent": np.zeros((3, 4), np.float32),
           "goal": np.zeros((3, 12), np.float32)}

    # Call edge_cost_steps, which internally calls eval_actions and critic_quantiles
    cost, spread = edge_cost_steps(agent, obs, step_cost=0.1)

    # Verify the actions received by the critic match what eval_actions produced
    np.testing.assert_array_equal(agent._critic.seen_actions, agent.seen_actions,
                                  err_msg="critic must receive policy actions from eval_actions")
