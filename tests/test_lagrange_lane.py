"""The lane-crossing Lagrange multiplier: cost carried in the buffer, priced at sample time.

User 2026-10-04: lane crossings as an enforced CONSTRAINT rather than a fixed fine or a
terminal. v49 showed the terminal plus the paper's lambda=0.01 makes waiting cheaper than
driving, so the goal arm idled (clock share 0 -> 0.62). A fixed fine cannot say whether
the agent will actually keep its lane. A multiplier that rises while the crossing rate is
above target and falls below it is a fine that tunes itself until the constraint holds.

THE DESIGN DECISION THESE TESTS PIN: the cost rides in the replay buffer and the
multiplier is applied when a batch is SAMPLED, never when a row is stored. This is an
off-policy learner with a 1M-row buffer and hindsight relabelling. Baking mu into the
stored reward would leave every row priced at whatever mu was when it was written, for as
long as it sits in the buffer -- the critic would fit a reward that keeps changing under
it. Carried as a cost, every row, live or hindsight, is priced at today's mu.

The cost rides in the observation dict for the same reason the BC mask does: the jaxrl2
buffer stores nothing else. No network may ever see it.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

from vae_nav.lane_priv import (COST_KEY, EXPERT_KEY, PRIV_KEY,  # noqa: E402
                               strip_expert, strip_priv)


def _batch(rewards, cost, mode):
    n = len(rewards)
    obs = {"latent": np.zeros((n, 4), np.float32),
           "mode": np.asarray(mode, np.float32).reshape(n, 1),
           COST_KEY: np.asarray(cost, np.float32).reshape(n, 1)}
    return {"observations": obs, "next_observations": dict(obs),
            "rewards": np.asarray(rewards, np.float32),
            "masks": np.ones(n, np.float32)}


# ------------------------------------------------------------- never seen --
def test_no_network_input_ever_contains_the_cost():
    """strip_priv feeds the actor and the image-only critics."""
    b = _batch([0.0], [1.0], [1.0])
    s = strip_priv(b)
    assert COST_KEY not in s["observations"]
    assert COST_KEY not in s["next_observations"]


def test_the_privileged_critic_never_sees_it_either():
    """strip_expert feeds the privileged critic. It keeps the lane block, so
    the cost must be removed explicitly or that critic would read it."""
    obs = {PRIV_KEY: np.zeros((1, 13)), EXPERT_KEY: np.zeros((1, 1)),
           COST_KEY: np.ones((1, 1))}
    s = strip_expert({"observations": obs, "next_observations": dict(obs)})
    assert COST_KEY not in s["observations"]
    assert PRIV_KEY in s["observations"]        # the privileged block survives


def test_a_bare_observation_dict_is_stripped_too():
    o = {"latent": np.zeros(4), COST_KEY: np.ones(1)}
    assert COST_KEY not in strip_priv(o)


# ------------------------------------------------------- the adjustment --
def test_reward_is_reduced_by_mu_times_cost():
    from vae_nav.gated_tqc_learner import apply_lane_cost
    b = _batch([5.0, 5.0, 5.0], [0.0, 1.0, 1.0], [1, 1, 1])
    cost = b["observations"][COST_KEY][:, 0]
    out = apply_lane_cost(b, cost, 20.0)
    np.testing.assert_allclose(np.asarray(out["rewards"]), [5.0, -15.0, -15.0])


def test_mu_zero_is_exactly_the_unpenalised_reward():
    from vae_nav.gated_tqc_learner import apply_lane_cost
    b = _batch([3.0, -1.0], [1.0, 1.0], [1, 1])
    out = apply_lane_cost(b, b["observations"][COST_KEY][:, 0], 0.0)
    np.testing.assert_allclose(np.asarray(out["rewards"]), [3.0, -1.0])


def test_no_cost_key_leaves_the_batch_untouched():
    """Every run without LAGRANGE_LANE must be byte-identical."""
    from vae_nav.gated_tqc_learner import apply_lane_cost
    b = _batch([3.0], [1.0], [1])
    out = apply_lane_cost(b, None, 50.0)
    assert out is b


def test_the_input_batch_is_not_mutated():
    """The explore critic is updated from the SAME batch afterwards and must see
    the unpenalised reward."""
    from vae_nav.gated_tqc_learner import apply_lane_cost
    b = _batch([5.0], [1.0], [1])
    apply_lane_cost(b, b["observations"][COST_KEY][:, 0], 20.0)
    np.testing.assert_allclose(b["rewards"], [5.0])


def test_the_same_row_is_priced_at_TODAYS_mu_not_the_mu_it_was_stored_at():
    """THE REASON FOR THE DESIGN. A row stored when mu was 10 and sampled when mu
    is 80 must be charged 80. If mu were baked in at insert this would read -5."""
    from vae_nav.gated_tqc_learner import apply_lane_cost
    stored = _batch([5.0], [1.0], [1])          # what the buffer holds: cost, not mu
    later = apply_lane_cost(stored, stored["observations"][COST_KEY][:, 0], 80.0)
    np.testing.assert_allclose(np.asarray(later["rewards"]), [-75.0])


# ---------------------------------------------- through the jitted update --
# These run the REAL GatedTQCLearner.update. The pure tests above cannot reach a
# trace-time key error or a shape mismatch inside jit; these can.
jax = pytest.importorskip("jax")
pytest.importorskip("jaxrl2")
_spaces = pytest.importorskip("gymnasium").spaces
from flax.core.frozen_dict import freeze  # noqa: E402

_ACT = _spaces.Box(np.array([-1.0, 0.0], np.float32), np.array([1.0, 1.0], np.float32))


def _learner():
    from vae_nav.gated_tqc_learner import GatedTQCLearner
    # Built from the ENV'S observation space, which has no cost key -- exactly as
    # the trainer builds it. The cost exists only in the buffer.
    space = _spaces.Dict({"goal": _spaces.Box(-np.inf, np.inf, (4,), np.float32),
                          "latent": _spaces.Box(-np.inf, np.inf, (16,), np.float32),
                          "mode": _spaces.Box(0.0, 1.0, (1,), np.float32)})
    return GatedTQCLearner(seed=0, observations=space.sample(), actions=_ACT.sample(),
                           discount=0.99, discount_e=0.99, hidden_dims=(32, 32),
                           num_qs=2, num_quantiles=5, top_quantiles_to_drop_per_net=1,
                           target_entropy=-2.0, action_space=_ACT,
                           value_floor=-500.0, value_ceiling=50.0)


def _jbatch(seed, cost, n=32, with_key=True):
    rng = np.random.default_rng(seed)

    def obs():
        o = {"goal": rng.normal(size=(n, 4)).astype(np.float32),
             "latent": rng.normal(size=(n, 16)).astype(np.float32),
             "mode": np.ones((n, 1), np.float32)}
        o["mode"][n // 2:] = 0.0                # half goal rows, half explore
        if with_key:
            o[COST_KEY] = np.full((n, 1), float(cost), np.float32)
        return o
    return freeze({"observations": obs(), "next_observations": obs(),
                   "actions": rng.uniform(0, 1, (n, 2)).astype(np.float32),
                   "rewards": np.ones((n,), np.float32),
                   "masks": np.ones((n,), np.float32),
                   "dones": np.zeros((n,), np.float32)})


def _params(tree):
    return np.concatenate([np.asarray(x).ravel()
                           for x in jax.tree_util.tree_leaves(tree)])


def test_update_runs_with_the_cost_and_builds_no_layer_for_it():
    a = _learner()
    a.lagrange_mu = 20.0
    a.update(_jbatch(0, cost=1.0))
    for net in (a._actor.params, a._critic.params, a._critic_e.params):
        paths = ["/".join(str(getattr(k, "key", k)) for k in p)
                 for p, _ in jax.tree_util.tree_flatten_with_path(net)[0]]
        assert not any(COST_KEY in p for p in paths), "a network grew a cost input"


def test_mu_moves_the_goal_critic_and_leaves_the_explore_critic_alone():
    a, b = _learner(), _learner()
    a.lagrange_mu, b.lagrange_mu = 0.0, 50.0
    a.update(_jbatch(1, cost=1.0))
    b.update(_jbatch(1, cost=1.0))
    assert not np.allclose(_params(a._critic.params), _params(b._critic.params)), \
        "mu had no effect on the goal critic"
    np.testing.assert_array_equal(_params(a._critic_e.params),
                                  _params(b._critic_e.params))


def test_mu_zero_is_identical_to_having_no_cost_at_all():
    """Every run without LAGRANGE_LANE must be unchanged by this feature."""
    a, b = _learner(), _learner()
    a.lagrange_mu = 0.0
    a.update(_jbatch(2, cost=1.0, with_key=True))
    b.update(_jbatch(2, cost=1.0, with_key=False))
    np.testing.assert_allclose(_params(a._critic.params), _params(b._critic.params),
                               rtol=0, atol=1e-6)
    np.testing.assert_allclose(_params(a._actor.params), _params(b._actor.params),
                               rtol=0, atol=1e-6)


def test_a_new_mu_does_not_recompile():
    """mu changes every [GatedEp] block. If it were a static argument, each new
    value would retrace the whole update -- a silent slowdown, not a failure."""
    from vae_nav.gated_tqc_learner import _gated_update_jit
    a = _learner()
    a.lagrange_mu = 10.0
    # WARM UP FIRST. The first few updates retrace on their own as learner state
    # settles (measured: cache 1 -> 2 -> 3, then flat, with mu held FIXED). The
    # first version of this test compared across that warm-up and blamed mu for
    # it. Only after the cache stops growing is a change attributable.
    for i in range(5):
        a.update(_jbatch(i, cost=1.0))
    before = _gated_update_jit._cache_size()
    for i, mu in enumerate((20.0, 35.0, 80.0, 0.0)):
        a.lagrange_mu = mu
        a.update(_jbatch(10 + i, cost=1.0))
    assert _gated_update_jit._cache_size() == before


# ----------------------------------------------------- trainer wiring --
# The trainer's wiring runs inside main(), which needs a live simulator, so it is
# checked here at the source level. Each check names a mistake that was made or
# nearly made while writing it; the live run is the real verification.
_TRAINER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "train_carla_tqc.py")


def _src():
    return open(_TRAINER).read()


def test_the_signal_is_the_SNAPSHOT_not_the_zeroed_counter():
    """Written wrong the first time: the counter is zeroed at the top of the
    episode-end block, so appending lane_cost_ep[i] afterwards always appends 0
    and the multiplier would never rise."""
    s = _src()
    assert 'recent["lane_cost_goal"].append(_lane_cost_this_ep)' in s
    assert 'recent["lane_cost_goal"].append(lane_cost_ep[i])' not in s
    snap = s.index("_lane_cost_this_ep = lane_cost_ep[i]")
    zero = s.index("lane_cost_ep[i] = 0.0")
    use = s.index('recent["lane_cost_goal"].append(_lane_cost_this_ep)')
    assert snap < zero < use


def test_the_signal_is_counted_crossings_not_the_end_reason_share():
    """With the terminal off, episodes no longer END on crossings, so `cross=`
    reads ~0 however often the car leaves its lane."""
    s = _src()
    block = s[s.index("DUAL ASCENT on the crossing constraint"):]
    block = block[:block.index("agent.lagrange_mu = lagrange_mu")]
    assert 'recent["lane_cost_goal"]' in block
    assert "_cross" not in block


def test_the_combinations_that_break_it_are_refused():
    s = _src()
    for guard in ("GATED_GOAL_LANE_TERMINAL", "GOAL_PROGRESS_FINES", "N_STEP"):
        assert f'os.environ.get("{guard}"' in s[s.index("LAGRANGE_LANE = "):
                                                 s.index("agent.lagrange_mu = lagrange_mu")]


def test_mu_survives_a_resume():
    """run_gated.sh resumes from the newest checkpoint on every retry. Restarting
    mu at mu0 would discard the price the run learned it needed."""
    s = _src()
    assert '"lagrange_mu": float(lagrange_mu)' in s
    assert 'lagrange_mu = float(ck["lagrange_mu"])' in s
