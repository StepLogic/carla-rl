"""Checkpoint round-trip for the gated learner's second critic and temperature.

A gated checkpoint carries three tensors a plain one does not. Two directions
have to work and a third has to be refused:

  gated  -> gated   full restore, including the explore critic and its target
  plain  -> gated   WARM START: actor and goal critic restore, explore critic
                    stays at its initialisation
  gated  -> plain   the extra keys are ignored rather than crashing eval

THE WARM START IS CURRENTLY THEORETICAL AND THE TEST SAYS SO. The gated
observation is 145-d and the two existing arms are 144-d and 132-d, so a real
carla_lanefollow checkpoint will fail the structure check before any of this
matters. The path is built because it costs little and because it is what a
future same-width checkpoint would need; it is not a way to reuse the arms we
already trained.

Uses stand-in train states rather than a real learner on purpose: this is
pytree bookkeeping, and building two JAX agents per test would add minutes to
the suite for no extra coverage. The masking behaviour is tested against the
real learner in test_gated_tqc.py.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

pytest.importorskip("jax")

from vae_nav.policy_io import agent_state, load_agent_state  # noqa: E402


class _TrainState:
    """The two things policy_io uses: .params and .replace(params=...)."""

    def __init__(self, params):
        self.params = params

    def replace(self, params):
        return _TrainState(params)


def _params(value, width=3):
    return {"Dense_0": {"kernel": np.full((width, width), float(value),
                                          np.float32),
                        "bias": np.full((width,), float(value), np.float32)}}


class _PlainAgent:
    def __init__(self, seed=0.0):
        self._actor = _TrainState(_params(seed + 1))
        self._critic = _TrainState(_params(seed + 2))
        self._temp = _TrainState(_params(seed + 3, width=1))
        self._target_critic_params = _params(seed + 2)


class _GatedAgent(_PlainAgent):
    def __init__(self, seed=0.0):
        super().__init__(seed)
        self._critic_e = _TrainState(_params(seed + 4))
        self._temp_e = _TrainState(_params(seed + 5, width=1))
        self._target_critic_e_params = _params(seed + 4)


def _flat(tree):
    import jax
    return np.concatenate([np.asarray(x).ravel()
                           for x in jax.tree_util.tree_leaves(tree)])


# ------------------------------------------------------------- saving ----

def test_a_gated_agent_saves_the_second_critic_and_temperature():
    state = agent_state(_GatedAgent())
    for key in ("actor", "critic", "temp", "critic_e", "temp_e",
                "target_critic_e"):
        assert key in state, key


def test_a_plain_agent_saves_no_gated_keys():
    # A plain checkpoint must stay readable by everything that reads one today.
    state = agent_state(_PlainAgent())
    assert set(state) == {"actor", "critic", "temp"}


# ---------------------------------------------------------- gated round ----

def test_gated_to_gated_restores_every_tensor():
    src, dst = _GatedAgent(seed=0.0), _GatedAgent(seed=100.0)
    load_agent_state(dst, agent_state(src))
    for attr in ("_actor", "_critic", "_temp", "_critic_e", "_temp_e"):
        assert np.array_equal(_flat(getattr(src, attr).params),
                              _flat(getattr(dst, attr).params)), attr


def test_gated_to_gated_restores_both_targets():
    src, dst = _GatedAgent(seed=0.0), _GatedAgent(seed=100.0)
    load_agent_state(dst, agent_state(src))
    assert np.array_equal(_flat(src._critic.params),
                          _flat(dst._target_critic_params))
    assert np.array_equal(_flat(src._critic_e.params),
                          _flat(dst._target_critic_e_params))


# ------------------------------------------------------------ warm start ----

def test_a_plain_checkpoint_warm_starts_a_gated_agent():
    src, dst = _PlainAgent(seed=0.0), _GatedAgent(seed=100.0)
    before_e = _flat(dst._critic_e.params)
    load_agent_state(dst, agent_state(src))
    assert np.array_equal(_flat(src._actor.params), _flat(dst._actor.params))
    assert np.array_equal(_flat(src._critic.params), _flat(dst._critic.params))
    # The explore critic has no counterpart in a plain checkpoint and must be
    # left at its initialisation rather than filled with the goal critic's
    # parameters, which are fitted to a return an order of magnitude larger.
    assert np.array_equal(before_e, _flat(dst._critic_e.params))


def test_the_warm_start_says_so_out_loud(capsys):
    load_agent_state(_GatedAgent(seed=100.0), agent_state(_PlainAgent()))
    out = capsys.readouterr().out
    assert "explore critic" in out.lower()


# ------------------------------------------------------ backward compat ----

def test_a_gated_checkpoint_loads_into_a_plain_agent():
    # Evaluation harnesses build a plain agent. The extra keys must be ignored,
    # not crash, or a gated checkpoint could never be evaluated by them.
    src, dst = _GatedAgent(seed=0.0), _PlainAgent(seed=100.0)
    load_agent_state(dst, agent_state(src))
    assert np.array_equal(_flat(src._actor.params), _flat(dst._actor.params))


def test_a_plain_checkpoint_still_loads_into_a_plain_agent():
    # The regression guard: nothing above may change the existing path.
    src, dst = _PlainAgent(seed=0.0), _PlainAgent(seed=100.0)
    load_agent_state(dst, agent_state(src))
    for attr in ("_actor", "_critic", "_temp"):
        assert np.array_equal(_flat(getattr(src, attr).params),
                              _flat(getattr(dst, attr).params)), attr


def test_a_mismatched_structure_is_still_refused():
    # The loud failure must survive. A checkpoint from a different observation
    # space unpickles happily and then acts confidently on a layout it never
    # trained on.
    dst = _GatedAgent()
    bad = agent_state(_GatedAgent())
    bad["critic_e"] = {"Dense_0": {"kernel": np.zeros((2, 2), np.float32)}}
    with pytest.raises(SystemExit, match="critic_e"):
        load_agent_state(dst, bad)
