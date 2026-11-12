"""Trace collection restarts at the nearest waypoint to where the last episode
ended, instead of teleporting to a fresh random spawn.

USER DECISION 2026-09-12: "when recording trace in exploration mode restart at
nearest waypoint where the agent crashed or truncated".

WHY IT MATTERS FOR THE MAP. A trace is the candidate set the graph is built
from. Restarting at a random spawn every episode samples the town in
disconnected islands: the explore policy drives ~60-100 m, dies, and reappears
somewhere unrelated. Continuing from where it stopped walks the town instead,
so the candidates form a connected route and the graph's edges have a chance of
describing reachability rather than coincidence.

COLLECT_CONTINUE=0 restores the old behaviour. The FIRST episode always uses
the normal sampler -- there is nowhere to continue from.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_flag_defaults_on_and_is_readable():
    import importlib
    import eval_placement as ep
    importlib.reload(ep)
    assert hasattr(ep, "COLLECT_CONTINUE")
    assert ep.COLLECT_CONTINUE is True


def test_next_origin_is_the_last_position(monkeypatch):
    """The helper picks the end of the previous episode, not a fresh spawn."""
    import importlib
    import eval_placement as ep
    importlib.reload(ep)
    assert hasattr(ep, "continue_origin")
    # no previous position -> None, so the sampler is left alone
    assert ep.continue_origin(None) is None
    p = np.array([12.5, -3.25])
    out = ep.continue_origin(p)
    assert out is not None
    assert np.allclose(out, p), out


def test_disabled_by_env(monkeypatch):
    monkeypatch.setenv("COLLECT_CONTINUE", "0")
    import importlib
    import eval_placement as ep
    importlib.reload(ep)
    assert ep.COLLECT_CONTINUE is False
    assert ep.continue_origin(np.array([1.0, 2.0])) is None
    monkeypatch.delenv("COLLECT_CONTINUE")
    importlib.reload(ep)


def test_experiment_is_found_through_a_wrapper_chain():
    """The accessor must walk .env, not trust .unwrapped.

    MEASURED 2026-09-12: `getattr(env, "unwrapped", env).experiment` raised
    AttributeError('GatedReward' object has no attribute 'experiment') on the
    collection env, so every episode fell back to the random sampler -- and
    did so SILENTLY, because the fallback is a caught exception.
    """
    import importlib
    import eval_placement as ep
    importlib.reload(ep)

    class Base:
        experiment = "EXP"
        core = "CORE"

    class Wrap:
        def __init__(self, inner):
            self.env = inner
        @property
        def unwrapped(self):        # lies, as the real wrapper does
            return self

    chain = Wrap(Wrap(Base()))
    assert ep.find_attr_owner(chain, "experiment").experiment == "EXP"
    assert ep.find_attr_owner(chain, "core").core == "CORE"
    assert ep.find_attr_owner(chain, "nope") is None
