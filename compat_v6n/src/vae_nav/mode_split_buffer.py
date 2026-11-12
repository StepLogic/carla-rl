"""Two replay buffers, one per mode, sampled at a fixed ratio.

THE FAILURE THIS EXISTS TO PREVENT, measured by the peer on their gated agent:
explore episodes ran longer than goal episodes, so ~90% of env steps were
exploration. A single uniformly-sampled buffer then gave the goal critic only
10-37% of each batch, and goal-mode evaluation collapsed from 40-65% to 15%. The
goal arm was not broken -- it was starved.

THE IMBALANCE INVERTS IN CARLA, and the design is agnostic to which way it runs.
Here the GOAL arm has the long cap -- MAX_EPISODE_STEPS=2000 in run_lanefollow.sh
against 500 in run_lanefollow_nogoal.sh -- so goal rows will dominate instead.
Fixing composition on SAMPLE rather than correcting a known skew on insert is
what makes that direction irrelevant, and tests/test_mode_split_buffer.py asserts
both directions.

SO THE SPLIT IS ON INSERT AND THE RATIO IS FIXED ON SAMPLE. Rows are routed by
info["mode"] as they arrive; every batch is then assembled as a fixed block of
goal rows plus a fixed block of explore rows, regardless of how many steps each
mode actually produced.

PER BATCH, NOT PER SAMPLE. A stochastic 50/50 mixture would give the same ratio
in expectation but a variable one per update, and the two critics are updated on
masked subsets of the batch -- a batch that happened to contain three goal rows
would give the goal critic a three-sample gradient. A fixed block cannot.

WHEN ONE SIDE IS EMPTY the whole batch is drawn from the other, rather than
padding or erroring: at the very start of training only one mode has been
visited, and refusing to sample would stall the run. jaxrl2's Dataset.sample
draws WITH replacement, so a thin buffer still fills its share.
"""
import numpy as np


class ModeSplitBuffer:
    """Mode-stratified replay. Quacks like jaxrl2's ReplayBuffer."""

    def __init__(self, observation_space, action_space, capacity,
                 goal_frac=0.5, mode_key="mode"):
        from jaxrl2.data import ReplayBuffer
        half = max(1, int(capacity) // 2)
        self._buf = {1: ReplayBuffer(observation_space, action_space, half),
                     0: ReplayBuffer(observation_space, action_space, half)}
        self.goal_frac = float(goal_frac)
        self.mode_key = mode_key
        self._n = {1: 0, 0: 0}

    def __len__(self):
        return len(self._buf[1]) + len(self._buf[0])

    def counts(self):
        return {"goal_rows": self._n[1], "explore_rows": self._n[0]}

    def insert(self, tr, mode=None):
        """Route one transition. `mode` may be passed explicitly or carried in
        the transition dict, so this works whether the caller has the info dict
        to hand or has already stripped it."""
        if mode is None:
            mode = tr.get(self.mode_key, 1)
        m = 1 if int(mode) == 1 else 0
        tr = {k: v for k, v in tr.items() if k != self.mode_key}
        self._buf[m].insert(tr)
        self._n[m] += 1

    def sample(self, batch_size):
        n_goal = int(round(batch_size * self.goal_frac))
        n_exp = batch_size - n_goal
        have_g, have_e = len(self._buf[1]) > 0, len(self._buf[0]) > 0
        if have_g and not have_e:
            return self._buf[1].sample(batch_size)
        if have_e and not have_g:
            return self._buf[0].sample(batch_size)
        if not (have_g or have_e):
            raise ValueError("ModeSplitBuffer is empty")
        a = self._buf[1].sample(max(n_goal, 1))
        b = self._buf[0].sample(max(n_exp, 1))
        return _concat(a, b)


def _concat(a, b):
    """Join two batches leaf-wise. Handles the nested obs dicts jaxrl2 uses.

    Mapping, NOT dict. jaxrl2 returns flax FrozenDicts, which implement Mapping
    but do NOT subclass dict -- an `isinstance(x, dict)` test silently fails on
    them, the recursion never fires, and np.concatenate is handed the whole
    observation mapping. That produces an object array rather than an error, so
    the bug surfaces later as an unindexable batch instead of here.
    """
    from collections.abc import Mapping

    def join(x, y):
        if isinstance(x, Mapping):
            return type(x)({k: join(x[k], y[k]) for k in x})
        return np.concatenate([np.asarray(x), np.asarray(y)], axis=0)
    return type(a)({k: join(a[k], b[k]) for k in a})
