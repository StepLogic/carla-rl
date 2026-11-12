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

N-STEP RETURNS (n_step > 1). With n_step=1 everything below is the plain
one-step buffer, byte-identical to every earlier run. With n_step=n, sample()
merges each drawn row with its up-to-(n-1) SUCCESSORS in the same SEGMENT into
one row: rewards become the discounted sum sum_k gamma^k r_{t+k} (gamma is the
mode's own critic discount, passed in), next_observations becomes s_{t+n'}, and
the batch gains an `n_steps` field carrying the ACTUAL merged length n', which
the learner raises its discount to (gamma**n') when bootstrapping. The fixed
point is unchanged -- the gamma-discounted return -- only the path to it.

SEGMENTS. A segment is a maximal run of rows inserted with the same segment id;
episode_her assigns ids that never cross an episode end, a leg end, or a
hindsight-goal run. A caller that inserts without an id gets single-row
segments, i.e. silent 1-step behaviour -- deliberate: an ungated or non-HER
caller has no episode boundaries to offer, and a wrong boundary is worse than
a short segment.

WHY POINTERS, NOT STORAGE ADJACENCY. Rows of one segment are NOT contiguous in
storage: two envs insert interleaved and HER flushes splice whole blocks
between an episode's steps. Walking forward by storage index would chain the
wrong rows silently. So insert() records, per slot, the storage slot of the
next row of the same segment (nxt) plus the successor's generation -- a
per-buffer insert counter stamped on every slot. A pointer is followed only
while gen[slot] still equals the generation recorded when the pointer was
written, so a ring-overwritten slot is detected even when its overwriter
belongs to the SAME segment id (ids are unique per env, but the slot's content
is not the row the pointer was written for). seg is kept as a second belt.

AND STILL ONE MORE HAZARD, found by the ring test: a pointer can be
generation-VALID yet stale in TIME. On a small ring the successor slot may
never have been overwritten -- slot 4 holding row 9, pointed at by row 13's
slot via a pointer written on an EARLIER ring pass -- so the generation check
passes and the walk would chain row 9 onto row 13, summing a row from a
previous pass through the ring. Successors are always strictly later inserts
(the counter increments monotonically), so the walk additionally requires
gen[successor] > gen[last]: a pointer that points BACKWARD in insert order is
dead, whatever its generation says.
"""
import numpy as np


class ModeSplitBuffer:
    """Mode-stratified replay. Quacks like jaxrl2's ReplayBuffer."""

    def __init__(self, observation_space, action_space, capacity,
                 goal_frac=0.5, mode_key="mode", n_step=1,
                 discount_goal=None, discount_explore=None):
        from jaxrl2.data import ReplayBuffer
        if int(n_step) < 1:
            raise ValueError(f"n_step must be >= 1, got {n_step!r}")
        if int(n_step) > 1 and (discount_goal is None
                                or discount_explore is None):
            raise ValueError(
                "n_step > 1 needs both critic discounts: the merged reward sum "
                "is discounted by the mode's OWN gamma, which only the caller "
                "(the trainer, from the agent) knows. Pass discount_goal and "
                "discount_explore.")
        half = max(1, int(capacity) // 2)
        self._buf = {1: ReplayBuffer(observation_space, action_space, half),
                     0: ReplayBuffer(observation_space, action_space, half)}
        self.goal_frac = float(goal_frac)
        self.mode_key = mode_key
        self.n_step = int(n_step)
        self._discounts = {1: None if discount_goal is None
                           else float(discount_goal),
                           0: None if discount_explore is None
                           else float(discount_explore)}
        self._n = {1: 0, 0: 0}
        # N-step bookkeeping, one parallel array set per mode buffer. -1 means
        # "no successor pointer written yet" / "no row ever written here".
        self._seg = {m: np.full(half, -1, np.int64) for m in (1, 0)}
        self._nxt = {m: np.full(half, -1, np.int64) for m in (1, 0)}
        self._nxt_gen = {m: np.full(half, -1, np.int64) for m in (1, 0)}
        self._gen = {m: np.zeros(half, np.int64) for m in (1, 0)}
        self._ins_ctr = {1: 0, 0: 0}
        self._last_of_seg = {1: {}, 0: {}}

    def __len__(self):
        return len(self._buf[1]) + len(self._buf[0])

    def counts(self):
        return {"goal_rows": self._n[1], "explore_rows": self._n[0]}

    def insert(self, tr, mode=None, segment=None):
        """Route one transition. `mode` may be passed explicitly or carried in
        the transition dict, so this works whether the caller has the info dict
        to hand or has already stripped it.

        `segment` is the n-step segment id, GLOBAL and UNIQUE per run ( callers
        tag their env index into the high bits -- see episode_her). None starts
        a fresh single-row segment."""
        if mode is None:
            mode = tr.get(self.mode_key, 1)
        m = 1 if int(mode) == 1 else 0
        tr = {k: v for k, v in tr.items() if k != self.mode_key}
        buf = self._buf[m]
        buf.insert(tr)
        if self.n_step > 1:
            slot = (buf._insert_index - 1) % buf._capacity
            self._ins_ctr[m] += 1
            gen = self._ins_ctr[m]
            self._gen[m][slot] = gen
            self._seg[m][slot] = int(segment if segment is not None
                                     else -self._ins_ctr[m] - 1)
            prev = self._last_of_seg[m].get(int(self._seg[m][slot]), -1)
            if prev != -1:
                self._nxt[m][prev] = slot
                self._nxt_gen[m][prev] = gen
            self._last_of_seg[m][int(self._seg[m][slot])] = slot
        self._n[m] += 1

    def sample(self, batch_size):
        n_goal = int(round(batch_size * self.goal_frac))
        n_exp = batch_size - n_goal
        have_g, have_e = len(self._buf[1]) > 0, len(self._buf[0]) > 0
        if have_g and not have_e:
            return (self._buf[1].sample(batch_size) if self.n_step <= 1
                    else self._assemble(1, batch_size))
        if have_e and not have_g:
            return (self._buf[0].sample(batch_size) if self.n_step <= 1
                    else self._assemble(0, batch_size))
        if not (have_g or have_e):
            raise ValueError("ModeSplitBuffer is empty")
        if self.n_step <= 1:
            a = self._buf[1].sample(max(n_goal, 1))
            b = self._buf[0].sample(max(n_exp, 1))
        else:
            a = self._assemble(1, max(n_goal, 1))
            b = self._assemble(0, max(n_exp, 1))
        return _concat(a, b)

    def _assemble(self, m, batch_size):
        """Merge each drawn row with up to n_step-1 same-segment successors.

        Vectorised over the batch: for k = 1..n_step-1 one vectorised step
        extends every row that is still extendable -- its last merged row
        bootstraps (mask 1, done 0), a successor pointer exists, and the
        successor slot still holds the exact row the pointer was written for
        (generation check). The merged row keeps the LAST merged row's
        masks/dones and its next_observations -- exactly the state the
        gamma**n' bootstrap in the learner must evaluate.
        """
        from flax.core import frozen_dict

        buf = self._buf[m]
        ds = buf.dataset_dict
        size = len(buf)
        indx = buf.np_random.integers(size, size=batch_size)
        gamma = self._discounts[m]

        rewards = np.asarray(ds["rewards"])
        masks = np.asarray(ds["masks"])
        dones = np.asarray(ds["dones"])
        seg = self._seg[m]
        nxt = self._nxt[m]
        nxt_gen = self._nxt_gen[m]
        gen = self._gen[m]

        r_acc = rewards[indx].astype(np.float64)
        mask_out = masks[indx].astype(np.float32)
        done_out = dones[indx].astype(np.float32)
        last = indx.copy()
        n_steps = np.ones(batch_size, np.float32)

        for k in range(1, self.n_step):
            can = (mask_out == 1.0) & (done_out == 0.0)
            j = nxt[last]
            j_safe = np.where(j >= 0, j, 0)
            ok = can & (j >= 0) & (gen[j_safe] == nxt_gen[last]) \
                & (seg[j_safe] == seg[last]) \
                & (gen[j_safe] > gen[last])
            r_acc += np.where(ok, float(gamma ** k) * rewards[j_safe], 0.0)
            mask_out = np.where(ok, masks[j_safe], mask_out)
            done_out = np.where(ok, dones[j_safe], done_out)
            last = np.where(ok, j_safe, last)
            n_steps = np.where(ok, np.float32(k + 1), n_steps)

        batch = {
            "observations": _gather(ds["observations"], indx),
            # THE BOOTSTRAP STATE is the successor of the LAST merged row --
            # valid even when the walk stopped short (segment cut or ring
            # overwrite): that row's stored next_observations is s_{end+1}.
            "next_observations": _gather(ds["next_observations"], last),
            "actions": np.asarray(ds["actions"])[indx],
            "rewards": r_acc.astype(np.float32),
            "masks": mask_out,
            "dones": done_out.astype(bool),
            "n_steps": n_steps,
        }
        return frozen_dict.freeze(batch)


def _gather(tree, idx):
    """Index a (possibly nested) dataset dict leaf-wise."""
    out = {}
    for k, v in tree.items():
        out[k] = _gather(v, idx) if isinstance(v, dict) else np.asarray(v)[idx]
    return out


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
