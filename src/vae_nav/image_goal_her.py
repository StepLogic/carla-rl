"""Hindsight relabelling for the image-goal arm: goal rewrite + done-at-entry.

Subclasses EpisodeRelabelBuffer and overrides flush_episode (the base's
single-row pair sweep cannot express done-at-entry -- see that method) and
_relabelled(). The Phase 2 design's core fact is that the image-goal reward is
GOAL-INDEPENDENT -- every component is a function of the transition, not the
goal -- so a hindsight row keeps the flown reward exactly. There is no recost
path, no k_goal payout, no free-positive hazard in the reward: the arrival
test (next state within goal_radius of the hindsight goal) sets dones=1 on
the entry transition only, and the task-relative done is exactly that entry.

Guard (a) -- min_disp >= goal_radius -- is inherited and still load-bearing:
it keeps hindsight goals far enough that reaching them took real travel.

The timeout final frame is never a goal source (flush_episode(truncated=True)
drops it): a clock artefact is not a destination. Physical terminals already
cannot appear mid-leg: success and hero destruction end the episode, and the
env_error path discards the whole episode in the trainer. z = the coll key
rides along untouched -- it depends on the trajectory up to t, never on which
goal the step is credited to.

THE DIRTY CUT (user 2026-09-20). A goal-mode episode's relabelable stream is
CUT at the FIRST dirty event -- a lane-invasion step, a collision (hit) step,
or an off-road step. Hindsight legs are drawn only from the clean prefix
BEFORE that index: the dirty step and everything after it are neither goal
sources nor inserted rows. A dirty episode therefore still CONTRIBUTES its
clean prefix (this supersedes "skip dirty episodes entirely"); one with no
clean prefix (dirty at step 0/1) contributes nothing. On-policy rows are
ALWAYS kept -- they were inserted at record() time and are untouched by the
cut. With no dirty event the flush is byte-identical to the pre-cut code.
"""
import numpy as np

from vae_nav.episode_her import EpisodeRelabelBuffer, GOAL


class ImageGoalRelabelBuffer(EpisodeRelabelBuffer):
    # The trainer's flush call site dispatches on this: base-class call sites
    # call flush_episode() bare; image-goal call sites pass truncated=.
    IMAGE_GOAL = True

    def __init__(self, buffer, state_fn, goal_radius, n_relabel=1,
                 min_disp=None, max_disp=np.inf, seed=0, max_steps=10000):
        # NO reward coefficients AT ALL: passing any would be dead weight at
        # best and a second reward function at worst. The superclass requires
        # gated=>time_to_goal only when gated=True, which is never the case
        # here.
        super().__init__(buffer, state_fn, goal_radius=goal_radius,
                         k_goal=0.0, k_collision=0.0, k_lane=0.0,
                         n_relabel=n_relabel, min_disp=min_disp,
                         max_disp=max_disp, seed=seed, max_steps=max_steps,
                         time_to_goal=False, gated=False)
        self.n_entry_done = 0
        # Episodes that were flown (at least partly) in EXPLORE mode. Hindsight
        # relabelling is defined only for goal-mode episodes: an explore row's
        # zeroed goal block and its reward are functions of the explore task,
        # and rewriting the goal key onto such a trajectory would fabricate
        # goal data the env never flew. These episodes are counted and skipped
        # at flush; their on-policy rows were already inserted at record().
        self.n_explore_eps = 0
        # Goal-mode episodes whose relabelable stream was cut at the first
        # dirty event (lane / hit / off-road); counted in flush_episode.
        self.n_dirty_prefix_eps = 0

    def _reset_episode(self):
        super()._reset_episode()
        self._ep_modes = []
        # Off-road is goal-INDEPENDENT like lane and hit: the vehicle left the
        # road whatever goal the step is later credited to. The base does not
        # stage it (its record() signature is shared lineage with the gated
        # arms), so this subclass stages it beside _ep_modes.
        self._ep_offroad = []

    def record(self, transition, fields, pos, hit, lane=False,
               speed_kmh=0.0, target_speed_kmh=0.0, mode=None, off_road=False):
        """Stage one step, tracking the commanded mode and the off-road flag.

        The base's `mode` argument routes the ON-POLICY row only (and only when
        gated=True, never here); this override additionally remembers it per
        step so flush_episode can refuse to relabel a non-goal episode. A
        step recorded without a mode defaults to GOAL, matching the base.
        `off_road` is staged per step for the flush-time dirty cut; it is NOT
        forwarded to super() -- the base now stages its own `_offroad` too
        (for the counting-regime recost, which this arm refuses), but its copy
        is unused here and keeping the two lists separate leaves the base's
        behaviour untouched.
        """
        self._ep_modes.append(int(mode) if mode is not None else GOAL)
        self._ep_offroad.append(bool(off_road))
        super().record(transition, fields, pos, hit, lane=lane,
                       speed_kmh=speed_kmh,
                       target_speed_kmh=target_speed_kmh, mode=mode)
        # Mirror the base's max_steps trim so _ep_modes and _ep_offroad stay
        # aligned with _txns (the base pops its oldest row before appending).
        if len(self._ep_modes) > self.max_steps:
            self._ep_modes.pop(0)
            self._ep_offroad.pop(0)

    def flush_episode(self, truncated=False):
        """Insert HINDSIGHT LEGS, one per relabel: rows from the leg start up
        to and including the first radius entry (the arrival row, done=1).

        WHY THE FULL LEG AND NOT THE BASE'S ONE-ROW PAIRS. The base draws a
        pair (i, tp) per i and relabels ONE transition; its entry test asks
        whether pos[i+1] enters the goal's radius. The goal is drawn
        >= min_disp (5 m) from pos[i] and one env step moves ~0.28 m, so
        pos[i+1] sits >= 4.7 m from the goal and the test is geometrically
        dead: the 1M-step v1 run logged her_entry_done = 0 across all
        245,025 relabelled rows. On this arm the flown reward is
        goal-INDEPENDENT, so those rows carry no goal signal at all -- no
        terminal, no distance shaping -- and the all-negative reward's only
        incentive (the cost stream ending at arrival) lived solely in the
        few on-policy success rows. Fix approved by the user 2026-09-19,
        with the 5-25 m retrain.

        VOLUME. The base would generate a pair for EVERY i; relabelling each
        as a full leg is O(n^2) rows per episode. This override draws ONE
        (start, goal) pair per leg per n_relabel and inserts the contiguous
        rows to the arrival -- order-leg-length rows per episode, the same
        order as the old single-row sweep.
        """
        # EXPLORE EPISODES ARE NEVER RELABELLED. Hindsight goals are drawn
        # from a trajectory's own positions; if that trajectory was flown in
        # explore mode the rewritten goal key would credit goal-mode reward to
        # transitions whose stored goal block and reward describe the explore
        # task. The on-policy rows are already in the buffer (inserted at
        # record() time), so there is nothing to insert and nothing to undo --
        # the episode is dropped from the hindsight stream entirely. A
        # mid-episode mode flip (must not happen) counts as explore: the safe
        # direction.
        if getattr(self, "_ep_modes", None) and 0 in self._ep_modes:
            self.n_explore_eps += 1
            self._ep_modes = []
            self._reset_episode()
            return 0
        self._truncated = bool(truncated)
        try:
            n = len(self._txns)
            if n < 2:
                return 0
            bounds = list(self._leg_ends)
            if not bounds or bounds[-1] != n:
                bounds.append(n)
            # THE DIRTY CUT. The first step that was dirty -- a lane crossing,
            # a collision, or an off-road step -- ends the relabelable stream:
            # hindsight legs are drawn only from the clean prefix before that
            # index. The bound cap does the whole job geometrically: every
            # goal candidate is drawn from range(start+1, end) with end <= the
            # cut and every inserted row sits in [start, goal), so neither the
            # goal frame nor a row can reach the dirty step or anything after
            # it. _insert_leg, _insert_leg_rows and _relabelled are untouched,
            # and with no dirty event the bounds are exactly today's, so the
            # flush is byte-identical. The truncated timeout-frame exclusion
            # stays relative to the full n: the capped bound already excludes
            # the dirty frames.
            cut = next((i for i in range(n)
                        if self._lane[i] or self._hit[i] or self._ep_offroad[i]),
                       None)
            if cut is not None:
                self.n_dirty_prefix_eps += 1
                capped = []
                start = 0
                for b in bounds:
                    b = min(b, cut)
                    if b > start:      # else: empty or starts at/after the cut
                        capped.append(b)
                        start = b
                bounds = capped
            inserted = 0
            self.last_pairs = []
            for _ in range(self.n_relabel):
                start = 0
                for end in bounds:
                    inserted += self._insert_leg(start, end, n)
                    start = end
                self.n_legs += len(bounds)
            self.n_relabelled += inserted
            self._reset_episode()
            return inserted
        finally:
            self._truncated = False

    def _insert_leg(self, start, end, n):
        """One (start, goal) pair in the leg [start, end), rows inserted up to
        the arrival. Returns the number of rows inserted, 0 when no frame
        pair qualifies.

        THE GOAL FRAME IS DRAWN FIRST, THEN THE START. A uniform START can
        sit within min_disp of every later frame near the leg's end (measured
        with real 0.28 m spacing: i = 91 of a 108-frame episode has no
        candidate >= 5 m ahead), so start-first sampling loses whole legs at
        a high rate. Goal-first, a uniform goal frame almost always has some
        start >= min_disp behind it; five shuffled tries, then the leg is
        skipped for this relabel draw. The timeout frame is excluded here
        (the subclass's _pick_goal filter, now inlined): a clock artefact is
        not a destination."""
        goal_ts = list(range(start + 1, end))
        if self._truncated:
            goal_ts = [t for t in goal_ts if t < n - 1]
        if not goal_ts:
            return 0
        self.rng.shuffle(goal_ts)
        for t in goal_ts[:5]:
            starts = [k for k in range(start, t)
                      if self.min_disp <= float(np.linalg.norm(
                          self._pos[t] - self._pos[k])) <= self.max_disp]
            if not starts:
                continue
            i = int(self.rng.choice(starts))
            self.last_pairs.append((i, t))
            return self._insert_leg_rows(i, t, n)
        return 0

    def _insert_leg_rows(self, i, tp, n):
        """Rows i..tp-1 under the hindsight goal pos[tp]; the leg ends at the
        FIRST radius entry. At k = tp-1 the entry is exact (the next state IS
        the goal frame, d = 0 < goal_radius); earlier if the trajectory
        revisits the goal's neighbourhood mid-leg, in which case the later
        rows are a different continuation and are not inserted."""
        goal = self._pos[tp]
        count = 0
        for k in range(i, tp):
            row = self._relabelled(k, goal, n)
            self._insert(row, GOAL)
            count += 1
            if float(row["dones"]) > 0.0:
                break
        return count

    def _pick_goal(self, i, n, pos):
        # The timeout final frame is not a destination.
        cand_hi = n - 1 if getattr(self, "_truncated", False) else n
        cand = [t for t in range(i + 1, cand_hi)
                if self.min_disp <= float(np.linalg.norm(pos[t] - pos[i]))
                <= self.max_disp]
        if not cand:
            return None
        return int(self.rng.choice(cand))

    def _relabelled(self, i, goal, n):
        """Goal key rewritten; flown reward kept; done only at the entry."""
        row = self._clean(self._txns[i])
        obs_goal = self.state_fn({**self._fields[i], "goal_pos": goal})
        nxt_goal = self.state_fn({**self._fields[min(i + 1, n - 1)],
                                  "goal_pos": goal})
        row["observations"] = dict(row["observations"])
        row["next_observations"] = dict(row["next_observations"])
        row["observations"]["goal"] = obs_goal
        row["next_observations"]["goal"] = nxt_goal

        # TASK-RELATIVE DONE: the transition whose NEXT state enters the
        # hindsight goal's region. Everything else bootstraps -- including a
        # row that was flown with dones=1 from a truncation, because the
        # hindsight goal is reached inside the actually-flown trajectory.
        d_next = float(np.linalg.norm(
            self._pos[min(i + 1, n - 1)] - goal))
        entry = d_next < self.goal_radius
        row["dones"] = np.float32(1.0 if entry else 0.0)
        row["masks"] = np.float32(0.0 if entry else 1.0)
        if entry:
            self.n_entry_done += 1
        return row

    def stats(self):
        out = super().stats()
        out["her_entry_done"] = self.n_entry_done
        out["n_explore_eps"] = self.n_explore_eps
        out["n_dirty_prefix_eps"] = self.n_dirty_prefix_eps
        return out