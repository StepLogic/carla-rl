"""Hindsight relabelling for the image-goal arm: goal rewrite + done-at-entry.

Subclasses EpisodeRelabelBuffer and overrides the ONLY thing that differs:
_relabeled(). The Phase 2 design's core fact is that the image-goal reward is
GOAL-INDEPENDENT -- every component is a function of the transition, not the
goal -- so a hindsight row keeps the flown reward exactly. There is no recost
path, no k_goal payout, no free-positive hazard in the reward: the arrival
test (pos[i+1] within goal_radius of the hindsight goal) sets dones=1 on the
entry transition only, and the task-relative done is exactly that entry.

Guard (a) -- min_disp >= goal_radius -- is inherited and still load-bearing:
it keeps hindsight goals far enough that reaching them took real travel.

The timeout final frame is never a goal source (flush_episode(truncated=True)
drops it): a clock artefact is not a destination. Physical terminals already
cannot appear mid-leg: success and hero destruction end the episode, and the
env_error path discards the whole episode in the trainer. z = the coll key
rides along untouched -- it depends on the trajectory up to t, never on which
goal the step is credited to.
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

    def flush_episode(self, truncated=False):
        self._truncated = bool(truncated)
        n = super().flush_episode()
        self._truncated = False
        return n

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
        return out