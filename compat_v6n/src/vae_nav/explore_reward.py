"""Coverage-exploration reward, matching the Habitat peer's gated explorer v1.

Their spec, ported term for term from `explore_mode_wrapper.py` in
robot-learning-tools (checkpoint
`robot_policy_pointgoal_gated_s42_snap/checkpoint_600000`, their policy of
record):

    r = k_cell * [cell entered for the FIRST time this episode]
        - k_step
        - k_collision * [contact this step]

    k_cell = 1.0   k_step = 0.01   k_collision = 0.1
    cell   = 0.5 m x 0.5 m ground grid, cell_of(pos) = (floor(x/.5), floor(y/.5))
    bonus  = ONE-SHOT per cell per episode; no decay, no visit count
    v1 has NO observed-cell term (k_seen = 0)
    no reward normalisation, no clipping
    gamma_e = 0.99, target_entropy = -2.0, init temperature 0.2
    terminated is ALWAYS False; the step cap (500) truncates
    a stuck-nose-in wedge truncates with NO extra penalty

TWO REWARD-HACKING RULES THEY PAID FOR, carried over as constraints rather than
comments:

  * any information/seen bonus must stay BELOW one entered cell per step, or
    the actor spins in place harvesting it. v1 has no seen bonus, so the rule is
    satisfied by construction here -- and K_SEEN is deliberately absent rather
    than present-and-zero, so it cannot be switched on without reading this.
  * pay coverage on TRANSITIONS, never on presence, or the agent dwells. Hence
    the one-shot visited set: standing still earns nothing after the first step.

WHY COVERAGE, GIVEN IT IS THE WRONG INSTRUMENT FOR JUNCTION DETECTION. The peer
said so themselves and we agree: this Q encodes how much unvisited area lies
down each branch, so it is exploration STATE, not topology, and the same
junction reads differently on a second pass. Their visit ring reaches only 3 m,
so the exploration MDP is Markov to a 3 m memory and Q is not a function of the
map alone. This module exists to match v1 exactly, which is what was asked for;
a topology-only signal would need a keep-driving reward instead.

THE ONE PLACE WE KNOWINGLY DIVERGE. A car on a road covers cells almost
linearly -- at the measured 0.283 m/step and a 0.5 m grid, driving straight
enters a new cell on roughly every other step, so r is about +0.5/step
sustained. Their agent moves on a free-space navmesh and can spread laterally.
The RATIO k_step/k_cell = 0.01 is what their tuning constrains, and it is
preserved; the achievable coverage RATE is not, and cannot be without changing
the grid. Measured before trusting the port -- see ExploreReward.stats().
"""
import os

import numpy as np

try:
    from gymnasium import Wrapper
except ImportError:
    from gym import Wrapper

from vae_nav.visit_ring import CELL_SIZE_M, cell_of

K_CELL = float(os.environ.get("EXPLORE_K_CELL", 1.0))
K_STEP = float(os.environ.get("EXPLORE_K_STEP", 0.01))
K_COLLISION = float(os.environ.get("EXPLORE_K_COLLISION", 0.1))
EXPLORE_MAX_STEPS = int(os.environ.get("EXPLORE_MAX_STEPS", 500))
# Wedge = no progress under forward throttle, their "stuck nose-in against
# geometry". Their action space has no reverse and neither does ours, so such a
# state is unrecoverable and must be truncated rather than waited out.
WEDGE_WINDOW = int(os.environ.get("EXPLORE_WEDGE_WINDOW", 40))
WEDGE_MIN_MOVE_M = float(os.environ.get("EXPLORE_WEDGE_MIN_MOVE_M", 0.5))
WEDGE_MIN_THROTTLE = float(os.environ.get("EXPLORE_WEDGE_MIN_THROTTLE", 0.1))


class ExploreReward(Wrapper):
    """Goal-free coverage exploration. Matches the peer's v1 term for term."""

    def __init__(self, env, k_cell=K_CELL, k_step=K_STEP,
                 k_collision=K_COLLISION, max_steps=EXPLORE_MAX_STEPS,
                 cell_size=CELL_SIZE_M):
        super().__init__(env)
        self.k_cell = float(k_cell)
        self.k_step = float(k_step)
        self.k_collision = float(k_collision)
        self.max_steps = int(max_steps)
        self.cell_size = float(cell_size)
        self.visited = set()
        self._t = 0
        self._recent = []          # (pos, throttle) for the wedge detector
        self._n_new = 0
        self._n_collision = 0
        print(f"[Explore] v1 coverage reward: k_cell={self.k_cell} "
              f"k_step={self.k_step} k_collision={self.k_collision} "
              f"cell={self.cell_size} m, max_steps={self.max_steps} "
              "| one-shot per cell per episode, terminated always False")

    # -- the visited set is per EPISODE in v1 -------------------------------
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.visited = set()
        self._t = 0
        self._recent = []
        self._n_new = 0
        self._n_collision = 0
        pos = info.get("hero_pos")
        if pos is not None:
            # The starting cell counts as visited, so the first step cannot
            # collect a bonus for standing still.
            self.visited.add(cell_of(pos, self.cell_size))
        info["explore_visited"] = len(self.visited)
        return obs, info

    def _wedged(self):
        """No net movement over the window while asking for forward throttle."""
        if len(self._recent) < self.WEDGE_WINDOW_N:
            return False
        window = self._recent[-self.WEDGE_WINDOW_N:]
        throttles = [t for _p, t in window]
        if max(throttles) < WEDGE_MIN_THROTTLE:
            return False           # not trying to move; that is idling, not a wedge
        pts = np.asarray([p for p, _t in window], np.float64)
        return float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0))) < WEDGE_MIN_MOVE_M

    WEDGE_WINDOW_N = WEDGE_WINDOW

    def step(self, action):
        obs, _r, _done, _trunc, info = self.env.step(action)
        self._t += 1

        pos = np.asarray(info.get("hero_pos", (0.0, 0.0)), np.float64)
        cell = cell_of(pos, self.cell_size)
        is_new = cell not in self.visited
        if is_new:
            self.visited.add(cell)
            self._n_new += 1

        hit = bool(info.get("collision", False))
        self._n_collision += int(hit)

        reward = (self.k_cell * float(is_new)
                  - self.k_step
                  - self.k_collision * float(hit))

        self._recent.append((pos, float(info.get("applied_throttle", 0.0))))
        wedged = self._wedged()

        # terminated is ALWAYS False in explore mode -- there is no goal to
        # arrive at. Only the step cap and the wedge end an episode.
        terminated = False
        truncated = wedged or self._t >= self.max_steps

        info["explore_new_cell"] = bool(is_new)
        info["explore_visited"] = len(self.visited)
        info["explore_wedged"] = bool(wedged)
        info["explore_step"] = self._t
        return obs, float(reward), terminated, truncated, info

    def stats(self):
        """Coverage rate, for checking the port against their tuning.

        Their step cost is ~1% of one entered cell. If a car enters a new cell
        on most steps, the step cost is a far smaller fraction of the per-step
        reward than it was for them, and the ratio they tuned no longer holds.
        """
        t = max(1, self._t)
        return {"steps": self._t,
                "new_cells": self._n_new,
                "new_cell_rate": self._n_new / t,
                "collision_rate": self._n_collision / t,
                "visited": len(self.visited)}
