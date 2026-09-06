"""Goal-free lane following, taken from the upstream reward this repo forked.

Source: nisheeth-golakiya/carla-rl, as vendored in
`src/configs/train_env_config_v4_image_goal.py:compute_reward`. The upstream
reward is goal-CONDITIONED; this keeps the goal-free half of it verbatim and
drops the rest. What is kept and what is dropped, explicitly:

    KEPT (goal-free)                        DROPPED (goal-dependent)
    -----------------------------------     ------------------------------
    -base_penalty            4e-3           delta_distance (progress to goal)
    -steer_change (smoothness)              +goal_reached_bonus       5.0
    +projected_distance, gated on speed     +checkpoint_bonus         2.0
    -collision_penalty       3.0            -wrong_junction_penalty   3.0
    -lane_departure_penalty  5.0
    -done_speed penalty      3.0

WHY GOAL-FREE AT ALL. Junction ambiguity is only a property of topology when
there is no goal. With a goal, one exit closes the route and the junction is
DECIDED, and which junctions look ambiguous moves as the goal moves. Measured on
the peer's maze data: 86% of their junction observations had the goal within 2
steps versus 0.9% of corridor observations, and the junction-vs-corridor sign
flips depending on whether you match on goal distance. That confound exists
because the policy was goal-driven.

THE ONE SUBSTITUTION, AND IT IS LOAD-BEARING. Upstream projects displacement
onto `core.destination.get_forward_vector()` when the vehicle is in a junction,
and onto the waypoint's forward vector otherwise. The junction branch is the
GOAL's heading, which would make this reward secretly goal-conditioned on
exactly the 60% of episodes JUNCTION_GOAL_FRAC routes through a junction. Here
the projection is always onto the LANE's own direction (`wp_forward`, published
by the config).

WHY THAT SUBSTITUTION IS ALSO WHAT CREATES THE SIGNAL WE ARE AFTER. On straight
road one heading aligns with the lane, so one steering band earns and the rest
do not -- the action-value surface is PEAKED. Inside a junction several lane
directions are locally present, so several headings earn -- the surface is
FLAT. That difference is the Q-gap this explorer exists to expose, and it comes
from the road, not from a goal.

TERMINATION DIVERGES FROM THE ENV, DELIBERATELY. The env's own `done` fires on
any lane crossing (`diff_lane`, raw), which would end an episode every time the
explorer turned through a junction -- junctions would read as catastrophic
rather than ambiguous, and the explorer could never learn to turn. So the
env's `done` is discarded and re-derived with the junction exemption already
used by the wedge and pinned by tests/test_wedge_lane_filter.py:
illegal markings (Solid/SolidSolid/SolidBroken/Curb/Grass) are fatal anywhere,
Broken/NONE are fatal only OUTSIDE a junction, off_road is always fatal.
"""
import os

import numpy as np

try:
    from gymnasium import Wrapper
except ImportError:
    from gym import Wrapper

# Upstream WEIGHTS, unchanged.
# NOVELTY BONUS. Paid per step on how unlike the buffer the current view is.
#
# WHY IT IS SMALL, AND WHY THAT BOUND IS THE WHOLE SAFETY ARGUMENT. The peer
# paid for this rule already: "any information/seen bonus must stay BELOW one
# entered cell per step, or the actor spins in place harvesting it." Our
# analogue: a stationary car cannot earn novelty (the view stops changing and
# max-similarity goes to 1), but a car driving tight circles CAN -- every
# heading is a different image. Forward progress along the lane pays
# ~0.283/step at speed, so the bonus ceiling must sit under that or circling
# beats driving. At K_NOVELTY=0.2 and a realistic novelty of ~0.31 for a
# genuinely new view, the bonus tops out near 0.06 -- roughly a fifth of the
# progress term.
K_NOVELTY = float(os.environ.get("K_NOVELTY", 0.2))
BASE_PENALTY = float(os.environ.get("LANE_BASE_PENALTY", 4e-3))
COLLISION_PENALTY = float(os.environ.get("LANE_COLLISION_PENALTY", 3.0))
LANE_DEPARTURE_PENALTY = float(os.environ.get("LANE_DEPARTURE_PENALTY", 5.0))
SPEED_PENALTY = float(os.environ.get("LANE_SPEED_PENALTY", 3.0))
LANE_EXPLORE_MAX_STEPS = int(os.environ.get("LANE_EXPLORE_MAX_STEPS", 500))

# Same set as core_state.py and the config's wedge. Pinned by test.
ILLEGAL_MARKS = {"Solid", "SolidSolid", "SolidBroken", "Curb", "Grass"}


def lane_departure_is_fatal(diff_lane, marking_types, in_junction, off_road):
    """The junction-forgiving rule, shared with the wedge.

    A Broken line crossed inside a junction is the turn; the same line crossed
    on open road is the vehicle drifting out of lane. Unknown markings fail
    CLOSED -- absent data is not evidence of a legal crossing.
    """
    if off_road:
        return True
    if not diff_lane:
        return False
    marks = [str(t).split(".")[-1] for t in (marking_types or [])]
    crossed_illegal = (not marks) or any(m in ILLEGAL_MARKS for m in marks)
    if crossed_illegal:
        return True
    return not bool(in_junction)


class LaneExploreReward(Wrapper):
    """Goal-free lane following. The exploration arm."""

    def __init__(self, env, max_steps=LANE_EXPLORE_MAX_STEPS,
                 novelty=None, encoder=None, k_novelty=K_NOVELTY):
        super().__init__(env)
        self.max_steps = int(max_steps)
        self.novelty = novelty
        self.encoder = encoder
        self.k_novelty = float(k_novelty)
        self._t = 0
        self._prev_pos = None
        self._last_raw = 0.0
        _nov = ("off" if novelty is None or encoder is None
                else f"k={self.k_novelty} persist={novelty.persist}")
        print(f"[LaneExplore] goal-free upstream lane reward | "
              f"base={BASE_PENALTY} collision={COLLISION_PENALTY} "
              f"lane={LANE_DEPARTURE_PENALTY} speed={SPEED_PENALTY} "
              f"max_steps={self.max_steps} novelty={_nov} | projection onto "
              "the LANE, junction crossings forgiven")
        if (novelty is not None and encoder is not None
                and not novelty.persist):
            print("[LaneExplore] WARNING: novelty memory resets every episode, "
                  "so there is NO pressure to pick a different exit than last "
                  "episode. Cross-episode turn diversity needs "
                  "NOVELTY_PERSIST=1.")

    def _novelty_bonus(self, info):
        """Bounded to [0, 1] before scaling.

        The raw signal returns 2.0 on an empty buffer ('unlike anything, since
        there is nothing'). That is right for a novelty QUERY and wrong for a
        reward -- it would pay a large one-off spike on the first step of every
        episode, which the agent can farm by resetting rather than exploring.
        """
        self._last_raw = 0.0
        if self.novelty is None or self.encoder is None:
            return 0.0, None
        rgb = info.get("rgb")
        if rgb is None:
            return 0.0, None
        emb = self.encoder.latent(rgb)
        # Handed up so ExploreObsWrapper does not encode the SAME frame a
        # second time. Two passes would cost ~2.7 ms/step for one image.
        info["latent_emb"] = emb
        raw, admitted = self.novelty.observe(emb)
        bonus = self.k_novelty * float(np.clip(raw, 0.0, 1.0))
        # Published so the observation wrapper can use the SAME value. It must
        # not re-query the buffer: observe() has already admitted this frame,
        # so a second query would compare it against itself and read 0.
        self._last_raw = float(np.clip(raw, 0.0, 1.0))
        return bonus, admitted

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._t = 0
        self._prev_pos = np.asarray(info.get("hero_pos", (0.0, 0.0)), np.float64)
        if self.novelty is not None:
            self.novelty.reset()     # a no-op when persisting
        return obs, info

    def step(self, action):
        # The env's own `done` is discarded -- see the module docstring.
        obs, _env_reward, _env_done, _env_trunc, info = self.env.step(action)
        self._t += 1

        pos = np.asarray(info.get("hero_pos", (0.0, 0.0)), np.float64)
        disp = pos - self._prev_pos
        self._prev_pos = pos

        # Displacement projected onto the LANE direction, not the goal's.
        fwd = np.asarray(info.get("wp_forward", (0.0, 0.0)), np.float64)
        projected = float(np.dot(disp, fwd))

        steer_change = float(info.get("action_smoothness_penalty", 0.0))
        speed_factor = float(info.get("speed_factor", 0.0))

        reward = -BASE_PENALTY - steer_change
        # Upstream gates the progress term on being under the target speed.
        if speed_factor < 1.0:
            reward += projected

        # HOW THIS PRODUCES TURNS AT JUNCTIONS -- and it is indirect.
        #
        # The bonus pays for seeing something new, not for turning. Turn
        # diversity is supposed to emerge through the CRITIC: at a junction,
        # the branch that leads to unseen road has more future novelty, so
        # Q(s, turn) > Q(s, straight) whenever the straight branch is already
        # covered. The agent turns because turning pays later, not because
        # turning is rewarded.
        #
        # THAT MECHANISM REQUIRES MEMORY THAT OUTLIVES THE EPISODE. With a
        # per-episode reset every episode starts blank, every branch looks
        # equally unseen, and there is no reason to pick a different exit than
        # last time -- the behaviour asked for simply cannot arise. Hence the
        # constructor warning and NOVELTY_PERSIST.
        novelty_bonus, admitted = self._novelty_bonus(info)
        reward += novelty_bonus

        collision = bool(info.get("collision", False))
        off_road = bool(info.get("off_road", False))
        over_speed = bool(info.get("done_speed", False))
        fatal_lane = lane_departure_is_fatal(
            bool(info.get("lane_any", info.get("lane_invasion_any", False))),
            info.get("lane_marking_types"),
            bool(info.get("in_junction", False)),
            off_road)

        if collision:
            reward -= COLLISION_PENALTY
        if fatal_lane:
            reward -= LANE_DEPARTURE_PENALTY
        if over_speed:
            reward -= SPEED_PENALTY

        reward = float(np.nan_to_num(reward, nan=0.0))

        # Driving ends when you can no longer drive. There is no goal to arrive
        # at, so nothing else terminates.
        terminated = bool(collision or fatal_lane)
        truncated = (not terminated) and self._t >= self.max_steps

        info["lane_explore_projected"] = projected
        info["lane_explore_fatal_lane"] = fatal_lane
        info["lane_explore_step"] = self._t
        info["novelty_bonus"] = novelty_bonus
        info["novelty_raw"] = self._last_raw
        info["novelty_admitted"] = admitted
        if self.novelty is not None:
            info["novelty_buffer"] = len(self.novelty)
        return obs, reward, terminated, truncated, info
