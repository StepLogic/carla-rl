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
Broken is fatal only OUTSIDE a junction, NONE (unpainted) is never fatal,
off_road is always fatal.
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

# WHAT THE PROGRESS TERM MEASURES.
#
#   projected  displacement . wp_forward   -- the LANE's direction (default)
#   heading    displacement . hero_forward -- the CAR's own direction
#   raw        |displacement|              -- any motion at all
#   none       no progress term
#
# WHY THE CHOICE MATTERS, AND IT IS NOT A STYLE PREFERENCE.
#
# `projected` reads wp_forward, which comes from CARLA's road graph -- PRIVILEGED
# map information. It is only in the REWARD, never the observation, so the
# deployed policy does not need it; but it does mean the reward cannot be
# computed on a robot with no map, and progress is only paid when motion agrees
# with the lane. That agreement is precisely what makes it a lane-following
# signal rather than a go-anywhere signal.
#
# `heading` needs no map: the car's own yaw is proprioceptive. It pays forward
# motion regardless of whether that is along the lane.
#
# `raw` pays ANY displacement, and that is the dangerous one. |displacement| is
# maximised by moving fast in any direction whatsoever, so a car driving tight
# circles earns exactly as much as one driving down the road. This module's own
# novelty rule was written against the same failure mode -- "a car driving tight
# circles CAN earn, every heading is a different image" -- and the lane-departure
# penalty is then the ONLY thing discouraging it.
#
# `none` leaves the reward with no positive term at all: every step costs
# BASE_PENALTY plus steer_change, and the cheapest available outcome is to end
# the episode. That is a guaranteed suicide incentive, not a risk of one, and it
# is refused below rather than offered.
PROGRESS_MODE = os.environ.get("LANE_PROGRESS_MODE", "projected").lower()
# PROGRESS SCALE (user decision 2026-09-11, "scale displacement by 10"): the
# progress term is multiplied by this. At 10 a step at ~8 km/h pays ~2.2, so a
# 500-step explore episode is worth ~+1100 and a terminal crossing at step 37
# forfeits ~1000 of it -- the exit that trapped v4 stops being cheap by SCALE,
# not by a separate bill. Applies to both LaneExploreReward and the gated
# explore half, which share this constant.
PROGRESS_SCALE = float(os.environ.get("LANE_PROGRESS_SCALE", 1.0))
if PROGRESS_MODE not in ("projected", "heading", "raw", "none"):
    raise ValueError(f"LANE_PROGRESS_MODE={PROGRESS_MODE!r} is not one of "
                     "projected | heading | raw | none")

# Same set as core_state.py and the config's wedge. Pinned by test.
ILLEGAL_MARKS = {"Solid", "SolidSolid", "SolidBroken", "Curb", "Grass"}
# AN UNPAINTED BOUNDARY IS NOT A LANE VIOLATION. LaneMarkingType.NONE means
# the sensor fired where NOTHING IS PAINTED -- a road edge, a segment
# transition, an unmarked boundary. There is no lane discipline to violate at
# a line that does not exist.
#
# THIS WAS MISSING HERE WHILE THE ENV CONFIG'S WEDGE HAD IT, so the two
# filters disagreed despite this module claiming to share "the junction
# exemption already used by the wedge". MEASURED on the gated arm with NONE
# fatal: the scripted PID oracle -- competent by construction -- failed 64%
# of its episodes on lane_any. The env config records the same experiment at
# 100 m goals, where the oracle scored 0.000.
UNPAINTED_MARKS = {"NONE"}
# WHETHER off_road ENDS AN EPISODE. Same variable name the env config's wedge
# uses, so the two cannot be set to disagree.
#
# WHY IT IS WORTH TURNING OFF. off_road is get_waypoint(project_to_road=False)
# is None -- a purely geometric probe, and the config records it as LEAST
# reliable inside junctions, where lane polygons meet with gaps and a vehicle
# mid-turn can fall between them WHILE DRIVING THE ROUTE CORRECTLY. A single
# tick of None ends the episode.
#
# MEASURED, and it is why this gate now exists: in the gated arm the scripted
# PID oracle -- competent by construction -- failed 43% of its episodes on
# off_road, split evenly @junc 0.21 / @road 0.21, and scored 0.000 success
# against 0.980 for the same oracle in the time-to-goal arm.
#
# WHAT IT COSTS, AND THIS IS NOT SMALL. off_road is the ONLY signal that catches
# leaving the road where nothing is painted -- the lane sensor fires on crossing
# PAINT, and CARLA junction interiors have none (measured: not one crossing has
# ever been tagged @junc, in any run). With this off, a kerb-hop onto grass or a
# median costs nothing but time, and under a minimum-time objective a shortcut
# across grass is REWARDED whenever it is shorter. Watch for it.
OFF_ROAD_FATAL = os.environ.get("WEDGE_ON_OFF_ROAD", "1") == "1"
# ANY CROSSING IS FATAL. No marking filter, no junction exemption.
#
# WHY THE FILTER IS WORTH BYPASSING: it is mostly dead. Across every run in this
# repo only TWO marking types have ever been observed -- NONE (3224 @road, 380
# @junc) and Broken (3180 @road, 488 @junc). Solid, SolidSolid, SolidBroken,
# Curb and Grass have NEVER fired, so ILLEGAL_MARKS never matches and
# `crossed_illegal` is reachable only through its fail-closed empty-list branch.
# Mounting a kerb therefore reports as NONE, which the filter forgives.
#
# THE MEASURED RISK, so it is not discovered later: making NONE fatal at 100 m
# goals took the scripted PID oracle to 0.000 success over its last 100
# episodes, and NONE@road alone accounted for ~0.8 of episodes there. At a
# 20-50 m band the exposure is smaller but the same mechanism applies -- watch
# lane_any/orc, which is the solvability check.
LANE_ANY_FATAL = os.environ.get("LANE_STRICT", "0") == "1"


def lane_departure_is_fatal(diff_lane, marking_types, in_junction, off_road,
                            strict=None):
    """The junction-forgiving rule, shared with the wedge.

    A Broken line crossed inside a junction is the turn; the same line crossed
    on open road is the vehicle drifting out of lane. Unknown markings fail
    CLOSED -- absent data is not evidence of a legal crossing.
    """
    if off_road and OFF_ROAD_FATAL:
        return True
    if not diff_lane:
        return False
    # `strict` overrides the module's LANE_STRICT: explore mode passes False so
    # the goal arm's strictness never leaks into the explorer's rule.
    if (LANE_ANY_FATAL if strict is None else strict):
        return True
    marks = [str(t).split(".")[-1] for t in (marking_types or [])]
    # LANE_UNKNOWN_MARKS_FATAL=0 (user 2026-09-30: "no marking data at all and
    # offroad false - non terminal") stops a crossing whose marking the sensor
    # did not report from ending the episode. off_road is decided ABOVE and is
    # unaffected, so the freed case is exactly "on the road, crossed something,
    # no marking data". Default 1 keeps every earlier run's rule. Read per call,
    # not at import, so it matches the env config's wedge copy of this rule.
    unknown_fatal = os.environ.get("LANE_UNKNOWN_MARKS_FATAL", "1") == "1"
    crossed_illegal = ((not marks) and unknown_fatal) or any(
        m in ILLEGAL_MARKS for m in marks)
    if crossed_illegal:
        return True
    # Only when EVERY reported marking is unpainted. A mixed Broken+NONE
    # crossing is judged as the Broken one -- the painted line is the real
    # boundary, and the junction rule below decides it.
    #
    # NO MARKING DATA joins this branch under LANE_UNKNOWN_MARKS_FATAL=0. It
    # must be decided HERE and not fall through: the final line treats anything
    # unresolved as a painted crossing, so an early return is the only way an
    # unreported marking is actually forgiven on open road.
    if not marks:
        return bool(unknown_fatal)
    if all(m in UNPAINTED_MARKS for m in marks):
        return False
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
              f"max_steps={self.max_steps} novelty={_nov} "
              f"progress={PROGRESS_MODE} | junction crossings forgiven")
        if PROGRESS_MODE == "none":
            raise ValueError(
                "LANE_PROGRESS_MODE=none leaves this reward with no positive "
                "term: every step costs BASE_PENALTY + steer_change and the "
                "cheapest outcome is to end the episode immediately. That is a "
                "guaranteed suicide incentive. Use 'heading' for a map-free "
                "progress term, or 'raw' if you accept that circling pays.")
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

        # Progress term. See PROGRESS_MODE for why the choice is load-bearing.
        if PROGRESS_MODE == "projected":
            # Onto the LANE direction, not the goal's.
            fwd = np.asarray(info.get("wp_forward", (0.0, 0.0)), np.float64)
            projected = float(np.dot(disp, fwd))
        elif PROGRESS_MODE == "heading":
            # Onto the CAR's own heading -- no map anywhere in the reward.
            yaw = float(info.get("hero_yaw", 0.0))
            projected = float(disp[0] * np.cos(yaw) + disp[1] * np.sin(yaw))
        elif PROGRESS_MODE == "raw":
            # ANY motion. Pays a tight circle as much as a straight road.
            projected = float(np.linalg.norm(disp))
        else:
            projected = 0.0

        steer_change = float(info.get("action_smoothness_penalty", 0.0))
        speed_factor = float(info.get("speed_factor", 0.0))

        reward = -BASE_PENALTY - steer_change
        # Upstream gates the progress term on being under the target speed.
        if speed_factor < 1.0:
            reward += PROGRESS_SCALE * projected      # LANE_PROGRESS_SCALE

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
