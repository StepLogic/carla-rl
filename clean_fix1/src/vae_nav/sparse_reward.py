"""Sparse goal reward for CARLA, matching the peer Habitat setup.

    r = +k_goal        when distance to goal < goal_threshold
    r = -k_collision   on collision (and the episode ends)
    r =  0             otherwise

Applied as a WRAPPER rather than by editing train_env_config_v4_image_goal.py,
because that config's dense reward is still what the CORL 2026 experiments run
on. Two reward functions, two files, no shared mutable state.

WHY SPARSE AT ALL. The CARLA config's dense reward pays `delta_distance +
projected_distance` every step, where projected_distance is progress along the
road direction. That term cannot be recomputed for a hindsight goal on a worker
thread -- it needs the road geometry -- so a dense reward and HER relabelling
are mutually exclusive unless you carry a route-distance cache. Going sparse is
what makes HER tractable, and it is what the setup being copied actually ran.

THE COST, STATED PLAINLY: sparse reward is much harder to learn from in CARLA
than in Habitat point-nav. Habitat's agent stumbles into a 0.8 m goal by random
walk; a car with bounded throttle and no reverse does not stumble into a 2 m
goal 30 m away. This is why EXPERT_PROB exists in the trainer -- their run had a
scripted oracle driving 20% of episodes to seed the buffer with successes.
Without something like that, expect a long flat stretch at zero reward.
"""
import os

import numpy as np

try:
    from gymnasium import Wrapper
except ImportError:
    from gym import Wrapper

# Defaults mirror the peer's monoring-v3 config of record, with the arrival
# radius raised from 0.8 m to CARLA's existing 2 m resample radius.
K_GOAL = float(os.environ.get("K_GOAL", 600.0))
GOAL_THRESHOLD = float(os.environ.get("GOAL_THRESHOLD", 2.0))
K_COLLISION = float(os.environ.get("K_COLLISION", 5.0))
# LANE DEPARTURE IS A FAILURE OF THE SAME KIND AS A CRASH, and is treated
# identically: terminal, and penalised. An earlier version made env-side
# endings mere truncations, which removed the bad bootstrap but left the agent
# with NO signal about lane keeping at all -- 100 episodes ended on lane
# invasion with reward exactly 0.0. Terminating without penalising teaches
# nothing; penalising without terminating lets the agent drive down the wrong
# side of the road while paying a toll.
K_LANE = float(os.environ.get("K_LANE", K_COLLISION))
K_LIVING = float(os.environ.get("K_LIVING", 0.0))

# PER-STEP PROGRESS TOWARD THE GOAL: r += k_dist * (d_prev - d_now).
#
# This is POTENTIAL-BASED shaping with potential Phi(s) = -distance_to_goal, so
# summed over any trajectory it telescopes to d_start - d_end and adds nothing
# to a closed loop. That matters: it cannot be farmed by driving in circles, and
# in the potential-based form it does not change which policy is optimal -- it
# only changes how fast the critic learns one. A raw "reward for being close"
# would have neither property.
#
# IT IS HER-COMPATIBLE, unlike the env's own dense reward. The CARLA config pays
# `delta_distance + projected_distance`, and projected_distance is progress
# along the road direction -- it needs road geometry, so it cannot be recomputed
# for a hindsight goal on a worker thread. Plain euclidean delta CAN: it is a
# pure function of two positions and a goal, which is exactly what
# EpisodeRelabelBuffer already stores.
#
# THE TIMESTEP MUST MATCH THE PAYOUT. For a transition (s_i, a_i, s_i+1) the
# arrival test reads pos[i+1]; the progress term therefore reads
# ||pos[i] - g|| - ||pos[i+1] - g||, the same step. The peer's implementation
# mixes pos[i-1]/pos[i] shaping with a pos[i+1] payout, so their two terms
# describe different steps -- reproduced here would be a bug, not fidelity.
K_DIST = float(os.environ.get("K_DIST", 0.0))

# SPEED GATE ON THE PROGRESS TERM. Pay it only while below a target speed, so
# the agent stops being rewarded for going faster once it is driving at pace.
# 0 (or unset) uses the env's own target_speed; a negative value disables the
# gate entirely.
K_DIST_TARGET_SPEED = float(os.environ.get("K_DIST_TARGET_SPEED", 0.0))

# GATE MODE, and this is a REWARD-HACKING HAZARD worth stating plainly.
#
# "both" applies the gate to the whole term, which is the literal reading of
# "pay the per-step reward only when below target speed". It also BREAKS the
# potential-based property that made this term safe: gating both signs means
# moving AWAY above target speed costs nothing, while moving TOWARD below
# target speed pays. A cycle of "retreat fast, approach slowly" then nets
# positive reward on a closed loop, and the shaping becomes farmable.
#
# "positive" gates only the reward and always charges the penalty for moving
# away. That keeps every closed loop at or below zero -- the anti-farming
# property survives -- while still removing any incentive to exceed the target
# speed, which is what the gate is for.
#
# Default is "positive". The literal "both" is available and warns at
# construction, because a farmable shaping term produces a policy that looks
# busy and goes nowhere, which is expensive to diagnose from a learning curve.
K_DIST_GATE_MODE = os.environ.get("K_DIST_GATE_MODE", "positive")

# How the per-step goal-distance delta is taken before the speed gate.
# Shared name and semantics with lane_follow_reward.DELTA_MODE.
K_DIST_DELTA_MODE = os.environ.get("DELTA_MODE", "positive")  # positive|signed|abs

# SPEEDING PENALTY: r -= k_speed * (speed - target) while over the target.
#
# THE TARGET IS SEPARATE FROM THE ENV'S target_speed, DELIBERATELY. The env
# config carries target_speed=5.0 km/h, but it uses it as a NORMALISER --
# clip(speed / target, 0, 1) -- where a small value just saturates the feature.
# MEASURED on a 15,000-step trace, the trained policy drives at a median of
# 24.3 km/h, p90 46.1, and is above 5.0 km/h on 81.7% of steps. Penalising
# against 5.0 would fire on four steps in five with a ~20 km/h average excess
# and would dominate every other term in the reward.
#
# SPEED_LIMIT_KMH is therefore its own knob with a driving-plausible default.
# 0 falls back to the env's target_speed, which is almost certainly not what
# anyone wants and warns.
SPEED_LIMIT_KMH = float(os.environ.get("SPEED_LIMIT_KMH", 30.0))

# SCALE. At the measured p90 of 46 km/h the excess over a 30 km/h limit is
# ~16 km/h. k_speed=0.02 makes that -0.32 per step, comparable in magnitude to
# the progress term (+0.33/step at 24 km/h) so the two trade off rather than one
# swamping the other. k_speed=1.0 would be -16/step, or -1600 over a 100-step
# episode against k_goal=600, and the agent would simply stop.
K_SPEED = float(os.environ.get("K_SPEED", 0.02))
COLLISION_TERMINAL = os.environ.get("COLLISION_TERMINAL", "1") == "1"

# EXPLORE MODE: lane departures do not end the episode.
#
# For TRACE COLLECTION the vehicle is surveying, not performing the driving
# task, and terminating on a lane crossing severs the trace every ~1.5 s. The
# Habitat analogue severs on being STUCK, not on a rule violation, which is a
# different thing entirely.
#
# MEASURED with lane termination left on: 503 segments over 15,000 steps, a
# median of 1.27 m travelled per segment, 98.8% of segments too short to yield
# even one node at 2 m spacing, and 660 m of total coverage. The resulting graph
# had 6 trace edges and was 0.7% routable. Collisions STAY terminal -- driving
# into a wall should still reset -- so the trace is severed by getting stuck or
# crashing, which is the intended analogue.
EXPLORE_MODE = os.environ.get("EXPLORE_MODE", "0") == "1"

# GOAL CHAINING. On arrival the episode CONTINUES toward a freshly sampled goal
# instead of ending, up to a cap.
#
# The CARLA env already did the hard half: reaching a goal sets passed_chckp,
# increments checkpoint_count and calls resample_goal(). Arrival is NOT one of
# its done conditions. This wrapper's `task_terminal = reached or ...` was the
# only thing ending the episode, so the resampled goal was discarded the instant
# it was created. (The Habitat setup needed the guard in TWO layers because its
# env terminates on arrival too; ours needs it only here.)
#
# CAP IS RAMPED 1 -> CHAIN_MAX_GOALS over CHAIN_CURRICULUM_STEPS, mirroring
# their habitat_wrappers ramp. Default CHAIN_MAX_GOALS=1 means chaining is OFF,
# so nothing changes unless it is asked for.
#
# CALIBRATION FROM THEIR RUNS, worth stating because it is not free: chaining
# COST them 23 points on single-goal training (73.3% -> 50.0%) while nearly
# doubling mapped long-horizon performance (6.7% -> 25.0% at 8-20 m). The gain
# appears at the subgoal handoff and nowhere else. Keep a no-chain control.
CHAIN_MAX_GOALS = int(os.environ.get("CHAIN_MAX_GOALS", 1))
CHAIN_CURRICULUM_STEPS = int(os.environ.get("CHAIN_CURRICULUM_STEPS", 0))


def max_goals_now():
    """Current chain cap, ramped by global training step.

    The trainer publishes TRAIN_STEP; eval reads the SAME value, which is
    deliberate. Their habitat_wrappers carries the note that collisions were
    once terminal in eval but not in training and "that mismatch cost a whole
    run" -- so the cap and the termination rules are shared by construction
    rather than by two call sites agreeing.
    """
    if CHAIN_MAX_GOALS <= 1:
        return 1
    if CHAIN_CURRICULUM_STEPS <= 0:
        return CHAIN_MAX_GOALS
    step = float(os.environ.get("TRAIN_STEP", 0))
    progress = min(1.0, step / float(CHAIN_CURRICULUM_STEPS))
    return 1 + int((CHAIN_MAX_GOALS - 1) * progress)


def tqc_kappa(k_goal=None):
    """The Huber knee, DERIVED from the reward scale -- not a free parameter.

    Their measurement: at kappa=1 against k_goal=600, every goal arrival
    produced |td| up to 523.8, which sits in Huber's linear regime, and Q
    stalled at +0.46 by 75k steps. Retuning k_goal without moving kappa
    reproduces that.
    """
    return max(1.0, float(K_GOAL if k_goal is None else k_goal) / 10.0)


def assert_no_suicide_incentive(k_collision=K_COLLISION, k_living=K_LIVING,
                                max_episode_steps=2500,
                                collision_terminal=COLLISION_TERMINAL):
    """With terminal collisions, a per-step cost can make crashing an ESCAPE.

    If living costs more over an episode than crashing does once, the optimal
    policy is to drive into a wall immediately. Their trainer raises SystemExit
    on this rather than warning, and so does this.
    """
    if collision_terminal and k_collision <= max_episode_steps * k_living:
        raise SystemExit(
            f"suicide incentive: k_collision={k_collision} <= "
            f"max_episode_steps({max_episode_steps}) * k_living({k_living}). "
            "With terminal collisions the agent is better off crashing on step "
            "one than paying the living cost. Lower k_living or raise "
            "k_collision.")


class SparseGoalReward(Wrapper):
    """Replaces the env's dense reward with the sparse goal/collision one."""

    def __init__(self, env, k_goal=K_GOAL, goal_threshold=GOAL_THRESHOLD,
                 k_collision=K_COLLISION, k_living=K_LIVING,
                 collision_terminal=COLLISION_TERMINAL, k_lane=K_LANE,
                 max_episode_steps=2500, k_dist=K_DIST,
                 k_dist_target_speed=K_DIST_TARGET_SPEED,
                 k_dist_gate_mode=K_DIST_GATE_MODE,
                 k_speed=K_SPEED, speed_limit_kmh=SPEED_LIMIT_KMH,
                 k_dist_delta_mode=K_DIST_DELTA_MODE):
        super().__init__(env)
        self.k_speed = float(k_speed)
        self.speed_limit_kmh = float(speed_limit_kmh)
        self.k_dist_delta_mode = str(k_dist_delta_mode)
        if self.k_dist_delta_mode not in ("positive", "signed", "abs"):
            raise SystemExit(f"DELTA_MODE={self.k_dist_delta_mode!r} is not one "
                             "of positive | signed | abs")
        self.k_dist = float(k_dist)
        self.k_dist_target_speed = float(k_dist_target_speed)
        self.k_dist_gate_mode = str(k_dist_gate_mode)
        if self.k_dist and self.k_dist_gate_mode == "both":
            print("[SparseReward] WARNING: K_DIST_GATE_MODE=both gates the "
                  "progress term on BOTH signs. Retreating above target speed "
                  "then approaching below it nets positive reward on a closed "
                  "loop -- the shaping is farmable. 'positive' avoids this.")
        self._prev_d = None
        self._prev_goal = None
        self.k_lane = float(k_lane)
        self.k_goal = float(k_goal)
        self.goal_threshold = float(goal_threshold)
        self.k_collision = float(k_collision)
        self.k_living = float(k_living)
        self.collision_terminal = bool(collision_terminal)
        assert_no_suicide_incentive(self.k_collision, self.k_living,
                                    max_episode_steps, self.collision_terminal)
        self.goals_reached = 0
        # A PER-STEP COST MAKES CRASHING AN ESCAPE when collisions are terminal.
        # k_living is checked for this already; a speeding penalty is the same
        # hazard with one mitigating difference -- it is AVOIDABLE, since the
        # agent can simply not exceed the limit, whereas k_living is charged
        # unconditionally. So it is warned about rather than refused, with the
        # worst case spelled out.
        if self.k_speed and self.collision_terminal:
            worst = self.k_speed * 30.0 * max_episode_steps   # ~30 km/h over
            if worst > self.k_collision:
                print(f"[SparseReward] NOTE: sustained speeding could cost up to "
                      f"{worst:.0f} over {max_episode_steps} steps against "
                      f"k_collision={self.k_collision}. Crashing is then cheaper "
                      "than speeding for a whole episode -- avoidable by "
                      "obeying the limit, but worth knowing if collisions rise.")
        print(f"[SparseReward] k_speed={self.k_speed}@{self.speed_limit_kmh}kmh "
              f"k_dist={self.k_dist} "
              f"gate={self.k_dist_gate_mode}@{self.k_dist_target_speed or 'env'}kmh "
              f"chain_max={CHAIN_MAX_GOALS} "
              f"ramp={CHAIN_CURRICULUM_STEPS} | "
              f"k_goal={self.k_goal} radius={self.goal_threshold}m "
              f"k_collision={self.k_collision} k_lane={self.k_lane} "
              f"k_living={self.k_living} "
              f"terminal={self.collision_terminal} -> tqc_kappa={tqc_kappa(self.k_goal):g}")

    def reset(self, **kwargs):
        self.goals_reached = 0
        self._prev_d = None
        self._prev_goal = None
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, _dense, terminated, truncated, info = self.env.step(action)

        d = float(np.linalg.norm(
            np.asarray(info["goal_pos"], np.float64)[:2]
            - np.asarray(info["hero_pos"], np.float64)[:2]))

        # Must be a genuine bool: a truthy dict here would mark every step a
        # collision, which is the bug the peer repo paid for in its HER buffer.
        hit = info.get("collision", False)
        if not isinstance(hit, (bool, np.bool_)):
            raise TypeError(
                f"info['collision'] must be a bool, got {type(hit).__name__}. "
                "A dict is truthy on every step.")
        lane = info.get("lane_invasion", False)
        if not isinstance(lane, (bool, np.bool_)):
            raise TypeError(
                f"info['lane_invasion'] must be a bool, got {type(lane).__name__}.")

        # Prefer the env's OWN arrival flag; fall back to distance only when
        # absent (a mock env in tests). Recomputing distance against a goal the
        # env has already resampled is how this reported zero successes across
        # hundreds of episodes while the oracle was in fact arriving.
        if "arrived" in info:
            reached = bool(info["arrived"])
        else:
            reached = d < self.goal_threshold
        # PROGRESS SHAPING, with the retarget discontinuity handled.
        #
        # On arrival the env resamples the goal, so d jumps by metres in one
        # step. Differencing across that jump produces a spurious spike larger
        # than k_goal itself, at exactly the moment the handoff is being taught.
        # Detecting the goal change and restarting the difference is what
        # prevents it -- the peer flagged this as a trap and it is real here
        # because chaining retargets mid-episode.
        goal_now = np.asarray(info["goal_pos"], np.float64)[:2]
        retargeted = (self._prev_goal is None
                      or float(np.linalg.norm(goal_now - self._prev_goal)) > 1e-6)
        progress = 0.0
        if self.k_dist and not retargeted and self._prev_d is not None:
            delta = self._prev_d - d          # per-step REDUCTION in goal distance
            # DELTA_MODE, matching lane_follow_reward so the two paths cannot
            # describe different rewards:
            #   positive (default) -- clip(delta, 0, None): retreating pays 0
            #   signed             -- the raw delta; the only loop-neutral form
            #   abs                -- pays the same for retreating; farmable
            # Closed loop (there and back), per unit of travel:
            #   positive +0.3   signed 0.0   abs +0.6
            if self.k_dist_delta_mode == "signed":
                progress = self.k_dist * delta
            elif self.k_dist_delta_mode == "abs":
                progress = self.k_dist * abs(delta)
            else:
                progress = self.k_dist * max(delta, 0.0)
            # SPEED GATE. Below target the term is paid in full; at or above it
            # the reward for further progress is withheld, so there is nothing
            # to gain from going faster still.
            target = self.k_dist_target_speed
            if target == 0.0:
                target = float(info.get("target_speed_kmh", 0.0) or 0.0)
            if target > 0.0:
                speed = float(info.get("speed_kmh", 0.0) or 0.0)
                if speed >= target:
                    if self.k_dist_gate_mode == "both":
                        progress = 0.0
                    else:
                        # Withhold the reward, keep the penalty: a closed loop
                        # can then never pay more than it costs.
                        progress = min(progress, 0.0)
        self._prev_d = d
        self._prev_goal = goal_now

        # SPEEDING PENALTY. Goal-INDEPENDENT, like collision and lane: the
        # vehicle was over the limit regardless of which goal the step is later
        # credited to, so HER recomputes it unchanged rather than per goal.
        limit = self.speed_limit_kmh or float(info.get("target_speed_kmh", 0.0) or 0.0)
        speed_now = float(info.get("speed_kmh", 0.0) or 0.0)
        over = max(0.0, speed_now - limit) if limit > 0 else 0.0
        speed_pen = self.k_speed * over

        reward = (self.k_goal if reached else 0.0) \
            - (self.k_collision if hit else 0.0) \
            - (self.k_lane if lane else 0.0) - self.k_living + progress \
            - speed_pen

        # TERMINATION vs TRUNCATION, and why the distinction is load-bearing.
        #
        # The underlying CarlaEnv ends episodes for reasons this reward says
        # NOTHING about: lane invasion, idle timeout, speed limits. Marking
        # those `terminated` sets mask=0, which tells the critic the trajectory
        # genuinely ended with no future value. MEASURED consequence: 100
        # episodes, every one ending at ~44 steps on lane invasion, every
        # reward 0.0 -- the critic was being taught that all trajectories end
        # at zero, from 100% zero-reward transitions. Nothing could be learned
        # from that.
        #
        # Only the two outcomes the reward actually models are terminal:
        # arriving, and crashing. Everything else is a TRUNCATION -- the
        # episode stopped, but the value estimate should still bootstrap, the
        # same treatment a time limit gets.
        # CHAINING. Full k_goal on EVERY arrival -- no decay by chain index,
        # matching their setup. A later hop is already worth less through gamma
        # alone; discounting it again in the reward double-counts.
        cap = max_goals_now()
        if reached:
            self.goals_reached += 1
        chain_exhausted = reached and self.goals_reached >= cap

        # Collisions and lane departures stay terminal MID-CHAIN, unchanged by
        # chaining. They measured non-terminal collisions producing a
        # goal-BLIND policy (60% correct goal / 51% wrong goal) and marked the
        # mechanism REJECTED. Terminal failure is only safe because k_living=0;
        # with any per-step tax the agent crashes deliberately to escape it,
        # which assert_no_suicide_incentive already refuses to allow.
        lane_terminal = lane and not EXPLORE_MODE
        task_terminal = bool(chain_exhausted
                             or ((hit or lane_terminal) and self.collision_terminal))

        info["goals_reached"] = int(self.goals_reached)
        info["chain_cap"] = int(cap)

        # SUPPRESS A DONE CAUSED ONLY BY A LEGAL LANE CROSSING.
        #
        # MEASURED: 90% of episodes ended on `lane_any` -- the env's RAW
        # diff_lane flag, which fires on crossing ANY marking including a legal
        # dashed centre line. Filtering the PENALTY was not enough: the env ORs
        # that raw flag into its own `done`, so a legal turn still ended the
        # episode, now silently at reward 0.0. Episodes died at ~70 steps and
        # the oracle could never reach a checkpoint.
        #
        # get_done_status only computes flags -- CarlaEnv tears nothing down on
        # done -- so declining to end the episode here is safe: the next step()
        # proceeds normally.
        reasons = info.get("done_reasons") or {}
        if EXPLORE_MODE and reasons:
            # The ENV also ORs its raw lane flag into `done`, so suppressing the
            # wrapper's terminal is not enough on its own -- the episode would
            # still end, now silently.
            real = any(v for k, v in reasons.items()
                       if k not in ("lane_any", "off_road"))
            if bool(terminated) and not real:
                terminated = False
                info["suppressed_explore_lane_done"] = True
        if reasons:
            real = any(v for k, v in reasons.items() if k != "lane_any")
            spurious_lane_only = (bool(terminated) and reasons.get("lane_any")
                                  and not real and not lane)
        else:
            spurious_lane_only = False
        if spurious_lane_only:
            terminated = False
            info["suppressed_legal_lane_done"] = True

        env_ended_for_other_reasons = bool(terminated) and not task_terminal

        # SUCCESS IS PER-GOAL, NOT ALL-OR-NOTHING (their point 6). With chaining
        # the episode usually ends on a failure or the step limit, so a naive
        # `is_success = reached` read at the FINAL step would report 0 for an
        # episode that reached three goals. Any arrival makes the episode a
        # success; goals_reached carries the chain length for the mean.
        info["is_success"] = bool(self.goals_reached > 0)
        info["arrived_this_step"] = bool(reached)
        info["distance_to_goal"] = d
        info["progress_reward"] = float(progress)
        info["speed_penalty"] = float(speed_pen)
        info["speed_over_kmh"] = float(over)
        info["progress_gated"] = bool(
            self.k_dist and float(info.get("speed_kmh", 0.0) or 0.0) >=
            (self.k_dist_target_speed or float(info.get("target_speed_kmh", 0.0) or 0.0) or 1e9))
        info["env_truncation"] = env_ended_for_other_reasons
        return (obs, float(reward), task_terminal,
                bool(truncated or env_ended_for_other_reasons), info)
