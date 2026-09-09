"""Observation wrapper: goal geometry + frozen VAE latent.

    obs = {"goal": (12,) float32, "latent": (128,) float32}

WHY TWO KEYS RATHER THAN ONE FLAT VECTOR. jaxrl2's PixelMultiplexer gives every
non-image observation key its own Dense(latent_dim) -> LayerNorm -> tanh
projection before concatenating. Two keys therefore means the VAE latent gets
its own normalisation instead of sharing one with the goal offsets.

That matters because the two blocks live on different scales. The goal offset is
in metres (order 1-30 in CARLA), the action history is in [-0.5, 0.5], and a VAE
posterior mean is roughly unit-normal. The peer Habitat repo put all of these in
one flat 28-d vector behind a single LayerNorm -- and their own audit found the
16-beam ring in that vector was raw metres in [0, 10] while the goal offset was
0-2.5, with nothing normalising between them. Splitting the keys avoids
inheriting that.

THE GOAL BLOCK IS DELIBERATELY GPS-DENIED. goal[0:2] is the offset in the
AGENT's frame, not world coordinates, so a policy cannot memorise absolute
positions in the training towns and must actually navigate. This is what makes
the held-out-town evaluation mean something.
"""
import os
from collections import deque

import numpy as np

try:
    from gymnasium import Wrapper, spaces
except ImportError:  # older gym in some of this repo's envs
    from gym import Wrapper, spaces

from vae_nav.latent_source import LatentSource

N_ACTION_HISTORY = 4
GOAL_DIM = 2 + 2 + 2 * N_ACTION_HISTORY      # offset, sin/cos yaw, action hist

# THE ENV'S 4-VECTOR: [steer, throttle, speed/target_speed, mask_goal].
#
# Publishing it is OPT-IN because adding an observation key changes the shape
# the agent is built against, and a checkpoint trained without it will not load
# into an agent built with it (policy_io.load_agent_state checks the pytree and
# fails loudly, which is the intended behaviour but still a break).
#
# It MUST be on whenever the reward is speed-gated. A reward that pays only
# below target speed is not learnable if the policy cannot observe its own
# speed -- the gate would be an unobservable coin flip from the agent's point of
# view. The goal block carries offsets and action history but no speed.
WITH_VECTOR = os.environ.get("OBS_WITH_VECTOR", "0") == "1"
VECTOR_DIM = 4

# OBS_WITH_GOAL=0 REMOVES THE GOAL SPECIFICATION FROM THE OBSERVATION.
#
# The key is OMITTED, not zero-filled. A permanently-zero 12-vector is not free:
# it is a channel the encoder must learn to ignore, and it makes a goal-free run
# indistinguishable from a goal-conditioned one by shape alone. Omitting it makes
# the difference visible in the observation space and in the [VAEGoal] line.
#
# CHECKPOINT COMPATIBILITY: this changes the pytree the agent is built against,
# so a checkpoint trained WITH the goal block will not load into an agent built
# without it. policy_io.load_agent_state fails loudly rather than silently
# mismatching, which is intended -- but it means goal-free runs need their own
# POLICY_FOLDER and cannot resume from carla_lanefollow.
#
# THE REWARD MUST MATCH. LaneFollowReward's main term is delta_distance, the
# per-step reduction in distance TO THE GOAL. Dropping the goal from the
# observation while keeping that reward leaves the agent paid for something it
# cannot see -- the return becomes unpredictable from the observation and the
# critic is being asked to fit noise. The coherent pairing already exists in this
# repo: LANE_EXPLORE=1 (LaneExploreReward) drops delta_distance and projects
# progress onto the LANE's own direction instead of the destination's. This flag
# is refused below unless the reward is goal-free, rather than letting the two
# disagree silently.
WITH_GOAL = os.environ.get("OBS_WITH_GOAL", "1") == "1"

# OBS_WITH_MODE=1 ADDS THE GATED POLICY'S COMMANDED MODE BIT.
#
# ITS OWN KEY, NOT A FIFTH SLOT IN `vector`. PixelMultiplexer gives every
# non-image key its own Dense -> LayerNorm -> tanh projection before
# concatenating. Appending the bit to `vector` would push a single 0/1 value
# through a LayerNorm shared with steer, throttle and speed, where it would be
# partly normalised away. Its own key gives it its own projection, and it leaves
# the 4-vector layout untouched so existing checkpoints stay readable.
#
# THIS FLAG DOES NOT OVERLOAD OBS_WITH_GOAL, and must not be made to. That flag
# has a load-bearing meaning -- omit the 12-d block, 144 -> 132 -- which
# run_lanefollow_nogoal.sh and every carla_lanefollow_nogoal checkpoint depend
# on. Gating needs the goal PRESENT and zeroed, because one actor cannot accept
# two input shapes. The two flags are orthogonal and the pairing is checked in
# check_gated_coherence below.
#
# COST, STATED PLAINLY: the gated observation is 145-d, which matches neither
# existing arm (144 goal-conditioned, 132 goal-free). policy_io.load_agent_state
# will refuse both, correctly. A gated run is therefore FROM SCRATCH, not a
# fine-tune, and that is the main reason the three-way comparison costs a full
# training budget.
WITH_MODE = os.environ.get("OBS_WITH_MODE", "0") == "1"
MODE_DIM = 1

# Must match vae_nav.gated_reward. Inverting these silently swaps which critic
# trains on which mode, and nothing downstream would raise.
GOAL, EXPLORE = 1, 0


def check_gated_coherence():
    """Reject gated configurations that cannot learn. Inert unless gating."""
    if not WITH_MODE:
        return
    if not WITH_GOAL:
        raise ValueError(
            "OBS_WITH_MODE=1 requires the goal block (OBS_WITH_GOAL=1). A gated "
            "policy switches between goal-reaching and goal-free driving. With "
            "no goal in the observation the mode bit selects between a goal the "
            "actor cannot see and the same observation again, so the two modes "
            "are indistinguishable to the network and the gate is decorative. "
            "Gating zeroes the goal block in explore mode; it does not remove "
            "it. If you want the goal genuinely absent, that is the SEPARATE "
            "goal-free arm (OBS_WITH_GOAL=0, OBS_WITH_MODE=0), which is the "
            "third leg of the ablation rather than a mode of this one.")
    if not WITH_VECTOR:
        raise ValueError(
            "OBS_WITH_MODE=1 requires the vector block (OBS_WITH_VECTOR=1). "
            "BOTH gated rewards are speed-gated -- they pay progress only below "
            "target speed -- so a policy that cannot observe its own speed sees "
            "the gate as an unobservable coin flip and the critic fits noise.")


def apply_mode(goal_block, mode):
    """Zero the goal block in explore mode, in place, and return it.

    ZEROED, NOT OMITTED. One actor cannot accept two input shapes. This is the
    dead-input cost of gating -- a permanently-zero channel the encoder must
    learn to ignore -- and measuring it against the separate goal-free arm is
    precisely why the gated arm exists.
    """
    if int(mode) == EXPLORE:
        goal_block[:] = 0.0
    return goal_block


class VAEGoalWrapper(Wrapper):
    """Publishes {"goal", "latent"} and owns the ONLY goal-block transform."""

    def __init__(self, env, source=None, latent_dim=128):
        super().__init__(env)
        self.source = source if source is not None else LatentSource(
            os.environ.get("VAE_CKPT", ""), latent_dim=latent_dim)
        self.latent_dim = self.source.latent_dim
        self._hist = deque([np.zeros(2, np.float32)] * N_ACTION_HISTORY,
                           maxlen=N_ACTION_HISTORY)

        self.with_vector = WITH_VECTOR
        self.with_goal = WITH_GOAL
        self.with_mode = WITH_MODE
        check_gated_coherence()
        if not self.with_goal:
            _goal_reward = (os.environ.get("LANE_FOLLOW", "0") == "1"
                            or os.environ.get("TIME_TO_GOAL", "0") == "1"
                            or float(os.environ.get("K_DIST", 0.0)) != 0.0)
            if _goal_reward:
                raise ValueError(
                    "OBS_WITH_GOAL=0 removes the goal from the observation, but "
                    "a GOAL-DEPENDENT REWARD is still enabled (LANE_FOLLOW / "
                    "TIME_TO_GOAL / K_DIST). That pays the agent for closing "
                    "distance to a goal it cannot observe, so the return is not "
                    "a function of the observation and the critic fits noise. "
                    "Use LANE_EXPLORE=1 (LaneExploreReward), which drops "
                    "delta_distance and projects onto the lane instead of the "
                    "destination.")
        space = {
            "latent": spaces.Box(-np.inf, np.inf, (self.latent_dim,), np.float32),
        }
        if self.with_goal:
            space["goal"] = spaces.Box(-np.inf, np.inf, (GOAL_DIM,), np.float32)
        if self.with_vector:
            # Bounds match the env's own declaration for this block.
            space["vector"] = spaces.Box(-5.1, 5.1, (VECTOR_DIM,), np.float32)
        if self.with_mode:
            space["mode"] = spaces.Box(0.0, 1.0, (MODE_DIM,), np.float32)
        self.observation_space = spaces.Dict(space)
        # THE BANNER MUST LIST EVERY KEY IN THE SPACE. It is the one place an
        # operator confirms the observation layout of a live run, and a gated
        # run that printed only three keys would read as an ordinary
        # goal-conditioned one. Pinned by tests/test_gated_obs.py.
        total = sum(int(np.prod(s.shape)) for s in space.values())
        print("[VAEGoal] obs={"
              + (f"'goal': ({GOAL_DIM},), " if self.with_goal else "")
              + f"'latent': ({self.latent_dim},)"
              + (f", 'vector': ({VECTOR_DIM},)" if self.with_vector else "")
              + (f", 'mode': ({MODE_DIM},)" if self.with_mode else "")
              + f"}} = {total}-d backend='{self.source.name}'"
              + ("  [GATED: goal block zeroed in explore mode]"
                 if self.with_mode else "")
              + ("" if self.with_goal else "  [GOAL-FREE: no goal in the obs]"))

    # -- THE transform. HER relabelling calls this, never a second copy. ----
    def goal_block(self, fields):
        """Build the 12-d goal block from plain per-step fields.

        `fields` carries only what the block needs:
            hero_pos  (2,)  world x, y
            hero_yaw  float radians
            goal_pos  (2,)  world x, y
            action_history  (N_ACTION_HISTORY, 2)

        Kept a pure function of those so hindsight relabelling can recompute it
        arithmetically on a worker thread, with no simulator access. This is the
        peer repo's PointGoalRelabelBuffer contract: relabelled states are
        produced by the wrapper's OWN transform, so they cannot drift from
        on-policy ones.
        """
        hero = np.asarray(fields["hero_pos"], np.float64)
        goal = np.asarray(fields["goal_pos"], np.float64)
        yaw = float(fields["hero_yaw"])

        d = goal - hero
        # CARLA IS LEFT-HANDED. Its basis vectors are
        #     forward = ( cos yaw,  sin yaw)
        #     right   = (-sin yaw,  cos yaw)
        # so the LEFT component is the negative of the projection onto `right`.
        #
        # The previous form used a textbook right-handed rotation and computed
        # d . right while calling it `left` -- the lateral bearing was MIRRORED.
        # VERIFIED against CARLA's own transform vectors: a goal placed 20 m to
        # the right reported left=+20.0 when the truth is -20.0. The policy was
        # told to steer away from the goal at every turn, in every run, which is
        # why it drove 110 m of a 500-step episode and never converged on a goal
        # 10-75 m away. Forward was always correct, which is why the failure
        # looked like "drives but does not arrive" rather than anything obvious.
        cy, sy = np.cos(yaw), np.sin(yaw)
        forward = d[0] * cy + d[1] * sy          # d . forward
        left = -(-d[0] * sy + d[1] * cy)         # -(d . right)

        hist = np.asarray(fields.get("action_history",
                                     np.zeros((N_ACTION_HISTORY, 2))), np.float32)
        hist = hist.reshape(-1)[:2 * N_ACTION_HISTORY]
        if hist.size < 2 * N_ACTION_HISTORY:
            hist = np.pad(hist, (0, 2 * N_ACTION_HISTORY - hist.size))

        return np.concatenate([
            np.array([forward, left], np.float32),
            np.array([np.sin(yaw), np.cos(yaw)], np.float32),
            hist.astype(np.float32),
        ]).astype(np.float32)

    # -- gym plumbing ------------------------------------------------------
    def _fields(self, info):
        return {
            "hero_pos": info["hero_pos"],
            "hero_yaw": info["hero_yaw"],
            "goal_pos": info["goal_pos"],
            "action_history": np.asarray(self._hist, np.float32),
        }

    def _observation(self, raw_obs, info):
        rgb = info.get("rgb", raw_obs.get("rgb") if isinstance(raw_obs, dict) else None)
        if rgb is None:
            raise KeyError(
                "VAEGoalWrapper needs the full-resolution RGB frame in info['rgb'] "
                "to encode. The env's downsampled 84x84 greyscale observation is "
                "not what the VAE was trained on.")
        out = {"latent": self.source.latent(rgb).astype(np.float32)}
        if self.with_goal:
            out["goal"] = self.goal_block(self._fields(info))
        if self.with_vector:
            vec = raw_obs.get("vector") if isinstance(raw_obs, dict) else None
            if vec is None:
                raise KeyError(
                    "OBS_WITH_VECTOR=1 but the env produced no 'vector' key. "
                    "train_env_config_v4_image_goal.get_observation returns "
                    "{'state', 'goal', 'vector'}; a wrapper below this one has "
                    "dropped it.")
            v = np.asarray(vec, np.float32).reshape(-1)[:VECTOR_DIM]
            if v.size < VECTOR_DIM:
                v = np.pad(v, (0, VECTOR_DIM - v.size))
            out["vector"] = v
        if self.with_mode:
            # DEFAULTS TO GOAL. An env without GatedReward below it publishes no
            # mode, and goal mode leaves the observation exactly as the
            # ungated arm's -- so a missing wrapper degrades to the familiar
            # behaviour rather than silently zeroing every goal.
            mode = int(info.get("mode", GOAL))
            if self.with_goal:
                out["goal"] = apply_mode(out["goal"], mode)
            out["mode"] = np.array([float(mode)], np.float32)
        return out

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        obs, info = out if isinstance(out, tuple) else (out, {})
        self._hist = deque([np.zeros(2, np.float32)] * N_ACTION_HISTORY,
                           maxlen=N_ACTION_HISTORY)
        return self._observation(obs, info), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # HISTORY OF APPLIED CONTROLS, not of actions.
        #
        # The peer's block carries a_{t-3..t}, and in Habitat the action IS the
        # control, so the two are the same thing. Here they are not: under
        # INCREMENTAL_ACTIONS the action is a delta of at most 0.01, so a
        # history of actions records only rates of change. Velocity is not
        # inferable from that -- but it IS inferable from the throttle LEVEL,
        # which is what the vehicle is actually holding.
        #
        # Falls back to the raw action when the env does not publish the
        # applied control, which keeps absolute-action runs identical (there
        # the two are equal by construction).
        applied = (info.get("applied_steer"), info.get("applied_throttle"))
        if applied[0] is not None and applied[1] is not None:
            self._hist.append(np.array(applied, np.float32))
        else:
            self._hist.append(np.asarray(action, np.float32)[:2])
        return self._observation(obs, info), reward, terminated, truncated, info

    def identity(self):
        """Merged into the RL checkpoint so eval cannot silently mismatch."""
        ident = self.source.identity()
        ident["goal_dim"] = GOAL_DIM
        ident["n_action_history"] = N_ACTION_HISTORY
        # Recorded so evaluation cannot silently pair a gated checkpoint with a
        # plain agent. The shapes differ (145 against 144), so the loader would
        # refuse anyway, but this names the reason.
        ident["with_mode"] = self.with_mode
        ident["with_vector"] = bool(self.with_vector)
        ident["with_goal"] = bool(self.with_goal)
        return ident
