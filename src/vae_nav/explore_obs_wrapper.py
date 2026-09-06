"""Observation for the goal-free explorer: memory of where it has been.

Publishes {"explore": (21,), "latent": (D,)} -- mirroring VAEGoalWrapper's
two-key layout, because jaxrl2 gives each non-image key its own
Dense -> LayerNorm -> tanh projection before concatenating, so splitting the
image latent from the state block is what stops the 384-d latent swamping a
21-d state vector.

    explore block, 21-d
        [0:2]   dx, dy      displacement from the episode start, scaled by
                            POS_SCALE_M
        [2:4]   sin, cos    heading
        [4:20]  visit ring  16 slots, fraction of sampled points already visited
        [20]    novelty     1 - max cosine similarity to the image buffer

TWO MEMORIES, DELIBERATELY, AND THEY FAIL DIFFERENTLY:

    visit ring     "have I BEEN at that coordinate"   directional, needs pose
    novelty        "have I SEEN this view"            scalar, pose-free

The ring is the richer signal but needs a pose. The novelty scalar has no pose
in it at all and is therefore the one that survives on hardware. Keeping both is
what lets an ablation say which is carrying the behaviour -- the four arms being
true pose / dead-reckoned pose / novelty only / no memory.

NO GOAL, AND NO MODE BIT. The peer's v1 carries [gx, gz] zeroed and a mode bit
because ONE actor serves both modes (MultiCriticAL). We train a separate
explore policy, so a permanently-zero pair of goal coordinates and a constant
bit would be three dead inputs. They are omitted rather than zero-filled: a
constant input is not free, it is a channel the encoder must learn to ignore.

THE POSE IS PRIVILEGED DURING TRAINING (EXPLORE_POSE=true). See the constant
below for why, and for the measured cost of the dead-reckoned alternative.

DISPLACEMENT, NOT ABSOLUTE POSITION. Absolute map coordinates in a single town
are a memorisation crutch: a policy can learn "at x=140 turn left" and carry
nothing to a new town. Displacement from the episode start is bounded by
episode length times speed and says the same thing locally.
"""
import os

import numpy as np

try:
    from gymnasium import ObservationWrapper, spaces
except ImportError:
    from gym import ObservationWrapper, spaces

from vae_nav.dead_reckoning import DeadReckoner
from vae_nav.visit_ring import N_SLOTS, cell_of, visit_ring

# 500 steps at ~0.283 m/step is ~140 m, so this keeps dx, dy within roughly
# [-1, 1] without clipping away real structure.
POS_SCALE_M = float(os.environ.get("EXPLORE_POS_SCALE_M", 140.0))
EXPLORE_DIM = 2 + 2 + N_SLOTS + 1        # 21

# WHICH POSE THE VISIT RING IS BUILT ON.
#
#   true            the simulator's transform -- PRIVILEGED, and the default
#   dead_reckoned   integrated from speed and steer alone
#
# Privileged during training is a deliberate choice, not an oversight. The
# deliverable here is a trace and a value function for the graph, not a
# deployable driver, and a ring built on drifting pose degrades exactly where
# it matters: MEASURED on the dead-reckoned run, error reached 36 m over 312
# steps once the policy drove far enough to turn through junctions, which is 72
# cells at the 0.5 m grid against a ring that only reaches 3 m. The ring stayed
# self-consistent but stopped corresponding to the world, so loop closure --
# recognising a place on a second visit -- was gone.
#
# The reckoner keeps running either way and its drift is still reported, so the
# cost of the non-privileged variant stays measurable and the ablation arm
# remains available.
EXPLORE_POSE = os.environ.get("EXPLORE_POSE", "true").lower()


def explore_block(fields):
    """Pure function of plain per-step fields -> (21,) float32.

    Kept pure and free of simulator access for the same reason
    VAEGoalWrapper.goal_block is: it makes hindsight relabelling arithmetically
    recomputable on a worker thread, and it means the wrapper owns the ONLY
    definition of this transform.
    """
    dx, dy = np.asarray(fields["displacement"], np.float64) / POS_SCALE_M
    yaw = float(fields["yaw"])
    ring = np.asarray(fields["visit_ring"], np.float32)
    novelty = float(fields.get("novelty", 0.0))
    return np.concatenate([
        np.array([dx, dy, np.sin(yaw), np.cos(yaw)], np.float32),
        ring.astype(np.float32),
        np.array([novelty], np.float32),
    ]).astype(np.float32)


class ExploreObsWrapper(ObservationWrapper):
    """Owns the dead reckoner and the geometric visited set."""

    def __init__(self, env, source, reckoner=None, cell_size=0.5,
                 pose=EXPLORE_POSE):
        super().__init__(env)
        self.source = source
        self.reckoner = reckoner or DeadReckoner()
        self.pose_mode = str(pose).lower()
        self.cell_size = float(cell_size)
        self._true_pos = np.zeros(2)
        self._true_yaw = 0.0
        self.visited = set()
        self._start = np.zeros(2)
        self._last_info = {}

        self.observation_space = spaces.Dict({
            "explore": spaces.Box(-np.inf, np.inf, (EXPLORE_DIM,), np.float32),
            "latent": spaces.Box(-np.inf, np.inf,
                                 (self.source.latent_dim,), np.float32),
        })
        print(f"[ExploreObs] explore={EXPLORE_DIM}d "
              f"(dxy, sin/cos yaw, {N_SLOTS}-slot visit ring, novelty) "
              f"+ latent={self.source.latent_dim}d | pose = "
              f"{'TRUE (privileged)' if self.pose_mode == 'true' else 'DEAD-RECKONED'}")

    @property
    def pose(self):
        """(pos, yaw) the ring and displacement are built on."""
        if self.pose_mode == "true":
            return self._true_pos, self._true_yaw
        return self.reckoner.pos, self.reckoner.yaw

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        pos = np.asarray(info.get("hero_pos", (0.0, 0.0)), np.float64)
        yaw = float(info.get("hero_yaw", 0.0))
        self._true_pos, self._true_yaw = pos.copy(), yaw
        # Anchor the estimate at the true start pose: a vehicle knows where it
        # began in its own frame, and drift is what accrues after.
        self.reckoner.reset(pos=pos, yaw=yaw)
        self._start = pos.copy()
        self.visited = {cell_of(self.pose[0], self.cell_size)}
        self._last_info = info
        return self.observation(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Advance the estimate from PROPRIOCEPTION ONLY.
        self.reckoner.step(speed_kmh=float(info.get("speed_kmh", 0.0)),
                           steer=float(info.get("applied_steer", 0.0)))
        self._true_pos = np.asarray(info.get("hero_pos", self._true_pos),
                                    np.float64)
        self._true_yaw = float(info.get("hero_yaw", self._true_yaw))
        self.visited.add(cell_of(self.pose[0], self.cell_size))

        # Drift is reported every step so it is measurable rather than assumed.
        true_pos = np.asarray(info.get("hero_pos", self.reckoner.pos), np.float64)
        info["dr_drift"] = self.reckoner.drift(
            true_pos, float(info.get("hero_yaw", self.reckoner.yaw)))
        self._last_info = info
        return self.observation(obs), reward, terminated, truncated, info

    def observation(self, obs):
        info = self._last_info
        # Reuse the embedding the reward wrapper already computed for the
        # novelty query; only encode ourselves if it is absent.
        latent = info.get("latent_emb")
        if latent is None:
            rgb = info.get("rgb")
            latent = (self.source.latent(rgb) if rgb is not None
                      else np.zeros(self.source.latent_dim, np.float32))
        _pos, _yaw = self.pose
        block = explore_block({
            "displacement": _pos - self._start,
            "yaw": _yaw,
            # The ring is queried at the ESTIMATED pose against cells recorded
            # at estimated poses, so it stays self-consistent as drift grows --
            # mixing estimated queries with true-pose cells would corrupt it.
            "visit_ring": visit_ring(_pos, _yaw, self.visited, self.cell_size),
            # Read, never re-queried: the reward wrapper has already admitted
            # this frame, so asking again would compare it against itself.
            "novelty": info.get("novelty_raw", 0.0),
        })
        return {"explore": block, "latent": np.asarray(latent, np.float32)}
