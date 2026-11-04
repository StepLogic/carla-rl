#!/usr/bin/env python3
"""The image-goal observation: DEBiT's binocular vector, plus the pixels it came from.

User 2026-10-12: train the image goal with SAC and DEBiT, 5-25 m band first. This
wrapper is the env half of that arm. It replaces the true-pose goal block
(VAEGoalWrapper.goal_block) with DEBiT-tiny's 3136-d binocular reading of
(current view, goal view), and it ALSO publishes both frames as uint8 pixels.

WHY THE PIXELS TRAVEL ALONGSIDE THE VECTOR. The user chose AdaptFormer adapters in
DEBiT's encoder blocks, so the encoder is no longer frozen. A replay that stores
only the precomputed binocular vector cannot train those adapters at all -- the
update needs the frames and a forward/backward through the encoder -- and a
goal-side feature cache goes stale the moment the adapter parameters move. The
pixels are also CHEAPER than the vector they replace: a 112x112x3 uint8 frame is
37.6 KB against 3136 fp16 (6.3 KB) plus, in the peer's Habitat rows, 49,152 VC-1
token values (98 KB). So the arm stores frames and encodes in the learner.

THE BAND. The goal pose comes from the env's own goal sampler, so the band is set
with GOAL_DIST_MIN / GOAL_DIST_MAX as everywhere else (5-25 m for this arm, the
user's "5-25 m first, then widen"). DEBiT compares two views and needs them to
overlap: our own probe chose 5-25 m for that reason, and past ~30 m a goal image
shares no pixels with the view and becomes a distractor.

THE WRONG-GOAL GATE, built in from the start rather than added later. v52's reward
has NO dense goal-directed term -- the goal arm is paid explore's road progress
plus an arrival bonus -- so "the policy ignores the image and just drives well" is
the most likely failure, and a success rate alone cannot see it. The peer's v2 arm
was goal-blind for 50M steps under a DENSE goal reward and only the paired gate
caught it. WRONG_GOAL_MODE substitutes a goal frame captured at least
WRONG_GOAL_MIN_M of route distance away, with the episode seed otherwise
identical, so SR(true) - SR(wrong) is a paired measurement over the same episodes.
"""
import os
import threading
from collections import deque
from typing import Optional

import numpy as np

try:                                            # gymnasium, else gym
    from gymnasium import Wrapper, spaces
except Exception:                                # pragma: no cover
    from gym import Wrapper, spaces

BINOC_DIM = 3136
CUR_KEY, GOAL_KEY = "debit_cur", "debit_goal"


def _as_uint8_rgb(frame) -> np.ndarray:
    """(H, W, 3) uint8. Refuses anything else loudly: a float frame scaled 0-1
    silently becomes a black image under DEBiT's uint8 preprocessing."""
    a = np.asarray(frame)
    if a.ndim != 3 or a.shape[2] != 3:
        raise ValueError(f"expected (H, W, 3), got {a.shape}")
    if a.dtype != np.uint8:
        raise ValueError(
            f"expected uint8 pixels, got {a.dtype}. DEBiT's preprocessing does "
            "the /255 itself, so a pre-normalised frame would be encoded as near "
            "black.")
    return a


# FOREIGN GOALS FOR EXPLORE EPISODES (v34, user 2026-11-02). With
# FOREIGN_GOAL_EXPLORE=1 an explore episode is shown the goal view of a
# DIFFERENT training town -- an unreachable goal -- while it is rewarded for
# driving forward in any direction and paid the goal reward at 100 m. With the
# mode bit hidden from the network (HIDE_MODE_BIT in train_carla_sac), the image
# alone says whether to navigate and stop or to keep driving: the pressure v29
# and v33 lacked (both learned to ignore the image).
#
# One pool per town, shared by every wrapper in the PROCESS: ThreadedVecEnv
# steps the training towns on threads, so a dict guarded by a lock is enough.
# Pools only ever hold views the training envs themselves rendered, so no
# held-out eval scenery reaches training.
FOREIGN_GOAL_EXPLORE = os.environ.get("FOREIGN_GOAL_EXPLORE", "0") == "1"
# GOAL_RELATCH_ON_MOVE (2026-11-03). The env resamples the goal when the car
# passes within the arrival radius too fast to count (carla_core.resample_goal),
# which in v30 happened in roughly one goal episode in two. The goal FRAME was
# latched at reset and never followed, so for the rest of such an episode the
# policy was rewarded for driving to a place that did not match the image it
# was shown -- label noise on exactly the signal this arm is meant to learn.
# With this on, the frame is re-latched whenever the goal moves more than
# RELATCH_MIN_M, so image and planner agree. Never in a wrong-goal or
# foreign-goal episode: there the mismatch IS the control. Default off.
GOAL_RELATCH_ON_MOVE = os.environ.get("GOAL_RELATCH_ON_MOVE", "0") == "1"
RELATCH_MIN_M = float(os.environ.get("GOAL_RELATCH_MIN_M", 1.0))
FOREIGN_POOL = {}
_FOREIGN_LOCK = threading.Lock()
_FOREIGN_CAP = 256


def _pool_add(town, frame):
    with _FOREIGN_LOCK:
        FOREIGN_POOL.setdefault(str(town), deque(maxlen=_FOREIGN_CAP)).append(
            _as_uint8_rgb(frame).copy())


def _pool_pick_foreign(town, rng):
    """A frame from some OTHER town's pool, or None if none has one yet."""
    with _FOREIGN_LOCK:
        others = [p for t, p in FOREIGN_POOL.items() if t != str(town) and len(p)]
        if not others:
            return None
        pool = others[int(rng.integers(len(others)))]
        return pool[int(rng.integers(len(pool)))]


class WrongGoalBank:
    """Goal frames seen earlier in this episode set, for the paired gate.

    A wrong goal must be a REAL goal view from somewhere else, not noise and not
    another episode's current view: the question the gate asks is whether the
    policy is reading the goal it was given, and the only honest distractor is a
    frame that looks exactly like a goal.

    Frames are kept with the pose they were captured at, so a candidate can be
    required to sit at least `min_m` away. Without that distance floor a "wrong"
    goal can be metres from the true one, the two views overlap, and the gate
    reports no gap for a policy that is in fact reading the image.
    """

    def __init__(self, capacity: int = 256, min_m: float = 5.0) -> None:
        self.frames = deque(maxlen=int(capacity))
        self.min_m = float(min_m)

    def add(self, frame: np.ndarray, pos) -> None:
        self.frames.append((_as_uint8_rgb(frame).copy(),
                            np.asarray(pos, np.float64)[:2].copy()))

    def pick(self, true_pos, rng: np.random.Generator) -> Optional[np.ndarray]:
        """A frame at least min_m from `true_pos`, or None if the bank has none
        yet. None is NOT substituted with a near frame: a gate that silently
        falls back to an overlapping distractor is worse than a missing number."""
        true_pos = np.asarray(true_pos, np.float64)[:2]
        far = [f for f, p in self.frames
               if np.linalg.norm(p - true_pos) >= self.min_m]
        if not far:
            return None
        return far[int(rng.integers(len(far)))]


class DebitGoalWrapper(Wrapper):
    """Observation = {"binoc": 3136-d, "state": ..., "mode": 1} + pixels in info.

    `encoder` is anything with .forward_tensors / .encode_batch; it is injected
    rather than constructed here so the learner can share ONE encoder with the
    env (the adapters must be the same parameters in both places) and so these
    tests need neither weights nor a GPU.
    """

    def __init__(self, env, encoder=None, state_keys=("vector",),
                 wrong_goal: bool = False, wrong_goal_min_m: float = None,
                 seed: int = 0, town: str = None) -> None:
        # state_keys DEFAULTS TO ("vector",), the env's proprioceptive block:
        # [steer/max_steer, throttle/max_throttle, speed/target_speed, mask flag].
        # NOT "state" -- that key holds the env's own 5400-d MobileNet features,
        # which this arm does not use, and feeding them as proprioception made
        # the replay row 43,035 wide instead of 37,639. Worth stating plainly:
        # `vector` carries NO goal bearing (the env computes a goal heading for
        # its own termination check and does not publish it), so an image-goal
        # policy here cannot read the goal direction from proprioception. If it
        # could, the wrong-goal gate would show no gap however blind the policy
        # was, and the arm's central question would be unanswerable.
        super().__init__(env)
        self.encoder = encoder
        self.state_keys = tuple(state_keys)
        self.wrong_goal = bool(wrong_goal)
        if wrong_goal_min_m is None:
            wrong_goal_min_m = float(os.environ.get("WRONG_GOAL_MIN_M", 5.0))
        self.bank = WrongGoalBank(min_m=float(wrong_goal_min_m))
        self._rng = np.random.default_rng(int(seed))
        self._goal_frame = None
        self._goal_pos = None            # pose the frame was latched for
        self._relatched = 0
        self._served_wrong = False
        self.town = town
        self._served_foreign = False
        self._foreign_fallback = False
        base = getattr(env, "observation_space", None)
        # ONLY the state keys survive, plus binoc and the mode. The env's own
        # "state" and "goal" entries are 5400-d MobileNet features this arm does
        # not use, and keeping them would put them in the replay row.
        inner = {k: v for k, v in (getattr(base, "spaces", {}) or {}).items()
                 if k in self.state_keys}
        inner["binoc"] = spaces.Box(-np.inf, np.inf, (BINOC_DIM,), np.float32)
        inner["mode"] = spaces.Box(0.0, 1.0, (1,), np.float32)
        self.observation_space = spaces.Dict(inner)

    # ----------------------------------------------------------------- frames --
    def _maybe_relatch(self, info):
        """Follow a goal the env moved mid-episode (GOAL_RELATCH_ON_MOVE)."""
        if not GOAL_RELATCH_ON_MOVE or self._served_wrong or self._served_foreign:
            return False
        gp = info.get("goal_pos")
        if gp is None or self._goal_pos is None or "goal_rgb" not in info:
            return False
        if float(np.linalg.norm(np.asarray(gp, np.float64)[:2] - self._goal_pos)) < RELATCH_MIN_M:
            return False
        self._goal_frame = _as_uint8_rgb(info["goal_rgb"]).copy()
        self._goal_pos = np.asarray(gp, np.float64)[:2].copy()
        self._relatched += 1
        # The new view is a real goal view, so it is an honest distractor later.
        self.bank.add(self._goal_frame, gp)
        return True

    def _frames(self, info):
        """(current, goal) uint8 frames for this step.

        The GOAL frame is LATCHED at reset. The env's goal camera re-renders it
        every step, and while the goal pose is fixed the frames differ anyway
        (weather, moving light, other traffic). An image goal that changes under
        the policy is a different task from the one we mean, so the episode is
        scored against the view it was given at the start.
        """
        cur = _as_uint8_rgb(info["rgb"])
        if self._goal_frame is None:
            raise RuntimeError("no goal frame latched: reset() was not called, "
                               "or the env published no info['goal_rgb']")
        return cur, self._goal_frame

    def _latch_goal(self, info):
        if "goal_rgb" not in info:
            raise KeyError(
                "info['goal_rgb'] is missing: this arm needs the env config's "
                "goal camera (sensor.goal). Without it there is no image goal.")
        true_frame = _as_uint8_rgb(info["goal_rgb"]).copy()
        goal_pos = info.get("goal_pos")
        if info.get("chain_soft_reset"):
            # GOAL CHAINING proof: the new goal's view must differ from the last.
            n = getattr(self, "_chain_logged", 0)
            if n < 5 and self._goal_frame is not None:
                self._chain_logged = n + 1
                diff = float(np.abs(true_frame.astype(np.int16)
                                    - self._goal_frame.astype(np.int16)).mean())
                print(f"[ChainGoal] soft reset {n + 1}: new goal at "
                      f"{np.round(np.asarray(goal_pos), 1).tolist() if goal_pos is not None else None}, "
                      f"{info.get('chain_goal_stats')}, goal-frame change {diff:.1f}/255",
                      flush=True)
        self._served_wrong = False
        self._goal_pos = (None if goal_pos is None
                          else np.asarray(goal_pos, np.float64)[:2].copy())
        if self.wrong_goal and goal_pos is not None:
            swap = self.bank.pick(goal_pos, self._rng)
            if swap is not None:
                self._goal_frame = swap
                self._served_wrong = True
        if not self._served_wrong:
            self._goal_frame = true_frame
        # FOREIGN GOAL in an explore episode (FOREIGN_GOAL_EXPLORE). info["mode"]
        # is GatedReward's draw for this episode (0 = explore).
        self._served_foreign = self._foreign_fallback = False
        if (FOREIGN_GOAL_EXPLORE and self.town is not None
                and int(round(float(info.get("mode", 1)))) == 0):
            far = _pool_pick_foreign(self.town, self._rng)
            if far is not None:
                self._goal_frame, self._served_foreign = far, True
            else:
                # No other town has contributed yet (the first resets): an
                # own-town view >= WRONG_GOAL_MIN_M away, flagged and counted.
                near = self.bank.pick(goal_pos, self._rng) if goal_pos is not None else None
                if near is not None:
                    self._goal_frame = near
                self._foreign_fallback = True
        if FOREIGN_GOAL_EXPLORE and self.town is not None:
            _pool_add(self.town, true_frame)
        # The TRUE frame always enters the bank, so later episodes have honest
        # distractors even in a wrong-goal run.
        if goal_pos is not None:
            self.bank.add(true_frame, goal_pos)

    # ------------------------------------------------------------ observation --
    def _observation(self, obs, info):
        self._maybe_relatch(info)
        cur, goal = self._frames(info)
        info[CUR_KEY], info[GOAL_KEY] = cur, goal
        info["served_wrong_goal"] = bool(self._served_wrong)
        info["served_foreign_goal"] = bool(self._served_foreign)
        info["foreign_goal_fallback"] = bool(self._foreign_fallback)
        info["goal_relatched"] = int(self._relatched)
        out = {k: np.asarray(obs[k], np.float32) for k in self.state_keys
               if k in obs}
        missing = [k for k in self.state_keys if k not in obs]
        if missing:
            raise KeyError(
                f"observation has no {missing} (it has {sorted(obs)}). The "
                "state block would silently be short and the learner's "
                "mode_index would land on the wrong column.")
        # THE MODE COMES FROM info, NOT the observation: GatedReward draws it per
        # episode and publishes it there, and this env's observation has no mode
        # key at all. It is added here so the observation is self-contained and
        # the learner's state block ends with it.
        if "mode" not in info:
            raise KeyError(
                "info has no 'mode': GatedReward must sit below this wrapper. "
                "Without it the gated learner cannot tell the two arms apart.")
        out["mode"] = np.asarray([float(info["mode"])], np.float32)
        if self.encoder is None:
            # NO ENCODER: the pixels still travel, so a collection run or a test
            # can exercise the whole path without weights or a GPU. A zero vector
            # is honest here only because it is paired with a flag.
            out["binoc"] = np.zeros(BINOC_DIM, np.float32)
            info["binoc_encoded"] = False
        else:
            out["binoc"] = np.asarray(
                self.encoder.encode(cur, goal), np.float32).reshape(-1)
            info["binoc_encoded"] = True
        return out

    def retarget(self, frame, info):
        """MAP ARMS (vae_nav.image_map): replace the latched goal mid-episode
        with a subgoal view and return the binocular vector for the CURRENT
        frame against it, so the observation already in hand is not left
        encoding the old goal for a step. info[GOAL_KEY] is updated too."""
        frame = _as_uint8_rgb(frame)
        if self._goal_frame is not None and frame.shape != self._goal_frame.shape:
            raise ValueError(f"subgoal frame {frame.shape} != goal frame "
                             f"{self._goal_frame.shape}: the map was built at "
                             "another image size")
        self._goal_frame = frame.copy()
        info[GOAL_KEY] = self._goal_frame
        if self.encoder is None:
            return np.zeros(BINOC_DIM, np.float32)
        return np.asarray(self.encoder.encode(info[CUR_KEY], self._goal_frame),
                          np.float32).reshape(-1)

    def latched_goal(self):
        return None if self._goal_frame is None else self._goal_frame.copy()

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._relatched = 0
        self._latch_goal(info)
        return self._observation(obs, info), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return (self._observation(obs, info), reward, terminated, truncated,
                info)
