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

    def __init__(self, env, encoder=None, state_keys=("state", "mode"),
                 wrong_goal: bool = False, wrong_goal_min_m: float = None,
                 seed: int = 0) -> None:
        super().__init__(env)
        self.encoder = encoder
        self.state_keys = tuple(state_keys)
        self.wrong_goal = bool(wrong_goal)
        if wrong_goal_min_m is None:
            wrong_goal_min_m = float(os.environ.get("WRONG_GOAL_MIN_M", 5.0))
        self.bank = WrongGoalBank(min_m=float(wrong_goal_min_m))
        self._rng = np.random.default_rng(int(seed))
        self._goal_frame = None
        self._served_wrong = False
        base = getattr(env, "observation_space", None)
        inner = dict(getattr(base, "spaces", {}) or {})
        inner.pop("latent", None)
        inner.pop("goal", None)
        inner["binoc"] = spaces.Box(-np.inf, np.inf, (BINOC_DIM,), np.float32)
        self.observation_space = spaces.Dict(inner)

    # ----------------------------------------------------------------- frames --
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
        self._served_wrong = False
        if self.wrong_goal and goal_pos is not None:
            swap = self.bank.pick(goal_pos, self._rng)
            if swap is not None:
                self._goal_frame = swap
                self._served_wrong = True
        if not self._served_wrong:
            self._goal_frame = true_frame
        # The TRUE frame always enters the bank, so later episodes have honest
        # distractors even in a wrong-goal run.
        if goal_pos is not None:
            self.bank.add(true_frame, goal_pos)

    # ------------------------------------------------------------ observation --
    def _observation(self, obs, info):
        cur, goal = self._frames(info)
        info[CUR_KEY], info[GOAL_KEY] = cur, goal
        info["served_wrong_goal"] = bool(self._served_wrong)
        out = {k: np.asarray(obs[k], np.float32) for k in self.state_keys
               if k in obs}
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

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._latch_goal(info)
        return self._observation(obs, info), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return (self._observation(obs, info), reward, terminated, truncated,
                info)
