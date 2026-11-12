"""A dm_control suite task as a gymnasium env (plan 2026-11-10 learner validation, Task 2).

`states`: the observation dict flattened in key order to float32. `pixels`: an RGB render
(image_size x image_size, camera 0) stacked over `frame_stack` steps channel-first, oldest
first, uint8 (the reset fills the stack with the first frame) -- the PlaNet / DrQ protocol is
84 px, 3 frames, action repeat 8. `action_repeat` sums rewards and stops early at the episode end.
DMC episodes end on a time limit, so the end is reported as truncated (the value bootstraps),
never terminated. Works under gymnasium 0.29 (the CleanRL control) and 1.x (our learners).

Registered ids (gymnasium.make("dmc_env:<id>") imports this module first):
  dmc/cartpole-swingup-states-v0   action repeat 1
  dmc/cartpole-swingup-pixels-v0   84 px, 3 frames, action repeat 8
Rendering needs MUJOCO_GL=egl on the cluster's CPU nodes (osmesa is not installed).
"""
import collections
from typing import Callable, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces


def _suite_factory(domain: str, task: str) -> Callable:
    def make(seed):
        from dm_control import suite
        return suite.load(domain, task, task_kwargs={"random": seed})
    return make


class DMCEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, domain: str = "cartpole", task: str = "swingup", obs_type: str = "states",
                 action_repeat: int = 1, frame_stack: int = 1, image_size: int = 84,
                 camera_id: int = 0, env_factory: Optional[Callable] = None):
        if obs_type not in ("states", "pixels"):
            raise ValueError(f"obs_type must be states or pixels, got {obs_type!r}")
        self.obs_type, self.action_repeat = obs_type, int(action_repeat)
        self.frame_stack, self.image_size, self.camera_id = int(frame_stack), int(image_size), int(camera_id)
        self._factory = env_factory or _suite_factory(domain, task)
        self._env = self._factory(None)
        spec = self._env.action_spec()
        self.action_space = spaces.Box(np.asarray(spec.minimum, np.float32),
                                       np.asarray(spec.maximum, np.float32), dtype=np.float32)
        if obs_type == "states":
            dim = int(sum(int(np.prod(s.shape)) for s in self._env.observation_spec().values()))
            self.observation_space = spaces.Box(-np.inf, np.inf, (dim,), np.float32)
        else:
            shape = (3 * self.frame_stack, self.image_size, self.image_size)
            self.observation_space = spaces.Box(0, 255, shape, np.uint8)
        self._frames = collections.deque(maxlen=self.frame_stack)

    def _states(self, ts) -> np.ndarray:
        return np.concatenate([np.asarray(v, np.float32).reshape(-1)
                               for v in ts.observation.values()])

    def _frame(self) -> np.ndarray:
        img = self._env.physics.render(self.image_size, self.image_size, camera_id=self.camera_id)
        return np.ascontiguousarray(np.asarray(img, np.uint8).transpose(2, 0, 1))

    def _obs(self, ts) -> np.ndarray:
        if self.obs_type == "states":
            return self._states(ts)
        return np.concatenate(list(self._frames), axis=0)

    def reset(self, *, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._env = self._factory(int(seed))
        ts = self._env.reset()
        if self.obs_type == "pixels":
            f = self._frame()
            self._frames.clear()
            for _ in range(self.frame_stack):
                self._frames.append(f)
        return self._obs(ts), {}

    def step(self, action):
        reward, ts = 0.0, None
        for _ in range(self.action_repeat):
            ts = self._env.step(np.asarray(action, np.float64))
            reward += float(ts.reward or 0.0)
            if ts.last():
                break
        if self.obs_type == "pixels":
            self._frames.append(self._frame())
        return self._obs(ts), reward, False, bool(ts.last()), {}


gym.register(id="dmc/cartpole-swingup-states-v0", entry_point=DMCEnv,
             kwargs={"domain": "cartpole", "task": "swingup", "obs_type": "states",
                     "action_repeat": 1})
gym.register(id="dmc/cartpole-swingup-pixels-v0", entry_point=DMCEnv,
             kwargs={"domain": "cartpole", "task": "swingup", "obs_type": "pixels",
                     "action_repeat": 8, "frame_stack": 3, "image_size": 84})
