"""REWARD_SCALE: the environment's reward times a constant (user 2026-11-08: "multiply
reward in environment", "i want implementation to plain SAC").

Reward scaling is the standard knob SAC uses to set how much the return weighs against
the entropy term (Haarnoja et al. 2018 tune a reward scale per task; with an
auto-tuned temperature it still sets the size of Q relative to alpha * log pi). It lives
here, in the environment, and NOWHERE in the learner:

  * ScaledReward wraps the env built by torch_rl.carla_env_build.build_env: step()
    returns reward * k and keeps the unscaled value in info["raw_reward"].
  * The two reward sources that never pass through env.step get the same k where they
    enter the replay: HER's relabelled rows (torch_rl.image_her, which recomputes from
    the RAW reward, then scales) and the preloaded demonstrations
    (torch_rl.expert_data.fill_buffer). Every row the critic sees is on one scale.
"""
import os

import gymnasium as gym


def reward_scale() -> float:
    """REWARD_SCALE from the environment (default 1: unchanged). Must be > 0."""
    k = float(os.environ.get("REWARD_SCALE", "") or 1.0)
    if not k > 0.0:
        raise ValueError(f"REWARD_SCALE={k:g}: the reward scale must be positive")
    return k


class ScaledReward(gym.Wrapper):
    """step() returns the wrapped env's reward times `scale`; info["raw_reward"] keeps
    the unscaled one (HER recomputes its relabelled rewards from it)."""

    def __init__(self, env, scale: float):
        super().__init__(env)
        self.scale = float(scale)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        info["raw_reward"] = float(reward)
        return obs, float(reward) * self.scale, terminated, truncated, info
