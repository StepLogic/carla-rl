"""The two losses that make IQL (Kostrikov, Nair, Levine 2021, "Offline
Reinforcement Learning with Implicit Q-Learning") query the critic only at
actions the replay holds (user 2026-11-06, after four SAC fine-tunes of the
clone collapsed as soon as its actor could climb the critic).

  expectile_loss   V(s) as an upper expectile of Q(s, a_replay): with tau > 0.5
                   it approaches the value of the better actions in the data,
                   without ever evaluating an action the data lacks.
  awr_weights      advantage-weighted regression: the actor imitates replay
                   actions, each weighted by exp(beta * A / scale), clipped.
"""
import torch


def expectile_loss(diff: torch.Tensor, tau: float) -> torch.Tensor:
    """Per-element asymmetric square |tau - 1(diff < 0)| * diff^2, diff = Q - V."""
    w = torch.where(diff < 0, 1.0 - float(tau), float(tau))
    return w * diff * diff


def awr_weights(adv: torch.Tensor, beta: float, scale: torch.Tensor,
                max_weight: float) -> torch.Tensor:
    """exp(beta * adv / scale), clipped at max_weight. `scale` (the learner's
    running Q std, actor_q_norm) makes beta independent of the reward units: our
    goal Q runs to ~100 with the arrival bonus, where a raw beta of 3 would turn
    every weight into 0 or the clip."""
    return torch.exp(float(beta) * adv / scale.clamp_min(1e-6)).clamp(max=float(max_weight))
