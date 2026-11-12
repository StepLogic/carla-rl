"""Two replays, one batch (user 2026-11-05: "keep expert buffer and rl buffer and
sample from both").

RLPD's symmetric sampling for a fine-tune with recorded expert data: the expert
transitions live in a replay of their own, which the agent's data never
overwrites, and every update draws `expert_frac` of its chunks from it and the
rest from the agent's own replay. With one shared replay the share is whatever
the row counts make it -- ~88% expert rows at the unfreeze of fine-tune job
463472, so the critic barely saw the clone's own states -- and in a 1M-row ring
the expert rows would be the first to go once the agent had added 600k.

Everything but sampling (add, end_episode, size, sampleable, ...) is the online
replay's, so a loop written for one buffer uses this one unchanged.
"""
from typing import Dict

import torch


class MixedReplay:
    def __init__(self, online, expert, expert_frac: float = 0.5) -> None:
        if not 0.0 < float(expert_frac) < 1.0:
            raise ValueError(f"expert_frac={expert_frac} must lie strictly between 0 and 1")
        self.online, self.expert, self.expert_frac = online, expert, float(expert_frac)

    def __getattr__(self, name):
        # only reached for names MixedReplay does not define itself
        return getattr(self.online, name)

    def sample_chunks(self, n_chunks: int, length: int, burn_in: int) -> Dict[str, torch.Tensor]:
        """`expert_frac` of the chunks (rounded) from the expert replay, the rest
        from the online one -- all of them expert until the online replay has
        an ended episode, as during the offline critic warm-up."""
        n = int(n_chunks)
        n_e = n if not self.online.episodes else min(n, max(1, round(n * self.expert_frac)))
        parts = [self.expert.sample_chunks(n_e, length, burn_in)]
        if n - n_e:
            parts.append(self.online.sample_chunks(n - n_e, length, burn_in))
        if len(parts) == 1:
            return parts[0]
        return {k: torch.cat([p[k] for p in parts], dim=0) for k in parts[0]}
