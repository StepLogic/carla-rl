#!/usr/bin/env python3
"""DEBiT as the learner's visual stack, so its adapters receive gradient.

User 2026-10-13: pixels in the replay. This is the module that makes that worth
doing -- the adapters chosen on 2026-10-12 live in DEBiT's encoder blocks, and a
replay holding a precomputed 3136-d binocular vector cannot train them at all.
The update needs the FRAMES and a forward/backward through the encoder, which is
what this provides.

IT FITS THE EXISTING SEAM EXACTLY, WHICH IS WHY NOTHING ELSE CHANGES. The learner
already has a per-chunk goal path (`goal_tokens`, built for stacked goal-image
tokens): _embed reshapes a row's visual block to (N, embed_dim, grid, grid),
broadcasts the per-chunk goal over the time axis, and channel-stacks the two
before ONE call to the visual module. With embed_dim=3 and grid=112 that stack is
(N, 6, 112, 112) -- the current frame in channels 0:3 and the goal frame in 3:6 --
and a binocular encoder is precisely a module that wants two images. So _embed,
_burn_in and _roll are untouched; this module splits the stack and pairs them.

THE TARGET NETWORK COMES FREE. The learner builds `compress_targ` as a frozen
copy of the visual module and soft-updates it, so the target branch encodes with
its own slowly-following adapters rather than the live ones. Nothing extra to
wire.

WHAT IS TRAINED: the adapters (2.3% of the encoder) and the projection from
DEBiT's 3136-d output to visual_dim. The CroCo encoder blocks' own weights and
the whole cross-attention decoder stay frozen -- the decoder deliberately, since
our probe credits it with the held-out generalisation.

PIXELS ARE STORED RAW. Rows carry values in 0-255 at 112x112, and the /255 and
the ImageNet normalisation happen HERE, on the GPU, per update. Storing
normalised floats instead would bake today's constants into every row in the
replay and make a change of preprocessing a buffer flush.
"""
from typing import Optional

import torch
import torch.nn as nn

IMG = 112
BINOC_DIM = 3136
# DEBiT's preprocessing, as DebitTinyEncoder applies it on the rollout path. Kept
# here as buffers so a checkpoint carries them and a device move takes them along.
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


class DebitVisual(nn.Module):
    """(N, 6, 112, 112) raw-pixel stack -> (N, visual_dim).

    `encoder` is a DebitTinyEncoder (or anything exposing `.net` and
    `.output_size`); it is shared with the env wrapper so the rollout and the
    update use the SAME adapter parameters. `adapters_only` leaves the backbone
    frozen, which is the arm's design and not a default to drift from.
    """

    def __init__(self, encoder, visual_dim: int = 512,
                 binoc_dim: Optional[int] = None) -> None:
        super().__init__()
        self.net = encoder.net if hasattr(encoder, "net") else encoder
        d = int(binoc_dim if binoc_dim is not None
                else getattr(encoder, "output_size", BINOC_DIM))
        self.binoc_dim = d
        self.visual_dim = int(visual_dim)
        self.proj = nn.Linear(d, self.visual_dim)
        self.register_buffer("mean", torch.tensor(_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(_STD).view(1, 3, 1, 1))

    def _prep(self, x: torch.Tensor) -> torch.Tensor:
        """Raw 0-255 pixels -> normalised, in the module's own dtype."""
        x = x.to(dtype=self.mean.dtype)
        return (x / 255.0 - self.mean) / self.std

    def forward(self, stack: torch.Tensor) -> torch.Tensor:
        if stack.dim() != 4 or stack.shape[1] != 6:
            raise ValueError(
                f"expected (N, 6, {IMG}, {IMG}) -- the current frame stacked on "
                f"the goal frame -- got {tuple(stack.shape)}. With "
                "mono_encoder='debit' the config must set embed_dim=3, "
                f"grid={IMG} and goal_tokens=True.")
        cur = self._prep(stack[:, :3])
        goal = self._prep(stack[:, 3:])
        binoc = self.net(cur, goal)
        return self.proj(binoc)

    def describe(self) -> str:
        n_tr = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_fr = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        return (f"DebitVisual(binoc {self.binoc_dim} -> proj {self.visual_dim}; "
                f"{n_tr:,} trainable / {n_fr:,} frozen; pair encoded per step, "
                "adapters in the ENCODER only)")
