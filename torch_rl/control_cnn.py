"""The CONTROL STREAM of the two-stream image-goal policy: a small trainable CNN on ONE frame,
integrated by a GRU, with an action head (behaviour cloning on the expert's labels) and a lane head
(offset and heading error) used only in pretraining.

WHY IT EXISTS (user 2026-11-13, after the day's measurements):
  * c0's frozen features carry lateral offset (R^2 0.70-0.81) but NOT heading error (camera latent
    0.046, binocular 0.06) -- and heading is the quantity a steering action changes. From raw pixels
    one frame gives heading R^2 0.042, two frames 0.317: the information is temporal. In c0 the only
    place it appeared at all was the GRU state (0.31-0.36).
  * With image features a trained critic's Q varies by ~1% across the whole steering grid, and
    ascending that gradient destroyed c0's own actor in 10k steps (60/80 -> 27/80 of the strict
    validation, 524-682 -> 50 m between fault ends). So the critic needs a representation in which
    the effect of a steering action is visible.
  * The user's call: ONE frame plus a GRU ("add gru so it learns speed  do not stack froms"), so the
    net infers speed and heading rate from its own recurrence instead of a frame stack.

WHAT STAYS FROZEN: DEBiT's binocular features remain the GOAL stream. They are what gives the policy
its wrong-goal gap (+0.65, p<1e-4) and a SAC gradient has no business in a 20M-parameter transformer
on 256-sample batches.

PLAIN-SAC NOTE: the encoder layout (d4pg conv stack, LayerNorm+tanh bottleneck) and the random-crop
augmentation are DrQ's, and DrQ is named in PLAIN_SAC_RULES as an allowed source. Pretraining the
encoder by imitation before RL is an initialisation, not a change to the SAC update.
"""
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

# DrQ's encoder: four 3x3 convolutions, 32 channels, the first with stride 2, then a linear
# bottleneck with LayerNorm and tanh -- the same bottleneck convention as c0's mono head (ln_tanh),
# so the critic's input range is unchanged from the runs so far.
CHANNELS = 32
PAD = 4                      # DrQ's random shift: pad by 4 and crop back


def random_shift(x: torch.Tensor, pad: int = PAD) -> torch.Tensor:
    """DrQ's augmentation: replicate-pad by `pad` and take a random crop of the original size. One
    shift per sample. pad=0 returns the input untouched, so a run can turn it off."""
    if pad <= 0:
        return x
    n, c, h, w = x.shape
    padded = F.pad(x, (pad, pad, pad, pad), mode="replicate")
    dh = torch.randint(0, 2 * pad + 1, (n,), device=x.device)
    dw = torch.randint(0, 2 * pad + 1, (n,), device=x.device)
    rows = (torch.arange(h, device=x.device)[None, :] + dh[:, None])      # n x h
    cols = (torch.arange(w, device=x.device)[None, :] + dw[:, None])      # n x w
    idx_r = rows[:, None, :, None].expand(n, c, h, w)
    out = torch.gather(padded, 2, idx_r.clamp(max=padded.shape[2] - 1))
    idx_c = cols[:, None, None, :].expand(n, c, h, w)
    return torch.gather(out, 3, idx_c.clamp(max=out.shape[3] - 1))


class ControlCNN(nn.Module):
    """One frame (B, 3, px, px), uint8 or float in [0, 255] -> a bounded latent (B, latent_dim)."""

    def __init__(self, latent_dim: int = 50, px: int = 84, in_channels: int = 3):
        super().__init__()
        self.px = int(px)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, CHANNELS, 3, stride=2), nn.ReLU(),
            nn.Conv2d(CHANNELS, CHANNELS, 3, stride=1), nn.ReLU(),
            nn.Conv2d(CHANNELS, CHANNELS, 3, stride=1), nn.ReLU(),
            nn.Conv2d(CHANNELS, CHANNELS, 3, stride=1), nn.ReLU(),
        )
        with torch.no_grad():
            flat = self.conv(torch.zeros(1, in_channels, self.px, self.px)).flatten(1).shape[1]
        self.head = nn.Sequential(nn.Linear(flat, latent_dim), nn.LayerNorm(latent_dim), nn.Tanh())
        self.latent_dim = int(latent_dim)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        x = frames.float()
        if x.max() > 1.5:                     # the net owns the scaling: uint8 and float agree
            x = x / 255.0
        return self.head(self.conv(x).flatten(1))


class ControlGRU(nn.Module):
    """ControlCNN + [speed] -> GRU -> hidden state, with the pretraining heads.

    `forward` takes ONE step for a batch of B independent streams and returns (output, hidden), so
    the same module serves a rollout (B envs, one step at a time) and a chunked replay batch (loop
    over the chunk). Speed is appended to the CNN latent, not read from a column the critic also
    sees, so the GRU can learn the relation between what it sees and how fast it is going.
    """

    def __init__(self, latent_dim: int = 50, hidden_dim: int = 64, px: int = 84,
                 state_dim: int = 1, n_frames: int = 1, in_channels: int = 3):
        super().__init__()
        # n_frames > 1 (user 2026-11-13 "ok stack images"): the last n frames concatenated on the
        # channel axis, DrQ's layout. Heading is visible only as motion BETWEEN frames -- raw-pixel
        # R^2 0.042 from one frame, 0.317 from two -- so random_shift applies ONE shift per sample
        # across the whole stack; a per-frame shift would fabricate motion.
        self.n_frames = int(n_frames)
        self.cnn = ControlCNN(latent_dim=latent_dim, px=px,
                              in_channels=int(in_channels) * self.n_frames)
        self.gru = nn.GRUCell(latent_dim + int(state_dim), hidden_dim)
        self.act_head = nn.Linear(hidden_dim, 2)          # steer, throttle (tanh-bounded)
        self.lane_head = nn.Linear(hidden_dim, 2)         # offset (m), heading error (deg)
        self.hidden_dim = int(hidden_dim)
        self.state_dim = int(state_dim)

    def reset_hidden(self, batch: int, device=None) -> torch.Tensor:
        dev = device or next(self.parameters()).device
        return torch.zeros(int(batch), self.hidden_dim, device=dev)

    def forward(self, frames: torch.Tensor, state: torch.Tensor,
                hidden: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if hidden is None:
            hidden = self.reset_hidden(frames.shape[0], frames.device)
        z = self.cnn(frames)
        h = self.gru(torch.cat([z, state.float().reshape(z.shape[0], self.state_dim)], dim=-1),
                     hidden)
        return h, h

    def action(self, out: torch.Tensor) -> torch.Tensor:
        """The behaviour-cloning head: the expert's action, bounded like the env's action space."""
        return torch.tanh(self.act_head(out))

    def lane(self, out: torch.Tensor) -> torch.Tensor:
        """The auxiliary head: lateral offset and heading error, supervised in PRETRAINING ONLY from
        the recorded positions and the town's OpenDRIVE map (torch_rl.lane_labels). Never a policy
        input, and dropped before RL -- its job is to force the missing variable into the latent."""
        return self.lane_head(out)
