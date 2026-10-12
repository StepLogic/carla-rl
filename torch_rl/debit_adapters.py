#!/usr/bin/env python3
"""AdaptFormer adapters for DEBiT's CroCo ENCODER blocks. The decoder stays frozen.

User 2026-10-12. They first chose the decoder, then reversed when the peer's
argument was put to them, and the argument is our own measurement: the
cross-attention decoder IS DEBiT's correspondence mechanism, and the probe
(scripts/probe_debit_goal.py) credits that correspondence for the held-out gain --
DEBiT-B reads goal distance at R2 0.81 on held-out Town03 where the monocular
seg-VAE control manages 0.64, while on Town02 the control matches it. So the
decoder is the part that earns the generalisation. Train it and a later gain
cannot be attributed: better adaptation and an eroded prior look the same from
the outside. The encoder blocks are where adaptation is cheap and reversible.

THE METHOD (Chen et al. 2022, AdaptFormer). A croco encoder block is

    x = x + drop_path(attn(norm1(x), xpos))
    x = x + drop_path(mlp(norm2(x)))

and AdaptFormer adds a bottleneck branch PARALLEL to the MLP sublayer, reading
the same post-norm input:

    x = x + drop_path( mlp(h) + s * up(act(down(h))) ),     h = norm2(x)

Because the branch is parallel and reads `h`, wrapping `block.mlp` reproduces it
exactly without editing croco -- which matters, since croco is a vendored
third-party checkout we do not want to fork.

WHY THE UP-PROJECTION IS ZERO-INITIALISED. At step 0 the adapted network must be
the pretrained network, bit for bit. Otherwise the first gradient step starts from
a randomly perturbed encoder, part of what the pretraining bought is thrown away,
and nothing downstream looks wrong enough to notice. tests/test_debit_adapters.py
pins this both per block and through a full forward of the real DEBiT-tiny.

COST. 2*d*r + r + d parameters per block. At DEBiT-tiny's d=384 with r=64 that is
about 49 K per block against the MLP's ~1.2 M, so roughly 4%, and 8 encoder blocks
come to ~0.4 M trainable parameters.

ONE CONSEQUENCE THE CALLER MUST HANDLE, not handled here: a trainable encoder
invalidates any cached goal-side features. DEBiT's binocular encoder is goal-
coupled and the peer's rollout caches the goal tower's output once per episode,
which is only sound while the encoder is frozen. Once these adapters train, cached
goal features are stale the moment the adapter parameters move, and a replay that
stores the precomputed 3136-d binocular vector cannot be used to train them at all
-- the update needs the goal PIXELS and a forward/backward through the encoder.
See the arm's design note before wiring this into a learner.
"""
from typing import Iterator, List

import torch
import torch.nn as nn


class AdaptMLP(nn.Module):
    """A frozen MLP plus a trainable parallel bottleneck. Drop-in for `block.mlp`.

    Holds the original module as a child, so the wrapped block keeps working with
    whatever signature croco's Mlp has and the pretrained weights stay where
    `load_state_dict` expects them (prefix `mlp.mlp.` after wrapping).
    """

    def __init__(self, mlp: nn.Module, dim: int, bottleneck: int = 64,
                 scale: float = 0.1) -> None:
        super().__init__()
        if int(bottleneck) < 1:
            raise ValueError(f"bottleneck must be >= 1, got {bottleneck}")
        self.mlp = mlp
        for p in self.mlp.parameters():
            p.requires_grad_(False)
        self.down = nn.Linear(int(dim), int(bottleneck))
        self.act = nn.ReLU(inplace=False)
        self.up = nn.Linear(int(bottleneck), int(dim))
        # Scale is a BUFFER, not a Python float: it rides into a compiled update
        # with the module, and the peer measured that changing Python floats
        # passed into a compiled step retrace it on every call.
        self.register_buffer("scale", torch.tensor(float(scale)))

        nn.init.kaiming_uniform_(self.down.weight, a=5 ** 0.5)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)          # identity at init
        nn.init.zeros_(self.up.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.mlp(h) + self.scale * self.up(self.act(self.down(h)))

    def extra_repr(self) -> str:
        return (f"bottleneck={self.down.out_features}, "
                f"scale={float(self.scale):g}, parallel=True")


def add_encoder_adapters(net: nn.Module, bottleneck: int = 64,
                         scale: float = 0.1) -> int:
    """Freeze `net`, then wrap every ENCODER block's MLP with an AdaptMLP.

    Returns the number of blocks adapted. Decoder blocks are left exactly as they
    are -- that is the user's decision and the only reason this function is not
    simply "adapt every block".
    """
    if not hasattr(net, "enc_blocks"):
        raise AttributeError(
            "expected a CroCo-style network with .enc_blocks (and .dec_blocks); "
            f"got {type(net).__name__}. The adapters address the encoder tower "
            "by name so the decoder cannot be adapted by accident.")
    if any(isinstance(b.mlp, AdaptMLP) for b in net.enc_blocks):
        raise RuntimeError(
            "encoder blocks are already adapted; injecting twice would stack two "
            "parallel branches per block and double the trainable count")

    for p in net.parameters():
        p.requires_grad_(False)

    dim = int(getattr(net, "enc_embed_dim", None)
              or net.enc_blocks[0].norm2.normalized_shape[0])
    for blk in net.enc_blocks:
        blk.mlp = AdaptMLP(blk.mlp, dim=dim, bottleneck=bottleneck, scale=scale)
    return len(net.enc_blocks)


def adapter_parameters(net: nn.Module) -> Iterator[nn.Parameter]:
    """The parameters the optimiser should see: the adapters and nothing else."""
    for mod in net.modules():
        if isinstance(mod, AdaptMLP):
            yield from (mod.down.weight, mod.down.bias,
                        mod.up.weight, mod.up.bias)


def describe_adapters(net: nn.Module) -> str:
    mods: List[AdaptMLP] = [m for m in net.modules() if isinstance(m, AdaptMLP)]
    if not mods:
        return "no adapters: the encoder is fully frozen"
    n_par = sum(p.numel() for p in adapter_parameters(net))
    frozen = sum(p.numel() for p in net.parameters() if not p.requires_grad)
    return (f"AdaptFormer on {len(mods)} ENCODER blocks (decoder FROZEN), "
            f"bottleneck={mods[0].down.out_features}, "
            f"scale={float(mods[0].scale):g}, "
            f"{n_par:,} trainable vs {frozen:,} frozen "
            f"({100.0 * n_par / max(1, n_par + frozen):.2f}%), "
            "identity at init")
