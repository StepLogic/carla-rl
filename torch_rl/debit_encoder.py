#!/usr/bin/env python3
"""DEBiT's frozen binocular encoder as the image-goal arm's goal path, in carla-rl.

Ported from the peer's robot-learning-tools version (their torch_rl is
env-agnostic), with the loading rewritten for THIS box: their copy does
`torch.load(path)["model"]`, which needs habitat_baselines importable and a
re-exported file. The released checkpoints here are habitat_baselines PPO
checkpoints under key "state_dict", so both of those assumptions fail.

It is goal-coupled: one CroCo transformer over (current frame, goal image), so it
cannot run on a single frame and it replaces BOTH the camera latent and the goal
block. User 2026-10-12 chose DEBiT-TINY (not B): the cross-attention decoder runs
every environment step, and tiny is what the peer's working 65-87% Habitat arms use.

THE THREE LOADING TRAPS, all of them measured here before (see the
debit-encoder-goal-probe note), all handled below:
 1. A plain torch.load dies with ModuleNotFoundError on habitat/omegaconf/hydra
    classes pickled into the `config` entry, BEFORE reaching a single tensor.
    scripts/inspect_debit_ckpt.py resolves every class from a missing module to an
    inert stub, so tensors unpickle and the config unpickles into stand-ins.
 2. The weights we want sit under `actor_critic.net.binocular_encoder.`. The same
    file also carries a monocular ResNet18 and a GRU state encoder, which this arm
    does not use; loading without stripping the prefix matches nothing.
 3. croco needs BOTH its parent directory on sys.path (so `croco.models` resolves
    as a package) AND its own root (croco.py does `from models.blocks import ...`).
    Either one alone fails.
"""
import hashlib
import os
import sys
from typing import Dict, Optional

import numpy as np
import torch
import torchvision.transforms as tf

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BINOC_PREFIX = "actor_critic.net.binocular_encoder."
BINOC_REPR_SIZE = 3136          # 49 patches x 64 channels, DEBiT's own head width

# From the published debit_tiny.pth config; verified against the checkpoint's own
# tensor shapes (8 encoder blocks, 2 decoder blocks, enc mlp fc1 (1536, 384)).
TINY_CROCO = {"img_size": 112, "patch_size": 16, "enc_embed_dim": 384, "enc_depth": 8,
              "enc_num_heads": 6, "dec_embed_dim": 256, "dec_depth": 2,
              "dec_num_heads": 8, "pos_embed": "cosine"}
BASE_CROCO = {"img_size": 112, "patch_size": 16, "enc_embed_dim": 384, "enc_depth": 12,
              "enc_num_heads": 6, "dec_embed_dim": 512, "dec_depth": 8,
              "dec_num_heads": 16, "pos_embed": "cosine"}


def _add_croco_to_path() -> str:
    """Put croco on sys.path, parent AND root (trap 3). Returns the root used."""
    roots = []
    if os.environ.get("DEBIT_SRC"):
        roots.append(os.path.join(os.environ["DEBIT_SRC"], "croco"))
    roots += [os.path.join(_HERE, "dependencies", "croco"),
              "/home/robotlab/scratch/debit/src/croco"]
    for root in roots:
        if os.path.isfile(os.path.join(root, "models", "croco.py")):
            for p in (os.path.dirname(root), root):
                if p not in sys.path:
                    sys.path.insert(0, p)
            return root
    raise SystemExit(
        "croco not found. Looked in: " + ", ".join(roots) +
        ". Set DEBIT_SRC to the directory CONTAINING croco/, or place the "
        "checkout at dependencies/croco.")


def load_binocular_state(path: str) -> Dict[str, torch.Tensor]:
    """The binocular encoder's tensors, prefix stripped (traps 1 and 2)."""
    scripts = os.path.join(_HERE, "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    from inspect_debit_ckpt import load_tensors
    tensors = load_tensors(path)
    state = {k[len(BINOC_PREFIX):]: v for k, v in tensors.items()
             if k.startswith(BINOC_PREFIX)}
    if not state:
        raise SystemExit(
            f"{path} has no tensors under {BINOC_PREFIX!r}. Keys look like: "
            + ", ".join(list(tensors)[:3]))
    return state


class _LinearHead(torch.nn.Module):
    """DEBiT's head: project every decoder token to `proj_channels`, then flatten."""

    def __init__(self, proj_channels: int) -> None:
        super().__init__()
        self.proj_channels = proj_channels

    def setup(self, croco_net) -> None:
        self.n_cls_token = getattr(croco_net, "n_cls_token", 0)
        self.proj = torch.nn.Linear(croco_net.dec_embed_dim, self.proj_channels)

    def forward(self, dec_out: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.proj(dec_out[:, self.n_cls_token:, :]).flatten(1, -1)


class DebitTinyEncoder:
    """Frozen by default. Call torch_rl.debit_adapters.add_encoder_adapters(enc.net)
    to make the ENCODER blocks trainable (the decoder stays frozen, per the user's
    2026-10-12 decision) -- and read that module's note about cached goal features
    first, because a trainable encoder invalidates them."""

    def __init__(self, weights: str, device: str = "cuda", half: bool = True,
                 croco: Optional[dict] = None) -> None:
        self.cfg = dict(TINY_CROCO if croco is None else croco)
        n_patches, rem = divmod(self.cfg["img_size"], self.cfg["patch_size"])
        assert rem == 0, "patch size must divide image size"
        proj_channels, rem = divmod(BINOC_REPR_SIZE, n_patches ** 2)
        assert rem == 0, "repr size must be a multiple of the patch count"
        self.output_size = BINOC_REPR_SIZE
        self.device = torch.device(device)
        self.half = bool(half) and self.device.type == "cuda"
        self.weights_path = weights
        self.weights_sha256 = hashlib.sha256(open(weights, "rb").read()).hexdigest()

        self.croco_root = _add_croco_to_path()
        from croco.models.croco_downstream import CroCoDownstreamBinocular

        net = CroCoDownstreamBinocular(_LinearHead(proj_channels), **self.cfg)
        missing, unexpected = net.load_state_dict(load_binocular_state(weights),
                                                  strict=False)
        # The head is TASK-SPECIFIC and is re-initialised here (DEBiT's own head
        # was trained for Habitat image-goal); everything else must match, or the
        # config and the checkpoint disagree and the encoder is not DEBiT's.
        missing = [k for k in missing if not k.startswith("head.")]
        unexpected = [k for k in unexpected if not k.startswith("head.")]
        if missing or unexpected:
            raise SystemExit(
                f"[DEBiT] weight mismatch: missing {missing[:6]} "
                f"({len(missing)}), unexpected {unexpected[:6]} ({len(unexpected)})")
        self.net = net.eval().to(self.device)
        if self.half:
            self.net = self.net.half()
        for p in self.net.parameters():
            p.requires_grad_(False)
        # DEBiT's preprocessing: resize to img_size, to float, ImageNet norm.
        self.pre = tf.Compose([tf.Resize(self.cfg["img_size"], antialias=True),
                               tf.ConvertImageDtype(torch.float),
                               tf.Normalize((0.485, 0.456, 0.406),
                                            (0.229, 0.224, 0.225))])

    def _prep(self, frames: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(np.ascontiguousarray(frames)).to(self.device)
        x = self.pre(x.permute(0, 3, 1, 2))
        return x.half() if self.half else x

    def encode_batch(self, cur: np.ndarray, goal: np.ndarray) -> np.ndarray:
        """ROLLOUT PATH: no gradient. The learner's update must NOT come through
        here once adapters train -- it needs the graph."""
        with torch.no_grad():
            out = self.net(self._prep(cur), self._prep(goal))
        return out.float().cpu().numpy()

    def encode(self, cur: np.ndarray, goal: np.ndarray) -> np.ndarray:
        return self.encode_batch(cur[None], goal[None])[0]

    def forward_tensors(self, cur: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        """TRAINING PATH: preprocessed tensors in, graph kept, so adapter
        gradients flow. Both frames go through the shared encoder tower."""
        return self.net(cur, goal)

    def describe(self) -> str:
        return (f"DebitTinyEncoder croco={self.cfg} out={self.output_size} "
                f"weights={os.path.basename(self.weights_path)} "
                f"sha256 {self.weights_sha256[:12]} "
                f"{'fp16' if self.half else 'fp32'} on {self.device} "
                f"croco={self.croco_root}")
