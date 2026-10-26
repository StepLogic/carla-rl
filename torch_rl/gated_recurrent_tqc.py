#!/usr/bin/env python3
"""Two-mode (goal / explore) recurrent TQC over VC-1 patch tokens.

The two-mode sibling of torch_rl/recurrent_tqc.py, with its conventions kept
(burn-in under no_grad, actor on the DETACHED core state, critic grads cleared
after the actor step, skip on a chunk with nothing to train). What is new:

  * the VC-1 paper's compression layer turns each step's (D, 8, 8) patch
    tokens into a 512-d vector; it, a previous-action embedding and the GRU
    are the "encoder", trained by the critic losses only;
  * one actor, two critic ensembles, two temperatures -- the gated semantics
    of jaxrl2/jaxrl2/agents/tqc/gated_tqc_learner.py. Rows are split by the
    BUFFER's per-chunk mode, and each critic reads its own state columns
    (neither sees the mode bit, Q_e never the goal). A view may be EMPTY for
    the explore critic: under layout D13 (state [goal 3 | mode 1]) Q_e reads
    the core state h and the action only, so whatever it knows of the places
    already visited is in the GRU;
  * the CARLA-v13 learner fixes that were active in its runs: goal-only
    target clip, per-mode actor Q-normalisation, explore-actor warm-up, and
    no temperature step for a mode absent from the batch.

A replay row is `[tokens (D, grid, grid) flattened C-order | prev_action 2 |
state]`: channels-first, exactly eai-vc's `reshape_embedding` output, so the
token slice feeds the Conv2d with a view and no permute.

The vc2 image-goal configuration (`goal_tokens=True, prev_action=False`) has a
different row and a per-chunk goal: the row is `[cur tokens D*64 | state]` with
NO previous-action slot (no `act_embed`), and the goal image's tokens (D, 8, 8)
ride beside it as one `(B, D*64)` tensor per chunk (the flattened C-order grid,
the layout of the row's token slice). Each step's tokens are channel-stacked
with that chunk's goal, `cat([cur, goal], dim=1)` -> a (2D, 8, 8) grid, and go
through ONE `Vc1Compression(2*D)`; its 512-d output is the GRU's only input.
The goal is broadcast over the time axis of the burn-in rows AND the trained
window. The heads are unchanged: actor `[h | state]`, critics `[h]` + action.
The defaults (`goal_tokens=False, prev_action=True`) are the v1 module exactly.

The vc3 SAC arm (GOAL_FEATURES=debit; `goal_tokens=False, prev_action=False`)
carries the goal IN the row: `[cur tokens D*64 | binoc 3136 | mode 1]`, state
`[binoc | mode]` (state_dim 3137, mode_index 3136), no goal tensor.
`core_state_cols` names state columns concatenated onto each step's GRU input,
`[visual (| act_embed(prev_a)) | state[core_state_cols]]` -- burn-in and window
rows, online and target nets alike -- so the 3136 binoc columns (DEBiT's
relation of the frame to the goal) reach the memory, as in the PPO arm. Q_g
reads `[h | binoc]` (goal_cols), Q_e `[h]` (explore_cols empty: no binoc column
at its HEAD). Unlike v1, whose core never read the state, h itself now carries
the binoc through the GRU, so Q_e is blind to the binoc only at its head, not
through h. The default `()` adds nothing: every v1 / v2 module, RNG draw and
forward is unchanged.

`mono_encoder` picks the trained visual stack, `build_visual` (shared with
torch_rl/recurrent_ppo.py): "vc1" (the default) is the `Vc1Compression` above
over frozen VC-1 tokens, built at the same point with the same draws, so every
v1 / v2 / v3 module and RNG draw is unchanged. "resnet" (vc3r, debit-only) is
DEBiT's monocular branch trained from scratch, torch_rl/resnet_mono.py's
`MonoResNetEncoder` (half-width ResNet-18, GroupNorm, fixed ImageNet
normalisation, then the same compression): the row's token block is then the
/255 RGB frame, (3, s, s) C-order, so `embed_dim` must be 3 and `grid` the
image size s (a multiple of 32); `_embed`'s reshape yields (N, 3, s, s) pixels
with no permute. Both stacks sit in `compress` / `compress_targ`: trained by
the critic losses only (opt_critic), soft-updated, autocast under bf16 by the
same context. `mono_baseplanes` / `mono_ngroups` size the ResNet (32 / 16 in
production; small in tests) and are ignored under "vc1".
"""
import contextlib
import copy
import dataclasses
import json
import math
import os
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from torch_rl import tqc_loss
from torch_rl.nets import Actor, GRUCore, QuantileCritics, soft_update

PREV_ACTION_DIM = 2
_HALF_LOG_2PI_E = 0.5 * math.log(2.0 * math.pi * math.e)
_CHUNK_KEYS = ("obs", "keep", "actions", "rewards", "masks", "valid", "mode")
_CHUNK_KEYS_V2 = _CHUNK_KEYS + ("goal",)     # vc2: + the per-chunk (B, D*64) goal tokens


class Vc1Compression(nn.Module):
    """eai-vc `compression_layer.py` plus the policy's first Linear:
    (N, D, grid, grid) -> Conv2d(D, C, 3, p1, no bias), C = round(2048 / grid^2)
    -> GroupNorm(1, C) -> ReLU -> Flatten (C * grid^2 = 2048 at grid 8)
    -> Linear(2048, out_dim) -> ReLU.

    `embed_dim` is the input channel count: D for a current-frame grid, and
    vc2 passes 2*D for channel-stacked [cur | goal] grids."""

    def __init__(self, embed_dim: int, grid: int = 8, out_dim: int = 512,
                 flat_size: int = 2048) -> None:
        super().__init__()
        self.embed_dim, self.grid = int(embed_dim), int(grid)
        channels = int(round(flat_size / self.grid ** 2))
        self.conv = nn.Conv2d(self.embed_dim, channels, kernel_size=3, padding=1, bias=False)
        self.norm = nn.GroupNorm(1, channels)
        self.act = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(channels * self.grid * self.grid, int(out_dim))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.fc(self.flatten(self.act(self.norm(self.conv(tokens))))))


MONO_ENCODERS = ("vc1", "resnet", "debit")


def mono_encoder_problems(mono_encoder: str, embed_dim: int, grid: int) -> Optional[str]:
    """The refusal both learners' configs raise (None when the stack fits):
    an unknown encoder, or a resnet row that is not a (3, s, s) image with s
    a multiple of 32. The vc1 default is never refused here."""
    if mono_encoder not in MONO_ENCODERS:
        return f"mono_encoder {mono_encoder!r} not in {MONO_ENCODERS}"
    if mono_encoder == "resnet":
        if int(embed_dim) != 3:
            return f"mono_encoder=resnet needs embed_dim 3 (RGB pixels), got {embed_dim}"
        if int(grid) % 32:
            return (f"mono_encoder=resnet needs grid (the image size) a multiple of 32, "
                    f"got {grid}")
    if mono_encoder == "debit":
        # The pixel path's geometry (carla-rl 2026-10-13). _embed reshapes a
        # row's visual block to (embed_dim, grid, grid) and channel-stacks the
        # per-chunk goal onto it, which is a frame PAIR only at 3 channels and
        # DEBiT's own 112x112. Wrong here, it fails deep inside croco where the
        # message says nothing about the cause.
        if int(embed_dim) != 3:
            return (f"mono_encoder=debit needs embed_dim 3 (RGB pixels): the "
                    f"row's visual block IS a frame, got {embed_dim}")
        if int(grid) != 112:
            return (f"mono_encoder=debit needs grid 112, DEBiT's img_size, "
                    f"got {grid}")
    return None


def build_visual(cfg, in_ch: int) -> nn.Module:
    """The trained visual stack for `cfg.mono_encoder`, (N, in_ch, grid, grid)
    -> (N, visual_dim): `Vc1Compression(in_ch, grid, visual_dim)` over VC-1
    tokens (vc1), or `MonoResNetEncoder(grid, visual_dim, mono_baseplanes,
    mono_ngroups)` over /255 pixels (resnet; grid is the image size). Imported
    lazily: resnet_mono imports Vc1Compression from this module."""
    if cfg.mono_encoder == "resnet":
        from torch_rl.resnet_mono import MonoResNetEncoder
        return MonoResNetEncoder(cfg.grid, cfg.visual_dim, cfg.mono_baseplanes, cfg.mono_ngroups,
                                 in_channels=in_ch)
    if cfg.mono_encoder == "debit":
        # The DEBiT binocular encoder as the whole visual stack (carla-rl
        # 2026-10-13, "pixels in replay"). It needs the encoder INSTANCE, which
        # the trainer owns and shares with the env wrapper so the rollout and the
        # update use the same adapter parameters; the config cannot carry a live
        # module, so it is passed through cfg.debit_encoder.
        from torch_rl.debit_visual import DebitVisual
        enc = getattr(cfg, "debit_encoder", None)
        if enc is None:
            raise ValueError(
                "mono_encoder='debit' needs cfg.debit_encoder set to the "
                "DebitTinyEncoder instance (the same one the env wrapper uses, "
                "so the rollout and the update share one set of adapters)")
        return DebitVisual(enc, visual_dim=cfg.visual_dim)
    return Vc1Compression(in_ch, cfg.grid, cfg.visual_dim)


def visual_describe(module: nn.Module) -> str:
    """One line naming a visual stack: MonoResNetEncoder.describe(), or
    "Vc1Compression(conv D->C, fc F->out)"."""
    if hasattr(module, "describe"):
        return module.describe()
    return (f"{type(module).__name__}(conv {module.conv.in_channels}->{module.conv.out_channels}, "
            f"fc {module.fc.in_features}->{module.fc.out_features})")


def pretanh_penalty(mean_u, free: float):
    """ACTOR OUTPUT PENALTY (user 2026-10-26, v16): sum over action dims of
    relu(|u| - free)^2 on the actor's PRE-tanh mean. Zero inside the free band;
    quadratic beyond it, where tanh is flat and the Q gradient that should pull
    the action back is cut by its slope (0.06-0.08 at v15's parked actions)."""
    return (mean_u.abs() - float(free)).clamp_min(0.0).pow(2).sum(dim=-1)


@dataclasses.dataclass
class GatedRecurrentTQCConfig:
    embed_dim: int
    goal_cols: Tuple[int, ...]
    explore_cols: Tuple[int, ...]           # may be () -- Q_e then reads [h | action]
    grid: int = 8
    # 36 = the pre-D13 ring layout, kept as the default for this module's tests
    # and old snapshots; the VC-1 trainer always passes its layout's value (D13: 4).
    state_dim: int = 36
    mode_index: int = 3
    action_dim: int = 2
    hidden: int = 512
    visual_dim: int = 512
    action_emb: int = 32
    mlp: Sequence[int] = (256, 256)
    n_critics: int = 5
    n_quantiles: int = 25
    top_quantiles_to_drop_per_net: int = 2
    lr: float = 3e-4
    discount: float = 0.99
    discount_e: float = 0.99
    tau: float = 0.005
    kappa: float = 1.0
    # GOAL critic target only; the explore critic is unclipped (v13). A
    # divergence guard, not a return bound: tqc_target always backs up the
    # entropy term, worth about +-alpha*|log pi|/(1 - gamma) ~ 20 at alpha 0.1.
    value_floor: Optional[float] = -100.0
    value_ceiling: Optional[float] = 100.0
    init_temperature: float = 0.1
    init_temperature_e: float = 0.1
    # Unit ([-1, 1]^2) space: the torch Actor's log_prob carries no box
    # Jacobian. target_entropy_box() prints the Box-space equivalent.
    target_entropy: float = -2.0
    target_entropy_e: float = -2.0
    actor_q_norm: bool = True
    q_norm_tau: float = 0.01
    q_norm_floor: float = 1e-3
    explore_actor_warmup: int = 5000
    # ACTOR OUTPUT PENALTY (v16): weight on pretanh_penalty in the actor loss,
    # and the |pre-tanh mean| below which it is zero. 0 = off (every earlier run).
    actor_pretanh_l2: float = 0.0
    actor_pretanh_free: float = 1.5
    burn_in: int = 8
    seed: int = 42
    compile: bool = True
    device: str = "cpu"
    # L40 tensor cores roughly double the compression conv and the actor /
    # critic MLPs in bf16 (Task 10). The GRU core, tqc_target, the quantile
    # Huber loss, masked means and the temperature / Q-scale math stay fp32
    # regardless -- see GatedRecurrentTQC._autocast.
    bf16: bool = False
    # vc2 image goal (appended last so an older config.json still round-trips):
    # goal_tokens stacks the chunk's goal-image tokens onto each step's tokens
    # (a (2D, grid, grid) compression input); prev_action=False drops the
    # previous-action slot from the row and the act_embed module.
    goal_tokens: bool = False
    prev_action: bool = True
    # vc3 (appended last, so an older config.json reloads as ()): state columns
    # concatenated onto the GRU input after the visual vector (and the action
    # embedding); the debit layout passes the 3136 binoc columns.
    core_state_cols: Tuple[int, ...] = ()
    # vc3r (appended last, so an older config.json reloads as the vc1 stack):
    # the trained visual stack, see build_visual. "resnet" reads (3, grid, grid)
    # /255 pixels; baseplanes / ngroups size it (production 32 / 16).
    mono_encoder: str = "vc1"
    mono_baseplanes: int = 32
    mono_ngroups: int = 16
    # carla-rl 2026-10-12 (appended last, so an older config.json reloads as
    # None = the peer's unclipped explore critic, which did not diverge in
    # Habitat). On CARLA it DID: our explore critic ran 20 -> 11,801 against a
    # realisable ~110 while the clipped goal critic held its range, so v52
    # clipped both heads and the SAC arm sets these.
    value_floor_e: Optional[float] = None
    value_ceiling_e: Optional[float] = None
    # mono_encoder="debit" (carla-rl 2026-10-13): the live DebitTinyEncoder, set
    # by the trainer rather than serialised. Excluded from a config round trip,
    # which is why it carries a default and is never read by anything but
    # build_visual.
    debit_encoder: Optional[object] = None
    # ONE CRITIC FOR BOTH REWARD MODES (user 2026-10-13: "I want a single critic
    # with two different reward modes"). Appended last, defaulting False, so the
    # two-critic arm is untouched. The two modes then differ only in what the
    # ENV pays -- GatedReward already pays a different arrival bonus per mode --
    # and the critic tells them apart from the mode bit in its own view, which
    # one_critic REQUIRES (see __post_init__).
    one_critic: bool = False

    def __post_init__(self) -> None:
        self.goal_cols = tuple(int(c) for c in np.asarray(self.goal_cols).ravel())
        self.explore_cols = tuple(int(c) for c in np.asarray(self.explore_cols).ravel())
        self.core_state_cols = tuple(int(c) for c in
                                     np.asarray(tuple(self.core_state_cols), dtype=np.int64).ravel())
        self.mlp = tuple(int(h) for h in self.mlp)
        # embed_dim and grid are checked by mono_encoder_problems below, which
        # both learners' configs already run. This is the one condition it cannot
        # see, because it takes no goal flag.
        if self.mono_encoder == "debit" and not self.goal_tokens:
            raise ValueError(
                "mono_encoder='debit' needs goal_tokens=True: the goal FRAME "
                "rides the per-chunk goal path, and the encoder is binocular -- "
                "without it there is no goal image to compare against.")
        if self.prev_action and int(self.action_dim) != PREV_ACTION_DIM:
            raise ValueError(f"the replay row carries a {PREV_ACTION_DIM}-d previous action; "
                             f"action_dim={self.action_dim} does not fit it")
        # The goal critic is a distance function of (state, goal): it must read
        # the goal. With goal_tokens the goal is an image the GRU has already
        # folded into h, so no state column is needed. The explore critic may
        # read no state column at all (D13).
        if not self.goal_tokens and not self.goal_cols:
            raise ValueError("goal_cols is empty: the goal critic must read the goal")
        for name, cols in (("goal_cols", self.goal_cols), ("explore_cols", self.explore_cols)):
            bad = [c for c in cols if not 0 <= c < self.state_dim]
            if bad:
                raise ValueError(f"{name} {bad} outside the {self.state_dim}-d state")
            # THE MODE BIT: FORBIDDEN WITH TWO CRITICS, REQUIRED WITH ONE.
            #
            # With two critics each sees only its own mode's rows, so the bit is
            # a constant there -- a bias, not a feature -- and on Q_g it would
            # break the distance reading. With ONE critic trained on both modes
            # the same column becomes essential: the two rewards differ (the
            # arrival bonus is 50 in goal mode and 10 at 100 m in explore), and
            # without the bit the critic is asked to predict two different
            # returns from otherwise identical inputs. It would not fail -- it
            # would learn their average and the run would look merely mediocre.
            # User 2026-10-13, asked and confirmed: "the mode bit should be
            # observed by critic".
            if self.one_critic:
                if name == "goal_cols" and self.mode_index not in cols:
                    raise ValueError(
                        f"one_critic=True needs the mode bit (state index "
                        f"{self.mode_index}) in goal_cols, which is the single "
                        "critic's view: it is what tells the two reward modes "
                        f"apart. Got goal_cols={cols}.")
            elif self.mode_index in cols:
                raise ValueError(f"{name} contains the mode bit (state index {self.mode_index})")
        bad = [c for c in self.core_state_cols if not 0 <= c < self.state_dim]
        if bad:
            raise ValueError(f"core_state_cols {bad} outside the {self.state_dim}-d state")
        # As the critic views: never the mode bit (the actor reads it from the
        # state; it is no memory input), and each column once -- a repeat would
        # only widen the GRU input past the layout's.
        if self.mode_index in self.core_state_cols:
            raise ValueError(f"core_state_cols contains the mode bit (state index "
                             f"{self.mode_index})")
        seen, dup = set(), set()
        for c in self.core_state_cols:
            (dup if c in seen else seen).add(c)
        if dup:
            dup = sorted(dup)
            raise ValueError(f"core_state_cols repeats column(s) {dup}: each state column enters "
                             "the GRU input at most once")
        problem = mono_encoder_problems(self.mono_encoder, self.embed_dim, self.grid)
        if problem:
            raise ValueError(problem)
        # The ResNet reads one RGB frame; vc2's channel-stacked goal grid is VC-1 tokens.
        if self.mono_encoder == "resnet" and self.goal_tokens:
            raise ValueError("mono_encoder=resnet is debit-only: goal_tokens must be False")

    @property
    def token_dim(self) -> int:
        return int(self.embed_dim) * int(self.grid) * int(self.grid)

    @property
    def prev_action_dim(self) -> int:
        return PREV_ACTION_DIM if self.prev_action else 0

    @property
    def goal_dim(self) -> int:
        """Width of the per-chunk goal tensor: the flattened (D, grid, grid)
        goal grid with goal_tokens, else 0 (no goal tensor)."""
        return self.token_dim if self.goal_tokens else 0

    @property
    def vec_dim(self) -> int:
        return self.prev_action_dim + int(self.state_dim)

    @property
    def obs_dim(self) -> int:
        return self.token_dim + self.vec_dim


def _bound_frac(target: torch.Tensor, bound: float, upper: bool) -> torch.Tensor:
    """Per row, how much of the target sits at a value bound -> (batch,).

    Rank-agnostic on purpose: TQC's target is (batch, n_quantiles) and the
    fraction is a mean over that axis; the SAC arm's target is (batch,) and the
    fraction is the indicator itself. A hardcoded mean(dim=1) made this telemetry
    the only thing standing between the two algorithms.

    A target pinned at a bound gives the actor no value gradient, and without
    this number a saturated critic and a flat loss look identical.
    """
    hit = ((target >= bound) if upper else (target <= bound)).float()
    return hit.mean(dim=1) if hit.dim() > 1 else hit


def _masked_mean(x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    return (x * m).sum() / m.sum().clamp_min(1.0)


def _masked_std(x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """Population std of x over the rows where m is 1."""
    mean = _masked_mean(x, m)
    return _masked_mean((x - mean) ** 2, m).sqrt()


def ema_q_scale(scale: torch.Tensor, initialised: torch.Tensor, std: torch.Tensor,
                tau: float, floor: float) -> torch.Tensor:
    """v13 ACTOR_Q_NORM: the first value is the batch std itself, then an EMA
    at `tau`; floored either way. Branch-free on the `initialised` flag, so a
    compiled update never guards on it."""
    first = torch.clamp_min(std, floor)
    ema = torch.clamp_min((1.0 - tau) * scale + tau * std, floor)
    return torch.where(initialised > 0, ema, first)


def _frozen_copy(module: nn.Module) -> nn.Module:
    twin = copy.deepcopy(module)
    for p in twin.parameters():
        p.requires_grad_(False)
    return twin


class GatedRecurrentTQC:
    _MODULES = ("compress", "act_embed", "core", "actor", "critics_g", "critics_e",
                "compress_targ", "act_embed_targ", "core_targ", "critics_g_targ",
                "critics_e_targ")
    _OPTIMISERS = ("opt_critic", "opt_actor", "opt_temp_g", "opt_temp_e")
    _TENSORS = ("log_alpha_g", "log_alpha_e", "q_scale_g", "q_scale_e", "q_scale_init_g",
                "q_scale_init_e", "n_updates")

    def __init__(self, cfg: GatedRecurrentTQCConfig, _test_cpu_autocast: bool = False) -> None:
        self.cfg = cfg
        # Test-only hook (never a config field, so it can never reach a saved
        # checkpoint or a production launch): lets tests/test_gated_recurrent_tqc.py
        # exercise the bf16 autocast plumbing without CUDA. Production code
        # must never set this -- CPU has no tensor cores, so CPU autocast buys
        # no speed and is not the deployment target for cfg.bf16.
        self._test_cpu_autocast = bool(_test_cpu_autocast)
        # Reseeds the GLOBAL torch RNG, as RecurrentTQC does: construction
        # order below is load-bearing for reproducibility.
        torch.manual_seed(cfg.seed)
        dev = torch.device(cfg.device)
        self.device = dev
        self.top_drop = int(cfg.top_quantiles_to_drop_per_net) * int(cfg.n_critics)
        self.goal_cols = torch.as_tensor(cfg.goal_cols, dtype=torch.long, device=dev)
        self.explore_cols = torch.as_tensor(cfg.explore_cols, dtype=torch.long, device=dev)
        self.core_state_cols = torch.as_tensor(cfg.core_state_cols, dtype=torch.long, device=dev)

        # vc2: the compression sees [cur | goal] channel-stacked (2D channels);
        # with prev_action=False there is no act_embed and the GRU reads the
        # visual vector alone. Nothing is drawn for an absent module, so the
        # default (v1) construction consumes the RNG exactly as before.
        # vc3r: build_visual returns that same Vc1Compression for mono_encoder
        # "vc1" (same point, same draws) and the MonoResNetEncoder for "resnet"
        # (3 RGB channels: goal_tokens is refused there).
        in_ch = (2 if cfg.goal_tokens else 1) * cfg.embed_dim
        self.compress = build_visual(cfg, in_ch).to(dev)
        self.act_embed = (nn.Linear(PREV_ACTION_DIM, cfg.action_emb).to(dev)
                          if cfg.prev_action else None)
        # vc3: + the core_state_cols (0 by default, so the v1 / v2 GRU is unchanged).
        self.core = GRUCore(cfg.visual_dim + (cfg.action_emb if cfg.prev_action else 0)
                            + len(cfg.core_state_cols), cfg.hidden).to(dev)
        self.actor = Actor(cfg.hidden + cfg.state_dim, cfg.action_dim, cfg.mlp).to(dev)
        self.critics_g = QuantileCritics(cfg.hidden + len(cfg.goal_cols), cfg.action_dim,
                                         cfg.n_critics, cfg.n_quantiles, cfg.mlp).to(dev)
        self.critics_e = QuantileCritics(cfg.hidden + len(cfg.explore_cols), cfg.action_dim,
                                         cfg.n_critics, cfg.n_quantiles, cfg.mlp).to(dev)
        self.compress_targ = _frozen_copy(self.compress)
        self.act_embed_targ = _frozen_copy(self.act_embed) if cfg.prev_action else None
        self.core_targ = _frozen_copy(self.core)
        self.critics_g_targ = _frozen_copy(self.critics_g)
        self.critics_e_targ = _frozen_copy(self.critics_e)
        # The module names save / load / the soft update iterate: the class
        # tuple (the pose default) minus the action embedding when it is absent.
        self.modules_list = tuple(m for m in self._MODULES
                                  if cfg.prev_action or not m.startswith("act_embed"))

        self.log_alpha_g = nn.Parameter(torch.tensor(float(np.log(cfg.init_temperature)), device=dev))
        self.log_alpha_e = nn.Parameter(torch.tensor(float(np.log(cfg.init_temperature_e)), device=dev))
        # WITH ONE CRITIC THE SECOND ENSEMBLE IS NOT IN THE OPTIMISER. It is
        # still constructed, so the module list, the checkpoint format and the
        # construction RNG order are identical either way, but it receives no
        # gradient and no Adam state: left in, its momentum would move
        # parameters that nothing reads, and a checkpoint would carry an
        # ensemble that had drifted for no reason.
        _critic_mods = ((self.compress, self.act_embed, self.core, self.critics_g)
                        if cfg.one_critic else
                        (self.compress, self.act_embed, self.core,
                         self.critics_g, self.critics_e))
        self.opt_critic = torch.optim.Adam(
            [p for mod in _critic_mods if mod is not None
             for p in mod.parameters()], lr=cfg.lr)
        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr)
        self.opt_temp_g = torch.optim.Adam([self.log_alpha_g], lr=cfg.lr)
        self.opt_temp_e = torch.optim.Adam([self.log_alpha_e], lr=cfg.lr)

        # Everything that changes between updates is an on-device tensor,
        # mutated in place: a Python scalar read inside the compiled update is
        # a dynamo guard, and a guard on a value that changes every update
        # recompiles every update.
        self.q_scale_g = torch.ones((), device=dev)
        self.q_scale_e = torch.ones((), device=dev)
        self.q_scale_init_g = torch.zeros((), device=dev)
        self.q_scale_init_e = torch.zeros((), device=dev)
        self.n_updates = torch.zeros((), dtype=torch.int64, device=dev)

        self._update = self._update_impl
        if cfg.compile and hasattr(torch, "compile"):
            # Default mode, NOT mode="reduce-overhead": CUDA Graph Trees never
            # hit their fast path with several backwards per update and
            # degraded job 450014 from 175 to 661 ms/step (see RecurrentTQC).
            self._update = torch.compile(self._update_impl)

    # ── properties / small helpers ───────────────────────────────────────

    @property
    def alpha_g(self) -> float:
        return float(self.log_alpha_g.detach().exp())

    @property
    def alpha_e(self) -> float:
        return float(self.log_alpha_e.detach().exp())

    @staticmethod
    def target_entropy_box(te: float, low, high) -> float:
        """The Box-space equivalent of a unit-space target entropy:
        te + sum(log(0.5 * (high - low))). Printed only: the torch Actor's
        log_prob is Gaussian minus the tanh Jacobian, with no affine Box
        Jacobian, so `target_entropy` is already the unit-space target that
        CARLA v13's TARGET_ENTROPY_UNIT=1 corrected jaxrl2 to mean (v13 box:
        -2 -> -4.303)."""
        half = 0.5 * (np.asarray(high, np.float64) - np.asarray(low, np.float64))
        return float(te) + float(np.sum(np.log(half)))

    def reset_hidden(self, batch: int = 1) -> torch.Tensor:
        return torch.zeros(1, batch, self.cfg.hidden, device=self.device)

    def _autocast(self):
        """bf16 autocast for the compression conv and the actor/critic MLPs
        (Task 10). CUDA only in production: with cfg.bf16 on a non-CUDA
        device this is a no-op UNLESS the private _test_cpu_autocast hook is
        set, which only tests/test_gated_recurrent_tqc.py does, to check the
        plumbing on CPU. Every caller casts the result back to fp32 before it
        reaches the GRU core, tqc_target, the quantile Huber loss, a masked
        mean, or the temperature / Q-scale math -- none of those are ever
        computed inside this context."""
        if not self.cfg.bf16:
            return contextlib.nullcontext()
        if self.device.type == "cuda":
            return torch.autocast("cuda", dtype=torch.bfloat16)
        if self._test_cpu_autocast:
            return torch.autocast("cpu", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    def _actor_forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        with self._autocast():
            mean, log_std = self.actor(x)
        return mean.float(), log_std.float()

    def _actor_sample(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        with self._autocast():
            a, logp = self.actor.sample(x)
        return a.float(), logp.float()

    def _state(self, rows: torch.Tensor) -> torch.Tensor:
        return rows[..., self.cfg.token_dim + self.cfg.prev_action_dim:]

    def _embed(self, compress: nn.Module, act_embed: Optional[nn.Module], rows: torch.Tensor,
               goal: Optional[torch.Tensor] = None) -> torch.Tensor:
        """(..., obs_dim) rows -> (..., visual_dim + action_emb) core inputs;
        with prev_action=False `act_embed` is None and the result is
        (..., visual_dim). By default the state is NOT an input: it reaches the
        heads only, which is what makes the per-critic views plain column
        slices. vc3: the state columns `core_state_cols` (fp32) are appended
        last, `[visual (| act_embed(prev_a)) | state[core_state_cols]]`, for
        every row this embeds -- burn-in and window, online and target alike;
        the critic views stay column slices of the heads' state input.
        The compression forward (online or target -- both call this)
        autocasts under bf16; its output is cast back to fp32 before the
        action embedding joins it, so the GRU core (fed by this method's
        return value) never sees bf16. Under mono_encoder="resnet" the same
        reshape yields the (N, 3, s, s) /255 frame and `compress` is the
        MonoResNetEncoder, inside the same autocast.

        vc2 (goal_tokens): `goal` (B, goal_dim) is one goal per leading batch
        index of `rows` (B, ...) and is broadcast over every trailing (time)
        axis -- burn-in and window rows alike -- then channel-stacked after the
        current tokens, `cat([cur, goal], dim=1)`, before the ONE compression."""
        cfg = self.cfg
        lead = rows.shape[:-1]
        flat = rows.reshape(-1, rows.shape[-1])
        tokens = flat[:, :cfg.token_dim].reshape(-1, cfg.embed_dim, cfg.grid, cfg.grid)
        if cfg.goal_tokens:
            if goal is None:
                raise ValueError("goal_tokens=True: _embed needs the per-chunk goal (B, goal_dim)")
            B = lead[0]
            L = flat.shape[0] // B                   # rows per batch entry: the time axis
            g = goal.to(device=tokens.device, dtype=tokens.dtype)
            g = g.reshape(B, 1, cfg.embed_dim, cfg.grid, cfg.grid)
            g = g.expand(B, L, cfg.embed_dim, cfg.grid, cfg.grid)
            tokens = torch.cat([tokens, g.reshape(-1, cfg.embed_dim, cfg.grid, cfg.grid)], dim=1)
        with self._autocast():
            visual = compress(tokens).float()
        if cfg.prev_action:
            prev_a = flat[:, cfg.token_dim:cfg.token_dim + PREV_ACTION_DIM]
            e = torch.cat([visual, act_embed(prev_a)], dim=-1)
        else:
            e = visual
        # A static shape, fixed at construction (as _q's empty-view branch): dynamo
        # guards the tensor's size, never the column tuple itself.
        if self.core_state_cols.numel() > 0:
            core_in = self._state(flat).index_select(-1, self.core_state_cols).float()
            e = torch.cat([e, core_in], dim=-1)
        return e.reshape(*lead, e.shape[-1])

    def _burn_in(self, compress: nn.Module, act_embed: Optional[nn.Module], core: GRUCore,
                 burn: torch.Tensor, keep: torch.Tensor,
                 goal: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Burn-in rows (B, K, obs_dim) and keep (B, K+1) -> the hidden state
        (1, B, hidden) entering the trained window, without gradients.

        `keep[:, j]` refers to episode row t0 - K + j and is 0 when that row is
        at or before episode row 0 (ruling R6). Columns 0..K-1 zero h BEFORE
        burn-in step j, so a chunk whose episode starts inside the burn-in
        enters row 0 with act()'s reset state. Column K (row t0) zeroes h
        before the window: it covers a window starting AT row 0 (t0 = 0),
        where every burn-in row is padding and h would otherwise leave the
        last padding step as GRU(embed(padding), 0) != 0."""
        B, K = burn.shape[0], burn.shape[1]
        if keep.shape[1] != K + 1:
            raise ValueError(f"keep must be (B, K+1) = (B, {K + 1}), got {tuple(keep.shape)}")
        h = burn.new_zeros(1, B, self.cfg.hidden)
        if K == 0:
            return h
        with torch.no_grad():
            e = self._embed(compress, act_embed, burn, goal)
            for j in range(K):
                h = h * keep[:, j][None, :, None]
                _, h = core.step(e[:, j], h)
            h = h * keep[:, K][None, :, None]
        return h

    def _roll(self, compress: nn.Module, act_embed: Optional[nn.Module], core: GRUCore,
              obs: torch.Tensor, keep: torch.Tensor,
              goal: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Chunk obs (B, K+T+1, obs_dim) -> core states of the trained window
        (B, T+1, hidden): burn-in step by step (keep needs a reset between
        steps), then the window through cuDNN in one call."""
        K = int(self.cfg.burn_in)
        h = self._burn_in(compress, act_embed, core, obs[:, :K], keep, goal)
        s_all, _ = core(self._embed(compress, act_embed, obs[:, K:], goal), h)
        return s_all

    # ------------------------------------------------------------------ seams --
    # THE ONLY TWO PLACES THE CRITIC ALGORITHM LIVES, extracted as methods so a
    # subclass can be a different algorithm without a second copy of this file.
    # torch_rl/gated_recurrent_sac.py overrides exactly these two to get the
    # user's expectation-SAC with a REDQ target (2026-10-12: "move away from
    # TQC", "USE 10 critics for SAC"). Pure delegation here -- the TQC arm must
    # behave as it did before the extraction, and a test pins that.

    def _critic_target(self, next_q, rewards, masks, discount, alpha,
                       next_log_probs, value_floor, value_ceiling, head):
        """Backed-up target from the target critics' output. `head` is "goal" or
        "explore", for subclasses that treat the two ensembles separately."""
        return tqc_loss.tqc_target(next_q, rewards, masks, discount=discount,
                                   top_drop=self.top_drop, alpha=alpha,
                                   next_log_probs=next_log_probs,
                                   value_floor=value_floor,
                                   value_ceiling=value_ceiling)

    def _critic_loss_per_sample(self, qs, target, head):
        """Per-row critic loss, reduced over critics but NOT over the batch: the
        caller weights each row by its mode mask."""
        return tqc_loss.quantile_huber_loss_per_sample(qs, target,
                                                       kappa=self.cfg.kappa)

    def _q(self, critics: QuantileCritics, h: torch.Tensor, state: torch.Tensor,
           cols: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Online or target critic forward (both call this): autocasts under
        bf16, and always returns fp32 -- every caller (the quantile Huber
        loss, tqc_target, the actor's Q term, the Q-scale EMA) is fp32 maths.
        An empty view (D13's Q_e) feeds h alone; the branch is on a static
        shape, fixed at construction, so it is no guard in the compiled update."""
        x = h if cols.numel() == 0 else torch.cat([h, state.index_select(-1, cols)], dim=-1)
        with self._autocast():
            out = critics(x, actions)
        return out.float()

    def goal_q(self, h: torch.Tensor, state: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Q_g quantiles (n_critics, N, n_quantiles) from core states and the
        FULL state; the goal view is applied here, so a caller cannot feed the
        goal critic the wrong columns."""
        return self._q(self.critics_g, h, state, self.goal_cols, actions)

    def explore_q(self, h: torch.Tensor, state: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Q_e quantiles, with the exploration view applied."""
        return self._q(self.critics_e, h, state, self.explore_cols, actions)

    def _goal_input(self, goal, n: int) -> Optional[torch.Tensor]:
        """The (n, goal_dim) float32 goal on the learner device for act /
        act_batch, or None when the config has no goal tensor. `goal` may be
        fp16 or fp32, a numpy array or a tensor on any device (Vc1VecEnv holds
        its goal grids as fp16 CUDA tensors). Required with goal_tokens; the
        reshape refuses any goal that is not exactly (n, goal_dim), so a single
        goal is never silently broadcast over n envs."""
        if not self.cfg.goal_tokens:
            return None
        if goal is None:
            raise ValueError(f"goal_tokens=True: act needs the goal (n, {self.cfg.goal_dim}) "
                             f"tokens of each env's goal image")
        return torch.as_tensor(goal, dtype=torch.float32, device=self.device).reshape(
            n, self.cfg.goal_dim)

    @torch.no_grad()
    def act(self, row: np.ndarray, hidden, deterministic: bool = False, goal=None,
            noise_scale: float = 1.0):
        """One (obs_dim,) row in, one action out; always batch 1. `goal`
        ((goal_dim,) or (1, goal_dim), required with goal_tokens): the goal
        image's flattened tokens."""
        x = torch.as_tensor(np.asarray(row, np.float32)[None], device=self.device)
        g = self._goal_input(goal, 1)
        s, hidden = self.core.step(self._embed(self.compress, self.act_embed, x, g), hidden)
        inp = torch.cat([s, self._state(x)], dim=-1)
        a = (self.actor.deterministic(inp) if deterministic
             else self.actor.sample(inp, noise_scale=noise_scale)[0])
        return a[0].cpu().numpy(), hidden

    @torch.no_grad()
    def act_batch(self, rows, hidden: torch.Tensor, deterministic: bool = False, goal=None,
                  noise_scale=None):
        """N (obs_dim,) rows in -> N actions out, one GRU step shared across the
        batch (12 parallel envs' worth in the 50M-step run). `hidden` is
        (1, N, hidden), one column per env, carried by the caller across calls
        exactly like N independent `act()` hiddens stacked on dim 1.

        `rows` may be a numpy array OR a torch tensor on any device (Vc1VecEnv
        hands back float16 CUDA tensors on the learner device): `torch.as_tensor`
        with an explicit dtype/device does the numpy-host-copy-and-cast case and
        the already-a-tensor case in one call, and leaves a tensor that already
        has the right dtype and device untouched -- no `np.asarray` on the input,
        which raises on a CUDA tensor (numpy cannot read GPU memory; a CPU
        tensor silently round-trips through it instead, which is why this only
        shows up once rows live on the GPU).

        Equal to N separate `act()` calls only with `deterministic=True`
        (both reduce to tanh(mean), no RNG draw). With sampling, this method
        draws ONE (N, action_dim) batch from the actor's Normal, one call to
        `torch.randn_like`, consuming the global torch RNG in one shot; a loop
        of N `act()` calls draws N separate (1, action_dim) samples instead.
        Those are different draws from the same stream (and, depending on the
        generator, not even reorderings of each other -- e.g. Box-Muller pairs
        drawn together vs. singly), so stochastic act_batch output is not
        expected to match a stochastic act() loop row for row, only the
        deterministic path is tested for exact equality.

        vc2 (goal_tokens): `goal` is REQUIRED, (N, goal_dim) -- row i's own
        goal image's flattened tokens, fp16 or fp32, numpy or a tensor on any
        device -- and is stacked with each row's current tokens, exactly as in
        `update`. A missing goal raises ValueError rather than acting blind.
        """
        x = torch.as_tensor(rows, dtype=torch.float32, device=self.device)
        g = self._goal_input(goal, x.shape[0])
        s, hidden = self.core.step(self._embed(self.compress, self.act_embed, x, g), hidden)
        inp = torch.cat([s, self._state(x)], dim=-1)
        if deterministic:
            a = self.actor.deterministic(inp)
        elif noise_scale is None:
            a = self.actor.sample(inp)[0]
        else:
            # Per-env scale (N,): 0 = that env's episode acts noise-free.
            ns = torch.as_tensor(np.asarray(noise_scale, np.float32), device=self.device)
            a = self.actor.sample(inp, noise_scale=ns)[0]
        return a.cpu().numpy(), hidden

    @staticmethod
    def zero_hidden_columns(hidden: torch.Tensor, env_ids) -> torch.Tensor:
        """Zero hidden[:, env_ids, :] IN PLACE and return `hidden` for chaining.

        For the envs in `env_ids` (an int, or any sequence/tensor of ints) --
        the ones whose episode just ended in a batched rollout -- the next
        `act_batch` call must see the same zero hidden state `reset_hidden`
        gives a fresh episode, while every other column (env) of the shared
        (1, N, hidden) tensor is untouched.
        """
        hidden[:, env_ids, :] = 0.0
        return hidden

    # ── update ───────────────────────────────────────────────────────────

    def update(self, chunk: Dict[str, torch.Tensor]) -> Dict[str, float]:
        keys = _CHUNK_KEYS_V2 if self.cfg.goal_tokens else _CHUNK_KEYS
        args = [chunk[k].to(device=self.device, dtype=torch.float32) for k in keys]
        if not self.cfg.goal_tokens:
            # A (B, 0) placeholder keeps the compiled update at 8 positional
            # arguments; with no goal_tokens nothing reads it (no ops).
            args.append(torch.zeros((chunk["obs"].shape[0], 0), device=self.device))
        out = self._update(*args)
        # ONE host sync for every metric (Phase 2 review P5): a float() per
        # metric was ~20 device syncs per update. Every metric is an fp32 0-d
        # tensor, so each value equals its float() exactly.
        vals = torch.stack([v.detach().reshape(()).float() for v in out.values()]).tolist()
        return dict(zip(out.keys(), vals))

    def _skip_metrics(self) -> Dict[str, torch.Tensor]:
        z = torch.zeros((), device=self.device)
        return {"critic_loss_g": z, "critic_loss_e": z, "actor_loss": z,
                "temp_loss_g": z, "temp_loss_e": z,
                "alpha_g": self.log_alpha_g.detach().exp(),
                "alpha_e": self.log_alpha_e.detach().exp(),
                "q_g": z, "q_e": z,
                "q_scale_g": self.q_scale_g.clone(), "q_scale_e": self.q_scale_e.clone(),
                "entropy_g": z, "entropy_e": z, "has_g": z, "has_e": z, "goal_frac": z,
                "valid_frac": z, "target_at_floor": z, "target_at_ceiling": z,
                "skipped": torch.ones((), device=self.device)}

    def _actor_q_scale(self, q: torch.Tensor, m: torch.Tensor, scale: torch.Tensor,
                       initialised: torch.Tensor) -> torch.Tensor:
        """Called only when the mode is present: advance that mode's scale from
        the detached Q on its rows, and return the UPDATED scale for this step."""
        cfg = self.cfg
        if not cfg.actor_q_norm:
            return torch.ones((), device=q.device)
        with torch.no_grad():
            std = _masked_std(q.detach(), m)
            scale.copy_(ema_q_scale(scale, initialised, std, cfg.q_norm_tau, cfg.q_norm_floor))
            initialised.fill_(1.0)
        return scale.clone()

    def _update_impl(self, obs, keep, actions, rewards, masks, valid, mode, goal=None):
        # `goal` (B, goal_dim) is the per-chunk goal-image tokens under
        # goal_tokens, else the (B, 0) placeholder update() appends (or None
        # for a direct call): both are ignored by _embed without goal_tokens.
        cfg, K = self.cfg, int(self.cfg.burn_in)

        # Row masks from the BUFFER's mode, never the obs: a padded row is an
        # all-zero obs whose mode bit would read "explore".
        B, T = valid.shape
        N = B * T
        if cfg.one_critic:
            # ONE CRITIC, BOTH MODES. Every valid row is the single critic's
            # row, whichever mode produced it, and the mode reaches it as a
            # FEATURE (its view includes the mode column) rather than as a mask.
            # Expressing it this way means the rest of this method is unchanged:
            # the "goal" critic is simply the only one, m_e stays zero so the
            # explore branch never runs, and the actor reads one Q, one alpha
            # and one Q-scale.
            m_g = valid.reshape(N)
            m_e = torch.zeros_like(m_g)
        else:
            m_g = (mode[:, None] * valid).reshape(N)
            m_e = ((1.0 - mode)[:, None] * valid).reshape(N)
        # One host sync, of BOOLS, never counts: a Python value live across a
        # graph break is guarded on its exact value by dynamo. MEASURED (round-0
        # review, torch 2.4.1): synced float counts recompiled the resume frame
        # on every new count (`L['___stack0'][0] == 12.0`) and ran eager after
        # config.cache_size_limit (8), and padded windows make the counts
        # differ almost every batch. Two bools specialise at most three ways.
        # Counts stay tensors (every masked mean divides by m.sum()).
        has_g, has_e = (torch.stack([m_g.sum(), m_e.sum()]) > 0).tolist()

        # Nothing valid to train on (mode is 0/1, so m_g + m_e == valid): skip
        # without touching any optimiser, so Adam's stale momentum cannot move
        # parameters on zero signal.
        if not (has_g or has_e):
            return self._skip_metrics()

        # Python branches below, deliberately: an absent mode's critic term
        # must not be computed at all, so its parameters keep grad None and
        # Adam skips them. A zero-weighted term would give a zero grad, and
        # Adam moves a parameter with a zero grad by its stale momentum.
        alpha_g = self.log_alpha_g.detach().exp()
        alpha_e = self.log_alpha_e.detach().exp()
        zero = torch.zeros((), device=obs.device)

        s_all = self._roll(self.compress, self.act_embed, self.core, obs, keep, goal)
        with torch.no_grad():
            s_all_t = self._roll(self.compress_targ, self.act_embed_targ, self.core_targ, obs,
                                 keep, goal)
        H = cfg.hidden
        s = s_all[:, :-1].reshape(N, H)
        s_next_t = s_all_t[:, 1:].reshape(N, H)
        st = self._state(obs[:, K:-1]).reshape(N, cfg.state_dim)
        st_next = self._state(obs[:, K + 1:]).reshape(N, cfg.state_dim)
        a = actions.reshape(N, cfg.action_dim)
        r, m = rewards.reshape(N), masks.reshape(N)

        with torch.no_grad():
            next_a, next_logp = self._actor_sample(torch.cat([s_next_t, st_next], dim=-1))

        # Critics: each regressed on its own rows, with its own discount and
        # temperature; the goal target alone is clipped.
        critic_loss = zero
        loss_g = loss_e = q_g_data = q_e_data = zero
        at_floor = at_ceiling = zero
        if has_g:
            with torch.no_grad():
                next_q = self._q(self.critics_g_targ, s_next_t, st_next, self.goal_cols, next_a)
                target_g = self._critic_target(
                    next_q, r, m, discount=cfg.discount, alpha=alpha_g,
                    next_log_probs=next_logp, value_floor=cfg.value_floor,
                    value_ceiling=cfg.value_ceiling, head="goal")
                # A target pinned at a bound gives the actor no value gradient;
                # without these a saturated critic and a flat loss look alike.
                if cfg.value_floor is not None:
                    at_floor = _masked_mean(
                        _bound_frac(target_g, cfg.value_floor, upper=False), m_g)
                if cfg.value_ceiling is not None:
                    at_ceiling = _masked_mean(
                        _bound_frac(target_g, cfg.value_ceiling, upper=True), m_g)
            qs_g = self._q(self.critics_g, s, st, self.goal_cols, a)
            loss_g = _masked_mean(
                self._critic_loss_per_sample(qs_g, target_g, head="goal"), m_g)
            q_g_data = _masked_mean(qs_g.detach().mean(dim=(0, 2)), m_g)
            critic_loss = critic_loss + loss_g
        if has_e:
            with torch.no_grad():
                next_q = self._q(self.critics_e_targ, s_next_t, st_next, self.explore_cols, next_a)
                target_e = self._critic_target(
                    next_q, r, m, discount=cfg.discount_e, alpha=alpha_e,
                    next_log_probs=next_logp, value_floor=cfg.value_floor_e,
                    value_ceiling=cfg.value_ceiling_e, head="explore")
            qs_e = self._q(self.critics_e, s, st, self.explore_cols, a)
            loss_e = _masked_mean(
                self._critic_loss_per_sample(qs_e, target_e, head="explore"), m_e)
            q_e_data = _masked_mean(qs_e.detach().mean(dim=(0, 2)), m_e)
            critic_loss = critic_loss + loss_e
        self.opt_critic.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.opt_critic.step()

        # Actor: no gradient into the memory (s is detached). Each row reads
        # its own mode's critic, temperature and Q-scale.
        s_d = s.detach()
        actor_in = torch.cat([s_d, st], dim=-1)
        with torch.no_grad():
            # Entropy Rule: the diagnostic is the analytic entropy of the base
            # Gaussian, never the sampled squashed -log pi. _actor_forward
            # already returns fp32, so this is exact whatever cfg.bf16 is.
            mean_u, log_std = self._actor_forward(actor_in)
            entropy = log_std.sum(dim=-1) + cfg.action_dim * _HALF_LOG_2PI_E
        new_a, logp = self._actor_sample(actor_in)
        # SATURATION DIAGNOSTICS (2026-10-19, after v7/v8/v10 all locked one
        # action at its limit while alpha_g fell to ~5e-4). Logged only; they
        # change nothing. The temperature tunes against the SAMPLED, SQUASHED
        # -log pi (neg_logp), not the base-Gaussian entropy above, so both are
        # needed to tell "random enough" from "stuck at +-1 with a wide
        # Gaussian behind it".
        with torch.no_grad():
            _a = new_a.detach()
            diag = {
                "neg_logp_g": _masked_mean(-logp.detach(), m_g),
                "sat_steer": _masked_mean((_a[..., 0].abs() > 0.99).float(), m_g),
                "sat_throttle": _masked_mean((_a[..., 1].abs() > 0.99).float(), m_g),
                "mean_abs_u_steer": _masked_mean(mean_u[..., 0].abs(), m_g),
                "mean_abs_u_throttle": _masked_mean(mean_u[..., 1].abs(), m_g),
                "log_std_g": _masked_mean(log_std.mean(dim=-1), m_g),
            }
        per_row = zero
        if has_g:
            q_g = self._q(self.critics_g, s_d, st, self.goal_cols, new_a).mean(dim=(0, 2))
            scale_g = self._actor_q_scale(q_g, m_g, self.q_scale_g, self.q_scale_init_g)
            per_row = per_row + (alpha_g * logp - q_g / scale_g) * m_g
        if has_e:
            q_e = self._q(self.critics_e, s_d, st, self.explore_cols, new_a).mean(dim=(0, 2))
            scale_e = self._actor_q_scale(q_e, m_e, self.q_scale_e, self.q_scale_init_e)
            # v13 warm-up: the exploration critic trains alone for the first
            # explore_actor_warmup updates. A tensor, so no per-update guard.
            w_e = (self.n_updates >= cfg.explore_actor_warmup).to(logp.dtype)
            per_row = per_row + w_e * (alpha_e * logp - q_e / scale_e) * m_e
        if cfg.actor_pretanh_l2 > 0.0:
            # A second actor forward WITH grad: the one above is no_grad.
            mean_live, _ = self._actor_forward(actor_in)
            pen = pretanh_penalty(mean_live, cfg.actor_pretanh_free)
            per_row = per_row + cfg.actor_pretanh_l2 * pen * (m_g + m_e)
            diag["pretanh_pen"] = _masked_mean(pen.detach(), m_g + m_e)
        else:
            diag["pretanh_pen"] = zero
        actor_loss = per_row.sum() / (m_g + m_e).sum().clamp_min(1.0)
        self.opt_actor.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.opt_actor.step()
        # actor_loss flowed through the critics: clear those grads so critic
        # grads are only ever from the critic loss (as in RecurrentTQC).
        self.opt_critic.zero_grad(set_to_none=True)

        # Temperatures: a mode absent from the batch takes no step at all.
        lp = logp.detach()
        temp_loss_g = temp_loss_e = zero
        if has_g:
            temp_loss_g = _masked_mean(self.log_alpha_g.exp() * (-lp - cfg.target_entropy), m_g)
            self.opt_temp_g.zero_grad(set_to_none=True)
            temp_loss_g.backward()
            self.opt_temp_g.step()
        if has_e:
            temp_loss_e = _masked_mean(self.log_alpha_e.exp() * (-lp - cfg.target_entropy_e), m_e)
            self.opt_temp_e.zero_grad(set_to_none=True)
            temp_loss_e.backward()
            self.opt_temp_e.step()

        with torch.no_grad():
            self.n_updates.add_(1)
        _pairs = [(self.compress_targ, self.compress),
                  (self.act_embed_targ, self.act_embed),
                  (self.core_targ, self.core),
                  (self.critics_g_targ, self.critics_g)]
        if not cfg.one_critic:
            # Nothing trains the second ensemble under one_critic, so following
            # it with a soft update would only copy its initialisation onto its
            # own target forever.
            _pairs.append((self.critics_e_targ, self.critics_e))
        for targ, src in _pairs:
            if src is not None:                        # no act_embed without prev_action
                soft_update(targ, src, cfg.tau)

        n_rows = (m_g + m_e).sum().clamp_min(1.0)
        return {"critic_loss_g": loss_g.detach(), "critic_loss_e": loss_e.detach(),
                "actor_loss": actor_loss.detach(),
                "temp_loss_g": temp_loss_g.detach(), "temp_loss_e": temp_loss_e.detach(),
                "alpha_g": self.log_alpha_g.detach().exp(),
                "alpha_e": self.log_alpha_e.detach().exp(),
                "q_g": q_g_data, "q_e": q_e_data,
                "q_scale_g": self.q_scale_g.clone(), "q_scale_e": self.q_scale_e.clone(),
                "entropy_g": _masked_mean(entropy, m_g), "entropy_e": _masked_mean(entropy, m_e),
                "has_g": (m_g.sum() > 0).float(), "has_e": (m_e.sum() > 0).float(),
                **diag,
                "goal_frac": m_g.sum() / n_rows, "valid_frac": valid.mean(),
                "target_at_floor": at_floor, "target_at_ceiling": at_ceiling,
                "skipped": zero}

    # ── checkpoint ───────────────────────────────────────────────────────

    def save(self, out_dir: str) -> str:
        """Every module and target, both temperatures, all four optimisers,
        the Q-scales with their init flags and the update counter -- a resume
        must not restart the warm-up or the Q-normalisation."""
        os.makedirs(out_dir, exist_ok=True)
        blob = {n: getattr(self, n).state_dict() for n in self.modules_list + self._OPTIMISERS}
        blob.update({n: getattr(self, n).detach().cpu() for n in self._TENSORS})
        path = os.path.join(out_dir, "params.pt")
        torch.save(blob, path)
        with open(os.path.join(out_dir, "config.json"), "w") as fh:
            json.dump(dataclasses.asdict(self.cfg), fh, indent=1, default=str)
        return path

    def load(self, in_dir: str) -> None:
        blob = torch.load(os.path.join(in_dir, "params.pt"), map_location=self.device,
                          weights_only=True)
        # Strict on names and shapes: a pose snapshot (D-channel compression,
        # an act_embed) refuses to load into an image-goal agent and vice versa.
        for n in self.modules_list + self._OPTIMISERS:
            getattr(self, n).load_state_dict(blob[n])
        # In place: the compiled update holds these very tensors.
        with torch.no_grad():
            for n in self._TENSORS:
                getattr(self, n).copy_(blob[n].to(self.device))
