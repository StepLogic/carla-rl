#!/usr/bin/env python3
"""Pack a step's frames into the replay row the learner's visual stack expects.

User 2026-10-13: pixels in the replay. The row layout is the learner's existing
one -- [visual block | previous action | state] -- with the visual block holding
the CURRENT frame as raw 0-255 pixels at 112x112x3, and the goal frame travelling
once per chunk through the buffer's `goal` field rather than once per row.

WHY THE GOAL IS PER CHUNK AND NOT PER ROW. It is constant for the whole episode
(DebitGoalWrapper latches it at reset), so storing it per row would multiply the
replay's size by two for no information. DeviceEpisodeBuffer already carries a
per-episode goal for exactly this reason.

SIZE, which is the reason this is affordable: 112*112*3 = 37,632 values per row.
Held as fp16 -- the buffer's dtype, and lossless here since every integer up to
2048 is exact in fp16 -- that is 75 KB per row, so 184k rows come to 13.8 GB on a
46 GB L40 beside CARLA's 1.7 GB. For comparison the peer's Habitat rows are
52,289 values of which 49,152 are VC-1 tokens this arm does not have.

THE RESIZE HAPPENS HERE, ONCE, at insert. The env's camera is 120x120 and the
goal camera is already 112x112; DEBiT wants 112. Resizing on the way into the
replay keeps the row width fixed and the per-update cost zero, and it means the
stored frame is exactly what the encoder will see.
"""
from typing import Tuple

import numpy as np

IMG = 112
PIX = IMG * IMG * 3
PREV_ACTION_DIM = 2


def to_debit_frame(frame: np.ndarray) -> np.ndarray:
    """(H, W, 3) uint8 -> (IMG, IMG, 3) float32 in 0-255, channels unchanged.

    Bilinear via PIL when the size differs, which matches torchvision's Resize on
    the rollout path closely enough that the stored frame and a freshly encoded
    one agree; an exact match is not required because BOTH the rollout and the
    update read the stored pixels in training.
    """
    a = np.asarray(frame)
    if a.ndim != 3 or a.shape[2] != 3:
        raise ValueError(f"expected (H, W, 3), got {a.shape}")
    if a.dtype != np.uint8:
        raise ValueError(
            f"expected uint8 pixels, got {a.dtype}: the row holds RAW 0-255 "
            "values and the normalisation happens in the learner, so a "
            "pre-normalised frame would be stored as near zero.")
    if a.shape[0] != IMG or a.shape[1] != IMG:
        from PIL import Image
        a = np.asarray(Image.fromarray(a).resize((IMG, IMG), Image.BILINEAR))
    return a.astype(np.float32)


def pixel_row(cur_frame: np.ndarray, prev_action, state) -> np.ndarray:
    """One replay row: [current frame (PIX) | previous action (2) | state]."""
    pix = to_debit_frame(cur_frame).reshape(-1)
    pa = np.asarray(prev_action, np.float32).reshape(-1)
    if pa.size != PREV_ACTION_DIM:
        raise ValueError(f"previous action must be {PREV_ACTION_DIM}-d, got {pa.size}")
    st = np.asarray(state, np.float32).reshape(-1)
    return np.concatenate([pix, pa, st]).astype(np.float32)


def goal_block(goal_frame: np.ndarray) -> np.ndarray:
    """The per-chunk goal: the latched goal frame, same layout as the row's
    visual block so _embed can channel-stack the two without reshaping."""
    return to_debit_frame(goal_frame).reshape(-1)


def row_dim(state_dim: int) -> int:
    return PIX + PREV_ACTION_DIM + int(state_dim)


def split_row(row: np.ndarray, state_dim: int) -> Tuple[np.ndarray, np.ndarray,
                                                        np.ndarray]:
    """Inverse of pixel_row, for tests and for anything inspecting a replay."""
    row = np.asarray(row)
    pix = row[..., :PIX]
    pa = row[..., PIX:PIX + PREV_ACTION_DIM]
    st = row[..., PIX + PREV_ACTION_DIM:PIX + PREV_ACTION_DIM + int(state_dim)]
    return pix, pa, st


def frame_from_block(block: np.ndarray) -> np.ndarray:
    """A stored visual block back to (IMG, IMG, 3), for inspection."""
    return np.asarray(block).reshape(IMG, IMG, 3)


# ---------------------------------------------------------------------------
# THE BINOCULAR + MONOCULAR LAYOUT (user 2026-10-14): "use the Debit binocular
# monocular setup - keep debit frozen and monocular as Resnet trained from
# scratch", and "get rid of all state vectors except mode bit".
#
#   row = [ mono frame 3x128x128 | previous action 2 | mode 1 | binoc 3136 ]
#
# THE MONO FRAME IS STORED /255 (see mono_row): the ported ResNet's contract.
#
# WHY THE FRAME AND THE BINOC BOTH TRAVEL, AND WHY ONLY THE FRAME IS PIXELS.
# The monocular ResNet is trained from scratch, so its input must be pixels the
# learner can backpropagate through. The DEBiT binocular encoder is FROZEN, so
# its 3136-d output is a fixed function of (view, goal) and can be computed once
# in the env and stored -- re-encoding it per update would cost 209 ms and buy
# nothing. That is the whole reason this arm is cheap again: the expensive
# encoder is frozen and precomputed, the trainable one is small.
#
# 128, not 112: MonoResNetEncoder requires a multiple of 32 and DEBiT's 112 is
# not one. The binocular encoder keeps its own 112 px input inside the wrapper.
#
# THE STATE BLOCK IS THE MODE BIT ALONE, then the binoc columns. No steer, no
# throttle, no speed: the actor used to receive its own last steer and throttle
# as observation, which is a feedback path a policy can latch onto -- and the
# run that collapsed sat at full left lock for 800k steps.
MONO_IMG = 128
MONO_PIX = MONO_IMG * MONO_IMG * 3
BINOC_DIM = 3136


def to_frame(frame: np.ndarray, size: int) -> np.ndarray:
    """(H, W, 3) uint8 -> (size, size, 3) float32 in 0-255."""
    a = np.asarray(frame)
    if a.ndim != 3 or a.shape[2] != 3:
        raise ValueError(f"expected (H, W, 3), got {a.shape}")
    if a.dtype != np.uint8:
        raise ValueError(
            f"expected uint8 pixels, got {a.dtype}: rows hold RAW 0-255 values "
            "and the normalisation happens in the learner.")
    if a.shape[0] != size or a.shape[1] != size:
        from PIL import Image
        a = np.asarray(Image.fromarray(a).resize((size, size), Image.BILINEAR))
    return a.astype(np.float32)


def mono_row(frame: np.ndarray, prev_action, mode, binoc) -> np.ndarray:
    """One replay row for the binocular+monocular arm.

    THE FRAME IS STORED /255, in [0, 1] -- the opposite of pixel_row above, and
    not a free choice: MonoResNetEncoder applies the ImageNet mean/std to what
    it is handed and REFUSES anything outside [0, 1] ("pixels must be /255 in
    [0, 1] (the row stores them scaled)"), which is how the first pilot of this
    stack failed at step 283. The DEBiT path stores raw 0-255 because its own
    preprocessing does the division; this path's does not. fp16 holds [0, 1]
    with far more precision than 8-bit pixels carry, so nothing is lost.
    """
    pix = (to_frame(frame, MONO_IMG) / 255.0).reshape(-1)
    pa = np.asarray(prev_action, np.float32).reshape(-1)
    if pa.size != PREV_ACTION_DIM:
        raise ValueError(f"previous action must be {PREV_ACTION_DIM}-d, got {pa.size}")
    b = np.asarray(binoc, np.float32).reshape(-1)
    if b.size != BINOC_DIM:
        raise ValueError(
            f"binoc must be {BINOC_DIM}-d (DEBiT's output), got {b.size}. An "
            "unencoded zero vector here would train the policy goal-blind.")
    m = np.asarray([float(mode)], np.float32)
    return np.concatenate([pix, pa, m, b]).astype(np.float32)


def mono_state_dim(binoc_dim: int = BINOC_DIM) -> int:
    """mode + binoc. The learner's mode_index is 0 and core_state_cols is the
    rest, so the GRU reads the binoc and the critic reads the mode."""
    return 1 + int(binoc_dim)


def mono_row_dim(binoc_dim: int = BINOC_DIM) -> int:
    return MONO_PIX + PREV_ACTION_DIM + mono_state_dim(binoc_dim)


def mono_core_state_cols(binoc_dim: int = BINOC_DIM):
    """The binoc columns inside the state block: everything after the mode bit."""
    return tuple(range(1, 1 + int(binoc_dim)))
