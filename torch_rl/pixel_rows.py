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
