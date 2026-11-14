"""Training chunks for the control stream, assembled from the recorded expert shards.

Pure array work, no I/O and no torch, because every defect here is SILENT: a one-step misalignment
between a frame and the action taken from it, a frame stack that bleeds the previous episode's
pixels into the first step, or an invalid lane label counted in the loss would all train quietly and
just give a worse network. tests/test_control_data.py pins each of those.

Shard layout (torch_rl.expert_record): frames (T+1, H, W, 3) uint8 -- ONE MORE than the actions,
since the last frame is the state after the final action -- act_label (T, 2), speed (T,), pos (T, 2).
Lane labels (torch_rl.lane_labels.episode_labels) are length T+1 with a `valid` mask that is False
at frame 0 and wherever the heading is undefined because the car was at rest.
"""
from typing import Dict

import numpy as np


def stack_frames(frames: np.ndarray, t: int, n_frames: int) -> np.ndarray:
    """The n_frames frames ending at t, channels-first, OLDEST FIRST: (3 * n_frames, H, W).

    At the start of an episode there is no frame -1, so the oldest available frame is repeated.
    Nothing outside this episode's array can ever enter the stack, which is what keeps the
    between-frame motion -- the heading signal the stack exists for -- honest.
    """
    idx = [max(0, t - k) for k in range(int(n_frames) - 1, -1, -1)]
    return np.concatenate([frames[i].transpose(2, 0, 1) for i in idx], axis=0)


def chunk_from_episode(ep: Dict[str, np.ndarray], start: int, length: int,
                       n_frames: int = 1) -> Dict[str, np.ndarray]:
    """`length` consecutive steps of one episode from `start`, ready for the GRU.

    Step k of the chunk is frame (start + k), the action taken FROM that frame, the speed at it, and
    its lane labels. Raises ValueError if the window would run past the last action.
    """
    n_act = int(ep["act_label"].shape[0])
    start, length = int(start), int(length)
    if start < 0 or length <= 0 or start + length > n_act:
        raise ValueError(f"chunk [{start}, {start + length}) outside the episode's {n_act} actions")
    sl = slice(start, start + length)
    frames = np.stack([stack_frames(ep["frames"], t, n_frames) for t in range(start, start + length)])
    lane = np.stack([np.asarray(ep["offset"], np.float32)[sl],
                     np.asarray(ep["heading_err"], np.float32)[sl]], axis=-1)
    valid = np.asarray(ep["valid"], bool)[sl].copy()
    # NON-FINITE LABELS ARE SCRUBBED, NOT JUST MASKED. episode_labels leaves NaN wherever the
    # heading is undefined (the car was at rest) or the lane query hit a fork, and in a masked loss
    # `nan * 0` is still nan: it poisons the first backward pass and NaNs every weight. So the mask
    # is widened to cover them AND the numbers are replaced, which makes the loss finite whatever
    # the consumer does with the mask.
    valid &= np.isfinite(lane).all(axis=-1)
    lane = np.nan_to_num(lane, nan=0.0, posinf=0.0, neginf=0.0)
    return {
        "frames": frames,                                              # (L, 3n, H, W) uint8
        "actions": np.asarray(ep["act_label"], np.float32)[sl],        # (L, 2)
        "speed": np.asarray(ep["speed"], np.float32)[sl],              # (L,)
        "lane": lane,                                                  # (L, 2) offset, heading err
        "lane_valid": valid,                                           # (L,)
    }
