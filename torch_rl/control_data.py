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


def transition_actions(shard, n: int) -> np.ndarray:
    """The action a VALUE LEARNER must condition on: the one that was EXECUTED.

    The shards carry two action columns and they are not interchangeable:
      * `act_label` is what the EXPERT would have done -- the behaviour-cloning target;
      * `act_exec` is what the car actually did (the driving policy's action, plus DART noise), so it
        is the action whose consequence the next frame records.
    They differ by a lot: on the DAgger chains the executed steering differs from the label with
    sd 0.199, which is 96% of the expert steering's own sd (DART sigma in {0, 0.1, 0.2}).

    Conditioning a critic on `act_label` is therefore a silent corruption: identical (s, a_label)
    rows map to DIFFERENT next states according to noise the critic cannot see, TD averages the
    action-dependence away, and Q comes out flat or monotone in the action -- which is exactly what
    the first IQL run measured (peak at the edge of every steering band, no interior optimum).
    torch_rl/frl/expert.py gets this right; scripts/pretrain_critic_iql.py did not, at first.
    """
    files = getattr(shard, "files", None) or shard.keys()
    key = "act_exec" if "act_exec" in files else "act_label"
    return np.asarray(shard[key], np.float32)[:int(n)]


def nstep_rows(actions: np.ndarray, rewards: np.ndarray, terminal_at_end: bool, k: int,
               gamma: float) -> Dict[str, np.ndarray]:
    """Non-overlapping k-step transitions for an ACTION-REPEAT test (user 2026-11-14: "ok use action
    repeat at 8"): (s_t, mean action over [t, t+k), sum_i gamma^i r_{t+i}, s_{t+k}).

    WHY. The measured problem is that Q is almost flat in the action: over the band the data lives
    in it spans 0.30% of |Q|, and the peak sits at the edge of any grid. One steering value held for
    0.05-0.1 s moves the car about a centimetre and the behaviour policy corrects it at the next
    step, so the return genuinely barely responds. Holding one action for k steps makes the
    perturbation sustained -- the advantage should grow with the action's duration (Tallec, Blier &
    Ollivier, ICML 2019). This builds that MDP from existing data so the prediction can be checked
    before spending GPU time in CARLA.

    CAVEAT, stated here because it bounds the conclusion: the recorded actions VARIED inside each
    window, so the mean is only an approximation of holding one action, and averaging washes out
    variation -- it biases the result toward a flatter Q. A span that GROWS under this is therefore
    real evidence; a null result is weak evidence.

    A partial tail window (fewer than k steps left) is dropped. `terminal_at_end` marks the episode
    as ending in a fault or an arrival, in which case only the final window loses its bootstrap.
    """
    k = int(k)
    n = int(len(rewards))
    starts = np.arange(0, n - k + 1, k, dtype=np.int64)
    if len(starts) == 0:
        z = np.zeros((0,), np.float32)
        return {"starts": starts, "nexts": starts, "actions": np.zeros((0, actions.shape[1]),
                np.float32), "rewards": z, "masks": z}
    disc = gamma ** np.arange(k, dtype=np.float64)
    rew = np.array([float((np.asarray(rewards[s:s + k], np.float64) * disc).sum())
                    for s in starts], np.float32)
    act = np.stack([np.asarray(actions[s:s + k], np.float64).mean(axis=0) for s in starts]
                   ).astype(np.float32)
    nexts = starts + k
    masks = np.ones(len(starts), np.float32)
    if terminal_at_end and nexts[-1] >= n:
        masks[-1] = 0.0
    return {"starts": starts, "nexts": nexts, "actions": act, "rewards": rew, "masks": masks}


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
