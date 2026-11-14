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


def stack_frames(frames: np.ndarray, t: int, n_frames: int, stride: int = 1) -> np.ndarray:
    """The n_frames frames ending at t, `stride` ticks apart, channels-first, OLDEST FIRST:
    (3 * n_frames, H, W).

    STRIDE MUST MATCH THE DECISION RATE. With ACTION_REPEAT=k the policy only sees frames at
    decision boundaries, k ticks apart, so a stream pretrained on CONSECUTIVE frames would be fed
    8x the apparent motion at k=8 -- the same optical flow meaning a different heading rate in
    training and deployment. It also helps: a wider baseline makes heading easier to read (the
    2026-11-09 probe gives heading R^2 0.042 from one frame and 0.317 from two consecutive ones).

    At the start of an episode there is no frame -stride, so the oldest available frame is repeated.
    Nothing outside this episode's array can ever enter the stack, which is what keeps the
    between-frame motion -- the heading signal the stack exists for -- honest.
    """
    s = max(1, int(stride))
    idx = [max(0, t - k * s) for k in range(int(n_frames) - 1, -1, -1)]
    return np.concatenate([frames[i].transpose(2, 0, 1) for i in idx], axis=0)


def chunk_from_episode(ep: Dict[str, np.ndarray], start: int, length: int,
                       n_frames: int = 1, stride: int = 1, step: int = 1) -> Dict[str, np.ndarray]:
    """`length` GRU steps of one episode from tick `start`, `step` ticks apart.

    Step j of the chunk is the frame at tick (start + j * step), the action taken FROM that frame,
    the speed at it, and its lane labels. `stride` is the spacing INSIDE each frame stack. Both must
    match the decision rate (ACTION_REPEAT=k -> stride = step = k): the policy then decides once
    every k ticks, so its GRU steps once per decision and its frame pair is k ticks apart, and a
    stream pretrained at one tick per step would meet a different time scale at deployment.
    Raises ValueError if the last step would fall past the last action.
    """
    n_act = int(ep["act_label"].shape[0])
    start, length, step = int(start), int(length), max(1, int(step))
    last = start + (length - 1) * step
    if start < 0 or length <= 0 or last > n_act - 1:
        raise ValueError(f"chunk ticks {start}..{last} (step {step}) outside the episode's "
                         f"{n_act} actions")
    sl = start + step * np.arange(length)
    frames = np.stack([stack_frames(ep["frames"], int(t), n_frames, stride) for t in sl])
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


# ---- how a pretrained stream is judged --------------------------------------------------------------
#
# THE RULE (2026-11-14, after getting it wrong): a number that says "the features carry X" must come
# from rows of EPISODES THE FIT NEVER SAW. The first control streams were gated on a ridge probe
# fitted and scored on the same rows -- "heading R^2 0.81" -- and compared with c0's 0.046 from
# scripts/probe_c0_features.py, which holds episodes out. Held out by episode the stream scored
# about 0.1 (its own lane head: 0.10; offset 0.38). Rows of one episode are a few seconds apart, so
# a few thousand rows from 30-50 episodes are 30-50 samples, and a 129-weight probe memorises them.
# Quote, in this order: the trained head's own R^2 on held-out episodes (r2_score, nothing fitted),
# then ridge_r2_episode_cv. ridge_r2_in_sample exists to show the gap, never as evidence.

def r2_score(target, pred) -> float:
    """1 - SSE / SST of a prediction nobody fitted here. 0 = no better than the mean; negative =
    worse than it."""
    y, p = np.asarray(target, np.float64).reshape(-1), np.asarray(pred, np.float64).reshape(-1)
    sst = float(((y - y.mean()) ** 2).sum())
    return float(1.0 - ((y - p) ** 2).sum() / sst) if sst > 0 else float("nan")


def _ridge_weights(H: np.ndarray, Y: np.ndarray, lam: float) -> np.ndarray:
    X = np.concatenate([H, np.ones((len(H), 1), H.dtype)], axis=1).astype(np.float64)
    return np.linalg.solve(X.T @ X + float(lam) * np.eye(X.shape[1]), X.T @ np.asarray(Y, np.float64))


def _ridge_predict(H: np.ndarray, W: np.ndarray) -> np.ndarray:
    return np.concatenate([H, np.ones((len(H), 1), H.dtype)], axis=1).astype(np.float64) @ W


def ridge_r2_in_sample(H: np.ndarray, Y: np.ndarray, lam: float = 1.0) -> list:
    """A ridge probe from features H (N, D) to each column of Y (N, K), FITTED AND SCORED ON THE
    SAME ROWS. Optimistic by construction; reported only next to ridge_r2_episode_cv."""
    P = _ridge_predict(H, _ridge_weights(H, Y, lam))
    return [r2_score(Y[:, k], P[:, k]) for k in range(Y.shape[1])]


def episode_folds(groups: np.ndarray, folds: int = 5, seed: int = 0) -> np.ndarray:
    """A fold number per row such that an episode is never split across folds."""
    groups = np.asarray(groups)
    ids = np.unique(groups)
    if len(ids) < 2:
        raise ValueError("holding episodes out needs at least two episodes")
    k = min(int(folds), len(ids))
    order = np.random.default_rng(seed).permutation(len(ids))
    fold_of = {ids[j]: i % k for i, j in enumerate(order)}
    return np.array([fold_of[g] for g in groups])


RIDGE_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)      # x the row count, as probe_c0_features._Ridge


def _standardise(Xtr: np.ndarray, *others):
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd = np.where(sd < 1e-6, 1.0, sd)
    return [(X - mu) / sd for X in (Xtr,) + others]


def _ridge_grid(X: np.ndarray, y: np.ndarray) -> list:
    """(weights, intercept) for every lambda of RIDGE_GRID from one eigendecomposition."""
    s, V = np.linalg.eigh(X.T @ X)
    ym = float(y.mean())
    vc = V.T @ (X.T @ (y - ym))
    return [(V @ (vc / (s + a * len(X))), ym) for a in RIDGE_GRID]


def ridge_r2_episode_cv(H: np.ndarray, Y: np.ndarray, groups: np.ndarray, folds: int = 5,
                        seed: int = 0) -> list:
    """The probe with WHOLE EPISODES held out: every row is predicted by a fit that saw no row of its
    episode, and R^2 is pooled over all rows. It follows scripts/probe_c0_features.ridge_cv -- features
    standardised on the training fold, lambda chosen on an inner split of 20% of the training fold's
    EPISODES -- so a stream scored here is comparable with c0's 2026-11-09 figures."""
    H, Y, groups = np.asarray(H, np.float64), np.asarray(Y, np.float64), np.asarray(groups)
    f = episode_folds(groups, folds, seed)
    rng = np.random.default_rng(seed + 1)
    P = np.zeros(Y.shape, np.float64)
    for i in np.unique(f):
        tr, te = np.flatnonzero(f != i), np.flatnonzero(f == i)
        ug = np.unique(groups[tr])
        inner_g = set(rng.choice(ug, max(1, len(ug) // 5), replace=False).tolist())
        iv = np.array([g in inner_g for g in groups[tr]])
        A, B = _standardise(H[tr[~iv]], H[tr[iv]])
        A2, C = _standardise(H[tr], H[te])
        for k in range(Y.shape[1]):
            ytr = Y[tr, k]
            scores = [r2_score(ytr[iv], B @ w + b) for w, b in _ridge_grid(A, ytr[~iv])]
            w, b = _ridge_grid(A2, ytr)[int(np.nanargmax(scores))]
            P[te, k] = C @ w + b
    return [r2_score(Y[:, k], P[:, k]) for k in range(Y.shape[1])]
