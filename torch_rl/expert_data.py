"""Expert episode shards for imitation (2026-11-04, imitation-first plan, Task 3).

One compressed .npz per episode at <root>/<town>/ep_<seed>.npz. Per step:
  frames    (T+1, 128, 128, 3) uint8   the 128 px camera frame the row uses
  binoc     (T+1, 3136)        float16 DEBiT's binocular vector for that frame
  act_exec  (T, 2)             float32 the action EXECUTED, actor space [-1, 1]
  act_label (T, 2)             float32 the expert's clean action, actor space
  reward    (T,)               float32 the env reward for that step
  speed     (T,)               float32 km/h, for analysis
  pos       (T, 2)             float32 hero x, y, for analysis
and per episode: mode (int8), terminal, truncated (bool), goal
(pxr.goal_block of the latched goal frame), sigma (float32), driver, end,
town (str), seed (int64).

TWO ACTIONS, deliberately. With recovery noise the car executes
act_exec = expert + noise while the label is the expert's own correction:
cloning trains on act_label; replay (the critic's data in the fine-tune) must
use act_exec, because that is the action the reward and next state came from.
"""
import glob
import os

import numpy as np

from torch_rl import pixel_rows as pxr

_STEP_KEYS = ("act_exec", "act_label", "reward", "speed", "pos")
_STR_KEYS = ("driver", "end", "town")


def _check(ep):
    frames = np.asarray(ep["frames"])
    if (frames.ndim != 4 or frames.shape[1:] != (pxr.MONO_IMG, pxr.MONO_IMG, 3)
            or frames.dtype != np.uint8):
        raise ValueError(f"frames must be (T+1, {pxr.MONO_IMG}, {pxr.MONO_IMG}, 3) "
                         f"uint8, got {frames.shape} {frames.dtype}")
    T = len(np.asarray(ep["act_exec"]))
    if T == 0:
        raise ValueError("empty episode (T == 0): nothing to learn from")
    if frames.shape[0] != T + 1:
        raise ValueError(f"frames has {frames.shape[0]} rows for T={T} steps (need T+1)")
    binoc = np.asarray(ep["binoc"])
    if binoc.shape != (T + 1, pxr.BINOC_DIM) or binoc.dtype != np.float16:
        raise ValueError(f"binoc must be (T+1, {pxr.BINOC_DIM}) float16, got "
                         f"{binoc.shape} {binoc.dtype}")
    for k, shape in (("act_exec", (T, 2)), ("act_label", (T, 2)), ("reward", (T,)),
                     ("speed", (T,)), ("pos", (T, 2))):
        a = np.asarray(ep[k])
        if a.shape != shape or a.dtype != np.float32:
            raise ValueError(f"{k} must be {shape} float32, got {a.shape} {a.dtype}")
    goal = np.asarray(ep["goal"])
    if goal.shape != (pxr.PIX,):
        raise ValueError(f"goal must be pxr.goal_block (({pxr.PIX},)), got {goal.shape}")
    if int(ep["mode"]) not in (0, 1):
        raise ValueError(f"mode must be 0 or 1, got {ep['mode']!r}")


def save_episode(path, ep):
    """Validate and write one episode atomically (tmp + rename)."""
    _check(ep)
    out = {k: np.asarray(ep[k]) for k in ("frames", "binoc", "goal") + _STEP_KEYS}
    out.update(mode=np.int8(ep["mode"]), terminal=np.bool_(ep["terminal"]),
               truncated=np.bool_(ep["truncated"]), sigma=np.float32(ep["sigma"]),
               seed=np.int64(ep["seed"]))
    out.update({k: np.str_(ep[k]) for k in _STR_KEYS})
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **out)
    os.replace(tmp, path)


def load_episode(path):
    """One episode as a dict: arrays as stored, per-episode scalars as numpy
    scalars and strings as str."""
    with np.load(path, allow_pickle=False) as z:
        ep = {k: z[k] for k in z.files}
    for k in ("mode", "terminal", "truncated", "sigma", "seed"):
        if k in ep:
            ep[k] = ep[k][()]
    for k in _STR_KEYS:
        if k in ep:
            ep[k] = str(ep[k][()])
    return ep


def episode_paths(root, towns):
    """All episode files under <root>/<town>/ for the given towns, sorted."""
    out = []
    for t in towns:
        out += glob.glob(os.path.join(root, t, "ep_*.npz"))
    return sorted(p for p in out if not p.endswith(".tmp.npz"))


def split_by_episode(paths, val_frac=0.05, seed=0):
    """Deterministic (train, val) split by whole episode; at least one
    validation episode when there are two or more."""
    paths = sorted(paths)
    if len(paths) < 2:
        return list(paths), []
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(paths))
    n_val = max(1, int(round(val_frac * len(paths))))
    val = set(idx[:n_val].tolist())
    return ([p for i, p in enumerate(paths) if i not in val],
            [p for i, p in enumerate(paths) if i in val])


def fill_buffer(buf, paths, which="label", env_id=0):
    """Load episodes into a DeviceEpisodeBuffer exactly as the trainer would
    have added them. `which`: "label" (cloning) or "exec" (replay). Returns the
    number of transitions added; empty episodes are skipped."""
    if which not in ("label", "exec"):
        raise ValueError(f"which must be 'label' or 'exec', got {which!r}")
    key = "act_label" if which == "label" else "act_exec"
    added = 0
    for path in paths:
        ep = load_episode(path)
        T = len(ep["act_exec"])
        if T == 0:
            continue
        mode = int(ep["mode"])
        # A plain float: mono_row does float(mode), which on a 1-element array
        # raises numpy's array-to-scalar DeprecationWarning (same row either way).
        rows = [pxr.mono_row(ep["frames"][t], float(mode), ep["binoc"][t].astype(np.float32))
                for t in range(T + 1)]
        term, trunc = bool(ep["terminal"]), bool(ep["truncated"])
        for t in range(T):
            last = t == T - 1
            buf.add(rows[t], ep[key][t], float(ep["reward"][t]), term and last,
                    trunc and last, rows[t + 1], mode, env_id=env_id,
                    goal=ep["goal"])
        buf.end_episode(env_id)
        added += T
    return added
