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
    """All episode files under <root>/<town>/ for the given towns, sorted. `root`
    may join several roots with os.pathsep (il_data:il_data_cover), each read
    with the same per-town layout."""
    out = []
    for r in (x for x in str(root).split(os.pathsep) if x):
        for t in towns:
            out += glob.glob(os.path.join(r, t, "ep_*.npz"))
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


STOP_RUN = 5          # consecutive stop labels that mark a stop-forever tail
STOP_MIN_KMH = 2.0    # ... starting while the car still moves


def stop_cut(act_label, speed, run=STOP_RUN, min_kmh=STOP_MIN_KMH):
    """Rows of an episode worth CLONING: all of them, unless the label turns
    into the expert's final-approach stop (steer 0, throttle 0 env = actor
    [0, -1]) for `run` consecutive steps starting while the car moves faster
    than `min_kmh` -- then everything before that run.

    That run is what DAgger recorded once the clone had driven PAST the goal:
    the expert's route was spent and its only answer, every step until the
    episode ended, was "stop" (9 round-1 episodes held 98% of all stop labels,
    ~950 rows each, given at 14-18 km/h). Those rows taught the clone to crawl
    (2-5 km/h). A single stop at the goal or at a route's end is a real label
    and stays."""
    lab = np.asarray(act_label)
    sp = np.asarray(speed)
    T = len(lab)
    if T < run:
        return T
    st = (np.abs(lab[:, 0]) < 1e-6) & (lab[:, 1] <= -0.999)
    full = np.convolve(st.astype(np.int32), np.ones(run, np.int32), mode="valid") == run
    n = T - run + 1
    cand = np.flatnonzero(full & st[:n] & (sp[:n] > min_kmh))
    return int(cand[0]) if len(cand) else T


def fill_buffer(buf, paths, which="label", env_id=0):
    """Load episodes into a DeviceEpisodeBuffer exactly as the trainer would
    have added them. `which`: "label" (cloning) or "exec" (replay). Returns the
    number of transitions added; empty episodes are skipped. For cloning, an
    episode is cut before a stop-forever label tail (stop_cut) and the cut
    episode counts as truncated; replay keeps every executed step."""
    if which not in ("label", "exec"):
        raise ValueError(f"which must be 'label' or 'exec', got {which!r}")
    key = "act_label" if which == "label" else "act_exec"
    added = 0
    for path in paths:
        ep = load_episode(path)
        T = len(ep["act_exec"])
        cut = False
        if which == "label":
            keep = stop_cut(ep["act_label"], ep["speed"])
            cut, T = keep < T, keep
        if T == 0:
            continue
        mode = int(ep["mode"])
        # The speed each frame was taken at (used only under SPEED_OBS): frame
        # t+1 was recorded with speed[t] from the same step, frame 0 is the
        # reset frame, taken at rest -- the pairing live rows read from
        # info["speed_kmh"].
        frame_kmh = np.concatenate([[0.0], np.asarray(ep["speed"], np.float32)])
        # A plain float: mono_row does float(mode), which on a 1-element array
        # raises numpy's array-to-scalar DeprecationWarning (same row either way).
        rows = [pxr.mono_row(ep["frames"][t], float(mode), ep["binoc"][t].astype(np.float32),
                             speed_kmh=float(frame_kmh[t]))
                for t in range(T + 1)]
        term, trunc = (False, True) if cut else (bool(ep["terminal"]), bool(ep["truncated"]))
        for t in range(T):
            last = t == T - 1
            buf.add(rows[t], ep[key][t], float(ep["reward"][t]), term and last,
                    trunc and last, rows[t + 1], mode, env_id=env_id,
                    goal=ep["goal"])
        buf.end_episode(env_id)
        added += T
    return added
