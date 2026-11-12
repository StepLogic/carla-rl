"""Save the replay buffer when training ends; restore it on resume.

WHY (user 2026-09-28: "save [replay] when training ends"). A resume restored
weights only and restarted with an EMPTY buffer, so extending a run (v26 to 1M)
trained its first steps on a handful of fresh transitions; v17's fresh-buffer
resume declined afterwards.

REPLAY_SAVE=1  at the end of training, pickle the whole buffer object to
               <run>/replay_<step>.pkl -- only if the disk has room: free space
               must exceed the buffer's size x 1.3 + 2 GB (the root disk sat at
               12 GB free, and a full disk corrupts checkpoints mid-write).
REPLAY_LOAD=1  on resume from checkpoint_<step>, load replay_<step>.pkl INTO
               the existing buffer (its attributes are replaced), so anything
               already holding a reference to the buffer keeps working.
"""
import os
import pickle
import shutil

import numpy as np


def buffer_nbytes(obj, _seen=None):
    """Bytes held in numpy arrays reachable from obj (dicts, lists, attrs)."""
    seen = set() if _seen is None else _seen
    if id(obj) in seen:
        return 0
    seen.add(id(obj))
    if isinstance(obj, np.ndarray):
        return int(obj.nbytes)
    if isinstance(obj, dict):
        return sum(buffer_nbytes(v, seen) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(buffer_nbytes(v, seen) for v in obj)
    if hasattr(obj, "__dict__"):
        return sum(buffer_nbytes(v, seen) for v in vars(obj).values())
    return 0


def save_replay(buffer, out_dir, step, margin=1.3, reserve_bytes=2 * 2**30):
    """Returns the path written, or None (with the reason printed) if skipped."""
    need = int(buffer_nbytes(buffer) * margin + reserve_bytes)
    free = shutil.disk_usage(out_dir).free
    if free < need:
        print(f"[Replay] NOT saved: {free / 2**30:.1f} GiB free, need "
              f"{need / 2**30:.1f} GiB (buffer x{margin} + reserve)", flush=True)
        return None
    path = os.path.join(out_dir, f"replay_{int(step)}.pkl")
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(buffer, f, protocol=4)
    os.replace(tmp, path)
    print(f"[Replay] saved {os.path.getsize(path) / 2**30:.2f} GiB -> {path}",
          flush=True)
    return path


def load_replay_into(buffer, out_dir, step):
    """True if replay_<step>.pkl was loaded into `buffer`."""
    path = os.path.join(out_dir, f"replay_{int(step)}.pkl")
    if not os.path.exists(path):
        print(f"[Replay] no {os.path.basename(path)}; buffer starts empty",
              flush=True)
        return False
    with open(path, "rb") as f:
        saved = pickle.load(f)
    if type(saved) is not type(buffer):
        print(f"[Replay] {path} holds a {type(saved).__name__}, this run a "
              f"{type(buffer).__name__}; buffer starts empty", flush=True)
        return False
    vars(buffer).clear()
    vars(buffer).update(vars(saved))
    print(f"[Replay] restored {len(buffer)} transitions from {path}", flush=True)
    return True
