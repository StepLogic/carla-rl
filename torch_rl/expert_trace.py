"""The single-trace evaluation maps' trace (user 2026-11-06: "CAN YOU CREATE MAP
FROM A SINGLE TRACE?" -> "I am talking about evaluations").

The Gibson protocol builds each scene's map from ONE continuous exploration
trace. Here the EXPERT (BasicAgent) drives one closed coverage tour of the town
(vae_nav.coverage_tour) in a single explore-mode episode with no respawn, and
the clone rides along at every step for the entropy the online-MAD map is
placed on. train_carla_sac.run_expert_trace drives; this module holds the pure
parts:

  * trace_dir: where the trace goes, refused anywhere training data is read
    from. This is an EVALUATION artefact in the held-out towns; it is written in
    the map-trace format (index.json + seg_*.npz), which no training loader
    reads, under its own root, and never under the expert-data root (il_data*).
  * ChunkWriter: run_trace's format, one file per `chunk` steps so a killed job
    loses at most one chunk. Chunks of one uninterrupted drive share a `drive`
    id and are not `jumped`, so image_map.build_graph joins them and the map
    builder resets the online-MAD buffer per DRIVE, not per file. A resume after
    a crash is a new drive starting where the last chunk stopped (the user's
    rule: traces resume at the crash), from the tour index it had reached.
"""
import json
import os

import numpy as np

FIELDS = ("frame", "pos", "yaw", "speed", "action", "reward", "entropy", "entropy_h0",
          "q_mean", "q_std", "t", "on_line")


def trace_dir(root, town, record_root):
    """<root>/trace_expert_<town>, refused inside the expert-data root or any
    il_data* directory: this trace must never be loadable as training data."""
    r = os.path.realpath(os.path.abspath(root))
    rec = os.path.realpath(os.path.abspath(record_root))
    parts = r.split(os.sep)
    if r == rec or r.startswith(rec + os.sep) or any(p.startswith("il_data") for p in parts):
        raise ValueError(f"EXPERT_TRACE root {root} is where training data is read "
                         f"from (expert-data root {record_root}): the held-out-town "
                         f"trace is for evaluation maps only")
    return os.path.join(os.path.abspath(root), f"trace_expert_{town}")


class ChunkWriter:
    """index.json + seg_XXXXX.npz in run_trace's format, plus per chunk the
    drive id, the tour index reached (`plan_index`) and running totals."""

    def __init__(self, out_dir, chunk=2000, meta=None):
        self.out, self.chunk = out_dir, int(chunk)
        os.makedirs(out_dir, exist_ok=True)
        self.idx_path = os.path.join(out_dir, "index.json")
        if os.path.exists(self.idx_path):
            with open(self.idx_path) as fh:
                self.index = json.load(fh)
        else:
            self.index = {**(meta or {}), "segments": [], "done": False}
        segs = self.index["segments"]
        self.done = bool(self.index.get("done"))
        self.drive = (segs[-1]["drive"] + 1) if segs else 0
        self.resume_plan_index = segs[-1]["plan_index"] if segs else 0
        self.steps = int(sum(s["steps"] for s in segs))
        self.metres = float(sum(s["metres"] for s in segs))
        self._rec = {k: [] for k in FIELDS}
        self._plan_index = self.resume_plan_index
        self._prev = None
        self._m = 0.0
        self._first_in_drive = True
        self._jump_next = False

    def add(self, sample, plan_index):
        pos = np.asarray(sample["pos"], np.float64)[:2]
        if self._prev is not None:
            self._m += float(np.linalg.norm(pos - self._prev))
        self._prev = pos
        for k in FIELDS:
            self._rec[k].append(sample[k])
        self._plan_index = int(plan_index)
        if len(self._rec["t"]) >= self.chunk:
            self.flush()

    def flush(self, end=None):
        """Write the samples since the last flush as one chunk (none: no-op)."""
        n = len(self._rec["t"])
        if not n:
            return
        seg = len(self.index["segments"])
        fname = f"seg_{seg:05d}.npz"
        arrays = {k: np.asarray(v) for k, v in self._rec.items()}
        np.savez(os.path.join(self.out, fname), **arrays)
        self.index["segments"].append({
            "seg": seg, "file": fname, "drive": self.drive,
            "continues": not self._first_in_drive,
            "jumped": bool(self._jump_next and self._first_in_drive),
            "start_xy": np.round(arrays["pos"][0][:2], 2).tolist(),
            "stop_xy": np.round(arrays["pos"][-1][:2], 2).tolist(),
            "end": end, "steps": n, "kept": n, "metres": round(self._m, 2),
            "plan_index": self._plan_index,
            "crossings_steps": int(np.sum(arrays["on_line"]))})
        self.steps += n
        self.metres += self._m
        self._m = 0.0
        self._first_in_drive = False
        self._jump_next = False
        self._rec = {k: [] for k in FIELDS}
        self._write_index()

    def new_drive(self, end, jumped=False):
        """The env ended the episode (`end`): close this drive and start the
        next one. `jumped` marks a restart that skipped unreached ground, so
        image_map.build_graph never joins across it."""
        self.flush(end=end)
        self.drive += 1
        self._first_in_drive = True
        self._jump_next = bool(jumped)
        self._prev = None

    def close(self, end, done):
        self.flush(end=end)
        if self.index["segments"] and end is not None:
            self.index["segments"][-1]["end"] = end
        self.index["done"] = bool(done)
        self.done = bool(done)
        self._write_index()

    def _write_index(self):
        tmp = self.idx_path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(self.index, fh, indent=1)
        os.replace(tmp, self.idx_path)


def drive_ids(trace, index):
    """Per-sample drive id of an image_map.load_trace trace: the segment ids
    the online-MAD buffer must reset on (a drive, not a chunk file)."""
    drive_of = {int(s["seg"]): int(s.get("drive", s["seg"])) for s in index["segments"]}
    return np.array([drive_of[int(s)] for s in np.asarray(trace["seg"])], int)
