"""Pure helpers for the exploration trace (train_carla_sac.run_trace).

User 2026-11-01: "first drive policy in exploration mode and store trace" --
the trace the map-based arms of the CARLA replication of the Gibson benchmark
are built from (the peer's best Gibson map is real-time entropy).
"""
import numpy as np


class CoverageGrid:
    """Fraction of a town's drivable waypoints within `radius` m of a kept
    sample. Waypoints are bucketed on a grid of cell = radius, so each sample
    only checks its 3x3 neighbourhood and coverage stays cheap as traces grow.
    """

    def __init__(self, waypoints_xy, radius=5.0):
        self.r = float(radius)
        self.wp = np.asarray(waypoints_xy, np.float64).reshape(-1, 2)
        self.covered = np.zeros(len(self.wp), bool)
        self.cells = {}
        for i, (x, y) in enumerate(self.wp):
            self.cells.setdefault(self._cell(x, y), []).append(i)

    def _cell(self, x, y):
        return (int(np.floor(x / self.r)), int(np.floor(y / self.r)))

    def add(self, xy):
        x, y = float(xy[0]), float(xy[1])
        cx, cy = self._cell(x, y)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for i in self.cells.get((cx + dx, cy + dy), ()):
                    if not self.covered[i]:
                        wx, wy = self.wp[i]
                        if (wx - x) ** 2 + (wy - y) ** 2 <= self.r ** 2:
                            self.covered[i] = True

    def fraction(self):
        return float(self.covered.mean()) if len(self.wp) else 0.0

    def nearest_uncovered(self, xy):
        """The uncovered waypoint closest to `xy`, or None when all are covered:
        where a stalled trace restarts, so coverage grows again."""
        left = np.flatnonzero(~self.covered)
        if not len(left):
            return None
        d = np.sum((self.wp[left] - np.asarray(xy, np.float64)[:2]) ** 2, axis=1)
        return self.wp[left[int(np.argmin(d))]]


class StuckDetector:
    """Fires when `limit` consecutive restarts land within `radius` m of the
    first of them, so the continuation rule cannot trace one failing corner
    forever. After firing it starts counting again from the next restart."""

    def __init__(self, radius=5.0, limit=4):
        self.r, self.limit = float(radius), int(limit)
        self.anchor, self.count = None, 0

    def restart(self, xy):
        p = np.asarray(xy, np.float64)[:2]
        if self.anchor is not None and np.linalg.norm(p - self.anchor) <= self.r:
            self.count += 1
        else:
            self.anchor, self.count = p, 1
        if self.count >= self.limit:
            self.anchor, self.count = None, 0
            return True
        return False
