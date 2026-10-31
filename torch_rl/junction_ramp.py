"""JUNCTION RAMP (v27, user 2026-10-30): goals start easy and harden.

v18 reached 0.725 on Town02 with half its goals past a junction; at 0.85 no run
has beaten 0.40. JunctionRamp starts JUNCTION_GOAL_FRAC at `start` and moves it
by `step` towards `target` as TRAINING goal success rises, and back towards
`start` when it falls -- v44 lost its run to a curriculum that could not demote.

THE MECHANISM IS os.environ. The goal sampler (rlib_integration carla_core)
reads JUNCTION_GOAL_FRAC at every reset, in this process (ThreadedVecEnv steps
the towns on threads), so setting the variable moves the next episode's goal.

A WINDOW per level: `window` goal episodes at the current fraction, judged on
their success rate, then the window restarts -- episodes sampled at the old
fraction say nothing about the new one.
"""
import os
from collections import deque

ENV_KEY = "JUNCTION_GOAL_FRAC"


class JunctionRamp:
    def __init__(self, start=0.5, target=0.85, step=0.05, up=0.3, down=0.1,
                 window=100):
        if target < start:
            raise ValueError(f"junction ramp target {target} is below its start {start}")
        self.start, self.target, self.step = float(start), float(target), float(step)
        self.up, self.down, self.window = float(up), float(down), int(window)
        self.value = self.start
        self._recent = deque(maxlen=self.window)
        self._set(self.value)

    @staticmethod
    def _set(v):
        os.environ[ENV_KEY] = f"{v:.2f}"

    def describe(self):
        return (f"JUNCTION RAMP: goals past a junction {self.start:.2f} -> "
                f"{self.target:.2f} in steps of {self.step:.2f}; up when training "
                f"goal success >= {self.up:.2f} over {self.window} goal episodes, "
                f"down when < {self.down:.2f}")

    def record(self, arrived):
        """One finished GOAL episode. Returns a log line when the level moves."""
        self._recent.append(1.0 if arrived else 0.0)
        if len(self._recent) < self.window:
            return None
        sr = sum(self._recent) / len(self._recent)
        old = self.value
        if sr >= self.up and self.value < self.target - 1e-9:
            self.value = min(self.target, self.value + self.step)
        elif sr < self.down and self.value > self.start + 1e-9:
            self.value = max(self.start, self.value - self.step)
        else:
            return None
        self._recent.clear()
        self._set(self.value)
        return (f"[Ramp] junction goals {old:.2f} -> {self.value:.2f} "
                f"(training goal success {sr:.2f} over {self.window})")
