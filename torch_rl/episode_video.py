#!/usr/bin/env python3
"""Episode videos for debugging: what the policy saw, beside what it was asked for.

User 2026-10-14: "Store some Videos for debugging". The useful frame for this arm
is not the camera alone -- it is the camera NEXT TO THE GOAL IMAGE, because the
question we keep asking of this policy is whether it is driving towards the view
it was given or merely driving well. A pair, with the step, mode, reward and
action burnt in, answers that by eye in a way no scalar does.

IT COSTS ALMOST NOTHING, which is why it can be on by default. Both frames are
already in memory every step (the replay stores them), so recording adds no
render, no GPU work and no simulator interaction -- only a memcpy and, once per
episode, an encode. This is deliberately unlike the VAE decode panel that killed
CARLA mid-sweep by taking 9.8 of 12.3 GB on the GPU: nothing here touches the GPU.

CHANNEL ORDER IS THE ONE EASY MISTAKE. With ENV_RGB_CHANNEL_FIX=1 the env's
frames are RGB, and cv2.VideoWriter expects BGR -- so the frames are converted on
the way out. Skip it and every video comes out with the sky orange and the road
blue, which looks like a perception bug and is not one.

BOUNDED BY DEFAULT: one episode in VIDEO_EVERY, at most VIDEO_MAX_FRAMES frames,
and at most VIDEO_KEEP files on disk -- 5, the user's number (2026-10-14: "Store
only 5 episodes at a time"), with the oldest deleted as each new one is written.
A debugging aid that fills a quota is a debugging aid someone turns off.
"""
import os
from typing import Optional

import numpy as np

PANEL_H = 240          # both frames are scaled to this height before joining


def _to_panel(frame: np.ndarray, height: int = PANEL_H) -> np.ndarray:
    import cv2
    a = np.asarray(frame)
    if a.dtype != np.uint8:
        a = np.clip(a, 0, 255).astype(np.uint8)
    h, w = a.shape[:2]
    return cv2.resize(a, (max(1, int(round(w * height / h))), height),
                      interpolation=cv2.INTER_NEAREST)


class EpisodeVideo:
    """Collects (current, goal) frames for one episode and writes one mp4.

    Usage per episode: start(), then add(...) per step, then save(). An episode
    that is not being recorded costs a single boolean test per step.
    """

    def __init__(self, out_dir: str, every: int = 50, max_frames: int = 1200,
                 keep: int = 5, fps: int = 10) -> None:
        self.out_dir = out_dir
        self.every = max(1, int(every))
        self.max_frames = int(max_frames)
        self.keep = int(keep)
        self.fps = int(fps)
        self.frames: list = []
        self.recording = False
        self.episode = 0
        os.makedirs(out_dir, exist_ok=True)

    # ------------------------------------------------------------ lifecycle --
    def start(self, episode: int) -> bool:
        """Begin an episode; returns whether it is being recorded."""
        self.episode = int(episode)
        self.frames = []
        self.recording = (self.episode % self.every) == 0
        return self.recording

    def add(self, cur: np.ndarray, goal: np.ndarray, step: int, mode: int,
            reward: float, action, ret: float) -> None:
        if not self.recording or len(self.frames) >= self.max_frames:
            return
        import cv2
        left, right = _to_panel(cur), _to_panel(goal)
        pair = np.concatenate([left, right], axis=1)
        pair = np.ascontiguousarray(pair)
        label = ("GOAL" if int(mode) == 1 else "EXPLORE")
        a = np.asarray(action, np.float32).ravel()
        lines = [f"ep {self.episode}  t {step}  {label}",
                 f"r {reward:+.2f}  ret {ret:+.1f}",
                 f"steer {a[0]:+.2f}  throttle {a[1]:+.2f}" if a.size >= 2 else ""]
        for i, text in enumerate(lines):
            if not text:
                continue
            # Drawn twice: black underneath, white on top, so the text stays
            # readable over both a bright sky and dark tarmac.
            for colour, thick in (((0, 0, 0), 3), ((255, 255, 255), 1)):
                cv2.putText(pair, text, (6, 18 + 16 * i),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, thick,
                            cv2.LINE_AA)
        # A divider, so it is obvious which half is the goal.
        pair[:, left.shape[1]:left.shape[1] + 2] = 255
        self.frames.append(pair)

    def save(self, end_reason: str = "") -> Optional[str]:
        """Write the episode's mp4 and return its path (None if not recording)."""
        if not self.recording or not self.frames:
            self.recording = False
            return None
        import cv2
        h, w = self.frames[0].shape[:2]
        tag = f"ep{self.episode:06d}_{end_reason or 'end'}"
        path = os.path.join(self.out_dir, f"{tag}.mp4")
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"),
                                 self.fps, (w, h))
        if not writer.isOpened():                      # no codec: not fatal
            self.recording = False
            self.frames = []
            return None
        for f in self.frames:
            # RGB -> BGR. cv2 writes BGR, and with ENV_RGB_CHANNEL_FIX=1 these
            # frames are RGB; without this every video has an orange sky.
            writer.write(f[:, :, ::-1])
        writer.release()
        self.frames = []
        self.recording = False
        self._prune()
        return path

    def _prune(self) -> None:
        """Keep the newest `keep` files, so a long run cannot fill a quota."""
        if self.keep <= 0:
            return
        try:
            vids = sorted(
                (os.path.join(self.out_dir, f) for f in os.listdir(self.out_dir)
                 if f.endswith(".mp4")), key=os.path.getmtime)
            for old in vids[:-self.keep]:
                os.remove(old)
        except OSError:                                # never fail a run for this
            pass
