"""Pose-free map navigation, as close to the Habitat peer's procedure as our
setup allows (user 2026-11-07: "make it match habitat as close as possible";
relayed procedure: memory habitat-map-procedure-2026-11).

  localisation  start node = top-1 image match of the car's view; goal node =
                top-1 image match of the goal image; no threshold, no radius
                (theirs: AnyLoc, max over a node's stored views -- ours store one)
  route         unit-cost shortest path; start == goal node or no route -> drive
                MAPLESS with the real goal image, counted normally
  subgoal       the next path node's stored image; the GRU is reset at every
                subgoal switch (theirs: DEBiT needed it)
  arrival       an image-match COMPETITION among the previous, current and next
                path nodes: the current subgoal is reached when it scores highest
                of the three AND at least `thr` on 2 consecutive steps. thr is
                OUR 2% far-false point (their 0.54 is theirs, set the same way).
                Theirs also ends a leg on DEBiT's STOP; c0 has no STOP action.
  timeout       150 steps on one leg -> relocalise (top-1 image) and replan to the
                same goal node, up to 3 times; then hand over the real goal image

ImageNavigator holds the pure state (tests/test_image_nav.py); the eval supplies
each step's cosine scores of the live view against every map node.
"""
import numpy as np


class ImageNavigator:
    """Subgoal state along a node path, driven by per-step image scores."""

    def __init__(self, path, thr, confirm_steps=2, leg_steps=150, max_replans=3, node_xy=None):
        # node_xy: map node positions, used ONLY by subgoal_m() for the episode
        # record (scoring / analysis); navigation never reads a pose
        self._xy = None if node_xy is None else np.asarray(node_xy, np.float64)
        self.path = list(path)
        self.thr = float(thr)
        self.confirm = int(confirm_steps)
        self.leg_steps = int(leg_steps)
        self.max_replans = int(max_replans)
        self.k = 1                    # index of the current subgoal in path
        self.switches = 0
        self.replans = 0
        self._wins = 0
        self._leg = 0
        self.gave_up = False

    @property
    def final(self):
        """True once the real goal image is handed over."""
        return self.gave_up or self.k >= len(self.path)

    def current(self):
        return None if self.final else self.path[self.k]

    def step(self, scores):
        """`scores`: cosine of the live view against every map node (indexable by
        node id). -> "switch" (subgoal reached), "timeout" (relocalise now),
        or None."""
        if self.final:
            return None
        self._leg += 1
        cur = self.path[self.k]
        rivals = [self.path[self.k - 1]] if self.k >= 1 else []
        if self.k + 1 < len(self.path):
            rivals.append(self.path[self.k + 1])
        s_cur = float(scores[cur])
        won = s_cur >= self.thr and all(s_cur > float(scores[r]) for r in rivals)
        self._wins = self._wins + 1 if won else 0
        if self._wins >= self.confirm:
            self.k += 1
            self.switches += 1
            self._wins = self._leg = 0
            return "switch"
        if self._leg >= self.leg_steps:
            self._wins = self._leg = 0
            return "timeout"
        return None

    def subgoal_m(self, pos):
        """ANALYSIS ONLY: true metres from `pos` to the current subgoal node (None
        when final or without node positions). Never used to navigate."""
        if self.final or self._xy is None:
            return None
        return float(np.hypot(*(self._xy[self.path[self.k]] - np.asarray(pos, np.float64)[:2])))

    def replan(self, new_path):
        """After a timeout: follow `new_path` (from a fresh top-1 relocalisation to
        the same goal node), or give up -> the real goal, once the replan budget
        is spent or no route exists. -> True if still navigating by the map."""
        if self.replans >= self.max_replans or not new_path or len(new_path) < 2:
            self.gave_up = True
            return False
        self.replans += 1
        self.path, self.k = list(new_path), 1
        self._wins = self._leg = 0
        return True


def top1(scores):
    """Index of the best-scoring node (no threshold, no radius: Habitat's rule)."""
    return int(np.argmax(np.asarray(scores)))
