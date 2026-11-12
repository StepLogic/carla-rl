"""Per-episode driving-rule accounting for offline evaluation.

WHY (user 2026-09-27, evaluating v17's 150k checkpoint: "is the goal mode strong
with no lane crossings and offroad", "explore mode can explore"). eval_arm.py
recorded only HOW an episode ended, so an arrival that cut across a solid line
and ran over the verge on the way counted the same as a clean one. This counts,
per episode, every fault step by kind, and the path driven.

CROSSINGS ARE CLASSIFIED WITH THE SAME MARKING SETS THE GRADED LANE FINE USES
(vae_nav.lane_explore_reward.ILLEGAL_MARKS / UNPAINTED_MARKS and the junction
flag), so "illegal" here means what gated_reward.goal_lane_fine charges in full:
    junction      broken line inside a junction -- the turn, legal
    edge          every mark unpainted (NONE)   -- the road edge, legal
    lane_change   broken line on open road      -- legal, but counted
    illegal       solid / double / kerb / grass, or no marking data (fails closed)

Pure: fed one step's info at a time, no CARLA needed, so it is unit-tested.
"""
import numpy as np

from vae_nav.lane_explore_reward import ILLEGAL_MARKS, UNPAINTED_MARKS

CROSSING_KINDS = ("junction", "edge", "lane_change", "illegal")


def classify_crossing(info):
    """None when no crossing this step, else one of CROSSING_KINDS."""
    if not bool(info.get("lane_any", info.get("lane_invasion_any", False))):
        return None
    marks = [str(m).split(".")[-1] for m in (info.get("lane_marking_types") or [])]
    if not marks or any(m in ILLEGAL_MARKS for m in marks):
        return "illegal"
    if all(m in UNPAINTED_MARKS for m in marks):
        return "edge"
    if bool(info.get("in_junction", False)):
        return "junction"
    return "lane_change"


class EpisodeFaults:
    """Accumulates one episode. Call step(info) after every env.step."""

    def __init__(self, start_pos=None):
        self.steps = 0
        self.off_road = 0
        self.collision = 0
        self.crossing = {k: 0 for k in CROSSING_KINDS}
        self.path_m = 0.0
        self._prev = None if start_pos is None else np.asarray(start_pos, float)[:2]

    def step(self, info):
        self.steps += 1
        self.off_road += int(bool(info.get("off_road", False)))
        self.collision += int(bool(info.get("collision", False)))
        kind = classify_crossing(info)
        if kind is not None:
            self.crossing[kind] += 1
        pos = info.get("hero_pos")
        if pos is not None:
            p = np.asarray(pos, float)[:2]
            if self._prev is not None:
                self.path_m += float(np.linalg.norm(p - self._prev))
            self._prev = p

    def summary(self, arrived):
        clean = (arrived and self.off_road == 0 and self.collision == 0
                 and self.crossing["illegal"] == 0)
        return {"steps": self.steps, "arrived": bool(arrived),
                "off_road_steps": self.off_road,
                "collision_steps": self.collision,
                "crossing_steps": dict(self.crossing),
                "path_m": round(self.path_m, 2),
                # arrived with NO off-road, NO collision and NO illegal crossing;
                # legal turns and lane changes allowed
                "clean_arrival": bool(clean),
                # and not even a lane change
                "strict_clean_arrival": bool(clean and self.crossing["lane_change"] == 0)}


def aggregate(episodes):
    """Across episodes: rates of any-fault episodes and mean fault steps."""
    n = max(1, len(episodes))
    frac = lambda f: round(sum(1 for e in episodes if f(e)) / n, 3)
    mean = lambda f: round(float(np.mean([f(e) for e in episodes])) if episodes else 0.0, 2)
    return {
        "n": len(episodes),
        "clean_arrival_rate": frac(lambda e: e["clean_arrival"]),
        "strict_clean_arrival_rate": frac(lambda e: e["strict_clean_arrival"]),
        "frac_any_off_road": frac(lambda e: e["off_road_steps"] > 0),
        "frac_any_collision": frac(lambda e: e["collision_steps"] > 0),
        "frac_any_illegal_crossing": frac(lambda e: e["crossing_steps"]["illegal"] > 0),
        "frac_any_lane_change": frac(lambda e: e["crossing_steps"]["lane_change"] > 0),
        "mean_off_road_steps": mean(lambda e: e["off_road_steps"]),
        "mean_illegal_crossing_steps": mean(lambda e: e["crossing_steps"]["illegal"]),
        "mean_path_m": mean(lambda e: e["path_m"]),
        "median_path_m": round(float(np.median([e["path_m"] for e in episodes])), 2)
        if episodes else 0.0,
    }
