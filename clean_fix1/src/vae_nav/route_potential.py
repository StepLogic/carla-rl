"""Remaining distance ALONG THE ROAD to the goal, for potential-based shaping.

WHY NOT EUCLIDEAN. The obvious potential is straight-line distance to the goal,
and it is actively wrong here. Measured on this project's own CORL 2026
evaluation set (Town02, 20 trajectories): trajectory 4's endpoints are 39.8 m
apart and 621.6 m by road, and trajectories 2, 9 and 3 sit at route/euclid
ratios of 8.8, 8.2 and 5.2. In those episodes -- a fifth of the set -- "reduce
straight-line distance" means driving AWAY from the correct route for a long
stretch. Potential-based shaping leaves the OPTIMUM unchanged either way (Ng et
al. 1999), but the dense gradient would point the wrong way in exactly the
episodes that are hard, which is the opposite of what shaping is for.

WHAT THIS PUBLISHES. Phi(s) = -remaining_m, where remaining_m is the distance
still to cover along the planned route. The shaping term is then

    F = Phi(s') - Phi(s) = remaining(s) - remaining(s')

i.e. simply the metres of route progress made this step, positive for progress
and negative for backtracking. It telescopes over an episode to
remaining(s_0) - remaining(s_T), so it adds a bounded constant to the return and
cannot be farmed by driving in circles.

THE VALUE FUNCTION STAYS INVERTIBLE. With r = -c + F and gamma = 1,

    V(s) = -c * E[steps to goal] + remaining(s) - E[remaining(s_T)]

so the cost-to-go is recoverable exactly, because `remaining` is a known
function of the state:

    steps = (remaining(s) - V(s)) / c        (arrival has remaining ~ 0)

vae_nav.value_distance MUST apply that correction before handing edge weights to
Dijkstra, or every edge is wrong by the local route distance.

MONOTONE POINTER, NOT NEAREST-NODE SEARCH. The route index only ever advances.
A nearest-node lookup would snap backwards wherever the route passes close to
itself -- a loop round a block, or the two carriageways of one street -- and the
potential would jump, injecting a large fake reward. Advancing only forwards
makes the potential monotone in progress by construction.
"""
import os

import numpy as np

# Route node spacing, metres. Matches ExpertOracle's default so the two agree
# about what a "node" is; a coarser route understates remaining distance.
SAMPLING_RESOLUTION_M = float(os.environ.get("ROUTE_SAMPLING_M", 2.0))
# How far ahead of the current pointer to look when advancing. Two nodes is
# enough at 2 m spacing for a car doing ~0.3 m/step, and bounds the cost.
ADVANCE_LOOKAHEAD = int(os.environ.get("ROUTE_ADVANCE_LOOKAHEAD", 12))

# HOW remaining() MEASURES (2026-09-26). "hop" is the original: the suffix of
# the NEAREST route node plus the straight hop to it. That hop is ADDED even when
# the node is BEHIND the car, overcounting by twice the gap, so the measure is a
# sawtooth. MEASURED on real Town05 routes, driving perfectly forward at the
# car's 0.28 m/step: 43.1% of steps paid NEGATIVE progress (typically -0.28 m,
# exactly backwards) and 13.6% a +2.04 m spike -- only the telescoped SUM was
# right (3346 vs 3304 m). Potential-based shaping tolerates that in theory; a
# reward that IS progress (GATED_GOAL_REWARD=progress) does not.
# "project" projects the car onto the route SEGMENT it is abreast of: smooth,
# and it never runs backwards for forward driving. Default stays "hop" so every
# existing run -- including a crash-retry of a live one -- is byte-identical.
REMAINING_MODE = os.environ.get("ROUTE_REMAINING_MODE", "hop").lower()
if REMAINING_MODE not in ("hop", "project"):
    raise ValueError(f"ROUTE_REMAINING_MODE={REMAINING_MODE!r} is not hop|project")
# Under "project", lateral distance from the route counts only beyond one lane
# width: a car in its own lane or the adjacent one is not further from the goal
# (a lane change is not a detour), but a car that has left the road network the
# route runs on is, and that must show up as distance.
LATERAL_DEADBAND_M = float(os.environ.get("ROUTE_LATERAL_DEADBAND_M", 3.5))


def _xy(loc):
    return np.array([loc.x, loc.y], np.float64)


class RoutePotential:
    """Remaining along-route distance to the goal, from a cached plan."""

    def __init__(self, world_map, sampling_resolution=SAMPLING_RESOLUTION_M):
        from src.global_route_planner import GlobalRoutePlanner
        self._planner = GlobalRoutePlanner(world_map, float(sampling_resolution))
        self._pts = None          # (N, 2) route node positions
        self._suffix = None       # (N,) metres from node i to the goal
        self._idx = 0
        self.plan_error = None

    # -- planning ----------------------------------------------------------
    def reset(self, origin_location, goal_location):
        """Plan origin -> goal and precompute the suffix distances.

        Returns the total route length in metres, or 0.0 if no route exists.
        A goal off the drivable graph is signalled by a 0.0 length and a
        recorded plan_error rather than an exception: the caller falls back to
        no shaping, which is always safe because shaping is optional by
        construction.
        """
        self._pts, self._suffix, self._idx = None, None, 0
        self.plan_error = None
        try:
            route = self._planner.trace_route(origin_location, goal_location)
        except Exception as e:  # noqa: BLE001
            self.plan_error = f"{type(e).__name__}: {e}"
            return 0.0
        if not route:
            self.plan_error = "trace_route gave 0 waypoints"
            return 0.0

        pts = np.array([_xy(wp.transform.location) for wp, _ in route], np.float64)
        # THE GOAL IS APPENDED. trace_route ends at the nearest ROUTE NODE, not
        # at the goal itself -- up to one sampling_resolution short. Without
        # this the potential bottoms out at a non-zero value and the shaping
        # never fully pays out for arriving.
        goal_xy = _xy(goal_location)
        if np.linalg.norm(pts[-1] - goal_xy) > 1e-6:
            pts = np.vstack([pts, goal_xy])

        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)     # (N-1,)
        # suffix[i] = metres from node i onward. suffix[-1] == 0 at the goal.
        suffix = np.concatenate([np.cumsum(seg[::-1])[::-1], [0.0]])
        self._pts, self._suffix = pts, suffix
        return float(suffix[0])

    # -- query -------------------------------------------------------------
    def remaining(self, hero_xy):
        """Metres left along the route from the hero's current position.

        Returns None when no route is available, which the caller must treat as
        "no shaping this step" rather than as zero -- zero would read as
        "arrived" and pay out the entire potential.
        """
        if self._pts is None or self._suffix is None:
            return None
        hero = np.asarray(hero_xy, np.float64)[:2]
        if REMAINING_MODE == "project":
            return self._remaining_project(hero)

        # ADVANCE ONLY. Search a bounded window ahead of the pointer for the
        # closest node and move up to it; never move back. See the module note
        # on why a global nearest-node search is wrong.
        hi = min(len(self._pts), self._idx + ADVANCE_LOOKAHEAD + 1)
        window = self._pts[self._idx:hi]
        if len(window) == 0:
            return float(self._suffix[-1])
        j = int(np.argmin(np.linalg.norm(window - hero, axis=1)))
        self._idx += j

        # Distance to the goal = distance to the node we are abreast of, plus
        # that node's suffix. Using the straight-line hop to the node keeps the
        # measure continuous as the car moves between nodes, instead of the
        # potential stepping down 2 m at a time and paying the shaping in
        # spikes.
        hop = float(np.linalg.norm(self._pts[self._idx] - hero))
        return float(self._suffix[self._idx] + hop)


def _project_onto_segments(pts, hero):
    """Distances and clamped projection fractions of `hero` onto each segment
    pts[k] -> pts[k+1]. Vectorised; zero-length segments project to their start."""
    a, b = pts[:-1], pts[1:]
    ab = b - a
    l2 = np.einsum("ij,ij->i", ab, ab)
    t = np.where(l2 > 0.0,
                 np.einsum("ij,ij->i", hero - a, ab) / np.where(l2 > 0.0, l2, 1.0),
                 0.0)
    t = np.clip(t, 0.0, 1.0)
    proj = a + t[:, None] * ab
    return np.linalg.norm(hero - proj, axis=1), t, np.sqrt(l2)


def _remaining_project(self, hero):
    """ROUTE_REMAINING_MODE=project: along-route distance from the car's
    projection onto the segment it is abreast of, plus lateral distance beyond
    one lane width. The pointer still only advances (same bounded look-ahead as
    "hop"), for the same reason: a global search snaps backwards wherever the
    route passes close to itself."""
    pts, suffix = self._pts, self._suffix
    if len(pts) < 2:
        return float(np.linalg.norm(pts[-1] - hero))
    lo = min(self._idx, len(pts) - 2)
    hi = min(len(pts) - 1, lo + ADVANCE_LOOKAHEAD + 1)        # segments lo..hi-1
    dist, t, seg_len = _project_onto_segments(pts[lo:hi + 1], hero)
    k = int(np.argmin(dist))
    self._idx = lo + k
    along = float(suffix[self._idx] - t[k] * seg_len[k])
    return along + max(0.0, float(dist[k]) - LATERAL_DEADBAND_M)


RoutePotential._remaining_project = _remaining_project


class XYRoute:
    """RoutePotential driven by (x, y) pairs, for callers that store 2-D poses.

    WHY IT EXISTS (2026-09-26). The hindsight relabeller keeps positions as
    (x, y) only, but must measure along-road progress toward a HINDSIGHT goal
    with exactly the algorithm the goal wrapper uses toward the real one: plan
    once, then a forward-only pointer plus a straight hop (RoutePotential). A
    fresh plan per position would instead quantise to the 2 m route nodes and
    turn per-step progress into 0-then-2 m spikes.

    HEIGHTS MATTER. trace_route localises by the nearest waypoint in 3-D, and
    Town05 is NOT flat: planning from z=0 snapped onto the wrong road for 17% of
    plans (detours >3x the true lane distance, up to 111x), against 1% from the
    true 3-D location. Callers pass the recorded z; 0.0 is only a fallback.
    """

    def __init__(self, world_map, sampling_resolution=SAMPLING_RESOLUTION_M):
        import carla
        self._loc = carla.Location
        self._rp = RoutePotential(world_map, sampling_resolution)

    def reset(self, origin_xy, goal_xy, origin_z=0.0, goal_z=0.0):
        return self._rp.reset(
            self._loc(x=float(origin_xy[0]), y=float(origin_xy[1]),
                      z=float(origin_z or 0.0)),
            self._loc(x=float(goal_xy[0]), y=float(goal_xy[1]),
                      z=float(goal_z or 0.0)))

    def remaining(self, xy):
        return self._rp.remaining(xy)

    @property
    def plan_error(self):
        return self._rp.plan_error
