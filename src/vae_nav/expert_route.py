"""Explore-mode routes for the expert, and the off-route test (2026-11-04).

Imitation-first plan, Task 2. In explore mode (mode bit 0) the policy is shown
no goal that decides which way to go, so the expert must not choose branches
at random: the clone would see conflicting labels at the same view, and a
mean-fitting actor averages left and right at a T-junction into straight
ahead, into the kerb. The rule is fixed instead:

  * take the exit with the smallest heading change;
  * where no exit is within `straight_tol_deg` of straight (a T-junction),
    turn right: a positive yaw change in CARLA's left-handed frame (x forward,
    y right, yaw growing clockwise seen from above);
  * where there is no right turn either, take the smallest turn.

An exit's heading is read where the branch LEAVES the junction (up to
`exit_lookahead_m` along it): CARLA's first in-junction waypoint of a turning
lane still points nearly straight, so the heading 2 m in would make every exit
look straight.

Pure functions over the CARLA waypoint interface (wp.next(d), wp.is_junction,
wp.transform.location, wp.transform.rotation.yaw), testable with a fake graph.
"""
import numpy as np

JUNCTION_OVERRUN_M = 40.0      # how far past `length_m` to walk out of a junction


def _yaw(wp):
    return float(wp.transform.rotation.yaw)


def _dyaw(a, b):
    """b - a in degrees, wrapped to (-180, 180]."""
    d = (float(b) - float(a) + 180.0) % 360.0 - 180.0
    return 180.0 if d == -180.0 else d


def _exit_yaw(wp, step_m, lookahead_m):
    """Heading of the branch starting at `wp`, read where it leaves the
    junction (or after `lookahead_m`, or at a dead end)."""
    d = 0.0
    while wp.is_junction and d < lookahead_m:
        nxt = wp.next(step_m)
        if not nxt:
            break
        wp, d = nxt[0], d + step_m
    return _yaw(wp)


def _pick(cur, candidates, step_m, straight_tol_deg, lookahead_m):
    if len(candidates) == 1:
        return candidates[0]
    here = _yaw(cur)
    turns = [(_dyaw(here, _exit_yaw(w, step_m, lookahead_m)), w) for w in candidates]
    straight = [t for t in turns if abs(t[0]) <= straight_tol_deg]
    if straight:
        return min(straight, key=lambda t: abs(t[0]))[1]
    right = [t for t in turns if t[0] > 0.0]
    if right:
        return min(right, key=lambda t: abs(t[0]))[1]
    return min(turns, key=lambda t: abs(t[0]))[1]


def explore_route(start_wp, length_m, step_m=2.0, straight_tol_deg=30.0,
                  exit_lookahead_m=15.0):
    """Waypoints from `start_wp` (inclusive) for `length_m` metres along lanes,
    branching by the fixed rule above. Never ends inside a junction (walks on
    out of one, at most JUNCTION_OVERRUN_M further); returns what it has at a
    dead end."""
    route, wp, covered = [start_wp], start_wp, 0.0
    while True:
        if covered >= length_m and (not wp.is_junction
                                    or covered >= length_m + JUNCTION_OVERRUN_M):
            break
        nxt = wp.next(step_m)
        if not nxt:
            break
        wp = _pick(wp, nxt, step_m, straight_tol_deg, exit_lookahead_m)
        route.append(wp)
        covered += step_m
    return route


def needs_replan(route_xy, xy, max_off_m):
    """True when the nearest route point is farther than `max_off_m` from `xy`,
    or the route is empty."""
    r = np.asarray(route_xy, np.float64).reshape(-1, 2)
    if len(r) == 0:
        return True
    p = np.asarray(xy, np.float64)[:2]
    return bool(np.min(np.hypot(r[:, 0] - p[0], r[:, 1] - p[1])) > float(max_off_m))
