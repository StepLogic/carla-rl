"""Junction zones and chained goals (user 2026-11-02).

The run: LANE FOLLOWING is the reward everywhere, and the GOAL-CONDITIONED
reward (route progress toward the goal, v30's 0.3 retreat charge) applies only
in a JUNCTION ZONE -- inside a junction and `zone_m` metres either side of it,
where the choice of branch is made. On each arrival a new goal is drawn
`min_m`..`max_m` past the NEXT junction from where the car stands, until the
chain has covered GOAL_CHAIN_M metres.

Pure helpers over the CARLA map interface (map.get_waypoint, wp.next(d),
wp.is_junction, wp.transform), so they are testable with a fake waypoint graph.
"""
import numpy as np

STEP_M = 2.0


def _wp(world_map, location):
    try:
        import carla
        return world_map.get_waypoint(location, project_to_road=True,
                                      lane_type=carla.LaneType.Driving)
    except ImportError:          # tests: a fake map
        return world_map.get_waypoint(location)


def junction_ahead_m(world_map, location, lookahead_m):
    """Metres along the lane to the next junction waypoint (0 inside one), or
    None if there is none within `lookahead_m`. At a fork before any junction
    the first successor is followed (lane splits outside junctions are rare)."""
    wp = _wp(world_map, location)
    if wp is None:
        return None
    if wp.is_junction:
        return 0.0
    d = 0.0
    while d < lookahead_m:
        nx = wp.next(STEP_M)
        if not nx:
            return None
        wp, d = nx[0], d + STEP_M
        if wp.is_junction:
            return d
    return None


def goal_past_next_junction(world_map, location, rng, min_m=10.0, max_m=30.0,
                            lookahead_m=150.0):
    """-> (waypoint, stats). Walk forward to the next junction (a random branch
    at every fork), through it, then a distance drawn uniformly from
    [min_m, max_m] beyond its exit. With no junction within `lookahead_m` the
    distance is drawn from the current position instead. A goal is never left
    inside a junction: the walk continues out of one it lands in. (None, stats)
    on a dead end."""
    wp = _wp(world_map, location)
    stats = {"through_junction": False, "to_junction_m": None,
             "past_junction_m": None, "metres": 0.0}
    if wp is None:
        return None, stats

    def step(w):
        nx = w.next(STEP_M)
        if not nx:
            return None
        return nx[int(rng.integers(len(nx)))] if len(nx) > 1 else nx[0]

    d = 0.0
    probe, pd = wp, 0.0
    while not probe.is_junction and pd < lookahead_m:   # find the junction
        probe = step(probe)
        if probe is None:
            break
        pd += STEP_M
    if probe is not None and probe.is_junction:
        wp, d = probe, pd
        while wp is not None and wp.is_junction:          # through it
            wp = step(wp)
            d += STEP_M
        if wp is None:
            return None, stats
        stats.update(through_junction=True, to_junction_m=pd)
    target = float(rng.uniform(min_m, max_m))
    past = 0.0
    while past < target or wp.is_junction:
        wp = step(wp)
        if wp is None:
            return None, stats
        past += STEP_M
    stats.update(past_junction_m=past, metres=d + past)
    return wp, stats


class JunctionZone:
    """In the zone = inside a junction, OR one is within `zone_m` ahead, OR the
    car left one at most `zone_m` of driven path ago."""

    def __init__(self, zone_m):
        self.zone_m = float(zone_m)
        self.reset()

    def reset(self):
        self._last_in_m = -np.inf

    def update(self, path_m, in_junction, ahead_m):
        if in_junction:
            self._last_in_m = float(path_m)
            return True
        if ahead_m is not None and ahead_m <= self.zone_m:
            return True
        return float(path_m) - self._last_in_m <= self.zone_m
