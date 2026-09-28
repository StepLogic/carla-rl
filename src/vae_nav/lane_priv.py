"""Privileged lane features for the asymmetric goal critic (OBS_PRIV=1).

WHY (user 2026-09-27). The Q-fault probe on v17@150k: the goal critic's Q gives
no warning before a fault (AUROC 0.50), while the true lane geometry from the
map predicts a fault 5-10 steps ahead at 0.81-0.82. The user's design: a
PRIVILEGED goal critic sees these features, is trained on the same minimum-time
reward, and drives the actor; the image-only goal critic is trained alongside
it on the same batches and is the one used for map edges. The actor never sees
this block (gated_tqc_learner strips it), so deployment needs no map.

Everything here is a property of the STATE, not of the goal, so hindsight
relabelling leaves it untouched.

Layout (PRIV_DIM = 13), in the lane's frame at the nearest driving waypoint:
    0  lateral offset from lane centre, metres, LEFT positive, clipped +-10
    1  |lateral| / (lane_width / 2)  -- > 1 means outside the lane, clipped 5
    2  sin(heading error)   hero yaw minus lane yaw
    3  cos(heading error)
    4  lane width, metres
    5  distance to the left lane edge, metres (negative = beyond it), clip +-10
    6  distance to the right lane edge, metres (same)
    7  left marking is illegal to cross   (solid / double / kerb / grass /
    8  right marking is illegal to cross   unknown -- the graded fine's set)
    9  left marking is unpainted (road edge)
   10  right marking is unpainted
   11  waypoint is inside a junction
   12  1.0 when a waypoint was found, else 0.0 (the rest are then zero)
"""
import math

import numpy as np

from vae_nav.lane_explore_reward import ILLEGAL_MARKS, UNPAINTED_MARKS

PRIV_KEY = "priv"
PRIV_DIM = 13


def _mark_name(marking):
    t = getattr(marking, "type", None) if marking is not None else None
    return str(t).split(".")[-1] if t is not None else ""


def lane_priv(hero_xy, hero_yaw_rad, wp):
    """The 13-d block from the hero pose and the nearest driving waypoint.

    `wp` is a carla.Waypoint (or anything with .transform.location.{x,y},
    .transform.rotation.yaw in degrees, .lane_width, .is_junction,
    .left_lane_marking.type, .right_lane_marking.type), or None.
    """
    out = np.zeros(PRIV_DIM, np.float32)
    if wp is None:
        return out
    tf = wp.transform
    wyaw = math.radians(float(tf.rotation.yaw))
    cy, sy = math.cos(wyaw), math.sin(wyaw)
    dx = float(hero_xy[0]) - float(tf.location.x)
    dy = float(hero_xy[1]) - float(tf.location.y)
    # CARLA is left-handed: right = (-sin, cos), so LEFT = -(d . right), the
    # same convention as vae_goal_wrapper.goal_block.
    lat = -(-dx * sy + dy * cy)
    half = max(0.1, float(wp.lane_width) / 2.0)
    herr = float(hero_yaw_rad) - wyaw
    herr = (herr + math.pi) % (2 * math.pi) - math.pi
    lm, rm = _mark_name(wp.left_lane_marking), _mark_name(wp.right_lane_marking)
    out[:] = [
        np.clip(lat, -10.0, 10.0),
        min(5.0, abs(lat) / half),
        math.sin(herr), math.cos(herr),
        float(wp.lane_width),
        np.clip(half - lat, -10.0, 10.0),
        np.clip(half + lat, -10.0, 10.0),
        float(lm == "" or lm in ILLEGAL_MARKS),
        float(rm == "" or rm in ILLEGAL_MARKS),
        float(lm in UNPAINTED_MARKS),
        float(rm in UNPAINTED_MARKS),
        float(bool(wp.is_junction)),
        1.0,
    ]
    return out


def lane_priv_from_map(world_map, location, hero_yaw_rad):
    """Query the map at a carla.Location and build the block."""
    import carla
    wp = world_map.get_waypoint(location, project_to_road=True,
                                lane_type=carla.LaneType.Driving)
    return lane_priv((location.x, location.y), hero_yaw_rad, wp)


def strip_priv(x):
    """Drop the privileged block from an observation dict, or from the
    observations / next_observations of a batch dict. Anything else is
    returned unchanged. The actor and the image-only critics are only ever
    called through this."""
    if not hasattr(x, "keys"):
        return x
    if PRIV_KEY in x:
        return {k: v for k, v in x.items() if k != PRIV_KEY}
    if "observations" in x and hasattr(x["observations"], "keys"):
        y = dict(x)
        for k in ("observations", "next_observations"):
            if k in y and hasattr(y[k], "keys") and PRIV_KEY in y[k]:
                y[k] = {kk: v for kk, v in y[k].items() if kk != PRIV_KEY}
        return y
    return x
