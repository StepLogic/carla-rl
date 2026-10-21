"""Make the goal camera the car's own camera, placed where the car would stand.

User 2026-10-21: "match the cameras". The image goal was rendered by a separate
sensor (rlib GoalImageSensor): 112x112 at CARLA's default 90 deg FOV, 2.0 m
above the road at the goal waypoint. The policy's view is the hero camera:
748x348 at 110 deg, 1.9 m forward and 1.7 m up from the car. Both are then
squashed to 120x120 by post_process_image, so view and goal differed in
field of view, aspect ratio and height. That broke two things:
  * HER's hindsight goals are VIEW frames, so they did not look like real goals;
  * DEBiT was trained (Habitat) on view/goal pairs from ONE camera, and its
    reading of where the goal is depends on that.

With GOAL_CAM_MATCH=1 the goal sensor gets the hero camera's resolution and
FOV (train_env_config_v4_image_goal), and its placement is overridden here to
the hero camera's mount applied at the goal: `fwd` metres ahead along the lane,
`up` metres above the road, same pitch. Arrival is measured from the car's
origin to the goal point, so a car that arrives sees almost exactly this frame.

The override replaces GoalImageSensor._compute_goal_transform in THIS process
only; rlib's shared carla_core is not edited, and with the flag off nothing is
patched.
"""
import math
import os

ENABLED = os.environ.get("GOAL_CAM_MATCH", "0") == "1"


def hero_mount(transform_str):
    """'x, y, z, roll?, pitch, yaw?' -> (fwd, up, pitch) from the hero rgb
    sensor's transform string, so the two cameras cannot drift apart."""
    v = [float(x) for x in str(transform_str).split(",")]
    # rlib transform strings are "x, y, z, roll, pitch, yaw"; the hero rgb is
    # "1.9, 0.0, 1.7, 0.0, -15.0, 0.0": pitch -15 is index 4.
    return v[0], v[2], v[4]


def matched_location(road_xyz, lane_yaw_deg, fwd, up):
    """The goal camera's location: `fwd` metres along the lane heading from the
    road point, `up` metres above it."""
    x, y, z = road_xyz
    yaw = math.radians(lane_yaw_deg)
    return x + fwd * math.cos(yaw), y + fwd * math.sin(yaw), z + up


def install(hero_rgb_transform):
    """Patch GoalImageSensor so it mounts like the hero camera. Idempotent."""
    import carla
    from rlib_integration.sensors.sensor import GoalImageSensor
    if getattr(GoalImageSensor, "_goal_cam_matched", False):
        return
    fwd, up, pitch = hero_mount(hero_rgb_transform)

    def _compute_goal_transform(self, transform):
        wp = self.town_map.get_waypoint(transform.location, project_to_road=True,
                                        lane_type=carla.LaneType.Driving)
        if wp is not None:
            loc, yaw = wp.transform.location, wp.transform.rotation.yaw
            road = (loc.x, loc.y, loc.z)
        else:
            # no lane: the incoming location (already raised 2 m by carla_core)
            yaw = transform.rotation.yaw
            road = (transform.location.x, transform.location.y,
                    transform.location.z - 2.0)
        x, y, z = matched_location(road, yaw, fwd, up)
        return carla.Transform(carla.Location(x, y, z),
                               carla.Rotation(pitch=pitch, yaw=yaw, roll=0.0))

    GoalImageSensor._compute_goal_transform = _compute_goal_transform
    GoalImageSensor._goal_cam_matched = True
