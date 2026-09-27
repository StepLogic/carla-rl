import os
"""Populates the `info` dict that the VAE/sparse-reward/HER stack reads.

CarlaEnv returns `self.info = dict()` -- empty. Everything downstream needs
(hero pose, goal position, collision flag, camera frame) lives on `core` and on
the experiment object instead, reachable only by reaching through the env.

This wrapper is the ONE place that reaching happens. Putting it here rather
than in each consumer means SparseGoalReward, VAEGoalWrapper and the HER
recorder all read plain dict keys, and a change to CarlaEnv's internals is a
one-file fix rather than a hunt.

It also means the shared train_env_config_v4_image_goal.py is left untouched --
the CORL 2026 experiments still run on it, and adding fields to its info dict
would be a change to code those results depend on.

FAILS LOUDLY, NOT QUIETLY. A missing hero or destination raises. The
alternative is a goal_pos silently defaulting to the origin, which yields a
reward that is wrong everywhere and looks merely difficult.
"""
import math

import numpy as np

try:
    from gymnasium import Wrapper
except ImportError:
    from gym import Wrapper


def _location_of(obj, what):
    """CARLA is inconsistent about Transform vs Waypoint; accept either."""
    if obj is None:
        raise SystemExit(
            f"{what} is None on the CARLA core. The sparse reward and the "
            "expert oracle both need it; refusing to train against a goal "
            "that does not exist.")
    if hasattr(obj, "location"):            # carla.Transform
        return obj.location
    if hasattr(obj, "transform"):           # carla.Waypoint
        return obj.transform.location
    if hasattr(obj, "x"):                   # carla.Location
        return obj
    raise SystemExit(f"cannot read a Location from {what}={type(obj).__name__}")


class CoreStateWrapper(Wrapper):
    """Injects hero pose, goal, collision flag and camera frame into `info`."""

    def __init__(self, env):
        super().__init__(env)
        self._last_action = np.zeros(2, np.float32)

    def _core(self):
        base = getattr(self.env, "unwrapped", self.env)
        core = getattr(base, "core", None)
        if core is None:
            raise SystemExit(
                "CoreStateWrapper must wrap a CarlaEnv (no .core found).")
        return core, getattr(base, "experiment", None)

    def _fill(self, info):
        core, experiment = self._core()

        hero = getattr(core, "hero", None)
        if hero is None:
            raise SystemExit("no hero actor on the CARLA core.")
        tf = hero.get_transform()
        goal_loc = _location_of(getattr(core, "destination", None), "core.destination")

        info = dict(info or {})
        info["hero_pos"] = np.array([tf.location.x, tf.location.y], np.float64)
        # HEIGHT, for the hindsight relabeller's route planner (2026-09-26):
        # planning from a 2-D pose at z=0 snapped onto the wrong road for 17% of
        # Town05 plans (detours >3x, up to 111x), against 1% from the true 3-D
        # location. Additive: nothing else reads it.
        info["hero_z"] = float(tf.location.z)
        # Radians. CARLA reports yaw in DEGREES; feeding those to sin/cos in the
        # goal block would silently scramble every bearing.
        info["hero_yaw"] = math.radians(float(tf.rotation.yaw))
        info["goal_pos"] = np.array([goal_loc.x, goal_loc.y], np.float64)
        info["goal_location"] = goal_loc          # the expert oracle wants the
                                                  # carla.Location itself
        # The experiment computes this as `'collision' in sensor_data.keys()`,
        # which is already a genuine bool. Coerced anyway: the HER buffer and
        # the sparse reward both reject a non-bool, because a truthy dict here
        # marks every step a collision.
        info["collision"] = bool(getattr(experiment, "collision", False))
        # Lane departure is a task failure of the same kind as a collision --
        # terminal and penalised -- so it is surfaced the same way and coerced
        # to a genuine bool for the same reason.
        # ILLEGAL crossings only. CARLA's lane-invasion sensor fires on ANY
        # marking crossed, including a Broken (dashed) centre line -- which is
        # legal driving, and which a route-following vehicle MUST cross to turn
        # at a junction. MEASURED with the unfiltered flag: the expert oracle
        # drove at exactly its 0.20 target and still scored 0 successes over 30
        # episodes, because it was terminated at -5 mid-turn every time.
        #
        # Solid / SolidSolid are real violations; Curb and Grass mean the
        # vehicle left the road entirely. Broken, BrokenBroken and BottsDots
        # are legal to cross.
        # ARRIVAL COMES FROM THE ENV, NOT FROM RECOMPUTING THE DISTANCE.
        #
        # get_done_status does this in a single step:
        #     if distance_to_goal < 2:
        #         self.passed_chckp = True
        #         core.resample_goal()        # <- the goal MOVES
        #
        # so by the time this wrapper reads core.destination the goal is
        # already the NEXT one, metres away, and a distance recomputed here can
        # never be under the arrival radius. The reward was literally
        # unwinnable: the oracle drove to the goal correctly and the arrival was
        # erased on the same step, every time -- 0 successes over hundreds of
        # episodes. `passed_chckp` is the env's own flag, set at the moment of
        # arrival and before the resample.
        info["arrived"] = bool(getattr(experiment, "passed_chckp", False))

        # WHY the episode ended. The env ORs seven conditions into one `done`
        # and reports none of them, so "29 of 30 episodes returned exactly 0"
        # was the entire symptom. Guessing which one fires has cost several
        # cycles today; this turns it into a measurement.
        info["done_reasons"] = {
            name: bool(getattr(experiment, attr, False))
            for name, attr in (("idle", "done_time_idle"),
                               ("falling", "done_falling"),
                               ("dist200", "done_dist"),
                               ("lane_any", "diff_lane"),
                               ("collision", "collision"),
                               ("speed", "done_speed"),
                               ("wrong_dir", "done_wrong_direction"),
                               ("off_road", "off_road"))}

        raw_lane = bool(getattr(experiment, "diff_lane", False))
        types = list(getattr(experiment, "lane_marking_types", []) or [])
        illegal = {"Solid", "SolidSolid", "SolidBroken", "Curb", "Grass"}
        info["lane_marking_types"] = types
        info["lane_invasion_any"] = raw_lane

        # OFF-ROAD IS A LANE INVASION, and it is the stronger of the two
        # signals. The sensor reports the EVENT of crossing a painted marking;
        # off_road reports the STATE of being on no drivable lane. They fail in
        # opposite directions:
        #
        #   - the sensor misses departures where nothing is painted (grass
        #     shoulder, median, kerb-hop) and cannot tell you the car is STILL
        #     out of its lane fifty steps later;
        #   - the state flag cannot distinguish a legal dashed-line turn from
        #     an illegal solid-line crossing, because during both the car is on
        #     a perfectly valid lane.
        #
        # So the marking filter applies ONLY to the sensor event, and off_road
        # is ORed in unconditionally -- there is no such thing as a legal
        # excursion onto no lane at all.
        # Speed in km/h, and the env's own target, for the progress-reward gate.
        # experiment.velocity is set every step by get_done_status; reading it
        # here rather than recomputing avoids a second get_velocity() RPC and
        # guarantees the reward sees the same value the env's done-conditions
        # did.
        # THE APPLIED CONTROL, not the action. Under INCREMENTAL_ACTIONS the
        # action is a DELTA on the control, so an action history made of
        # actions records rates of change and reveals neither the throttle
        # level nor, through it, the vehicle's speed. compute_action caches the
        # control it actually sent to CARLA; that is the quantity from which
        # velocity is inferable.
        info["applied_steer"] = float(getattr(experiment, "steer", 0.0) or 0.0)
        info["applied_throttle"] = float(getattr(experiment, "throttle", 0.0) or 0.0)

        info["speed_kmh"] = float(getattr(experiment, "velocity", 0.0) or 0.0)
        info["target_speed_kmh"] = float(getattr(experiment, "target_speed", 0.0) or 0.0)

        off_road = bool(getattr(experiment, "off_road", False))
        sensor_illegal = bool(
            raw_lane and (not types or any(
                t.split(".")[-1] in illegal for t in types)))
        info["off_road"] = off_road
        info["lane_invasion_illegal"] = sensor_illegal

        # STRICT BY DEFAULT: ANY crossing is a lane-keeping failure.
        #
        # The marking-type filter above exists for exactly one reason -- a
        # route-following ORACLE cannot turn at a junction without crossing a
        # dashed centre line, and terminating on that made it impossible for
        # the expert to reach a checkpoint. With EXPERT_PROB=0 there is no
        # oracle, so the exemption now only hides the signal: MEASURED on the
        # live run, lane_any fired in 15% of episodes and none of them ended,
        # because crossing a Broken line was classified legal.
        #
        # The task is goal following WITH LANE KEEPING. Under that definition a
        # dashed line the agent was not asked to cross is still a departure, so
        # the default is strict and the filter is opt-in.
        #
        # THE COST IS REAL AND WAS MEASURED: with every crossing terminal, an
        # earlier run ended 90% of its episodes on lane_any at ~44-70 steps.
        # Short episodes mean less exploration per episode and fewer goals
        # reached, so if this starves learning, LANE_STRICT=0 restores the
        # filter without touching anything else.
        strict = os.environ.get("LANE_STRICT", "1") == "1"
        info["lane_invasion"] = bool(
            (raw_lane if strict else sensor_illegal) or off_road)

        # The full-resolution frame is not retained by CarlaEnv, so the VAE gets
        # what the env kept: obs_img, already through post_process_image. That
        # is exactly what vae_nav/preprocess.py makes collection reproduce.
        obs_img = getattr(experiment, "obs_img", None)
        if obs_img is None:
            raise SystemExit(
                "experiment.obs_img is None -- the VAE has no frame to encode. "
                "Check that the env config's get_img_obs ran this step.")
        info["rgb"] = obs_img
        info["action_history"] = self._last_action
        return info

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        obs, info = out if isinstance(out, tuple) else (out, {})
        self._last_action = np.zeros(2, np.float32)
        return obs, self._fill(info)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._last_action = np.asarray(action, np.float32)[:2]
        return obs, reward, terminated, truncated, self._fill(info)
