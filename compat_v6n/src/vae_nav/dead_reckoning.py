"""Pose from the vehicle's own senses, so the visit ring stops being privileged.

WHAT WAS WRONG WITH TRUE POSE. The 16-slot visit ring is computed against a
cell grid indexed by the simulator's transform. A real vehicle has GPS and an
IMU, so having *a* pose is not the cheat -- having a DRIFT-FREE one is. The
Habitat peer reached the same conclusion and moved the same way: their spec
records pose as "truth in v1, dead-reckoned in v3".

Note what is and is not privileged here, because the two get conflated. The
ROAD GRAPH -- waypoints, is_junction, lane topology -- is genuinely privileged:
it is the map the experiment is trying to build, so exploring with it assumes
the answer. Ego pose is not in that category. It is merely optimistic.

THE MODEL. Kinematic bicycle, integrated from proprioception only:

    v         = speed_kmh / 3.6                    wheel odometry
    delta     = steer * max_steer_angle            steering command
    yaw_rate  = (v / wheelbase) * tan(delta)
    yaw      += yaw_rate * dt
    x        += v * cos(yaw) * dt                  CARLA is LEFT-HANDED:
    y        += v * sin(yaw) * dt                  forward = (cos yaw, sin yaw)

No exteroceptive sensing at all -- no camera, no lidar, no map. That is the
strictest form of dead reckoning and therefore the honest hard case: heading
error integrates without bound, and position error integrates on top of it.

ONE PARAMETER, NOT TWO. `wheelbase` and `max_steer_angle` are not separately
identifiable from (speed, steer, yaw-rate) data -- for small angles the yaw
rate depends only on their ratio:

    yaw_rate ~= (max_steer_angle / wheelbase) * v * steer

so `calibrate()` fits that single gain by least squares and leaves the tan()
form for large-angle fidelity. Guessing both separately would invite two wrong
numbers whose ratio happens to be right, which is harder to debug than one
fitted constant.

THE DEFAULTS ARE A STARTING POINT, NOT A MEASUREMENT. vehicle.micro.microlino
is a very short car; the defaults below are plausible for it and have NOT been
fitted. Run calibrate() on a recorded trace before trusting any drift figure --
drift is what this module exists to expose, and an unfitted gain manufactures
it.
"""
import os

import numpy as np

DT = float(os.environ.get("SIM_TIMESTEP", 0.1))
# Plausible for a microlino; see the docstring -- fit these, do not trust them.
WHEELBASE_M = float(os.environ.get("DR_WHEELBASE_M", 1.5))
MAX_STEER_RAD = float(os.environ.get("DR_MAX_STEER_RAD", 1.22))   # ~70 degrees


class DeadReckoner:
    """Integrates pose from speed and steering. No exteroceptive input."""

    def __init__(self, wheelbase=WHEELBASE_M, max_steer=MAX_STEER_RAD, dt=DT):
        self.wheelbase = float(wheelbase)
        self.max_steer = float(max_steer)
        self.dt = float(dt)
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self._n = 0

    @property
    def yaw_gain(self):
        """The single identifiable parameter: max_steer / wheelbase."""
        return self.max_steer / self.wheelbase

    def reset(self, pos=(0.0, 0.0), yaw=0.0):
        """Anchor at the episode start.

        A real vehicle knows where it started in its own frame; drift is what
        accumulates AFTER that. Anchoring to the true start pose is therefore
        not cheating -- resetting per episode is also what bounds the error,
        which is why the peer's visited set resets per episode too.
        """
        self.x, self.y = float(pos[0]), float(pos[1])
        self.yaw = float(yaw)
        self._n = 0

    def step(self, speed_kmh, steer):
        """Advance one timestep. -> (x, y, yaw)."""
        v = float(speed_kmh) / 3.6
        delta = float(steer) * self.max_steer
        yaw_rate = (v / self.wheelbase) * np.tan(delta)
        # Integrate heading FIRST, then translate along it. Translating on the
        # stale heading lags every turn and biases the path outward.
        self.yaw += yaw_rate * self.dt
        self.yaw = float(np.arctan2(np.sin(self.yaw), np.cos(self.yaw)))
        self.x += v * np.cos(self.yaw) * self.dt
        self.y += v * np.sin(self.yaw) * self.dt
        self._n += 1
        return self.x, self.y, self.yaw

    @property
    def pos(self):
        return np.array([self.x, self.y], np.float64)

    def drift(self, true_pos, true_yaw):
        """Error against ground truth. THE number this module exists to report."""
        d = float(np.linalg.norm(self.pos - np.asarray(true_pos, np.float64)))
        dyaw = float(np.arctan2(np.sin(self.yaw - true_yaw),
                                np.cos(self.yaw - true_yaw)))
        return {"position_m": d, "heading_rad": abs(dyaw), "steps": self._n}


def calibrate(speeds_kmh, steers, yaws, dt=DT):
    """Least-squares fit of the yaw gain from a recorded trace.

    Fits `yaw_rate = gain * v * steer` on the small-angle form, which is the
    only combination of wheelbase and max_steer the data can identify.

    Returns the gain, and the residual std so a bad fit is visible rather than
    silently applied.
    """
    v = np.asarray(speeds_kmh, np.float64) / 3.6
    s = np.asarray(steers, np.float64)
    yaw = np.asarray(yaws, np.float64)
    if v.size < 3:
        raise ValueError("need at least 3 samples to fit a gain")

    # Unwrap before differencing, or every wrap at +-pi becomes a huge fake
    # yaw rate and drags the fit.
    yaw_rate = np.diff(np.unwrap(yaw)) / dt
    x = (v * s)[:-1]
    ok = np.abs(x) > 1e-6
    if ok.sum() < 3:
        raise ValueError("trace has no turning at speed; gain unidentifiable")

    gain = float(np.sum(x[ok] * yaw_rate[ok]) / np.sum(x[ok] ** 2))
    resid = yaw_rate[ok] - gain * x[ok]
    return {"yaw_gain": gain,
            "residual_std": float(np.std(resid)),
            "n": int(ok.sum())}
