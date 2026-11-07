"""Drive CARLA from ViNT's predictions, for run_eval (EVAL_POLICY=vint).

User 2026-11-07: train ViNT on c0's expert data and evaluate it like c0. ViNT
(Shah et al. 2023) predicts 5 waypoints and the temporal distance to the goal
image; it predicts no controls. They are executed by the EXPERT'S OWN low-level
controller -- the vendored CARLA LocalPlanner defaults that BasicAgentExpert
drove every training episode with (dependencies/carla_agents/.../controller.py
and local_planner.py) -- so ViNT is judged on what it predicts, not on a
controller tuned for it:

  lateral       PID 1.95 / 0.05 / 0.2, dt 0.05, on the signed angle to the
                target (+ = right, CARLA's convention); steering moves at most
                0.1 per step and stays within +-0.8 (VehiclePIDController);
  target        the first predicted waypoint at least 3 m + 0.5 s x speed away
                (LocalPlanner's purge distance), else the last one;
  longitudinal  PID 1.0 / 0.05 / 0 toward BasicAgentExpert's approach profile
                (cruise, slowing from 10 m to 6 km/h at 3 m), fed with ViNT's
                PREDICTED goal distance where the expert used the true one;
  action        throttle <= 0.75, brake <= 0.3 carried as a negative throttle
                (ExpertOracle._to_action), clipped to the env's box.

ViNT's frame is x forward, y LEFT (vae_nav.vint_data flipped CARLA's y for
training), so a target at y > 0 steers left (negative).
"""
import json
import math
import os
from collections import deque

import numpy as np

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class PID:
    """CARLA's PID (agents/navigation/controller.py): 10-error memory, finite
    differences over dt, output clipped to [-1, 1]."""

    def __init__(self, k_p, k_i, k_d, dt=0.05):
        self.k_p, self.k_i, self.k_d, self.dt = float(k_p), float(k_i), float(k_d), float(dt)
        self.errors = deque(maxlen=10)

    def step(self, error):
        self.errors.append(float(error))
        if len(self.errors) >= 2:
            de = (self.errors[-1] - self.errors[-2]) / self.dt
            ie = sum(self.errors) * self.dt
        else:
            de = ie = 0.0
        return float(np.clip(self.k_p * error + self.k_d * de + self.k_i * ie, -1.0, 1.0))


def signed_angle(target_xy):
    """PIDLateralController's error: the angle between the car's heading and the
    vector to the target, positive when the target is to the RIGHT. In ViNT's
    frame (y left) that is atan2(-y, x)."""
    x, y = float(target_xy[0]), float(target_xy[1])
    if x == 0.0 and y == 0.0:
        return 0.0
    return math.atan2(-y, x)


def pick_waypoint(waypoints_m, min_dist_m):
    """The first waypoint at least min_dist_m away, else the last."""
    w = np.asarray(waypoints_m, np.float64)
    far = np.flatnonzero(np.hypot(w[:, 0], w[:, 1]) >= float(min_dist_m))
    return w[far[0]] if len(far) else w[-1]


class ViNTDriver:
    """Context frames in, env action out. `policy(ctx_frames, goal_frame)` ->
    (goal distance in metres, (L, 2) waypoints in metres, ViNT frame)."""

    drives_from_pixels = True

    def __init__(self, policy, low, high, *, context_size=5, stride=3, cruise_kmh=15.0,
                 lateral=(1.95, 0.05, 0.2), longitudinal=(1.0, 0.05, 0.0), dt=0.05,
                 max_steer=0.8, max_throttle=0.75, max_brake=0.3,
                 base_lookahead_m=3.0, lookahead_s=0.5):
        self.policy = policy
        self.low = np.asarray(low, np.float32)
        self.high = np.asarray(high, np.float32)
        self.context_size, self.stride = int(context_size), int(stride)
        self.cruise_kmh = float(cruise_kmh)
        self._lat_gains, self._lon_gains, self.dt = lateral, longitudinal, float(dt)
        self.max_steer, self.max_throttle, self.max_brake = max_steer, max_throttle, max_brake
        self.base_lookahead_m, self.lookahead_s = float(base_lookahead_m), float(lookahead_s)
        self.frames = deque(maxlen=self.context_size * self.stride + 1)
        self.last = {}
        self.reset_hidden(1)

    def reset_hidden(self, n=1):
        """A new episode: no frame history, fresh controllers, wheels straight."""
        self.frames.clear()
        self.lat = PID(*self._lat_gains, dt=self.dt)
        self.lon = PID(*self._lon_gains, dt=self.dt)
        self.past_steer = 0.0
        return None

    def context(self):
        """context_size past frames `stride` steps apart, then the current one;
        before the history is long enough the oldest frame repeats (the car is
        at rest at an episode start)."""
        n = len(self.frames)
        return [self.frames[max(0, n - 1 - j * self.stride)]
                for j in range(self.context_size, -1, -1)]

    def act_env(self, cur_frame, goal_frame, speed_kmh):
        from vae_nav.expert_oracle import BasicAgentExpert
        self.frames.append(np.asarray(cur_frame))
        dist_m, wps = self.policy(self.context(), np.asarray(goal_frame))
        speed = max(0.0, float(speed_kmh))
        target = pick_waypoint(wps, self.base_lookahead_m + self.lookahead_s * speed / 3.6)
        angle = signed_angle(target)
        steer = self.lat.step(angle)
        steer = min(max(steer, self.past_steer - 0.1), self.past_steer + 0.1)
        steer = max(-self.max_steer, min(self.max_steer, steer))
        self.past_steer = steer
        target_kmh = BasicAgentExpert.approach_speed_kmh(max(0.0, float(dist_m)), self.cruise_kmh)
        acc = self.lon.step(target_kmh - speed)
        throttle = min(acc, self.max_throttle) if acc >= 0.0 else -min(-acc, self.max_brake)
        action = np.clip(np.array([steer, throttle], np.float32), self.low, self.high)
        self.last = {"dist_m": float(dist_m), "target": [float(target[0]), float(target[1])],
                     "angle": angle, "target_kmh": float(target_kmh),
                     "steer": float(action[0]), "throttle": float(action[1])}
        return action


def load_vint_policy(ckpt_dir, device="cuda"):
    """EVAL_FROM=<dir> holding model.pth (a ViNT training checkpoint, {"model":
    ViNT, ...}) and vint.json (architecture, image_size, stride, metric_step_m,
    as trained). -> (policy for ViNTDriver, meta). Needs vint_train and
    efficientnet_pytorch on the path (slurm/vint_eval_pair.sh)."""
    import torch
    from PIL import Image
    from torchvision import transforms
    from vint_train.data.data_utils import resize_and_aspect_crop
    from vint_train.models.vint.vint import ViNT

    with open(os.path.join(ckpt_dir, "vint.json")) as fh:
        meta = json.load(fh)
    model = ViNT(context_size=meta["context_size"], len_traj_pred=meta["len_traj_pred"],
                 learn_angle=meta["learn_angle"], obs_encoder=meta["obs_encoder"],
                 obs_encoding_size=meta["obs_encoding_size"], late_fusion=meta["late_fusion"],
                 mha_num_attention_heads=meta["mha_num_attention_heads"],
                 mha_num_attention_layers=meta["mha_num_attention_layers"],
                 mha_ff_dim_factor=meta["mha_ff_dim_factor"])
    ck = torch.load(os.path.join(ckpt_dir, "model.pth"), map_location="cpu", weights_only=False)
    src = ck["model"]
    src = getattr(src, "module", src)
    model.load_state_dict(src.state_dict(), strict=True)
    model.to(device).eval()
    norm = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    size = tuple(int(v) for v in meta["image_size"])
    scale = float(meta["metric_step_m"]) * int(meta["stride"])

    def prep(frame):
        return norm(resize_and_aspect_crop(Image.fromarray(np.asarray(frame, np.uint8)), size))

    @torch.no_grad()
    def policy(ctx, goal):
        obs = torch.cat([prep(f) for f in ctx], dim=0)[None].to(device)
        g = prep(goal)[None].to(device)
        dist, act = model(obs, g)
        return float(dist.reshape(-1)[0]) * scale, act[0, :, :2].float().cpu().numpy() * scale

    return policy, meta
