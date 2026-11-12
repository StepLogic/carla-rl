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


def oracle_aim_points(lane_priv, spacing=1.1346, n=5):
    """PRIVILEGED diagnostic (VINT_ORACLE_AIM=1; development set only, never a
    result): the true lane-centre aim points j * spacing ahead along a straight
    lane, in ViNT's frame (x forward, y left), from the env's lane state
    (vae_nav.lane_priv: [0] offset LEFT positive, [2]/[3] sin/cos of hero yaw -
    lane yaw in CARLA's clockwise yaw, [12] 1.0 when a lane was found). Driving the
    controller on these isolates it from ViNT's predictions. None without a lane."""
    lp = np.asarray(lane_priv, np.float64).reshape(-1)
    if lp.shape[0] < 13 or lp[12] < 0.5:
        return None
    off = float(lp[0])
    h_left = -math.atan2(float(lp[2]), float(lp[3]))   # car vs lane, counter-clockwise +
    c, s = math.cos(h_left), math.sin(h_left)
    d = spacing * np.arange(1, n + 1, dtype=np.float64)
    return np.stack([c * d - s * off, -s * d - c * off], 1)


def approach_kmh_from_time(t_s, cruise_kmh, final_kmh=None, slow_from_m=None, final_at_m=None):
    """BasicAgentExpert's approach profile, indexed by TIME to the goal.

    ViNT's distance head predicts temporal distance -- frames, learned from the
    expert slowing down into its goals -- not metres. The expert's target falls
    linearly with distance, from cruise at slow_from_m to final_kmh at final_at_m,
    and stays at final_kmh inside. Driving that profile, the time left at
    distance d is T(d) = final_at_m / v_f inside, plus ln(v(d) / v_f) / k on the
    ramp (k = dv/ds); inverting T gives the speed for a predicted time, so a
    frame count is never read as metres at cruise speed (which near the goal,
    where the expert is slow, would arrive too fast)."""
    from vae_nav.expert_oracle import BasicAgentExpert as E
    cruise = float(cruise_kmh) / 3.6
    final = min(E.FINAL_APPROACH_KMH if final_kmh is None else float(final_kmh),
                float(cruise_kmh)) / 3.6
    lo = E.APPROACH_FINAL_AT_M if final_at_m is None else float(final_at_m)
    hi = E.APPROACH_SLOW_FROM_M if slow_from_m is None else float(slow_from_m)
    t = max(0.0, float(t_s))
    t_inside = lo / final
    if t <= t_inside or cruise <= final:
        return final * 3.6 if t <= t_inside else cruise * 3.6
    k = (cruise - final) / (hi - lo)
    return min(final * math.exp(k * (t - t_inside)), cruise) * 3.6


def pick_waypoint(waypoints_m, min_dist_m):
    """The first waypoint at least min_dist_m away, else the last."""
    w = np.asarray(waypoints_m, np.float64)
    far = np.flatnonzero(np.hypot(w[:, 0], w[:, 1]) >= float(min_dist_m))
    return w[far[0]] if len(far) else w[-1]


class ViNTDriver:
    """Context frames in, env action out. `policy(ctx_frames, goal_frame)` ->
    (goal distance in ViNT steps, (L, 2) waypoints in metres, ViNT frame); one
    ViNT step is `step_s` seconds (stride x the 0.1 s frame interval)."""

    drives_from_pixels = True

    def __init__(self, policy, low, high, *, context_size=5, stride=3, step_s=0.3, cruise_kmh=15.0,
                 lateral=(1.95, 0.05, 0.2), longitudinal=(1.0, 0.05, 0.0), dt=0.05,
                 max_steer=0.8, max_throttle=0.75, max_brake=0.3,
                 base_lookahead_m=3.0, lookahead_s=0.5):
        self.policy = policy
        self.low = np.asarray(low, np.float32)
        self.high = np.asarray(high, np.float32)
        self.context_size, self.stride = int(context_size), int(stride)
        self.step_s = float(step_s)
        self.cruise_kmh = float(cruise_kmh)
        self._lat_gains, self._lon_gains, self.dt = lateral, longitudinal, float(dt)
        self.max_steer, self.max_throttle, self.max_brake = max_steer, max_throttle, max_brake
        self.base_lookahead_m, self.lookahead_s = float(base_lookahead_m), float(lookahead_s)
        self.frames = deque(maxlen=self.context_size * self.stride + 1)
        self.last = {}
        self.reset_hidden(1)

    def reset_hidden(self, n=1):
        """A new episode: no frame history, fresh controllers, wheels straight, and
        the goal image is the episode's own (final) goal."""
        self.frames.clear()
        self.lat = PID(*self._lat_gains, dt=self.dt)
        self.lon = PID(*self._lon_gains, dt=self.dt)
        self.past_steer = 0.0
        # final_goal (map navigation): False while the goal image is an
        # intermediate map node, which is driven at cruise -- nodes ~6 m apart
        # would otherwise each read as "goal 2 s away" and the car would crawl
        # the whole route, a speed change the mapless arm does not get. The
        # planner's own state, not a privileged signal.
        self.final_goal = True
        return None

    def context(self):
        """context_size past frames `stride` steps apart, then the current one;
        before the history is long enough the oldest frame repeats (the car is
        at rest at an episode start)."""
        n = len(self.frames)
        return [self.frames[max(0, n - 1 - j * self.stride)]
                for j in range(self.context_size, -1, -1)]

    def act_env(self, cur_frame, goal_frame, speed_kmh):
        """One step toward `goal_frame`: push the frame, ask ViNT, steer."""
        self.push(cur_frame)
        dist_steps, wps = self.policy(self.context(), np.asarray(goal_frame))
        return self.control(dist_steps, wps, speed_kmh)

    def push(self, cur_frame):
        """The current camera frame joins the context (once per env step)."""
        self.frames.append(np.asarray(cur_frame))

    def control(self, dist_steps, wps, speed_kmh):
        """ViNT's prediction (goal distance in steps, waypoints in metres) -> env
        action, through the expert's controller. The topological navigator calls
        this with the prediction for the node it chose."""
        dist_s = max(0.0, float(dist_steps)) * self.step_s
        speed = max(0.0, float(speed_kmh))
        target = pick_waypoint(wps, self.base_lookahead_m + self.lookahead_s * speed / 3.6)
        angle = signed_angle(target)
        steer = self.lat.step(angle)
        steer = min(max(steer, self.past_steer - 0.1), self.past_steer + 0.1)
        steer = max(-self.max_steer, min(self.max_steer, steer))
        self.past_steer = steer
        target_kmh = (approach_kmh_from_time(dist_s, self.cruise_kmh) if self.final_goal
                      else self.cruise_kmh)
        acc = self.lon.step(target_kmh - speed)
        throttle = min(acc, self.max_throttle) if acc >= 0.0 else -min(-acc, self.max_brake)
        action = np.clip(np.array([steer, throttle], np.float32), self.low, self.high)
        self.last = {"dist_steps": float(dist_steps), "dist_s": dist_s,
                     "wps": np.round(np.asarray(wps, np.float64), 2).tolist(),
                     "target": [float(target[0]), float(target[1])],
                     "angle": angle, "target_kmh": float(target_kmh),
                     "steer": float(action[0]), "throttle": float(action[1])}
        return action


class VintPolicy:
    """ViNT inference. Distances stay in ViNT steps (time); waypoints come back in
    metres. Frames are either camera frames (any size; ViNT's crop is applied) or
    already-cropped (H, W, 3) uint8 images at ViNT's input size -- the form the
    topological map stores its nodes in, so thousands score cheaply."""

    def __init__(self, model, meta, device="cuda", crop_fn=None, chunk=256):
        import torch
        from torchvision import transforms
        self.torch, self.model, self.meta, self.device = torch, model, meta, device
        self.size = tuple(int(v) for v in meta["image_size"])          # (W, H)
        self.scale = float(meta["metric_step_m"]) * int(meta["stride"])
        self.chunk = int(chunk)
        self.normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        if crop_fn is None:
            from vint_train.data.data_utils import resize_and_aspect_crop
            crop_fn = lambda img: resize_and_aspect_crop(img, self.size)  # noqa: E731
        self.crop_fn = crop_fn

    # ---------------------------------------------------------------- images --
    def crop85(self, frame):
        """ViNT's input crop as (H, W, 3) uint8, before normalisation (lossless:
        ViNT's crop is a PIL resize followed by to_tensor's /255)."""
        from PIL import Image
        a = np.asarray(frame, np.uint8)
        if a.shape[:2] == (self.size[1], self.size[0]):
            return a
        t = self.crop_fn(Image.fromarray(a))
        return np.round(t.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)

    def _tensor(self, crops):
        """(B, H, W, 3) uint8 crops -> normalised (B, 3, H, W) float on the device."""
        t = self.torch.from_numpy(np.ascontiguousarray(crops)).permute(0, 3, 1, 2).float() / 255.0
        return self.normalize(t).to(self.device)

    def _obs(self, ctx):
        """Context frames -> (1, 3 * (context + 1), H, W)."""
        crops = np.stack([self.crop85(f) for f in ctx])
        return self._tensor(crops).reshape(1, -1, self.size[1], self.size[0])

    # ------------------------------------------------------------- inference --
    def __call__(self, ctx, goal):
        d, w = self.dists_to_goals(ctx, self.crop85(goal)[None])
        return float(d[0]), w[0]

    def dists_to_goals(self, ctx, goals):
        """The car's context against N goal images -> (dists (N,), waypoints (N, L, 2) m)."""
        with self.torch.no_grad():
            obs = self._obs(ctx)
            goals = np.asarray(goals, np.uint8)
            ds, ws = [], []
            for i in range(0, len(goals), self.chunk):
                g = self._tensor(np.stack([self.crop85(x) for x in goals[i:i + self.chunk]]))
                dist, act = self.model(obs.expand(len(g), -1, -1, -1), g)
                ds.append(dist.reshape(-1).float().cpu().numpy())
                ws.append(act[:, :, :2].float().cpu().numpy() * self.scale)
        return np.concatenate(ds), np.concatenate(ws)

    def pair_dists(self, ctxs, goals):
        """Context i against goal i: (N, context + 1, H, W, 3) and (N, H, W, 3) crops
        -> dists (N,). The map builder scores drive neighbours and revisit
        candidates this way."""
        with self.torch.no_grad():
            ctxs, goals = np.asarray(ctxs, np.uint8), np.asarray(goals, np.uint8)
            n, c = ctxs.shape[:2]
            ds = []
            for i in range(0, n, self.chunk):
                part = ctxs[i:i + self.chunk]
                obs = self._tensor(part.reshape(-1, *part.shape[2:]))
                obs = obs.reshape(len(part), c * 3, self.size[1], self.size[0])
                dist, _ = self.model(obs, self._tensor(goals[i:i + self.chunk]))
                ds.append(dist.reshape(-1).float().cpu().numpy())
        return np.concatenate(ds) if ds else np.zeros(0)

    def dists_from_contexts(self, ctxs, goal):
        """N contexts (N, context + 1, H, W, 3) crops against one goal -> dists (N,)."""
        with self.torch.no_grad():
            g = self._tensor(self.crop85(goal)[None])
            ctxs = np.asarray(ctxs, np.uint8)
            n, c = ctxs.shape[:2]
            ds = []
            for i in range(0, n, self.chunk):
                part = ctxs[i:i + self.chunk]
                obs = self._tensor(part.reshape(-1, *part.shape[2:]))
                obs = obs.reshape(len(part), c * 3, self.size[1], self.size[0])
                dist, _ = self.model(obs, g.expand(len(part), -1, -1, -1))
                ds.append(dist.reshape(-1).float().cpu().numpy())
        return np.concatenate(ds)


def load_vint_policy(ckpt_dir, device="cuda"):
    """EVAL_FROM=<dir> holding model.pth (a ViNT training checkpoint, {"model":
    ViNT, ...}) and vint.json (architecture, image_size, stride, metric_step_m,
    as trained). -> (VintPolicy, meta). Needs vint_train and efficientnet_pytorch
    on the path (slurm/vint_eval_pair.sh, slurm/vint_bench_map.sh)."""
    import torch
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
    return VintPolicy(model, meta, device=device), meta
