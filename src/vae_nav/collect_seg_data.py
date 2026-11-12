"""Collect (RGB, dense semantic label) pairs for the segmentation VAE.

User 2026-09-28: "pretrain resnet-9 masked VAE auto encoder (JAX) with [CARLA's
instance segmentation sensor] and then use for training"; "remove town02 and
add town07"; "dense segmentation", "single vector", "local".

WHAT IS NEW AGAINST collect_lane_data.py (whose camera, frame-exact reads,
distance gating and stuck recovery are reused unchanged):
  * sensor.camera.instance_segmentation -- its R channel is the semantic tag
    (G/B the instance id, unused here);
  * DENSE labels: every pixel mapped to one of N_CLASSES groups, stored at
    LABEL_SIZE x LABEL_SIZE with nearest-neighbour resize, so training can pool
    them into soft targets at the image size (thin lane lines survive);
  * STEERING DISTURBANCES: with probability PERTURB_PROB per tick the autopilot
    hands over for PERTURB_TICKS ticks of random steering, then recovers. The
    policy fails in drifted, near-the-line states that a clean autopilot
    drive never visits; the encoder must see them;
  * the CARLA server is started from Python per (town, quality).
The image goes through preprocess.env_frame exactly as the RL env's frame does
(748x348 FOV 110, 1.9/0/1.7 pitch -15, squashed to 120x120 RGB), so the
encoder is fitted on the distribution the policy sees.

usage: python -m vae_nav.collect_seg_data --out DIR --towns Town01,Town04 \
           --qualities Low,Epic --frames 5000 [--port 2000]
"""
import argparse
import json
import os
import socket
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vae_nav.collect_lane_data import (CAM_FOV, CAM_H, CAM_TRANSFORM, CAM_W,  # noqa: E402
                                       MIN_MOVE_M, STUCK_TICKS, _frame_at,
                                       _to_array, _weather_cycle)
from vae_nav.lane_mask import TAGS  # noqa: E402
from vae_nav.preprocess import env_frame  # noqa: E402
from vae_nav.town_split import assert_not_heldout, load_split  # noqa: E402

LABEL_SIZE = int(os.environ.get("SEG_LABEL_SIZE", 240))
IMAGE_SIZE = int(os.environ.get("ENV_IMAGE_SIZE", 120))
PERTURB_PROB = float(os.environ.get("PERTURB_PROB", 0.02))
PERTURB_TICKS = int(os.environ.get("PERTURB_TICKS", 12))
CARLA_ROOT = os.environ.get("CARLA_ROOT", "/home/robotlab/Apps/CARLA_0.9.15")
# CAMERA FOV VARIATION (user 2026-09-28: "use different fovs for cameras as
# well"): the encoder must transfer, not memorise one lens. A new RGB+seg pair
# at a freshly drawn FOV every FOV_SWITCH_FRAMES kept frames; the policy's own
# camera is 110.
SEG_FOVS = tuple(float(f) for f in os.environ.get("SEG_FOVS", "90,100,110,120").split(","))
FOV_SWITCH_FRAMES = int(os.environ.get("FOV_SWITCH_FRAMES", 500))

# CLASS GROUPS. Names are CARLA 0.9.15 semantic labels (lane_mask.TAGS is
# resolved from the installed CARLA, so the numbers are never hardcoded).
CLASS_NAMES = ("other", "road", "lane_line", "sidewalk", "vehicle", "person",
               "structure", "nature", "sky")
_GROUPS = {
    "road": ("Roads", "Road"),
    "lane_line": ("RoadLines", "RoadLine"),
    "sidewalk": ("Sidewalks", "SideWalk", "Sidewalk"),
    "vehicle": ("Car", "Truck", "Bus", "Train", "Motorcycle", "Bicycle", "Vehicles"),
    "person": ("Pedestrian", "Pedestrians", "Rider"),
    "structure": ("Building", "Buildings", "Wall", "Walls", "Fence", "Fences",
                  "Pole", "Poles", "TrafficLight", "TrafficSign", "TrafficSigns", "Static",
                  "Dynamic", "GuardRail", "Bridge", "RailTrack"),
    "nature": ("Vegetation", "Terrain", "Ground", "Water"),
    "sky": ("Sky",),
}
N_CLASSES = len(CLASS_NAMES)


def tag_to_class_lut():
    """256-entry lookup: semantic tag -> class index (0 = other)."""
    lut = np.zeros(256, np.uint8)
    for ci, cname in enumerate(CLASS_NAMES):
        for tname in _GROUPS.get(cname, ()):
            if tname in TAGS:
                lut[TAGS[tname]] = ci
    return lut


def labels_from_instance(inst_rgb, lut, size=LABEL_SIZE):
    """(H, W, 3) instance-seg image in RGB order -> (size, size) uint8 classes.
    The squash to a square mirrors env_frame's resize of the RGB image."""
    import cv2
    cls = lut[np.asarray(inst_rgb)[:, :, 0]]
    return cv2.resize(cls, (size, size), interpolation=cv2.INTER_NEAREST)


def start_server(port, quality, timeout=180.0):
    """Start CARLA from Python on `port` at `quality`, windowed; wait until the
    RPC answers (not just the TCP port). Returns the Popen."""
    import carla
    cmd = [f"{CARLA_ROOT}/CarlaUE4.sh", f"-carla-port={port}",
           f"-quality-level={quality}", "-windowed", "-ResX=640", "-ResY=360",
           "-prefernvidia"]
    os.makedirs("logs", exist_ok=True)
    log = open(f"logs/carla_seg_{port}_{int(time.time())}.log", "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True)
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"CARLA exited during startup rc={proc.returncode}")
        with socket.socket() as sk:
            sk.settimeout(1.0)
            if sk.connect_ex(("localhost", port)) == 0:
                try:
                    c = carla.Client("localhost", port)
                    c.set_timeout(4.0)
                    c.get_world().get_map()
                    print(f"[CARLA] {quality} ready on {port} after "
                          f"{time.time() - t0:.0f} s", flush=True)
                    return proc
                except Exception:  # noqa: BLE001
                    pass
        time.sleep(1.0)
    proc.kill()
    raise RuntimeError(f"CARLA did not answer on {port} in {timeout}s")


def stop_server(proc):
    import signal
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:  # noqa: BLE001
        pass
    time.sleep(5)


def collect_block(carla, port, town, quality, n_frames, seed, sink):
    """One (town, quality) block, appended into `sink` (lists owned by the
    caller, so a sensor timeout mid-block keeps what was already collected)."""
    client = carla.Client("localhost", port)
    client.set_timeout(120.0)
    world = client.get_world()
    if world.get_map().name.split("/")[-1] != town:
        world = client.load_world(town)
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)
    tm = client.get_trafficmanager(port + 6000)
    tm.set_synchronous_mode(True)
    tm.set_random_device_seed(seed)
    rng = np.random.default_rng(seed)
    spawns = world.get_map().get_spawn_points()
    rng.shuffle(spawns)
    bl = world.get_blueprint_library()
    hero = None
    for tf in spawns[:20]:
        hero = world.try_spawn_actor(bl.filter("vehicle.micro.microlino")[0], tf)
        if hero is not None:
            break
    if hero is None:
        raise RuntimeError(f"could not spawn the hero in {town}")
    hero.set_autopilot(True, tm.get_port())
    print(f"  [{town}/{quality}] hero {hero.id} spawned", flush=True)
    cam_tf = carla.Transform(
        carla.Location(x=CAM_TRANSFORM["x"], y=CAM_TRANSFORM["y"], z=CAM_TRANSFORM["z"]),
        carla.Rotation(pitch=CAM_TRANSFORM["pitch"], yaw=CAM_TRANSFORM["yaw"],
                       roll=CAM_TRANSFORM["roll"]))
    import queue
    cams, qs, retired = [], [], []

    def spawn_pair(fov):
        """Retire the current pair (stop only -- destroying sensors in sync
        mode aborted the process) and attach a new RGB+seg pair at `fov`."""
        for c in cams:
            try:
                c.stop()
            except Exception:  # noqa: BLE001
                pass
        retired.extend(cams)          # keep references: no out-of-scope warnings
        cams.clear()
        qs.clear()
        for bp_name in ("sensor.camera.rgb", "sensor.camera.instance_segmentation"):
            bp = bl.find(bp_name)
            bp.set_attribute("image_size_x", str(CAM_W))
            bp.set_attribute("image_size_y", str(CAM_H))
            bp.set_attribute("fov", str(fov))
            cam = world.spawn_actor(bp, cam_tf, attach_to=hero)
            q = queue.Queue()
            cam.listen(q.put)
            cams.append(cam)
            qs.append(q)

    cur_fov = float(rng.choice(SEG_FOVS))
    spawn_pair(cur_fov)
    lut = tag_to_class_lut()
    rgbs, labels, perturbed, weathers = (sink["rgb"], sink["label"],
                                         sink["perturbed"], sink["weather"])
    fovs = sink.setdefault("fov", [])
    kept, tick, last, stuck, pert_left, widx = 0, 0, None, 0, 0, 0
    cur_weather = _weather_cycle(carla, world, widx, rng)

    def tick_once(control=None):
        if control is not None:
            hero.apply_control(control)
        fid = world.tick()
        return _frame_at(qs[0], fid, "rgb"), _frame_at(qs[1], fid, "instance")

    try:
        while kept < n_frames:
            # STEERING DISTURBANCE: hand over from the autopilot for a few ticks
            if pert_left == 0 and rng.random() < PERTURB_PROB:
                pert_left = PERTURB_TICKS
                pert_steer = float(rng.uniform(-0.35, 0.35))
                hero.set_autopilot(False, tm.get_port())
            if pert_left > 0:
                ctrl = carla.VehicleControl(throttle=0.4, steer=pert_steer)
                pert_left -= 1
                if pert_left == 0:
                    hero.set_autopilot(True, tm.get_port())
            else:
                ctrl = None
            rgb_img, inst_img = tick_once(ctrl)
            tick += 1
            if tick < 30:
                continue
            loc = hero.get_transform().location
            if last is not None and loc.distance(last) < MIN_MOVE_M:
                stuck += 1
                if stuck >= STUCK_TICKS:
                    hero.set_transform(spawns[rng.integers(len(spawns))])
                    hero.set_target_velocity(carla.Vector3D(0, 0, 0))
                    for _ in range(10):
                        tick_once()
                    last, stuck = None, 0
                continue
            stuck, last = 0, loc
            if kept and kept % FOV_SWITCH_FRAMES == 0:
                cur_fov = float(rng.choice(SEG_FOVS))
                spawn_pair(cur_fov)
                for _ in range(5):
                    tick_once()
            if kept and kept % 250 == 0:
                widx += 1
                cur_weather = _weather_cycle(carla, world, widx, rng)
                for _ in range(10):
                    tick_once()
            rgbs.append(env_frame(_to_array(rgb_img), IMAGE_SIZE))
            labels.append(labels_from_instance(_to_array(inst_img), lut))
            perturbed.append(pert_left > 0)
            fovs.append(cur_fov)
            weathers.append(cur_weather)
            kept += 1
            if kept % 500 == 0 or kept in (1, 50):
                print(f"  [{town}/{quality}] {kept}/{n_frames} weather={cur_weather} "
                      f"fov={cur_fov:g}",
                      flush=True)
    finally:
        # NO ACTOR CLEANUP. Stopping/destroying the sensors in synchronous
        # mode aborted the process ("trying to operate on a destroyed actor",
        # a C++ terminate) after every block; the server is killed right after
        # this returns, which removes every actor anyway.
        print(f"  [{town}/{quality}] block done at kept={kept} tick={tick}", flush=True)
    return sink


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--towns", default="Town01,Town04,Town05,Town07,Town10HD")
    ap.add_argument("--qualities", default="Low,Epic")
    ap.add_argument("--frames", type=int, default=5000)
    # MORE LOW-QUALITY DATA (user 2026-09-28: "collect additional data with
    # quality low for generalization"): per-quality frame counts.
    ap.add_argument("--frames-low", type=int, default=None)
    ap.add_argument("--frames-epic", type=int, default=None)
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    sys.path.append(f"{CARLA_ROOT}/PythonAPI/carla")
    import carla
    # THE RL RUNS' SPLIT (configs/town_split_v7.json: only Town03 held out).
    # The default split also holds out Town10HD, which v25-v27 train on.
    _train_towns, heldout = load_split(os.environ.get(
        "TOWN_SPLIT", os.path.join(os.path.dirname(__file__), "..", "..",
                                   "configs", "town_split_v7.json")))
    os.makedirs(a.out, exist_ok=True)
    for ti, town in enumerate(a.towns.split(",")):
        assert_not_heldout(town, heldout)
        for qi, quality in enumerate(a.qualities.split(",")):
            path = os.path.join(a.out, f"shard_{town}_{quality}.npz")
            if os.path.exists(path):
                print(f"SKIP {path}", flush=True)
                continue
            print(f"=== {town} {quality} ===", flush=True)
            n = {"Low": a.frames_low, "Epic": a.frames_epic}.get(quality) or a.frames
            sink = {"rgb": [], "label": [], "perturbed": [], "weather": [], "fov": []}
            proc = start_server(a.port, quality)
            try:
                collect_block(carla, a.port, town, quality, n,
                              a.seed + 100 * ti + qi, sink)
            except Exception as e:  # noqa: BLE001 - keep a partial block
                print(f"  [{town}/{quality}] block stopped early: "
                      f"{type(e).__name__}: {e} -- keeping {len(sink['rgb'])} frames",
                      flush=True)
                if len(sink["rgb"]) < 500:
                    path = path.replace(".npz", ".partial_too_small.skip")
            finally:
                stop_server(proc)
            if path.endswith(".skip"):
                continue
            d = dict(rgb=np.stack(sink["rgb"]), label=np.stack(sink["label"]),
                     perturbed=np.array(sink["perturbed"]),
                     weather=np.array(sink["weather"]), fov=np.array(sink["fov"]))
            np.savez_compressed(path + ".tmp.npz", **d)
            os.replace(path + ".tmp.npz", path)
            counts = np.bincount(d["label"].ravel(), minlength=N_CLASSES) / d["label"].size
            print(f"saved {path}: {len(d['rgb'])} frames, perturbed "
                  f"{d['perturbed'].mean():.2f}, class share "
                  + " ".join(f"{n}={c:.3f}" for n, c in zip(CLASS_NAMES, counts)),
                  flush=True)
    json.dump({"classes": CLASS_NAMES, "label_size": LABEL_SIZE,
               "image_size": IMAGE_SIZE, "towns": a.towns, "qualities": a.qualities},
              open(os.path.join(a.out, "meta.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
