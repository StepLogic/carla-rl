"""Collect (RGB, [lane, obstacle]-mask) pairs across the TRAINING towns only.

Diversity is the point, not just volume: four towns x thirteen weather/lighting
presets x traffic and pedestrians. The VAE has to work on two towns it has never
seen, and appearance shift (wet specular road, low sun, rain) breaks a camera
model at least as readily as unfamiliar geometry does.

Deliberately bypasses CarlaEnv and drives the CARLA API directly. The RL env
downsamples observations to 84x84 greyscale, which is the policy's diet, not the
VAE's -- training a mask decoder against pre-greyscaled 84x84 input throws away
exactly the colour and resolution that distinguish a lane marking from a kerb.

The two cameras are spawned at the SAME transform with the SAME fov, so RGB and
mask are pixel-aligned by construction. Any mismatch there is irreducible label
noise the VAE would have to model.

HELD-OUT TOWNS ARE NEVER LOADED HERE. The peer Habitat repo's trainer globbed
every scene when its list was unset and trained on its own eval maps with
nothing in the logs to show it; assert_not_heldout() is the guard against the
CARLA version of that.
"""
import argparse
import json
import os
import queue
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vae_nav.lane_mask import (CHANNEL_NAMES, check_mask_plausible,  # noqa: E402
                               pos_weight_for, semantic_targets)
from vae_nav.preprocess import env_frame, mask_to_vae_target, to_vae_input  # noqa: E402
from vae_nav.town_split import assert_not_heldout, load_split  # noqa: E402

CAM_TRANSFORM = dict(x=1.9, y=0.0, z=1.7, pitch=-15.0, yaw=0.0, roll=0.0)
CAM_W, CAM_H, CAM_FOV = 748, 348, 110

# WEATHER AND LIGHTING DIVERSITY.
#
# The VAE is trained on four towns and must work on two it has never seen. Town
# geometry is only half the domain shift; the other half is appearance. A model
# fitted on ClearNoon alone learns "lane markings are bright lines on grey" and
# breaks at sunset, in rain (specular road, reflections that look like
# markings), and at night (markings visible only under headlights).
#
# Cycled rather than sampled per frame: weather changes are applied to the
# world, and thrashing them every tick would be slow and would smear the
# transition across frames. One preset per block of frames gives clean
# coverage.
# NIGHT IS DELIBERATELY A SMALL MINORITY, and this is not a stylistic choice.
# The semantic-segmentation target is ground truth from the simulator, so it is
# LIGHTING-INVARIANT: at night the mask still marks every lane marking, including
# the ones outside the headlight cone that the RGB camera genuinely cannot see.
# Those frames therefore carry targets that are not recoverable from the input,
# which is label noise rather than useful supervision. A little of it teaches
# the model to be uncertain in the dark; a lot of it just degrades the lane
# channel everywhere. Two of thirteen presets (~15%) rather than four of fifteen
# (~27%).
WEATHER_PRESETS = (
    "ClearNoon", "CloudyNoon", "WetNoon", "WetCloudyNoon",
    "SoftRainNoon", "MidRainyNoon", "HardRainNoon",
    "ClearSunset", "CloudySunset", "WetSunset", "SoftRainSunset",
    "ClearNight", "WetNight",
)
FRAMES_PER_WEATHER = int(os.environ.get("FRAMES_PER_WEATHER", 250))


def _weather_cycle(carla, world, index, rng):
    """Apply the index-th weather preset, with a jittered sun angle.

    Returns the name applied, for the provenance record. Presets absent from
    this CARLA build are skipped rather than raising -- the night variants in
    particular only exist from 0.9.13 onward.
    """
    names = [n for n in WEATHER_PRESETS
             if hasattr(carla.WeatherParameters, n)]
    if not names:
        return "default"
    name = names[index % len(names)]
    weather = getattr(carla.WeatherParameters, name)
    try:
        # Jitter the sun so shadows fall differently within a preset: hard
        # shadow edges across a lane marking are a common false positive.
        w = carla.WeatherParameters(
            cloudiness=weather.cloudiness, precipitation=weather.precipitation,
            precipitation_deposits=weather.precipitation_deposits,
            wind_intensity=weather.wind_intensity,
            sun_azimuth_angle=float(rng.uniform(0, 360)),
            sun_altitude_angle=weather.sun_altitude_angle,
            fog_density=float(rng.uniform(0, 25)),
            wetness=weather.wetness)
        world.set_weather(w)
    except Exception:  # noqa: BLE001  older signature
        world.set_weather(weather)
    return name


def _spawn_cameras(world, vehicle, carla):
    """RGB + semantic at one pose. Returns (rgb_cam, seg_cam, rgb_q, seg_q)."""
    bl = world.get_blueprint_library()
    tf = carla.Transform(
        carla.Location(x=CAM_TRANSFORM["x"], y=CAM_TRANSFORM["y"], z=CAM_TRANSFORM["z"]),
        carla.Rotation(pitch=CAM_TRANSFORM["pitch"], yaw=CAM_TRANSFORM["yaw"],
                       roll=CAM_TRANSFORM["roll"]))

    def _mk(bp_name):
        bp = bl.find(bp_name)
        bp.set_attribute("image_size_x", str(CAM_W))
        bp.set_attribute("image_size_y", str(CAM_H))
        bp.set_attribute("fov", str(CAM_FOV))
        return world.spawn_actor(bp, tf, attach_to=vehicle)

    rgb_cam = _mk("sensor.camera.rgb")
    seg_cam = _mk("sensor.camera.semantic_segmentation")
    rgb_q, seg_q = queue.Queue(), queue.Queue()
    rgb_cam.listen(rgb_q.put)
    seg_cam.listen(seg_q.put)
    return rgb_cam, seg_cam, rgb_q, seg_q


def _frame_at(q, frame_id, name, timeout=20.0):
    """Drain a sensor queue up to `frame_id` and return that image.

    THE BUG THIS REPLACES. The loop used to do:

        world.tick()
        rgb_img, seg_img = rgb_q.get(), seg_q.get()

    which assumes both sensors deliver exactly one frame per tick, forever, in
    lockstep. The moment they desync by a single frame the mismatch check hits
    `continue`, the next iteration ticks and pulls one MORE from each, and the
    offset is preserved rather than corrected. Every subsequent pair then
    mismatches, `kept` stops advancing, and one queue eventually starves into a
    timeout. Measured: Town02 died at the same frame twice with the SERVER
    STILL HEALTHY -- no segfault, just a client that had desynchronised itself.

    Draining to an explicit frame id makes a dropped frame self-correcting
    instead of permanent.
    """
    while True:
        img = q.get(timeout=timeout)
        if img.frame == frame_id:
            return img
        if img.frame > frame_id:
            # Overshot: this sensor never produced the requested frame. Report
            # rather than silently pairing mismatched poses.
            raise RuntimeError(
                f"{name} sensor overshot frame {frame_id} (got {img.frame}); "
                "the sensor dropped a frame the other one delivered")


def _to_array(image):
    """CARLA Image -> (H, W, 3) uint8, RGB order.

    Matches rlib_integration's BaseCamera.parse exactly, so the semantic tag
    lands in channel 0 the same way it does at RL time. Diverging here would
    mean the VAE trains on one channel convention and infers on another.
    """
    arr = np.frombuffer(image.raw_data, dtype=np.uint8)
    arr = arr.reshape((image.height, image.width, 4))[:, :, :3]
    return arr[:, :, ::-1]


# MINIMUM DISPLACEMENT BETWEEN KEPT FRAMES, and why a tick stride is not enough.
#
# The only sampling gate used to be `tick % stride`, which counts SIMULATOR
# TIME. A hero on autopilot spends much of its life not moving -- red lights,
# queued traffic, wedged behind a bus that never clears -- and every one of
# those ticks wrote another copy of the same view.
#
# MEASURED over the 30,000-frame corpus this produced (consecutive-frame mean
# absolute difference < 1.0 counted as a repeat):
#     shard            stuck%   distinct fraction
#     Town04_Epic       94.3        0.020
#     Town05_Epic       92.3        0.020
#     Town02_Epic       85.1        0.073     <- 91.9% of frames had NO lane
#     Town01_Epic       73.4        0.149        pixels at all: the hero spent
#     Town04_Low        47.5        0.342        the drive behind a bus
# Thirty thousand frames held roughly 1,000-1,500 distinct ones. Worse, a
# random train/val split over a corpus that is 90% duplicates puts near-copies
# of training frames in validation, so the reported val_iou measured memory
# rather than generalisation.
#
# Gating on DISTANCE TRAVELLED instead makes the corpus size mean what it says:
# n frames are n samples at least MIN_MOVE_M apart along the road.
MIN_MOVE_M = float(os.environ.get("MIN_MOVE_M", 1.5))

# A hero that has not moved for this many ticks is not going to. Autopilot
# deadlocks against stopped traffic and stays deadlocked; the Town02 collection
# spent most of 5000 frames pinned behind one bus. Recovery teleports the hero
# to a fresh spawn point rather than waiting the run out.
STUCK_TICKS = int(os.environ.get("STUCK_TICKS", 200))


def collect_town(carla, town, n_frames, out_dir, size, seed, warmup=30,
                 stride=None, n_traffic=None):
    """Drive one town on autopilot, keeping frames MIN_MOVE_M apart.

    `stride` (every Nth tick) used to be the sampling gate. It is now a coarse
    pre-filter ahead of the displacement gate, and defaults to 1 -- evaluating
    every tick -- because subsampling TIME while the real criterion is DISTANCE
    only throws away candidate frames and slows collection. At stride=4 a
    Town02 pass kept 200 frames while rejecting 818 stationary ones, a rate
    that projected to roughly sixteen hours for the corpus.
    """
    stride = int(os.environ.get("COLLECT_STRIDE", 1) if stride is None else stride)
    # TRAFFIC DENSITY IS A DEADLOCK KNOB, not just scene dressing. Thirty
    # autopilot vehicles in a town as tight as Town02 gridlock: the measured
    # pass rejected 92% of candidate frames as stationary and needed two
    # respawns before frame 200. Some traffic is wanted -- it is what puts
    # vehicles in the obstacle channel -- but past a point it stops the hero
    # from ever seeing new road.
    n_traffic = int(os.environ.get("N_TRAFFIC", 0) if n_traffic is None
                    else n_traffic)
    import cv2

    client = carla.Client(os.environ.get("CARLA_HOST", "localhost"),
                          int(os.environ.get("CARLA_PORT", 2000)))
    client.set_timeout(float(os.environ.get("CARLA_CLIENT_TIMEOUT", 120.0)))

    # Skip load_world when the server already has the right map.
    #
    # MEASURED: passing the town as a positional argument to CarlaUE4.sh does
    # NOT work on 0.9.15 -- the server boots into Town10HD_Opt regardless, so
    # this branch effectively always calls load_world. It is kept because the
    # check is free, it makes the load explicit in the log, and other CARLA
    # versions do honour the argument.
    #
    # load_world is the operation that segfaults this server unpredictably (it
    # killed two collection runs). Since it cannot be avoided, the queue wraps
    # each town in a retry against a FRESH server instead.
    world = client.get_world()
    current = world.get_map().name.split("/")[-1]
    if current != town:
        print(f"  [{town}] server is on {current}; calling load_world "
              "(this is the operation that has crashed the server before)")
        world = client.load_world(town)
    else:
        print(f"  [{town}] server already booted into {current}; no load_world")

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    tm = client.get_trafficmanager()
    tm.set_synchronous_mode(True)
    tm.set_random_device_seed(seed)

    rng = np.random.default_rng(seed)
    spawns = world.get_map().get_spawn_points()
    rng.shuffle(spawns)

    bl = world.get_blueprint_library()

    # A MICRO CAR AS THE HERO. The Model 3 is long and wide, and its bonnet
    # occupies the bottom of a forward camera pitched down 15 degrees -- pixels
    # spent on the ego vehicle rather than on the road the VAE is meant to
    # read. The Microlino's short body puts the near road surface, and the lane
    # markings on it, back in frame.
    #
    # Falls back down a chain rather than indexing blindly: an absent blueprint
    # would otherwise be an IndexError several lines later with nothing in the
    # message about which name was missing.
    hero = None
    for name in (os.environ.get("HERO_BP", "vehicle.micro.microlino"),
                 "vehicle.micro.microlino", "vehicle.mini.cooper_s",
                 "vehicle.tesla.model3"):
        bps = bl.filter(name)
        if not bps:
            continue
        hero = world.try_spawn_actor(bps[0], spawns[0])
        if hero is not None:
            print(f"  [{town}] hero = {bps[0].id}")
            break
    if hero is None:
        raise SystemExit(
            f"could not spawn a hero in {town}; tried micro/mini/tesla "
            f"against {len(spawns)} spawn points")
    hero.set_autopilot(True, tm.get_port())

    # NO OTHER ACTORS. Traffic and pedestrians are not spawned.
    #
    # They were there to put vehicles and people into the OBSTACLE channel, and
    # that is a real cost of removing them: the channel now carries only static
    # scene content -- buildings, walls, poles, vegetation, sidewalks -- so the
    # VAE learns "obstacle" as built environment and has never seen a car or a
    # pedestrian labelled as one.
    #
    # The benefit is that the corpus stops being about traffic. With 30
    # autopilot vehicles, Town02 gridlocked: 92% of candidate frames were
    # rejected as stationary and 91.9% of the frames that survived contained no
    # lane pixels at all, because the camera spent the drive against the side
    # of a bus. An empty town drives freely, so every frame is new road, and
    # the lane channel -- the one that was actually weak, at 0.438 IoU on
    # Town02 -- gets the coverage it never had.
    traffic, walkers = [], []
    if n_traffic:
        print(f"  [{town}] WARNING: N_TRAFFIC={n_traffic}; collection is "
              "intended to run in an empty town")
        for tf in spawns[1:1 + n_traffic]:
            v = world.try_spawn_actor(rng.choice(bl.filter("vehicle.*")), tf)
            if v is not None:
                v.set_autopilot(True, tm.get_port())
                traffic.append(v)

    rgb_cam, seg_cam, rgb_q, seg_q = _spawn_cameras(world, hero, carla)

    rgbs, masks, kept, tick = [], [], 0, 0
    last_kept_loc, stuck_ticks, n_unstuck, n_skipped = None, 0, 0, 0
    weather_counts, weather_idx = {}, 0
    cur_weather = _weather_cycle(carla, world, weather_idx, rng)
    try:
        while kept < n_frames:
            # tick() returns the frame id it produced; both sensors are then
            # drained TO that id, so the two streams cannot drift apart.
            frame_id = world.tick()
            tick += 1
            rgb_img = _frame_at(rgb_q, frame_id, "rgb")
            seg_img = _frame_at(seg_q, frame_id, "semantic")
            if tick < warmup or tick % stride:
                continue

            # DISPLACEMENT GATE. Keep this frame only if the hero has actually
            # travelled since the last one kept. See MIN_MOVE_M.
            loc = hero.get_transform().location
            if last_kept_loc is not None:
                moved = loc.distance(last_kept_loc)
                if moved < MIN_MOVE_M:
                    n_skipped += 1
                    stuck_ticks += 1
                    # UNSTICK. Autopilot does not recover from a deadlock on
                    # its own, so waiting costs the whole collection.
                    if stuck_ticks >= STUCK_TICKS:
                        spawns = world.get_map().get_spawn_points()
                        hero.set_transform(spawns[rng.integers(len(spawns))])
                        hero.set_target_velocity(carla.Vector3D(0, 0, 0))
                        for _ in range(10):
                            fid = world.tick()
                            _frame_at(rgb_q, fid, "rgb")
                            _frame_at(seg_q, fid, "semantic")
                        last_kept_loc, stuck_ticks = None, 0
                        n_unstuck += 1
                        print(f"  [{town}] hero stuck; respawned "
                              f"(#{n_unstuck})", flush=True)
                    continue
            stuck_ticks = 0
            last_kept_loc = loc
            # Now guaranteed by construction, kept as an assertion: a mismatch
            # here would mean _frame_at is broken, and pairing an RGB with a
            # mask from a different pose is invisible label noise.
            assert rgb_img.frame == seg_img.frame
            # Rotate weather every block so each preset gets clean coverage.
            if kept and kept % FRAMES_PER_WEATHER == 0:
                weather_idx += 1
                cur_weather = _weather_cycle(carla, world, weather_idx, rng)
                for _ in range(10):      # let the transition settle
                    fid = world.tick()
                    _frame_at(rgb_q, fid, "rgb")
                    _frame_at(seg_q, fid, "semantic")
            rgb = _to_array(rgb_img)
            seg = _to_array(seg_img)
            m = semantic_targets(seg)   # (H, W, 2) [lane, obstacle]
            # THROUGH THE ENV'S TRANSFORM FIRST. At RL time the VAE is handed
            # experiment.obs_img, which post_process_image has already squashed
            # to 84x84. Collecting straight from full resolution would fit the
            # encoder to a distribution it never sees online -- see
            # vae_nav/preprocess.py. The mask follows the SAME two-step path so
            # image and target stay pixel-aligned.
            rgbs.append((to_vae_input(env_frame(rgb), size) * 255).astype(np.uint8))
            masks.append(mask_to_vae_target(env_frame(m * 255.0) / 255.0, size))
            kept += 1
            weather_counts[cur_weather] = weather_counts.get(cur_weather, 0) + 1
            if kept % 200 == 0:
                print(f"  [{town}] {kept}/{n_frames}  weather={cur_weather}  "
                      f"skipped_stationary={n_skipped}  respawns={n_unstuck}",
                      flush=True)
    finally:
        # SAFE TEARDOWN. Calling .destroy() on an actor CARLA has already
        # reaped throws a C++ std::runtime_error, which is NOT catchable by a
        # Python except clause -- it reaches std::terminate and aborts the
        # process. That is exactly how a completed 5000-frame Town01 collection
        # was lost: every frame captured, then killed while tidying up.
        #
        # Sensors are stopped first (a live listen callback firing into a
        # destroyed queue is its own hazard), then everything is destroyed in
        # ONE batch command, which the server tolerates for already-dead actors.
        for cam in (rgb_cam, seg_cam):
            try:
                if cam is not None and cam.is_alive:
                    cam.stop()
            except Exception:  # noqa: BLE001
                pass
        try:
            alive = [a for a in [rgb_cam, seg_cam, hero, *traffic, *walkers]
                     if a is not None and getattr(a, "is_alive", False)]
            if alive:
                client.apply_batch([carla.command.DestroyActor(a) for a in alive])
        except Exception as e:  # noqa: BLE001
            print(f"  [{town}] teardown warning (ignored): {e}")
        try:
            settings.synchronous_mode = False
            world.apply_settings(settings)
        except Exception:  # noqa: BLE001
            pass

    print(f"  [{town}] weather coverage: {weather_counts}")
    return (np.asarray(rgbs, np.uint8), np.asarray(masks, np.float32),
            weather_counts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/lane_vae")
    ap.add_argument("--frames-per-town", type=int, default=5000)
    ap.add_argument("--size", type=int, default=int(os.environ.get("VAE_IMAGE_SIZE", 120)))
    ap.add_argument("--seed", type=int, default=0)
    # Render quality is a SERVER launch flag, not a per-frame setting, so one
    # pass collects one quality and the tag keeps the shards apart. Running the
    # collector once per quality accumulates a corpus spanning both.
    ap.add_argument("--tag", default=os.environ.get("CARLA_QUALITY", "Epic"),
                    help="render-quality tag for shard names")
    # ONE TOWN PER PROCESS, against a FRESH server.
    #
    # CARLA 0.9.15's server segfaults on repeated client.load_world() within a
    # single session: Town01 collected cleanly twice, then the server died with
    # Signal 11 on the transition to Town02, and the client saw only a 60s
    # TimeoutException. Loading exactly one map per server session avoids the
    # transition entirely. The queue restarts a server per town; the shard cache
    # makes that cheap, since finished towns are skipped.
    ap.add_argument("--town", default=None,
                    help="collect only this town, then exit (no corpus assembly)")
    ap.add_argument("--assemble-only", action="store_true",
                    help="skip collection; build the corpus from existing shards")
    args = ap.parse_args()

    train_towns, heldout = load_split()
    if args.town:
        if args.town not in train_towns:
            raise SystemExit(f"{args.town} is not a training town {train_towns}")
        train_towns = [args.town]
    os.makedirs(args.out, exist_ok=True)

    all_weather = {}
    if not args.assemble_only:
        import carla
    for i, town in enumerate(train_towns if not args.assemble_only else []):
        assert_not_heldout(town, heldout)
        shard = os.path.join(args.out, f"shard_{town}_{args.tag}.npz")

        # RESUME. A per-town shard is written as soon as that town finishes, so
        # a crash costs one town rather than the whole corpus, and a rerun skips
        # what is already on disk.
        if os.path.exists(shard):
            d = np.load(shard, allow_pickle=True)
            rgb, mask = d["rgb"], d["mask"]
            wcount = json.loads(str(d["weather"]))
            print(f"[Collect] {town} ({i + 1}/{len(train_towns)}) -- reusing "
                  f"existing shard, {len(rgb)} frames")
        else:
            print(f"[Collect] {town} ({i + 1}/{len(train_towns)})")
            rgb, mask, wcount = collect_town(carla, town, args.frames_per_town,
                                             args.out, args.size, args.seed + i)
            check_mask_plausible(mask, name=town)
            np.savez_compressed(shard, rgb=rgb, mask=mask,
                                weather=json.dumps(wcount))
            print(f"[Collect] wrote shard {shard} ({len(rgb)} frames)")

        all_weather[town] = wcount

    if args.town:
        print(f"[Collect] {args.town} done (single-town mode); "
              "corpus assembly deferred")
        return

    # Assemble from EVERY shard present, not just the ones this pass produced.
    # A second pass at another render quality therefore extends the corpus
    # rather than replacing it, and the VAE sees both -- which is the point:
    # a perception module fitted to one renderer degrades on the other, and
    # that degradation is invisible in the RL logs.
    import glob
    shards = sorted(glob.glob(os.path.join(args.out, "shard_*.npz")))
    all_rgb, all_mask, tags = [], [], {}
    for sp in shards:
        d = np.load(sp, allow_pickle=True)
        all_rgb.append(d["rgb"])
        all_mask.append(d["mask"])
        name = os.path.basename(sp)[len("shard_"):-len(".npz")]
        tags[name] = int(len(d["rgb"]))
    print(f"[Collect] assembling corpus from {len(shards)} shards:")
    for k, v in tags.items():
        print(f"           {k}: {v} frames")

    rgb = np.concatenate(all_rgb)
    mask = np.concatenate(all_mask)
    frac = check_mask_plausible(mask, name="corpus")

    path = os.path.join(args.out, "lane_dataset.npz")
    np.savez_compressed(path, rgb=rgb, mask=mask,
                        towns=np.array(train_towns),
                        weather=json.dumps(all_weather),
                        shards=json.dumps(tags),
                        pos_weight=pos_weight_for(mask),
                        positive_fraction=np.atleast_1d(frac))
    print(f"[Collect] wrote {len(rgb)} frames -> {path}")
    pw = pos_weight_for(mask)
    print("[Collect] pos_weight per channel: "
          + ", ".join(f"{n}={float(w):.1f}" for n, w in zip(CHANNEL_NAMES, np.atleast_1d(pw))))


if __name__ == "__main__":
    main()
