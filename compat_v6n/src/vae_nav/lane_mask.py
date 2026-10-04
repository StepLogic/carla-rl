"""Lane-marking masks from CARLA's semantic segmentation camera.

WHY THIS IS FREE SUPERVISION. CARLA tags RoadLine separately from Road, so a
pixel-accurate lane-marking mask falls out of the simulator with no hand
labelling. This is the CARLA analogue of the peer Habitat repo's depth target:
there, a MonoDepthNet regressed metric depth from RGB; here a VAE reconstructs
the lane mask from RGB.

TAG LOCATION. rlib_integration's BaseCamera.parse reads the BGRA buffer, keeps
the first three channels and reverses them, so the semantic tag -- which CARLA
writes into the RED byte -- lands in channel 0 of the returned array. No
CityScapesPalette conversion is applied anywhere in this repo, which is what
makes the raw tag recoverable at all. If anyone adds a palette conversion to
the sensor path, this module breaks silently and every mask goes empty, so
mask_from_semantic() range-checks rather than trusting the input.
"""
import os

import numpy as np

# SEMANTIC TAGS ARE RESOLVED FROM THE INSTALLED CARLA, NOT HARDCODED.
#
# The numbering changed between CARLA 0.9.11 and 0.9.14+, and the change is
# silent and catastrophic. Under 0.9.11 RoadLine was 6 and Road was 7; under
# 0.9.15 tag 6 is Poles, tag 7 is TrafficLight, and RoadLines moved to 24. This
# repo's README says 0.9.11 while /home/robotlab/Apps holds 0.9.15, so a
# hardcoded table would have trained the "lane" channel on POLES and marked the
# road surface and the sky as obstacles -- with a perfectly plausible positive
# pixel fraction, so check_mask_plausible would have passed it.
#
# Resolving by NAME from carla.CityObjectLabel makes the mapping correct on
# whichever version is actually installed, and makes a future renumbering a
# loud KeyError instead of silent mislabelling.
_FALLBACK_0_9_15 = {
    "NONE": 0, "Roads": 1, "Sidewalks": 2, "Buildings": 3, "Walls": 4,
    "Fences": 5, "Poles": 6, "TrafficLight": 7, "TrafficSigns": 8,
    "Vegetation": 9, "Terrain": 10, "Sky": 11, "Pedestrians": 12, "Rider": 13,
    "Car": 14, "Truck": 15, "Bus": 16, "Train": 17, "Motorcycle": 18,
    "Bicycle": 19, "Static": 20, "Dynamic": 21, "Other": 22, "Water": 23,
    "RoadLines": 24, "Ground": 25, "Bridge": 26, "RailTrack": 27,
    "GuardRail": 28,
}


def _resolve_tags():
    """Name -> tag id, from the installed CARLA when importable."""
    try:
        import carla
        lbl = carla.CityObjectLabel
        table = {n: int(getattr(lbl, n)) for n in dir(lbl)
                 if not n.startswith("_") and isinstance(getattr(lbl, n), lbl)}
        if "RoadLines" in table or "RoadLine" in table:
            return table
    except Exception:  # noqa: BLE001  (carla absent on an analysis box)
        pass
    return dict(_FALLBACK_0_9_15)


TAGS = _resolve_tags()


def _tag(*names):
    """First matching name, so one table serves both naming generations."""
    for n in names:
        if n in TAGS:
            return TAGS[n]
    raise KeyError(f"none of {names} in this CARLA's CityObjectLabel: "
                   f"{sorted(TAGS)}")


TAG_ROADLINE = _tag("RoadLines", "RoadLine")
TAG_ROAD = _tag("Roads", "Road")
TAG_SIDEWALK = _tag("Sidewalks", "SideWalk")
TAG_SKY = _tag("Sky")

# OBSTACLES: everything a car cannot drive through.
#
# This is the channel that carries what the peer Habitat repo's 16-beam depth
# ring carried -- where the free space ends. Roads, RoadLines, Ground, Terrain
# and Bridge are excluded as drivable or traversable surface; Sky is excluded
# because it is not an obstacle at any range. Sidewalks is deliberately IN:
# mounting a kerb is a failure, not a shortcut.
_OBSTACLE_NAMES = (
    "Sidewalks", "Buildings", "Walls", "Fences", "Poles", "TrafficLight",
    "TrafficSigns", "Vegetation", "Pedestrians", "Rider", "Car", "Truck",
    "Bus", "Train", "Motorcycle", "Bicycle", "Static", "Dynamic", "Water",
    "RailTrack", "GuardRail",
)
OBSTACLE_TAGS = tuple(sorted(TAGS[n] for n in _OBSTACLE_NAMES if n in TAGS))

# A drivable tag leaking into the obstacle set is the failure mode that started
# all this, so it is asserted rather than trusted.
for _drivable in (TAG_ROAD, TAG_ROADLINE, TAG_SKY):
    if _drivable in OBSTACLE_TAGS:
        raise SystemExit(
            f"tag {_drivable} is both drivable/sky and in OBSTACLE_TAGS -- the "
            "semantic tag table is being misread.")

# Channel order of the VAE target. Fixed here and nowhere else -- the wrapper,
# the trainer and any visualisation all index through these names, so the order
# cannot drift between training and inference.
CHANNEL_LANE = 0
CHANNEL_OBSTACLE = 1
N_CHANNELS = 2
CHANNEL_NAMES = ("lane", "obstacle")

# A mask this empty means the tag channel is probably not what we think it is
# (palette conversion, wrong channel, camera pointed at sky). Cheaper to fail
# here than to train a VAE against all-zero targets for six hours.
MIN_PLAUSIBLE_ROADLINE_FRAC = 1e-4


def mask_from_semantic(semantic_rgb, tag=TAG_ROADLINE, strict=True):
    """Binary (H, W) float32 mask of `tag` from a parsed semantic image.

    `semantic_rgb` is what BaseCamera.parse returns: (H, W, 3) uint8 with the
    semantic tag in channel 0.
    """
    arr = np.asarray(semantic_rgb)
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError(f"expected (H, W, 3) semantic image, got {arr.shape}")

    tags = arr[:, :, 0]
    # Tags are small integers. An RGB image that slipped through here instead
    # would be spread across the full byte range.
    if strict and tags.max() > 40:
        raise ValueError(
            f"channel 0 max is {tags.max()}, which is not a CARLA semantic tag. "
            "The sensor is probably returning palette-converted colour rather "
            "than raw tags -- check BaseCamera.parse.")

    return (tags == tag).astype(np.float32)


def lane_mask(semantic_rgb, strict=True):
    """The training target: RoadLine markings only."""
    return mask_from_semantic(semantic_rgb, TAG_ROADLINE, strict=strict)


def drivable_mask(semantic_rgb, strict=True):
    """Road surface including markings. Useful as an auxiliary channel."""
    tags = np.asarray(semantic_rgb)[:, :, 0]
    return np.isin(tags, (TAG_ROAD, TAG_ROADLINE)).astype(np.float32)


def obstacle_mask(semantic_rgb, strict=True):
    """Binary mask of everything the vehicle cannot drive through."""
    arr = np.asarray(semantic_rgb)
    tags = arr[:, :, 0]
    if strict and tags.max() > 40:
        raise ValueError(
            f"channel 0 max is {tags.max()}, which is not a CARLA semantic tag.")
    return np.isin(tags, OBSTACLE_TAGS).astype(np.float32)


# LANE DILATION, and why it is not cheating.
#
# MEASURED on 5000 real Town01 frames without it: lane markings occupy 0.11% of
# the target, the median non-zero value is 0.046, and only 3.2% of non-zero lane
# pixels exceed 0.5. Implied pos_weight 936. A target that never asserts itself
# cannot be learned.
#
# THE CAUSE IS NOT THE DOWNSAMPLE. A calibration over 300 real Town01 frames
# put the FULL-RESOLUTION lane fraction at 0.318% and the k=1 (undilated)
# 64x64 fraction at 0.316% -- INTER_AREA preserves the mean faithfully. Lane
# markings are simply that rare: Town01 has many unmarked segments, and a
# forward camera at -15 deg spends most of its frame on road, buildings and sky.
#
# Dilating at FULL resolution before any resize is what makes the channel
# learnable. What the policy needs is where the lane structure IS, not a
# sub-pixel-accurate width, so widening a marking preserves the information that
# matters and discards only precision the 64x64 grid could never represent.
# Standard practice for thin-structure segmentation.
#
# CALIBRATION (300 Town01 frames, fraction of 64x64 pixels after downsample):
#     k    mean%   >0.25%   >0.5%   p90 val   pos_weight
#     1    0.316    0.545   0.191     0.513        315
#     3    0.420    0.685   0.297     0.603        237
#     5    0.600    0.906   0.503     0.728        166   <- chosen
#     7    0.721    1.075   0.676     0.791        138
#     9    0.923    1.352   0.887     0.879        107
# k=5 balances a tractable imbalance against geometric fidelity; k=9 is easier
# to fit but triples marking width for diminishing returns.
#
# Applied to the LANE channel only. Obstacles are already dense (49% in Town01)
# and dilating them would just erode the free space the policy needs to see.
LANE_DILATE_PX = int(os.environ.get("LANE_DILATE_PX", 5))


def _dilate(mask, ksize):
    if ksize <= 1:
        return mask
    import cv2
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    return cv2.dilate(mask.astype(np.uint8), k, iterations=1).astype(np.float32)


def semantic_targets(semantic_rgb, strict=True, lane_dilate=None):
    """The VAE target: (H, W, 2) float32, [lane, obstacle].

    Lane and obstacle are SEPARATE channels rather than one 3-way label because
    they answer different questions and the policy needs both: the lane channel
    says where the vehicle should be, the obstacle channel says where it cannot
    go. Collapsing them into one map would force the decoder to choose, and a
    junction with a car in it needs both answers at once.

    The two channels are not mutually exclusive by construction (a pixel is at
    most one tag, so in practice they are disjoint), which is why the loss is
    per-channel binary rather than a softmax over classes.
    """
    lane = mask_from_semantic(semantic_rgb, TAG_ROADLINE, strict=strict)
    lane = _dilate(lane, LANE_DILATE_PX if lane_dilate is None else lane_dilate)
    obst = obstacle_mask(semantic_rgb, strict=False)
    return np.stack([lane, obst], axis=-1).astype(np.float32)


def check_mask_plausible(masks, name="lane"):
    """Guard against training on empty targets.

    Called once over a collected dataset rather than per frame: individual
    frames legitimately contain no lane markings (inside a junction, facing a
    wall), but a whole corpus that does is a bug.
    """
    masks = np.asarray(masks)
    # Per channel: obstacles cover a third of a frame while lane markings cover
    # a few percent, so a combined mean would hide an empty lane channel behind
    # a healthy obstacle one.
    if masks.ndim == 4 and masks.shape[-1] == N_CHANNELS:
        fracs = [float(masks[..., c].mean()) for c in range(N_CHANNELS)]
    else:
        fracs = [float(masks.mean())]

    for c, frac in enumerate(fracs):
        label = CHANNEL_NAMES[c] if len(fracs) == N_CHANNELS else name
        if frac < MIN_PLAUSIBLE_ROADLINE_FRAC:
            raise SystemExit(
                f"{name}/{label} masks are {frac:.6f} positive across "
                f"{len(masks)} frames -- effectively empty. Training against "
                "this yields a VAE that outputs all zeros and scores well. "
                "Check the semantic camera tag channel.")
        print(f"[LaneMask] {name}/{label}: {frac * 100:.2f}% positive pixels "
              f"over {len(masks)} frames")
    return fracs if len(fracs) > 1 else fracs[0]


def pos_weight_for(masks):
    """Per-channel BCE positive-class weight for the measured imbalance.

    Lane markings occupy a few percent of a forward-facing frame. Unweighted
    BCE on that is minimised by predicting zero everywhere, which is exactly
    the degenerate solution that looks like a converged loss curve. Obstacles
    are far denser and need a much smaller weight, so this is per channel --
    one shared weight would over-correct the obstacle channel into predicting
    obstacles everywhere.
    """
    masks = np.asarray(masks)
    if masks.ndim == 4 and masks.shape[-1] == N_CHANNELS:
        fr = np.clip(masks.reshape(-1, N_CHANNELS).mean(axis=0), 1e-6, 1 - 1e-6)
        return ((1.0 - fr) / fr).astype(np.float32)
    frac = min(max(float(masks.mean()), 1e-6), 1 - 1e-6)
    return np.float32((1.0 - frac) / frac)


# Sensor block to merge into an env config's "sensors" dict. Pose is deliberately
# identical to the rgb camera: the VAE learns RGB -> mask, so any disagreement in
# extrinsics between the two cameras becomes irreducible label noise.
def semantic_sensor_block(rgb_block):
    """Build a semantic camera spec matching an existing rgb camera spec."""
    keys = ("image_size_x", "image_size_y", "fov", "transform")
    missing = [k for k in keys if k not in rgb_block]
    if missing:
        raise ValueError(f"rgb sensor block is missing {missing}; cannot mirror its pose")
    block = {k: rgb_block[k] for k in keys}
    block["type"] = "sensor.camera.semantic_segmentation"
    return block
