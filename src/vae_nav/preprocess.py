"""The ONE image transform, shared by dataset collection and RL inference.

WHY THIS FILE EXISTS. The VAE is trained offline on frames captured straight
from a CARLA camera, and used online on frames the RL env has already
processed. If those two paths differ by so much as a resize order, the encoder
sees a different distribution at inference than it was fitted on, the latent
degrades, and nothing in the RL logs says so -- the policy simply learns less
than it should.

The env's path is fixed and not ours to change (train_env_config_v4_image_goal
shares it with the CORL 2026 experiments):

    raw 748x348 BGRA
      -> rlib_integration.helper.post_process_image(
             normalized=False, crop=False, grayscale=False, image_size=84)
         i.e. cv2.resize(..., (84, 84), INTER_AREA) then BGR->RGB
      -> experiment.obs_img            # (84, 84, 3) uint8, RGB

so collection must reproduce it exactly and then both sides take the SAME final
step down to the VAE's input size. That is what to_vae_input does.

Note the aspect ratio: 748x348 squashed to a square is a heavy distortion, but
it is the distortion the policy's camera already has. Matching it matters more
than correcting it -- a VAE trained on undistorted frames would be the one out
of distribution.
"""
import numpy as np

import os

# RESOLUTION IS ONE CHAIN, NOT TWO KNOBS.
#
# The frame reaches the VAE as
#     raw 748x348  ->  post_process_image(ENV_IMAGE_SIZE)  ->  VAE_IMAGE_SIZE
# so VAE_IMAGE_SIZE can never recover detail that ENV_IMAGE_SIZE has already
# thrown away. Raising only the second value UPSAMPLES -- at env 84 a "120x120"
# VAE input is 84x84 of information stretched over 120x120 of pixels, costing
# compute and adding nothing. The two move together or not at all.
#
# This matters here because the lane channel's errors sit exactly on the
# markings' edges, which is a resolution symptom: a lane line is a few pixels
# wide at full res and survives a squash to 84 only as a smear.
ENV_IMAGE_SIZE = int(os.environ.get("ENV_IMAGE_SIZE", 120))
VAE_IMAGE_SIZE = int(os.environ.get("VAE_IMAGE_SIZE", 120))
if VAE_IMAGE_SIZE > ENV_IMAGE_SIZE:
    raise SystemExit(
        f"VAE_IMAGE_SIZE={VAE_IMAGE_SIZE} exceeds ENV_IMAGE_SIZE="
        f"{ENV_IMAGE_SIZE}. The env frame is the source; asking the VAE for "
        "more pixels than it carries only interpolates.")


def env_frame(raw_rgb, size=ENV_IMAGE_SIZE):
    """Raw camera frame -> the env's (size, size, 3) uint8 RGB, bit-for-bit.

    Mirrors post_process_image(normalized=False, crop=False, grayscale=False).
    Input is expected already in RGB order, as BaseCamera.parse returns it.
    """
    import cv2
    arr = np.asarray(raw_rgb)
    if arr.shape[0] != size or arr.shape[1] != size:
        arr = cv2.resize(arr, (size, size), interpolation=cv2.INTER_AREA)
    return arr.astype(np.uint8)


def to_vae_input(image, size=VAE_IMAGE_SIZE):
    """Env-shaped frame -> (size, size, 3) float32 in [0, 1] for the VAE.

    Accepts either a raw camera frame or an already-84x84 obs_img: both are
    resized to `size`, so the only requirement is that the caller has not
    applied a DIFFERENT resize first.
    """
    import cv2
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.shape[0] != size or arr.shape[1] != size:
        arr = cv2.resize(arr, (size, size), interpolation=cv2.INTER_AREA)
    arr = arr.astype(np.float32)
    # A caller handing over 0-255 floats would saturate every conv and produce
    # a latent with the right shape and no information.
    if arr.max() > 1.5:
        arr = arr / 255.0
    return arr


def mask_to_vae_target(mask, size=VAE_IMAGE_SIZE):
    """(H, W, C) binary mask -> (size, size, C) float32 coverage fractions.

    Resized per channel with INTER_AREA so a one-pixel lane marking at full
    resolution becomes a small non-zero value rather than vanishing. The
    resulting soft targets in [0, 1] are valid BCE targets.
    """
    import cv2
    m = np.asarray(mask, np.float32)
    if m.ndim == 2:
        m = m[..., None]
    if m.shape[0] == size and m.shape[1] == size:
        return m
    return np.stack(
        [cv2.resize(m[:, :, c], (size, size), interpolation=cv2.INTER_AREA)
         for c in range(m.shape[2])], axis=-1).astype(np.float32)
