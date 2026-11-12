"""Evaluation video recording, with the VAE's own view alongside the camera.

WHAT GOES IN THE FRAME. Left: the RGB the policy actually sees. Right: the
lane and obstacle masks the frozen VAE decodes from it. Recording only the
camera would show you that an episode failed; recording the decode shows you
WHY -- a policy that drives into a wall because its obstacle channel went blank
looks identical, on a plain camera video, to one that simply chose badly.

That distinction is the whole point of a perception-substitution experiment, so
it is worth the extra decoder pass on the small number of episodes recorded.

RESOLUTION IS A LIE, DELIBERATELY. The policy's frame is 84x84 and the VAE's is
64x64; both are upscaled with INTER_NEAREST for viewing. Nearest-neighbour
rather than a smooth interpolation so that what you see is the actual pixel
grid the network had, not a prettier version of it.
"""
import os

import numpy as np

# Recording every episode of a 60-episode eval at every checkpoint fills a disk
# quickly and nobody watches 60 videos. A handful per eval is enough to see
# what the perception is doing.
DEFAULT_MAX_VIDEOS = int(os.environ.get("EVAL_VIDEO_EPISODES", 6))
FPS = int(os.environ.get("EVAL_VIDEO_FPS", 20))
PANEL = 256


def _upscale(img, size=PANEL):
    import cv2
    arr = np.asarray(img)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.dtype != np.uint8:
        arr = np.clip(arr * 255.0 if arr.max() <= 1.5 else arr, 0, 255).astype(np.uint8)
    return cv2.resize(arr, (size, size), interpolation=cv2.INTER_NEAREST)


def _mask_panel(masks):
    """(H, W, 2) probabilities -> an RGB panel. Lane green, obstacle red."""
    if masks is None:
        return np.zeros((PANEL, PANEL, 3), np.uint8)
    m = np.asarray(masks)
    h, w = m.shape[:2]
    rgb = np.zeros((h, w, 3), np.float32)
    rgb[..., 1] = m[..., 0]                       # lane -> green
    if m.shape[-1] > 1:
        rgb[..., 0] = m[..., 1]                   # obstacle -> red
    return _upscale(rgb)


class EvalVideoRecorder:
    """Writes one mp4 per recorded evaluation episode."""

    def __init__(self, out_dir, step, max_videos=DEFAULT_MAX_VIDEOS,
                 latent_source=None, decode=True):
        # decode=False drops the VAE mask panel. That panel is a full
        # encode->decode of the frozen VAE ON THE GPU for EVERY frame, and it
        # is the only GPU cost video adds: measured 2026-09-11, a sweep
        # peaked at 9.8 GB of 12.3 with it against 7.4 GB without, and the
        # simulator (Vulkan, same GPU) died mid-run in every video arm and in
        # no video-free one. The policy-view half of the video is pure CPU.
        self.decode = bool(decode)
        self.dir = os.path.join(out_dir, "videos")
        os.makedirs(self.dir, exist_ok=True)
        self.step = int(step)
        self.max_videos = int(max_videos)
        self.latent_source = latent_source
        self.written = 0
        self._frames = []
        self._active = False
        self._meta = {}

    def start_episode(self, index, town):
        self._active = self.written < self.max_videos
        self._frames = []
        self._meta = {"index": index, "town": town}
        return self._active

    def record(self, rgb, info=None):
        """One frame: the policy's view, plus the VAE decode when available."""
        if not self._active or rgb is None:
            return
        import cv2

        left = _upscale(rgb)
        if self.decode:
            right = _mask_panel(self._decode(rgb))
            frame = np.concatenate([left, right], axis=1)
        else:
            frame = np.ascontiguousarray(left)

        if info:
            d = info.get("distance_to_goal")
            txt = f"d={d:.1f}m" if isinstance(d, (int, float)) else ""
            if info.get("collision"):
                txt += " HIT"
            cv2.putText(frame, txt, (6, PANEL - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, "policy view", (6, 16), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (255, 255, 255), 1, cv2.LINE_AA)
        if self.decode:
            cv2.putText(frame, "VAE lane/obst", (PANEL + 6, 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
                        cv2.LINE_AA)
        self._frames.append(frame)

    def _decode(self, rgb):
        """Run the frozen VAE's full encode->decode for the mask panel.

        Returns None for the zero-latent ablation, which has no decoder -- the
        panel is then black, which correctly represents "no perception".
        """
        src = self.latent_source
        if src is None or getattr(src, "_model", None) is None:
            return None
        try:
            import jax
            import jax.nn as jnn
            x = src._prepare(rgb)
            logits, _, _ = src._model.apply({"params": src._params}, x,
                                            deterministic=True)
            return np.asarray(jax.device_get(jnn.sigmoid(logits))[0], np.float32)
        except Exception:  # noqa: BLE001
            # A decode failure must never take down an evaluation run.
            return None

    def end_episode(self, success):
        if not self._active or not self._frames:
            self._active = False
            return None
        import cv2

        tag = "success" if success else "fail"
        name = (f"eval_step{self.step}_ep{self._meta['index']:03d}_"
                f"{self._meta['town']}_{tag}.mp4")
        path = os.path.join(self.dir, name)
        h, w = self._frames[0].shape[:2]
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
        for f in self._frames:
            writer.write(f[:, :, ::-1])   # RGB -> BGR for OpenCV
        writer.release()
        n_frames = len(self._frames)
        self.written += 1
        self._frames = []
        self._active = False
        print(f"[Video] {path} ({n_frames} frames, {tag})")
        return path
