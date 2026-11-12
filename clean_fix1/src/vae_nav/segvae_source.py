"""PERCEPTION=segvae: the frozen ResNet-9 segmentation VAE's latent (mu).

User 2026-09-28: "VC1 is overkill"; the seg-VAE (vae_nav.seg_vae) was trained on
dense CARLA segmentation. Its 512-d mu locates the car in its lane better than
VC-1's pooled readout on the lane-position probe (distance from lane centre R^2
0.59 vs 0.41; signed offset inside the lane 0.63 vs 0.38), in a vector 1/7 the
size.

The frame is the env's info["rgb"] (120x120 uint8 RGB after the channel fix),
fed exactly as training fed the collector's frames: uint8 / 255, no resize.
"""
import os
import pickle

import numpy as np


class SegVaeSource:
    def __init__(self, ckpt_path):
        if not ckpt_path or not os.path.exists(ckpt_path):
            raise SystemExit(f"PERCEPTION=segvae needs SEGVAE_CKPT; got {ckpt_path!r}")
        import jax
        from vae_nav import seg_vae as sv
        c = pickle.load(open(ckpt_path, "rb"))
        self.latent_dim = int(c["latent_dim"])
        self.image_size = 120
        self._model = sv.SegVAE(latent_dim=self.latent_dim,
                                dec_start=c.get("dec_start", 15),
                                skips=c.get("skips", False))
        self._params = jax.tree_util.tree_map(np.asarray, c["params"])
        tag = os.path.basename(os.path.dirname(os.path.abspath(ckpt_path)))
        self.name = f"segvae_{tag}_s{c['step']}"
        self.trained_on_towns = ["Town01", "Town04", "Town05", "Town07", "Town10HD"]
        self.channels = ["other", "road", "lane_line", "sidewalk", "vehicle",
                         "person", "structure", "nature", "sky"]
        model, params = self._model, self._params
        self._enc = jax.jit(lambda x: model.apply(params, x, method=sv.SegVAE.encode))
        print(f"[Latent] backend '{self.name}' from {ckpt_path} (dim={self.latent_dim}, "
              f"frozen seg-VAE, trained on {self.trained_on_towns})", flush=True)

    def latent(self, rgb):
        """(120, 120, 3) uint8 RGB -> (latent_dim,) float32 mu."""
        x = np.asarray(rgb)
        if x.shape[:2] != (self.image_size, self.image_size):
            import cv2
            x = cv2.resize(x, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        x = x.astype(np.float32)
        if x.max() > 1.5:
            x = x / 255.0
        return np.asarray(self._enc(x[None])[0], np.float32)

    def identity(self):
        return {"latent_backend": self.name, "latent_dim": self.latent_dim,
                "image_size": self.image_size,
                "vae_trained_on_towns": list(self.trained_on_towns),
                "channels": list(self.channels)}
