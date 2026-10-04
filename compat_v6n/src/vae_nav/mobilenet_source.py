"""Frozen MobileNetV3-Small early blocks as the policy's perception backend.

Replaces the lane+obstacle VAE. Same contract as LatentSource -- `.latent(rgb)`
-> a float32 vector, `.identity()` recorded into the RL checkpoint -- so
`LatentSource.assert_matches` still refuses to evaluate a policy against a
backend it never trained on. That guard matters more here, not less: the two
backends produce observation vectors of different width AND different meaning.

WHY EARLY BLOCKS RATHER THAN A DEEPER CUT OR A ViT. The goal policy needs local
drivable geometry -- where the lane edge is, where the obstacle is -- which is
spatial and low-level. `features[:4]` keeps a 15x15 feature map, so WHERE a
thing is survives. A DINOv2 CLS token discards spatial layout entirely and its
patch stride is coarser, and it costs ~50-100x the FLOPs on a path that runs
once per environment step, 500k times. DINOv2's strength is semantic
correspondence, which is the GRAPH's question (is this node the same place as
that one), not the controller's -- and the Habitat peer runs it batched at
flush for exactly that reason rather than in their control loop.

Reuses the encoder already in this repo (sptm_train.FrozenMobileNetV3Encoder):
mobilenet_v3_small(weights=DEFAULT).features[:4], frozen, eval.

THE POOL IS NOT COSMETIC -- IT IS WHY THIS FITS IN MEMORY. Measured:

    120x120 -> (1, 24, 15, 15) = 5400-d -> 43.2 GB of replay at 1M capacity
     84x84  -> (1, 24, 11, 11) = 2904-d -> 23.2 GB
    pooled 4x4                 =  384-d ->  3.1 GB
    (the VAE's 128-d was 1.0 GB)

The raw flatten is not deployable at this buffer size, and this repo has
already filled a root filesystem once. Adaptive-average-pooling to 4x4 keeps
the coarse left/centre/right and near/far structure that steering actually
needs while making the buffer affordable. MOBILENET_POOL=0 restores the
faithful flatten for anyone who wants it and can pay for it.

IMAGENET NORMALISATION IS REQUIRED. These are pretrained weights and they
expect ImageNet statistics. Feeding [0,1] without it is a silent distribution
mismatch that degrades features rather than raising. The env's camera is
`sensor.camera.rgb`, so natural-image statistics are the right prior here --
this would NOT hold for a semantic-segmentation camera.

TORCH RUNS ON CPU BY DEFAULT. JAX owns the GPU (XLA_PYTHON_CLIENT_MEM_FRACTION
=0.5) and torch CUDA is broken on this box anyway ("the NVIDIA driver on your
system is too old"). Four mobile blocks on 120x120 are cheap on CPU; contending
for GPU memory with the learner is not worth the milliseconds.
"""
import os

import numpy as np

# 0 disables pooling and flattens the raw feature map -- see the docstring.
MOBILENET_POOL = int(os.environ.get("MOBILENET_POOL", 4))
MOBILENET_DEVICE = os.environ.get("MOBILENET_DEVICE", "cpu")
# One thread by default: the env loop and the JAX learner are already competing
# for cores, and torch's default pool oversubscribes them.
MOBILENET_THREADS = int(os.environ.get("MOBILENET_THREADS", 1))

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


class MobileNetSource:
    """Frozen MobileNetV3-Small features[:4], optionally pooled."""

    def __init__(self, image_size=120, pool=MOBILENET_POOL,
                 device=MOBILENET_DEVICE):
        import torch
        import torchvision.models as models

        torch.set_num_threads(max(1, MOBILENET_THREADS))
        self.torch = torch
        self.image_size = int(image_size)
        self.pool = int(pool)
        self.device = device

        net = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        self.features = net.features[:4].eval().to(device)
        for p in self.features.parameters():
            p.requires_grad = False

        # DERIVED, NEVER HARDCODED. A previous version of this repo carried a
        # literal 2904 that was only correct at 84x84; changing the image size
        # silently produced a shape error thousands of steps into training.
        with torch.no_grad():
            probe = self._forward(torch.zeros(1, 3, self.image_size,
                                              self.image_size, device=device))
        self.latent_dim = int(probe.numel())
        self.name = (f"mobilenet_v3_small_f4"
                     f"{'_pool%d' % self.pool if self.pool else '_flat'}")
        self.trained_on_towns = []          # ImageNet weights, no town provenance
        self.channels = []

        print(f"[Latent] backend '{self.name}' "
              f"(dim={self.latent_dim}, image={self.image_size}, "
              f"device={device}, frozen ImageNet weights)")

    def _forward(self, x):
        feat = self.features(x)
        if self.pool:
            feat = self.torch.nn.functional.adaptive_avg_pool2d(
                feat, (self.pool, self.pool))
        return feat.flatten(1)

    # -- the one call the wrapper makes ------------------------------------
    def latent(self, rgb):
        """(H, W, 3) uint8 or float RGB -> (latent_dim,) float32."""
        torch = self.torch
        x = self._prepare(rgb)
        with torch.no_grad():
            out = self._forward(x)
        return out.cpu().numpy()[0].astype(np.float32)

    def _prepare(self, rgb):
        from vae_nav.preprocess import to_vae_input
        # SAME resize/scale path the VAE used, so the two backends differ in the
        # encoder and nothing else.
        arr = np.asarray(to_vae_input(rgb, self.image_size), np.float32)
        if arr.ndim == 4:
            arr = arr[0]
        if arr.shape[-1] != 3:                      # tolerate (C, H, W)
            arr = np.transpose(arr, (1, 2, 0))
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD  # required; see docstring
        chw = np.transpose(arr, (2, 0, 1))[None]
        return self.torch.from_numpy(np.ascontiguousarray(chw)).to(self.device)

    # -- provenance --------------------------------------------------------
    def identity(self):
        return {
            "latent_backend": self.name,
            "latent_dim": self.latent_dim,
            "image_size": self.image_size,
            "vae_trained_on_towns": [],
            "channels": [],
        }
