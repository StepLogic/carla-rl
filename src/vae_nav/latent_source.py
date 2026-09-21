"""Where the perception backend is chosen -- exactly one place.

Modelled directly on the peer Habitat repo's ring_source.RingSource, whose
author recommended copying the pattern. The reasoning transfers unchanged: if
more than one call site can decide where the policy's perception comes from,
then a difference between two runs can come from the sampler rather than from
the perception itself, and the substitution claim stops being testable.

WHAT THIS REPLACES. In their setup a frozen MonoDepthNet produced metric depth
which ring_extract reduced to a 16-beam ring, and that ring was a slice of the
policy's observation vector. Here a frozen lane+obstacle VAE produces a 128-d
posterior mean which is published as its own observation key. The reduction
step has no analogue -- a lane mask has no natural projection onto bearings the
way a depth image does -- so the latent goes in whole.

IDENTITY TRAVELS WITH THE POLICY. `self.name` is written into the RL checkpoint
so a policy can never be silently evaluated against a perception backend it
never trained on. Swapping backends changes the observation distribution;
swappability belongs at the interface, not in the checkpoint.
"""
import os
import pickle

import numpy as np

LATENT_DIM = 128
IMAGE_SIZE = int(os.environ.get("VAE_IMAGE_SIZE", 120))


class LatentSource:
    """Produces the policy-facing latent, from a frozen VAE or a stub."""

    def __init__(self, ckpt_path=None, latent_dim=LATENT_DIM, deterministic=True):
        self.latent_dim = int(latent_dim)
        self.deterministic = bool(deterministic)
        self._model = None
        self._params = None
        self.image_size = IMAGE_SIZE
        self.trained_on_towns = []
        self.channels = []

        if not ckpt_path:
            # Explicit zero-latent stub. Exists so an ablation can measure what
            # the policy achieves with NO perception at all, which is the only
            # honest baseline for "does the latent help". Named distinctly so
            # it can never be mistaken for a real backend in a checkpoint.
            self.name = "zero_latent"
            return

        with open(ckpt_path, "rb") as f:
            ckpt = pickle.load(f)

        # Fail loudly on a checkpoint whose latent width disagrees with what the
        # observation space was built for: the alternative is a reshape error
        # thousands of steps into training, or worse, a silent broadcast.
        ck_dim = int(ckpt.get("latent_dim", self.latent_dim))
        if ck_dim != self.latent_dim:
            raise SystemExit(
                f"VAE checkpoint {ckpt_path} has latent_dim={ck_dim} but the "
                f"observation space was built for {self.latent_dim}. These must "
                "match; rebuild one or the other.")

        self.name = str(ckpt.get("backend", "lane_obstacle_vae"))
        self.image_size = int(ckpt.get("image_size", IMAGE_SIZE))
        self.trained_on_towns = list(ckpt.get("trained_on_towns", []))
        self.channels = list(ckpt.get("channels", []))
        self._params = ckpt["params"]

        # Imported lazily: an ablation running the zero-latent stub should not
        # need JAX resident just to construct the wrapper.
        from vae_nav.lane_vae import LaneVAE
        # image_size comes from the CHECKPOINT, so a model trained at 64 and
        # one trained at 120 both load correctly and neither is silently
        # reshaped to whatever the current default happens to be.
        self._model = LaneVAE(latent_dim=self.latent_dim,
                              image_size=self.image_size)

        print(f"[Latent] backend '{self.name}' from {ckpt_path} "
              f"(dim={self.latent_dim}, image={self.image_size}, "
              f"val_iou={ckpt.get('val_iou', float('nan')):.4f}, "
              f"trained on {self.trained_on_towns})")

    # -- the one call the wrapper makes ------------------------------------
    def latent(self, rgb):
        """(H, W, 3) uint8 or float RGB -> (latent_dim,) float32."""
        if self._model is None:
            return np.zeros(self.latent_dim, np.float32)

        x = self._prepare(rgb)
        import jax
        mu = self._encode(x)
        return np.asarray(jax.device_get(mu)[0], np.float32)

    def _encode(self, x):
        # Cached jit: re-tracing per step would dominate the env step time.
        if not hasattr(self, "_enc_fn"):
            import jax
            model, params = self._model, self._params

            @jax.jit
            def _fn(arr):
                return model.apply({"params": params}, arr, method=model.encode)

            self._enc_fn = _fn
        return self._enc_fn(x)

    def _prepare(self, rgb):
        # The SAME transform collection used. Any divergence here is a
        # train/inference mismatch that degrades the latent silently.
        from vae_nav.preprocess import to_vae_input
        return to_vae_input(rgb, self.image_size)[None]

    # -- provenance --------------------------------------------------------
    def identity(self):
        """Recorded into the RL checkpoint. See module docstring."""
        return {
            "latent_backend": self.name,
            "latent_dim": self.latent_dim,
            "image_size": self.image_size,
            "vae_trained_on_towns": list(self.trained_on_towns),
            "channels": list(self.channels),
        }

    @staticmethod
    def assert_matches(checkpoint_identity, source):
        """Refuse to evaluate a policy against a backend it never trained on."""
        want = checkpoint_identity.get("latent_backend")
        got = source.name
        if want is not None and want != got:
            raise SystemExit(
                f"policy was trained with latent backend '{want}' but eval is "
                f"using '{got}'. That changes the observation distribution, so "
                "the comparison would be meaningless. Pass the matching "
                "VAE_CKPT or evaluate a different checkpoint.")


def from_env(latent_dim=LATENT_DIM):
    """Build the source the same way in training and eval.

    Both paths call this rather than constructing LatentSource themselves,
    which is what keeps them from drifting apart -- the failure their
    map_harness.py has to guard against by staying byte-identical to the
    wrapper.

    PERCEPTION selects the backend, and this stays the ONLY place that chooses.
    That is the module's whole thesis: if two call sites could pick, a
    difference between runs could come from the sampler rather than the
    perception, and the substitution claim stops being testable.

        PERCEPTION=vae        frozen lane+obstacle VAE, 128-d      (needs VAE_CKPT)
        PERCEPTION=mobilenet  frozen MobileNetV3-Small features[:4]
        PERCEPTION=vc1        frozen Meta VC-1 ViT-B CLS, 768-d

    The two produce vectors of different width AND different meaning, so
    assert_matches -- which compares `latent_backend` recorded in the
    checkpoint -- is what stops a policy being evaluated against the wrong one.
    """
    backend = os.environ.get("PERCEPTION", "vae").lower()
    if backend in ("mobilenet", "mobilenetv3", "mnv3"):
        from vae_nav.mobilenet_source import MobileNetSource
        return MobileNetSource(image_size=IMAGE_SIZE)
    if backend in ("vc1", "vc1_vitb"):
        from vae_nav.vc1_source import VcSource
        # VC1_DEVICE=cpu: two windowed CARLA servers + torch + jax do not fit
        # in 12.3 GB (CARLA alone ~11.2 GB, measured). Identical computation
        # on a different device -- the encoder is frozen either way.
        return VcSource(device=os.environ.get("VC1_DEVICE") or None)
    if backend not in ("vae", ""):
        raise SystemExit(
            f"PERCEPTION='{backend}' is not a known backend. Use 'vae', "
            "'mobilenet' or 'vc1'. Failing loudly rather than silently "
            "falling back to the VAE, because a silent fallback would make a "
            "backend-ablation compare a run against itself.")
    return LatentSource(os.environ.get("VAE_CKPT", ""), latent_dim=latent_dim)
