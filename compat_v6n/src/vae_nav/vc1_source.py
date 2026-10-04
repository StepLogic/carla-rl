"""The VC-1 visual-cortex encoder as a LatentSource backend.

Frozen Meta VC-1 ViT-B (eai-vc), 224x224 input, 768-d CLS embedding. Chosen
because VC-1's training data includes image-goal navigation; the embedding is
the goal-facing representation this arm wants. The backbone is FROZEN --
eval() and no_grad everywhere, never trained.

WHY A CLASS BESIDE LatentSource AND NOT INSIDE IT. LatentSource is the VAE
path (JAX, pickle checkpoints); MobileNetSource is the torch-features path.
VC-1 needs its own preprocessing (resize to 224, the vit_transforms normalise)
and its own failure mode -- a ~400 MB checkpoint download on first use -- so
it is its own class implementing the same small interface, registered in
latent_source.from_env, which stays the ONE place a backend is chosen.

TRANSFORM CONTRACT (verified against vc_models' source and empirically in
scripts/vc1_encoder_check.py). vc_models.transforms.vit_transforms is
Resize(256, bicubic) -> CenterCrop(224) -> ToTensorIfNot -> Normalize
(ImageNet stats). ToTensorIfNot passes a torch tensor through UNCHANGED -- no
/255, no permute -- and applies the standard PIL/tensor conversion only to
non-tensor input, so a float CHW tensor already scaled to [0, 1] is the right
input and no PIL round-trip is needed. The transform accepts a batched
(1, 3, H, W) tensor directly (torchvision's (..., H, W) convention).

NOTE ON EVAL MODE. vc_models.models.load_model calls `model.train()` on its
way out (it probes the embedding width under no_grad, then leaves the module
in train mode), so this class MUST call .eval() itself -- which it does in
__init__, together with requires_grad=False on every parameter.
"""
import os

import numpy as np

VC1_MODEL_NAME = "vc1_vitb"
VC1_EMBED_DIM = 768


class VcSource:
    def __init__(self, model_name=VC1_MODEL_NAME, device=None):
        self.name = model_name
        self.latent_dim = VC1_EMBED_DIM
        # The trainer's town-leak check reads trained_on_towns on EVERY
        # backend (train_carla_tqc.py: a perception module trained on a
        # held-out town leaks it like a policy does). VC-1 is a web-scale
        # foundation model (Ego4D and similar) with NO CARLA town provenance,
        # so the honest value is empty -- no held-out leak, which is exactly
        # the encoder-clean transfer this arm exists to produce (the VAE
        # arms cannot give that number; their encoder saw four towns).
        self.trained_on_towns = []
        try:
            import torch
            from hydra import compose, initialize
            import hydra
        except ImportError as e:
            raise SystemExit(
                f"PERCEPTION=vc1 needs torch, hydra-core and vc_models "
                f"(pip install -e dependencies/eai-vc/vc_models): {e}")

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # Compose vc_models' hydra config; compose() wraps the group under a
        # top-level "model" key, hence ["model"]. The transform arrives as the
        # third element of load_model's return, so it is loaded HERE, at
        # construction, with the model -- no lazy fallback needed.
        with initialize(version_base=None):
            cfg = compose(config_name=f"model/{model_name}")
        model, embed_dim, transform, _metadata = hydra.utils.instantiate(
            cfg["model"])
        if int(embed_dim) != VC1_EMBED_DIM:
            raise SystemExit(
                f"vc_models reported embed_dim={embed_dim}, expected "
                f"{VC1_EMBED_DIM}. The observation space is built for "
                f"{VC1_EMBED_DIM}.")
        # load_model leaves the module in TRAIN mode (see module docstring);
        # eval() here is load-bearing, not cosmetic.
        self._model = model.to(device).eval()
        self._transform = transform

        # THE ENCODER MUST NOT TRAIN. A single .train() call by anything
        # sharing the module would re-enable dropout/BN updates.
        for p in self._model.parameters():
            p.requires_grad = False

        print(f"[Latent] backend '{self.name}' (dim={self.latent_dim}, "
              f"device={device}, frozen vc_models VC-1)")

    def latent(self, rgb):
        """(H, W, 3) uint8 or float RGB -> (768,) float32."""
        import torch
        with torch.no_grad():
            x = self._prepare(rgb).to(self.device)
            z = self._model(x)
        return z.squeeze(0).cpu().numpy().astype(np.float32)

    def _prepare(self, rgb):
        import torch
        x = torch.from_numpy(np.asarray(rgb, np.float32) / 255.0)
        x = x.permute(2, 0, 1)[None]                  # HWC -> 1CHW
        # vit_transforms resizes to 224 and normalises; it expects a batched
        # CHW float tensor in [0, 1] (ToTensorIfNot passes tensors through).
        return self._transform(x)

    def identity(self):
        return {"latent_backend": self.name, "latent_dim": self.latent_dim,
                "image_size": 224}

    @staticmethod
    def assert_matches(checkpoint_identity, source):
        # Lazy import: latent_source imports nothing from here at module
        # scope, but keeping this import inside the method makes the
        # direction explicit and immune to future import-order changes.
        from vae_nav.latent_source import LatentSource
        return LatentSource.assert_matches(checkpoint_identity, source)