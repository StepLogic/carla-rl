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

SPATIAL READOUT (VC1_READOUT, read at import time like the repo's other
module-level flags). "cls" (the default) is today's 768-d CLS embedding,
byte-identical to the pre-readout code path. "grid2" instead returns the
196 patch tokens (14x14x768, the model's own pre-pooling sequence minus the
CLS token) average-pooled 2x2 to a (7, 7, 768) = 37632-d spatial grid, for
the patch-grid observation. "pyramid" (2026-09-21, the pooled v8) returns a
3840-d vector: the mean over ALL 196 patch tokens concatenated with the mean
of each 14x14 QUADRANT -- global scene semantics plus which quadrant the
road / junction / obstacle sits in, the layout a steering critic needs, at
1/10 the buffer footprint of grid2 (user choice among pooling options).
The token extraction is verified against the
live model by scripts/vc1_grid_probe.py: the CLS token of the normed
sequence equals model(x) exactly (max abs diff 0.0), and
model.classifier_feature is asserted to be "use_cls_token" -- anything else
and the norm-then-slice recipe is wrong for the checkpoint.
"""
import os

import numpy as np

# VC1_MODEL selects the VC-1 backbone (user 2026-09-25: "run v13 with vit-l").
# Both are patch16 at 224, so both emit a 14x14 = 196 patch grid; they differ
# ONLY in width, so every readout below is written against VC1_EMBED_DIM
# rather than a literal. Default vc1_vitb keeps every existing run identical.
#
# THE WIDTH IS NOT COSMETIC. It sets the observation: pyramid goes 3840 ->
# 5120 and grid2 37632 -> 50176, so a checkpoint trained under one backbone
# CANNOT load under the other -- policy_io refuses it, which is the intended
# behaviour. A ViT-L run starts from scratch and is its own baseline.
#
# Per Meta's CortexBench table, ViT-L leads ViT-B on ImageNav (70.3 +- 0.7 vs
# 68.9 +- 0.7) and on mean rank (2.4 vs 6.9) -- but the two rows also differ
# in pre-training data (Ego4D+MNI vs Ego4D+MN), so it is not a size-only
# comparison.
VC1_EMBED_DIMS = {"vc1_vitb": 768, "vc1_vitl": 1024}
VC1_MODEL_NAME = os.environ.get("VC1_MODEL", "vc1_vitb").lower()
if VC1_MODEL_NAME not in VC1_EMBED_DIMS:
    raise SystemExit(
        f"VC1_MODEL={VC1_MODEL_NAME!r} is not one of {sorted(VC1_EMBED_DIMS)}")
VC1_EMBED_DIM = VC1_EMBED_DIMS[VC1_MODEL_NAME]
VC1_TOKENS = 14  # 224 / patch 16, the same for both backbones

READOUT = os.environ.get("VC1_READOUT", "cls").lower()
if READOUT not in ("cls", "grid2", "pyramid"):
    raise SystemExit(f"VC1_READOUT={READOUT!r} is not cls|grid2|pyramid")
VC1_GRID_DIM = (VC1_TOKENS // 2) ** 2 * VC1_EMBED_DIM   # 7x7xD
VC1_PYRAMID_DIM = 5 * VC1_EMBED_DIM  # global mean + 4 quadrant means


def pool_grid(tokens, pool=2):
    """(196, D) patch tokens -> (7*7*D,) spatial grid, 2x2 cell average."""
    n, d = VC1_TOKENS, VC1_EMBED_DIM
    t = np.asarray(tokens, np.float32).reshape(n, n, d)
    g = t.reshape(n // pool, pool, n // pool, pool, d).mean(axis=(1, 3))
    return g.reshape(-1).astype(np.float32)


def pool_pyramid(tokens):
    """(196, D) patch tokens -> (5*D,) spatial pyramid.

    Global mean over all 196 tokens, then the mean of each 7x7 quadrant of
    the 14x14 patch map (top-left, top-right, bottom-left, bottom-right).
    Order is [global, tl, tr, bl, br] -- the docstring's contract, and the
    order any consumer of the 3840-d vector must assume.
    """
    n, d, h = VC1_TOKENS, VC1_EMBED_DIM, VC1_TOKENS // 2
    t = np.asarray(tokens, np.float32).reshape(n, n, d)
    quads = (t[:h, :h].mean(axis=(0, 1)), t[:h, h:].mean(axis=(0, 1)),
             t[h:, :h].mean(axis=(0, 1)), t[h:, h:].mean(axis=(0, 1)))
    return np.concatenate((t.mean(axis=(0, 1)),) + quads).astype(np.float32)


def backend_name():
    return VC1_MODEL_NAME if READOUT == "cls" else f"{VC1_MODEL_NAME}_{READOUT}"


class VcSource:
    def __init__(self, model_name=VC1_MODEL_NAME, device=None):
        self.name = backend_name()
        self.latent_dim = VC1_GRID_DIM if READOUT == "grid2" \
            else VC1_PYRAMID_DIM if READOUT == "pyramid" else VC1_EMBED_DIM
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

        self._model, self._transform = self._load_model()

        print(f"[Latent] backend '{self.name}' (dim={self.latent_dim}, "
              f"readout={READOUT}, device={device}, frozen vc_models VC-1)")

    def _load_model(self):
        """Hydra-compose + instantiate the vc_models checkpoint; returns
        (model, transform) on the class's device. Kept as a method so tests
        can patch the (expensive) load out and wire attributes directly.
        Both readouts load the SAME vc1_vitb checkpoint -- backend_name()
        is the observation-identity name, not a config name."""
        from hydra import compose, initialize
        import hydra
        with initialize(version_base=None):
            cfg = compose(config_name=f"model/{VC1_MODEL_NAME}")
        model, embed_dim, transform, _metadata = hydra.utils.instantiate(
            cfg["model"])
        if int(embed_dim) != VC1_EMBED_DIM:
            raise SystemExit(
                f"vc_models reported embed_dim={embed_dim}, expected "
                f"{VC1_EMBED_DIM}. The observation space is built for "
                f"{VC1_EMBED_DIM}.")
        # load_model leaves the module in TRAIN mode (see module docstring);
        # eval() here is load-bearing, not cosmetic.
        model = model.to(self.device).eval()
        # The grid recipe slices the normed pre-pooling sequence; verified
        # for this head type by scripts/vc1_grid_probe.py.
        if READOUT in ("grid2", "pyramid") \
                and model.classifier_feature != "use_cls_token":
            raise SystemExit(
                f"VC1_READOUT={READOUT} needs classifier_feature "
                f"'use_cls_token', got {model.classifier_feature!r} -- the "
                f"patch-token recipe is wrong for this checkpoint.")
        for p in model.parameters():
            p.requires_grad = False
        return model, transform

    def latent(self, rgb):
        """(H, W, 3) uint8 or float RGB -> (768,) float32 (cls),
        (37632,) float32 (grid2) or (3840,) float32 (pyramid)."""
        import torch
        with torch.no_grad():
            x = self._prepare(rgb).to(self.device)
            if READOUT in ("grid2", "pyramid"):
                # The patch-token recipe verified by scripts/vc1_grid_probe.py
                # (vit.py forward_features without handle_outcome): the CLS
                # token of this sequence equals self._model(x) exactly.
                B = x.shape[0]
                h = self._model.patch_embed(x)
                cls_tok = self._model.cls_token.expand(B, -1, -1) \
                    + self._model.pos_embed[:, :1, :]
                h = torch.cat(
                    (cls_tok, h + self._model.pos_embed[:, 1:, :]), dim=1)
                h = self._model.blocks(h)
                h = self._model.norm(h)
                tokens = h[0, 1:].cpu().numpy()
                if READOUT == "pyramid":
                    return pool_pyramid(tokens)
                return pool_grid(tokens)
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
                "image_size": 224, "readout": READOUT}

    @staticmethod
    def assert_matches(checkpoint_identity, source):
        # Lazy import: latent_source imports nothing from here at module
        # scope, but keeping this import inside the method makes the
        # direction explicit and immune to future import-order changes.
        from vae_nav.latent_source import LatentSource
        return LatentSource.assert_matches(checkpoint_identity, source)