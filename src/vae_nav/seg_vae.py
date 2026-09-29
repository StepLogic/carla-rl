"""ResNet-9 masked VAE that decodes dense CARLA segmentation from RGB (JAX/Flax).

User 2026-09-28: "pretrain resnet-9 masked VAE auto encoder (JAX) with [CARLA's
instance segmentation sensor] and then use for training"; "dense segmentation",
"single vector"; "use heavy masking"; "the encoder [must be] robust and
transferable in carla - use different data augmentations"; "VC1 is overkill".

WHY. The perception probe showed the policy's VC-1 pyramid input loses WHERE
the car sits in its lane (distance from lane centre R^2 0.41; the patch grid
reached 0.65). A small encoder whose latent must reconstruct the scene's
segmentation -- lane lines weighted up -- is forced to keep that geometry.

  input      (B, 120, 120, 3) float32 in [0, 1]
  masking    training only: MASK_RATIO of the 8x8 patches zeroed (MAE-style,
             heavy by default) so the latent must infer layout, not copy pixels
  encoder    ResNet-9 (latent 512): prep 64 -> [128 + pool + res] -> [256 + pool]
             -> [512 + pool + res] -> global mean -> mu, logvar (LATENT_DIM)
             GroupNorm throughout: no batch statistics to freeze or leak
  decoder    z -> 15x15x256 -> up 30x30x128 -> up 60x60x64 -> up 120x120x32
             -> N_CLASSES logits
  loss       soft-label cross-entropy (class-weighted) + beta * KL
The policy consumes mu only (encode()), so no sampling noise reaches RL.
"""
import os

import flax.linen as nn
import jax
import jax.numpy as jnp

LATENT_DIM = int(os.environ.get("SEG_LATENT_DIM", 512))  # user 2026-09-28: "make latent 512"
N_CLASSES = 9
IMAGE_SIZE = 120
PATCH = 8
MASK_RATIO = float(os.environ.get("SEG_MASK_RATIO", 0.75))
# DECODER START GRID (user 2026-09-28: "Start the decoder at 30x30 instead of
# 15x15"). 15: z -> 15x15x256 -> three 2x upsamplings (the first model).
# 30: z -> 30x30x64 (the same 57,600 values, so the same Dense size) -> two
# upsamplings: finer room to place thin lane lines; the latent is unchanged.
DEC_START = int(os.environ.get("SEG_DEC_START", 30))
# U-NET SKIPS (user 2026-09-28 "use skip connections"): the decoder also reads
# the encoder's 15/30/60 px feature maps. RL reads ONLY the latent mu, and a
# skip lets the decoder bypass it, so SKIP_DROP zeroes each sample's skips with
# this probability during training: the latent must still carry the scene.
SKIPS = os.environ.get("SEG_SKIPS", "0") == "1"
SKIP_DROP = float(os.environ.get("SEG_SKIP_DROP", 0.5))


def _gn(c):
    return nn.GroupNorm(num_groups=min(32, c // 4))


class ConvBlock(nn.Module):
    features: int
    pool: bool = False

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(self.features, (3, 3), padding="SAME", use_bias=False)(x)
        x = nn.relu(_gn(self.features)(x))
        if self.pool:
            x = nn.max_pool(x, (2, 2), (2, 2))
        return x


class Residual(nn.Module):
    features: int

    @nn.compact
    def __call__(self, x):
        return x + ConvBlock(self.features)(ConvBlock(self.features)(x))


class Encoder(nn.Module):
    latent_dim: int = LATENT_DIM
    return_feats: bool = False

    @nn.compact
    def __call__(self, x):
        x = ConvBlock(64)(x)                       # 120
        x = ConvBlock(128, pool=True)(x)           # 60
        f60 = x = Residual(128)(x)
        f30 = x = ConvBlock(256, pool=True)(x)     # 30
        x = ConvBlock(512, pool=True)(x)           # 15
        f15 = x = Residual(512)(x)
        h = x.mean(axis=(1, 2))
        mu = nn.Dense(self.latent_dim, name="mu")(h)
        logvar = jnp.clip(nn.Dense(self.latent_dim, name="logvar")(h), -10.0, 10.0)
        if self.return_feats:
            return mu, logvar, (f15, f30, f60)
        return mu, logvar


class SkipDecoder(nn.Module):
    """z -> 15x15x256, then at 15/30/60 px concatenate the (1x1-projected)
    encoder map before a conv block, upsampling to 120. `drop_rng` given ->
    each sample's skips are zeroed with probability SKIP_DROP (training)."""
    n_classes: int = N_CLASSES
    skip_drop: float = SKIP_DROP

    @nn.compact
    def __call__(self, z, feats, drop_rng=None):
        b = z.shape[0]
        x = nn.Dense(15 * 15 * 256)(z).reshape(b, 15, 15, 256)
        x = nn.relu(_gn(256)(x))
        keys = (jax.random.split(drop_rng, 3) if drop_rng is not None else (None,) * 3)
        for i, (f, skip, pc) in enumerate(zip((256, 128, 64), feats, (128, 64, 32))):
            if i > 0:
                x = nn.ConvTranspose(f, (4, 4), strides=(2, 2), padding="SAME")(x)
                x = nn.relu(_gn(f)(x))
            s = nn.Conv(pc, (1, 1), name=f"skip_proj_{i}")(skip)
            if keys[i] is not None:
                keep = jax.random.bernoulli(keys[i], 1.0 - self.skip_drop, (b, 1, 1, 1))
                s = s * keep.astype(s.dtype)
            x = ConvBlock(f)(jnp.concatenate([x, s], axis=-1))
        x = nn.ConvTranspose(32, (4, 4), strides=(2, 2), padding="SAME")(x)   # 120
        x = ConvBlock(32)(nn.relu(_gn(32)(x)))
        return nn.Conv(self.n_classes, (1, 1))(x)


class Decoder(nn.Module):
    n_classes: int = N_CLASSES
    start: int = DEC_START

    @nn.compact
    def __call__(self, z):
        b = z.shape[0]
        c0, ups = {15: (256, (128, 64, 32)), 30: (64, (64, 32))}[self.start]
        x = nn.Dense(self.start * self.start * c0)(z).reshape(b, self.start, self.start, c0)
        x = nn.relu(_gn(c0)(x))
        for f in ups:                              # up to 120
            x = nn.ConvTranspose(f, (4, 4), strides=(2, 2), padding="SAME")(x)
            x = nn.relu(_gn(f)(x))
            x = ConvBlock(f)(x)
        return nn.Conv(self.n_classes, (1, 1))(x)


class SegVAE(nn.Module):
    latent_dim: int = LATENT_DIM
    n_classes: int = N_CLASSES
    dec_start: int = DEC_START
    skips: bool = SKIPS

    def setup(self):
        self.encoder = Encoder(self.latent_dim, return_feats=self.skips)
        self.decoder = (SkipDecoder(self.n_classes) if self.skips
                        else Decoder(self.n_classes, self.dec_start))

    def __call__(self, x, rng):
        """Training forward: sampled z; skips dropped per sample."""
        k1, k2 = jax.random.split(rng)
        if self.skips:
            mu, logvar, feats = self.encoder(x)
            z = mu + jnp.exp(0.5 * logvar) * jax.random.normal(k1, mu.shape)
            return self.decoder(z, feats, k2), mu, logvar
        mu, logvar = self.encoder(x)
        z = mu + jnp.exp(0.5 * logvar) * jax.random.normal(k1, mu.shape)
        return self.decoder(z), mu, logvar

    def predict(self, x):
        """Deterministic logits: z = mu, every skip kept (validation, renders)."""
        if self.skips:
            mu, _, feats = self.encoder(x)
            return self.decoder(mu, feats)
        return self.decoder(self.encoder(x)[0])

    def encode(self, x):
        """The latent RL consumes: mu only."""
        return self.encoder(x)[0]


def patch_mask(rng, x, ratio=MASK_RATIO, patch=PATCH):
    """Zero `ratio` of the patch x patch cells of each image (same mask over
    channels). 120 / 8 = 15 x 15 cells."""
    b, h, w, _ = x.shape
    gh, gw = h // patch, w // patch
    keep = jax.random.uniform(rng, (b, gh, gw, 1)) >= ratio
    keep = jnp.repeat(jnp.repeat(keep, patch, axis=1), patch, axis=2)
    return x * keep.astype(x.dtype)


def soft_targets(labels, n_classes=N_CLASSES, out=IMAGE_SIZE):
    """(B, L, L) uint8 class ids at label resolution -> (B, out, out, C) class
    coverage fractions by area pooling, so a one-pixel lane line at 240 px
    survives at 120 px as a partial target instead of vanishing."""
    oh = jax.nn.one_hot(labels, n_classes, dtype=jnp.float32)
    b, l = labels.shape[0], labels.shape[1]
    f = l // out
    return oh.reshape(b, out, f, out, f, n_classes).mean(axis=(2, 4))


def seg_vae_loss(logits, target, mu, logvar, class_w, beta):
    """Class-weighted soft cross-entropy (per pixel, normalised by the weights
    present) + beta * mean-per-dim KL. Returns (loss, parts)."""
    logp = jax.nn.log_softmax(logits, axis=-1)
    w = (target * class_w).sum(-1)
    ce = -(target * logp * class_w).sum(-1).sum() / jnp.maximum(w.sum(), 1e-6)
    kl = (-0.5 * (1 + logvar - mu ** 2 - jnp.exp(logvar))).mean()
    return ce + beta * kl, {"ce": ce, "kl": kl}


def soft_dice_loss(logits, target, present):
    """1 - soft multi-class Dice, averaged over the classes in `present`
    (user 2026-09-28 "add dice loss"). Dice is computed over the whole batch
    per class, so every class counts equally however few pixels it has --
    thin lane lines are scored on overlap, not on pixel count."""
    p = jax.nn.softmax(logits, axis=-1)
    inter = (p * target).sum(axis=(0, 1, 2))
    denom = p.sum(axis=(0, 1, 2)) + target.sum(axis=(0, 1, 2))
    dice = (2.0 * inter + 1.0) / (denom + 1.0)
    return ((1.0 - dice) * present).sum() / jnp.maximum(present.sum(), 1.0)


def class_iou(logits, target):
    """Per-class IoU of argmax prediction vs argmax target, over the batch."""
    p = jax.nn.one_hot(logits.argmax(-1), logits.shape[-1])
    t = jax.nn.one_hot(target.argmax(-1), logits.shape[-1])
    inter = (p * t).sum(axis=(0, 1, 2))
    union = ((p + t) > 0).sum(axis=(0, 1, 2))
    return inter / jnp.maximum(union, 1)


# ---- augmentation (training only), all on the GPU ---------------------------
def augment(rng, img, tgt):
    """Photometric on the image; the same geometric transform on image and
    soft target. img (B,H,W,3) in [0,1]; tgt (B,H,W,C)."""
    b = img.shape[0]
    ks = jax.random.split(rng, 12)

    def u(k, lo, hi, shape=(b, 1, 1, 1)):
        return jax.random.uniform(k, shape, minval=lo, maxval=hi)

    # --- photometric ---
    x = img * u(ks[0], 0.6, 1.4)                                    # brightness
    m = x.mean(axis=(1, 2, 3), keepdims=True)
    x = (x - m) * u(ks[1], 0.6, 1.4) + m                            # contrast
    g = x.mean(-1, keepdims=True)
    x = (x - g) * u(ks[2], 0.4, 1.6) + g                            # saturation
    x = x + u(ks[3], -0.06, 0.06, (b, 1, 1, 3))                     # colour cast
    x = jnp.clip(x, 0.0, 1.0) ** u(ks[4], 0.7, 1.4)                 # gamma
    grey = jax.random.bernoulli(ks[5], 0.1, (b, 1, 1, 1))
    x = jnp.where(grey, x.mean(-1, keepdims=True).repeat(3, -1), x)
    fog = jax.random.bernoulli(ks[6], 0.25, (b, 1, 1, 1)) * u(ks[7], 0.0, 0.5)
    x = x * (1 - fog) + fog * 0.75                                  # haze
    blur = jax.random.bernoulli(ks[8], 0.3, (b, 1, 1, 1))
    k = jnp.ones((3, 3, 1, 1), x.dtype) / 9.0
    xb = jax.lax.conv_general_dilated(
        x.transpose(3, 0, 1, 2).reshape(-1, *x.shape[1:3], 1), k, (1, 1), "SAME",
        dimension_numbers=("NHWC", "HWIO", "NHWC")).reshape(3, *x.shape[:3]).transpose(1, 2, 3, 0)
    x = jnp.where(blur, xb, x)
    x = x + jax.random.normal(ks[9], x.shape) * u(ks[10], 0.0, 0.04)  # noise
    x = jnp.clip(x, 0.0, 1.0)

    # --- geometric: random scale/shift + horizontal flip, same for target ---
    kk = jax.random.split(ks[11], 3)
    scale = jax.random.uniform(kk[0], (b,), minval=0.9, maxval=1.15)
    shift = jax.random.uniform(kk[1], (b, 2), minval=-8.0, maxval=8.0)
    flip = jax.random.bernoulli(kk[2], 0.5, (b,))

    def geo(a, s, t, f):
        h, w = a.shape[:2]
        out = jax.image.scale_and_translate(
            a, a.shape, (0, 1), jnp.array([s, s]),
            jnp.array([t[0] + (1 - s) * h / 2, t[1] + (1 - s) * w / 2]), "linear")
        return jnp.where(f, out[:, ::-1], out)

    x = jax.vmap(geo)(x, scale, shift, flip)
    tgt = jax.vmap(geo)(tgt, scale, shift, flip)
    # renormalise targets (edges become all-zero after translation: mark them
    # as 'other' so every pixel still sums to one)
    s = tgt.sum(-1, keepdims=True)
    tgt = jnp.where(s > 1e-3, tgt / jnp.maximum(s, 1e-6),
                    jax.nn.one_hot(jnp.zeros(tgt.shape[:3], jnp.int32), tgt.shape[-1]))
    return x, tgt
