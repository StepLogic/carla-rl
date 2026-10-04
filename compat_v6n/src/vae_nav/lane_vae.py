"""VAE that reconstructs a lane-marking mask from RGB, in JAX/Flax.

ROLE. This is the CARLA analogue of the peer Habitat repo's MonoDepthNet: a
task-specific perception head, frozen at RL time, whose output replaces a
hand-built sensor in the policy's observation. There, RGB -> metric depth ->
16-beam ring. Here, RGB -> lane mask, and the 128-d posterior mean IS the
policy input -- we skip their ring-extraction step because a lane mask has no
natural reduction to bearings the way a depth image does.

TWO DESIGN CHOICES INHERITED FROM THEIR MEASURED FAILURES:

1. LINEAR OUTPUT HEAD. They tried sigmoid*10 and softplus for the depth head;
   both overshot and saturated, and a linear head with the bias opened near
   mid-range was the fix. The same reasoning applies to a mask head, so the
   decoder emits LOGITS and the sigmoid lives in the loss (and only at
   inference). Nothing here is bounded by an activation.

2. WEIGHTED BCE. Lane markings are a few percent of a forward-facing frame.
   Unweighted BCE is minimised by predicting zero everywhere, which converges
   beautifully and learns nothing. pos_weight comes from the measured corpus
   (lane_mask.pos_weight_for), not a guess.
"""
import flax.linen as nn
import jax
import jax.numpy as jnp

LATENT_DIM = 128
IMAGE_SIZE = int(__import__("os").environ.get("VAE_IMAGE_SIZE", 120))


class Encoder(nn.Module):
    latent_dim: int = LATENT_DIM
    features: tuple = (32, 64, 128, 256)

    @nn.compact
    def __call__(self, x):
        # x: (B, H, W, 3) float32 in [0, 1]
        for f in self.features:
            x = nn.Conv(f, (3, 3), strides=(2, 2), padding="SAME")(x)
            x = nn.GroupNorm(num_groups=min(8, f))(x)
            x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        mu = nn.Dense(self.latent_dim)(x)
        logvar = nn.Dense(self.latent_dim)(x)
        # Keep the posterior from collapsing to a delta or exploding. Both ends
        # produce a latent the policy cannot use: a delta carries no gradient
        # signal, an exploding one drowns the goal block after LayerNorm.
        logvar = jnp.clip(logvar, -6.0, 2.0)
        return mu, logvar


class Decoder(nn.Module):
    features: tuple = (256, 128, 64, 32)
    out_channels: int = 2      # [lane, obstacle] -- see lane_mask.CHANNEL_NAMES
    image_size: int = IMAGE_SIZE

    @nn.compact
    def __call__(self, z):
        # THE SIDE LENGTH IS THE ENCODER'S, MIRRORED -- not image_size // 16.
        #
        # The encoder's convolutions use padding="SAME" with stride 2, so each
        # halving is a CEILING division: 120 -> 60 -> 30 -> 15 -> 8. Integer
        # division would say 120 // 16 = 7, and the decoder would then upsample
        # 7 -> 112 and never reach 120. The old form was correct only because
        # 64 happens to be divisible by 16; any size that is not (120 among
        # them) silently produced a decoder output of the wrong shape.
        side = self.image_size
        for _ in self.features:
            side = -(-side // 2)              # ceil division, mirroring SAME
        x = nn.Dense(side * side * self.features[0])(z)
        x = x.reshape((z.shape[0], side, side, self.features[0]))
        for f in self.features[1:]:
            x = nn.ConvTranspose(f, (3, 3), strides=(2, 2), padding="SAME")(x)
            x = nn.GroupNorm(num_groups=min(8, f))(x)
            x = nn.relu(x)
        x = nn.ConvTranspose(self.features[-1], (3, 3), strides=(2, 2),
                             padding="SAME")(x)
        x = nn.relu(x)
        # LINEAR head, bias opened slightly negative so the model starts by
        # predicting "mostly not a lane marking" -- which is true -- instead of
        # starting at p=0.5 and having to climb down through the imbalance.
        x = nn.Conv(self.out_channels, (3, 3), padding="SAME",
                    bias_init=nn.initializers.constant(-2.0))(x)
        # Four doublings from a ceil-divided side OVERSHOOT when image_size is
        # not a multiple of 16: 8 -> 128 for a 120 target. Crop rather than
        # resize -- an interpolation here would blur the very lane edges this
        # resolution increase exists to sharpen, and would put the decoder's
        # output on a different pixel grid than the target mask.
        if x.shape[1] != self.image_size or x.shape[2] != self.image_size:
            x = x[:, :self.image_size, :self.image_size, :]
        return x


class LaneVAE(nn.Module):
    latent_dim: int = LATENT_DIM
    image_size: int = IMAGE_SIZE

    def setup(self):
        self.encoder = Encoder(latent_dim=self.latent_dim)
        self.decoder = Decoder(image_size=self.image_size)

    def __call__(self, x, rng=None, deterministic=False):
        mu, logvar = self.encoder(x)
        z = mu if deterministic or rng is None else reparam(rng, mu, logvar)
        return self.decoder(z), mu, logvar

    def encode(self, x):
        """Inference path: the posterior MEAN is the policy-facing latent.

        Sampling here would make the observation stochastic for a fixed scene,
        which the replay buffer would record as environment noise.
        """
        mu, _ = self.encoder(x)
        return mu


def reparam(rng, mu, logvar):
    return mu + jnp.exp(0.5 * logvar) * jax.random.normal(rng, mu.shape)


def vae_loss(logits, target, mu, logvar, pos_weight=1.0, beta=1.0):
    """Weighted BCE-with-logits + beta * KL, both per-sample SUMS.

    THE SCALING MATTERS AND IS EASY TO GET WRONG. An earlier version of this
    meaned the BCE over pixels while the KL summed over latent dims. At 64x64
    that made reconstruction ~4000x smaller than the KL term, so even beta=0.5
    drove the posterior to collapse: measured active_units 1.00 -> 0.00 within
    three epochs, KL -> 0, and validation IoU pinned at 0.0 while the total
    loss fell smoothly and looked converged.

    Summing both puts them in the same units (nats per sample), which is what
    makes beta=1.0 the principled default rather than an arbitrary knob.

    `pos_weight` may be a scalar or a per-channel vector.

    Returns (loss, metrics). The terms are reported separately because a
    falling KL against a flat BCE is the signature of collapse, and that is
    invisible in the total.
    """
    # Numerically stable weighted BCE with logits:
    #   (1 - t)*x + (1 + (w-1)*t) * (log(1 + exp(-|x|)) + max(-x, 0))
    #
    # THE max(-x, 0) TERM MUST BE INSIDE THE WEIGHT. An earlier version wrote
    #   max(x, 0) - x*t + (1 + (w-1)*t) * log1pexp
    # which is correct at w=1 but silently wrong otherwise: it weights only the
    # log1pexp part, leaving the linear tail unweighted. For a positive pixel at
    # logit -3 with w=31 that returns 4.5 instead of 94.5, so confidently-wrong
    # positives produce almost no gradient. Measured effect: the lane channel
    # never rose above p=0.056 and obstacle IoU DECAYED 0.115 -> 0.016 over 400
    # steps as the decoder drifted back to the marginal.
    #
    # The |x| inside log1pexp keeps this finite for large-magnitude logits,
    # which a linear head can and does produce.
    x, t = logits, target
    w = jnp.asarray(pos_weight, jnp.float32)          # scalar or (C,)
    log1pexp = jnp.logaddexp(0.0, -jnp.abs(x))
    bce = (1.0 - t) * x + (1.0 + (w - 1.0) * t) * (log1pexp + jnp.maximum(-x, 0.0))
    # SUM over pixels and channels, matching the KL's sum over latent dims.
    bce = bce.reshape(bce.shape[0], -1).sum(axis=-1)

    kl = -0.5 * jnp.sum(1.0 + logvar - mu ** 2 - jnp.exp(logvar), axis=-1)

    loss = jnp.mean(bce + beta * kl)
    return loss, {
        "bce": jnp.mean(bce),
        "kl": jnp.mean(kl),
        "loss": loss,
        # Fraction of latent dims that have gone inert. Their equivalent of a
        # canary for a latent the policy would find useless.
        "active_units": jnp.mean((jnp.var(mu, axis=0) > 1e-2).astype(jnp.float32)),
    }


CHANNEL_NAMES = ("lane", "obstacle")


def mask_metrics(logits, target, threshold=0.5, target_threshold=0.25):
    """PER-CHANNEL IoU and recall on the positive class.

    Accuracy is useless at this imbalance -- predicting all-zero scores ~97% on
    the lane channel. IoU and recall are what distinguish a working model from
    a lazy one.

    Reported per channel rather than pooled: obstacles cover a third of a frame
    and are easy, lane markings cover a few percent and are the hard part. A
    pooled IoU would sit at a comfortable-looking number while the lane channel
    was empty, which is precisely the failure worth catching.
    """
    pred = (jax.nn.sigmoid(logits) > threshold).astype(jnp.float32)
    # BINARISE THE TARGET. After INTER_AREA downsampling the targets are
    # coverage FRACTIONS, not labels -- a lane pixel is typically 0.05-0.35. An
    # IoU computed against those soft values is not an IoU at all: the
    # intersection term sum(pred * t) is scaled by coverage, so a perfect
    # prediction scores far below 1.0 and the number cannot be compared to
    # anything. Threshold at target_threshold so "is there lane structure in
    # this cell" is the question being scored.
    tgt = (target > target_threshold).astype(jnp.float32)
    out = {}
    ious = []
    for c in range(logits.shape[-1]):
        p, t = pred[..., c], tgt[..., c]
        inter = jnp.sum(p * t, axis=(1, 2))
        union = jnp.sum(jnp.maximum(p, t), axis=(1, 2))
        pos = jnp.sum(t, axis=(1, 2))
        name = CHANNEL_NAMES[c] if c < len(CHANNEL_NAMES) else str(c)
        iou = jnp.mean(inter / jnp.maximum(union, 1.0))
        out[f"{name}_iou"] = iou
        out[f"{name}_recall"] = jnp.mean(inter / jnp.maximum(pos, 1.0))
        out[f"{name}_pred_frac"] = jnp.mean(p)
        ious.append(iou)
    # Selection metric: the MINIMUM across channels, so a checkpoint is only
    # "best" when every channel works, not when a strong obstacle channel
    # carries a dead lane one.
    out["iou"] = jnp.min(jnp.stack(ious))
    return out
