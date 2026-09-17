"""Train the lane+obstacle VAE on collected (RGB, semantic-mask) pairs.

Run after src/vae_nav/collect_lane_data.py. Produces a frozen checkpoint that
the RL wrapper loads and never updates.

    python train_lane_vae.py --data data/lane_vae/lane_dataset.npz \
        --out checkpoints/lane_vae/best.pkl

TWO THINGS THIS GUARDS AGAINST, both of which produce a loss curve that looks
converged while the model has learned nothing:

  * ALL-ZERO COLLAPSE. Lane markings are a few percent of a frame, so the
    unweighted BCE optimum is "predict no lane anywhere". pos_weight is read
    from the corpus, and validation reports IoU/recall rather than accuracy so
    the degenerate solution is visibly worthless (IoU 0.0) rather than
    invisibly excellent (accuracy 0.97).

  * POSTERIOR COLLAPSE. If KL dominates early the encoder stops carrying
    information and every frame maps to the same latent -- a policy fed that
    latent is effectively blind, and nothing in the RL logs would say so. beta
    is annealed from 0, and active_units is logged as the canary.
"""
import argparse
import os
import pickle
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from vae_nav.lane_mask import CHANNEL_NAMES  # noqa: E402
from vae_nav.lane_vae import LaneVAE, mask_metrics, reparam, vae_loss  # noqa: E402


EVAL_BATCH = int(os.environ.get("EVAL_BATCH", 64))


def make_state(rng, size, lr, latent_dim):
    model = LaneVAE(latent_dim=latent_dim, image_size=size)
    params = model.init(rng, jnp.zeros((1, size, size, 3)), deterministic=True)["params"]
    tx = optax.adam(lr)
    return model, train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=tx)


@jax.jit
def _train_step(state, rgb, mask, rng, pos_weight, beta):
    def loss_fn(params):
        mu, logvar = state.apply_fn({"params": params}, rgb,
                                    method=lambda m, x: m.encoder(x))
        z = reparam(rng, mu, logvar)
        logits = state.apply_fn({"params": params}, z,
                                method=lambda m, z_: m.decoder(z_))
        return vae_loss(logits, mask, mu, logvar, pos_weight, beta)

    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    return state.apply_gradients(grads=grads), metrics


@jax.jit
def _eval_counts(state, rgb, mask, pos_weight, beta, target_threshold=0.25):
    """Per-batch loss terms plus RAW intersection/union counts.

    Counts rather than ratios because the validation set is evaluated in
    batches and IoU does not average: mean(I_b/U_b) over batches is not
    I_total/U_total, and the gap is worst exactly where it matters -- a batch
    containing no lane pixels contributes a 0/0 that a ratio-average either
    drops or scores as zero. Summing the counts and dividing once at the end
    is the same number a single full-set pass would produce.
    """
    logits, mu, logvar = state.apply_fn({"params": state.params}, rgb,
                                        deterministic=True)
    _, metrics = vae_loss(logits, mask, mu, logvar, pos_weight, beta)
    pred = (jax.nn.sigmoid(logits) > 0.5).astype(jnp.float32)
    tgt = (mask > target_threshold).astype(jnp.float32)
    inter = jnp.sum(pred * tgt, axis=(0, 1, 2))
    union = jnp.sum(jnp.maximum(pred, tgt), axis=(0, 1, 2))
    pos = jnp.sum(tgt, axis=(0, 1, 2))
    return metrics, inter, union, pos


def evaluate(state, rgb_va, mask_va, pos_weight, beta, n_channels,
             batch=EVAL_BATCH):
    """Validation over the whole set, in batches.

    MEASURED failure of the one-shot version: at 120x120 a single decoder
    activation for 1200 frames is 1200 x 32 x 128 x 128 x 4 bytes = 2.5 GB, and
    cuDNN could not allocate it -- training died at the first validation with
    RESOURCE_EXHAUSTED, having written no checkpoint. It fit at 64x64 purely
    because the same tensor was 3.5x smaller.
    """
    tot, seen = {}, 0
    inter = np.zeros(n_channels, np.float64)
    union = np.zeros(n_channels, np.float64)
    pos = np.zeros(n_channels, np.float64)
    for i in range(0, len(rgb_va), batch):
        r = jnp.asarray(rgb_va[i:i + batch])
        m = jnp.asarray(mask_va[i:i + batch])
        met, bi, bu, bp = _eval_counts(state, r, m, pos_weight, beta)
        n = int(r.shape[0])
        for k, v in met.items():
            tot[k] = tot.get(k, 0.0) + float(v) * n
        inter += np.asarray(bi, np.float64)
        union += np.asarray(bu, np.float64)
        pos += np.asarray(bp, np.float64)
        seen += n
    out = {k: v / max(seen, 1) for k, v in tot.items()}
    ious = []
    for c in range(n_channels):
        iou = float(inter[c] / max(union[c], 1.0))
        out[f"{CHANNEL_NAMES[c]}_iou"] = iou
        out[f"{CHANNEL_NAMES[c]}_recall"] = float(inter[c] / max(pos[c], 1.0))
        ious.append(iou)
    # Selection metric is the MIN across channels: a checkpoint is only "best"
    # when EVERY channel works, so a saturated obstacle channel cannot carry a
    # dead lane one.
    out["iou"] = min(ious) if ious else 0.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/lane_vae/lane_dataset.npz")
    ap.add_argument("--out", default="checkpoints/lane_vae/best.pkl")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--latent-dim", type=int, default=128)
    ap.add_argument("--beta", type=float, default=1.0, help="final KL weight")
    ap.add_argument("--beta-warmup", type=int, default=10,
                    help="epochs to anneal beta 0 -> --beta")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    d = np.load(args.data)
    rgb = d["rgb"].astype(np.float32) / 255.0
    mask = d["mask"].astype(np.float32)
    if mask.ndim == 3:
        mask = mask[..., None]
    # Per-channel weight vector. One shared scalar would over-correct the
    # dense obstacle channel into predicting obstacles everywhere while still
    # under-correcting the sparse lane channel.
    pos_weight = (np.asarray(d["pos_weight"], np.float32) if "pos_weight" in d
                  else np.float32([32.0, 2.0]))
    size = rgb.shape[1]
    print(f"[Data] {len(rgb)} frames {rgb.shape[1:]} -> mask {mask.shape[1:]}")
    for c, name in enumerate(CHANNEL_NAMES[:mask.shape[-1]]):
        print(f"[Data]   {name}: {mask[..., c].mean() * 100:.2f}% positive | "
              f"pos_weight {np.atleast_1d(pos_weight)[c]:.1f}")

    # Split by a CONTIGUOUS BLOCK FROM EVERY SHARD.
    #
    # Contiguity is what keeps the split honest: consecutive frames of a drive
    # are near-duplicates, so a random split puts near-copies of every
    # validation frame into train and reports a training IoU under a
    # validation label. That part was already right.
    #
    # Taking the tail of the CONCATENATED corpus was not. Shards are stored in
    # sorted order, so a 10% tail was the last 3000 frames of exactly one shard
    # -- Town05_Epic, the most monotonous drive in the set (92.3% of its frames
    # identical to their predecessor, ~0.020 distinct fraction). Validation was
    # therefore roughly sixty distinct frames of straight road in a single
    # town, and the reported val_iou described that, not the corpus. Town02
    # sat at lane IoU 0.438 with nothing in the number to reveal it.
    #
    # Holding out the tail of EACH shard keeps every town and every render
    # quality represented in validation while preserving the contiguity that
    # makes the number mean something.
    bounds, off = [], 0
    if "shards" in d:
        import json
        for _name, cnt in json.loads(str(d["shards"])).items():
            bounds.append((off, off + int(cnt)))
            off += int(cnt)
    if off != len(rgb):                       # unknown layout: one block
        bounds = [(0, len(rgb))]

    tr_idx, va_idx = [], []
    for lo, hi in bounds:
        n_v = max(1, int((hi - lo) * args.val_frac))
        tr_idx.extend(range(lo, hi - n_v))
        va_idx.extend(range(hi - n_v, hi))
    tr_idx, va_idx = np.asarray(tr_idx), np.asarray(va_idx)
    rgb_tr, mask_tr = rgb[tr_idx], mask[tr_idx]
    rgb_va, mask_va = rgb[va_idx], mask[va_idx]
    print(f"[Data] train {len(rgb_tr)} | val {len(rgb_va)} "
          f"(contiguous tail of each of {len(bounds)} shard(s))")

    rng = jax.random.PRNGKey(args.seed)
    rng, init_rng = jax.random.split(rng)
    model, state = make_state(init_rng, size, args.lr, args.latent_dim)

    steps = max(1, len(rgb_tr) // args.batch_size)
    best_iou, best_params = -1.0, None
    np_rng = np.random.default_rng(args.seed)

    for epoch in range(args.epochs):
        beta = args.beta * min(1.0, (epoch + 1) / max(1, args.beta_warmup))
        perm = np_rng.permutation(len(rgb_tr))
        t0, agg = time.time(), {}
        for i in range(steps):
            idx = perm[i * args.batch_size:(i + 1) * args.batch_size]
            rng, step_rng = jax.random.split(rng)
            state, m = _train_step(state, jnp.asarray(rgb_tr[idx]),
                                   jnp.asarray(mask_tr[idx]), step_rng,
                                   pos_weight, beta)
            for k, v in m.items():
                agg[k] = agg.get(k, 0.0) + float(v)

        vm = evaluate(state, rgb_va, mask_va, pos_weight, beta,
                      mask.shape[-1])
        # Selection metric is the MIN across channels (see mask_metrics), so a
        # checkpoint is only "best" when every channel works.
        iou = float(vm["iou"])
        per_ch = "  ".join(
            f"{n}_iou={float(vm[f'{n}_iou']):.3f}/rec={float(vm[f'{n}_recall']):.3f}"
            for n in CHANNEL_NAMES[:mask.shape[-1]])
        print(f"[{epoch:3d}] beta={beta:.2f} "
              f"loss={agg['loss'] / steps:9.1f} "
              f"bce={agg['bce'] / steps:8.1f} kl={agg['kl'] / steps:7.2f} "
              f"active={agg['active_units'] / steps:.2f} | "
              f"min_iou={iou:.4f}  {per_ch} "
              f"({time.time() - t0:.1f}s)", flush=True)

        if iou > best_iou:
            best_iou, best_params = iou, jax.device_get(state.params)

    if best_iou <= 0.0:
        raise SystemExit(
            "best validation IoU is 0.0 on at least one channel -- the decoder "
            "predicts nothing there. Raise that channel's pos_weight or lower "
            "beta; do NOT ship this checkpoint, a policy fed its latent would "
            "be blind in that channel.")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump({
            "params": best_params,
            "latent_dim": args.latent_dim,
            "image_size": size,
            # Identity travels WITH the weights. RingSource in the peer repo
            # records the backend name into the checkpoint so a policy can never
            # be silently evaluated against a sensor it never trained on; the
            # same reasoning applies to a latent.
            "backend": "lane_obstacle_vae",
            "channels": list(CHANNEL_NAMES),
            "trained_on_towns": [str(t) for t in d["towns"]] if "towns" in d else [],
            "pos_weight": np.asarray(pos_weight),
            "val_iou": best_iou,
            "latent_mode": "mu",
        }, f)
    print(f"[Save] val_iou={best_iou:.4f} -> {args.out}")


if __name__ == "__main__":
    main()
