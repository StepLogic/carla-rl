#!/usr/bin/env python3
"""Train the ResNet-9 masked segmentation VAE (vae_nav.seg_vae).

usage: train_seg_vae.py DATA_DIR OUT_DIR [--steps 30000] [--batch 128]

Data: every shard_*.npz from vae_nav.collect_seg_data (rgb (N,120,120,3)
uint8, label (N,240,240) uint8). Validation = the LAST 5% of each shard --
consecutive frames are near-duplicates, so a random split would score memory.
Training batches: soft targets -> augment (photometric + shared geometric)
-> heavy patch masking -> VAE. Validation: no augmentation, no masking.
Class weights: (median freq / freq)^0.5 clipped to [0.5, 10], lane lines x2.
"""
import argparse
import glob
import json
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))


def load(data_dir, val_frac=0.05):
    tr_x, tr_y, va_x, va_y, names = [], [], [], [], []
    for f in sorted(glob.glob(os.path.join(data_dir, "shard_*.npz"))):
        z = np.load(f)
        n = len(z["rgb"])
        cut = int(n * (1 - val_frac))
        tr_x.append(z["rgb"][:cut]); tr_y.append(z["label"][:cut])
        va_x.append(z["rgb"][cut:]); va_y.append(z["label"][cut:])
        names.append((os.path.basename(f), n))
    return (np.concatenate(tr_x), np.concatenate(tr_y),
            np.concatenate(va_x), np.concatenate(va_y), names)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data"); ap.add_argument("out")
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--beta", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    # user 2026-09-28 "keep all classes same weight": weight 1 for every class
    # present (the weighted run painted thin lane lines as wide blobs)
    ap.add_argument("--equal-weights", action="store_true")
    # user 2026-09-28 "add dice loss": + dice_w * (1 - soft Dice over present classes)
    ap.add_argument("--dice-w", type=float, default=0.0)
    # warm start: parameters from an earlier checkpoint, fresh optimiser
    ap.add_argument("--init", default=None)
    a = ap.parse_args()
    import jax
    import jax.numpy as jnp
    import optax
    from vae_nav import seg_vae as sv

    trx, try_, vax, vay, names = load(a.data)
    print(f"[data] train {len(trx)}  val {len(vax)}  shards {names}", flush=True)
    freq = np.bincount(try_[::7].ravel(), minlength=sv.N_CLASSES).astype(np.float64)
    freq = freq / freq.sum()
    present = freq > 0
    med = np.median(freq[present])
    w = np.where(present, np.clip((med / np.maximum(freq, 1e-9)) ** 0.5, 0.5, 10.0), 0.0)
    w[2] *= 2.0                                               # lane lines
    if a.equal_weights:
        w = np.where(present, 1.0, 0.0)
    print("[data] class freq " + " ".join(f"{f:.3f}" for f in freq), flush=True)
    print("[data] class w    " + " ".join(f"{x:.2f}" for x in w), flush=True)
    class_w = jnp.asarray(w, jnp.float32)
    present_j = jnp.asarray(present, jnp.float32)
    print(f"[loss] CE {'equal weights' if a.equal_weights else 'weighted'} "
          f"+ {a.dice_w} x Dice + {a.beta} x KL", flush=True)

    model = sv.SegVAE()
    rng = jax.random.PRNGKey(a.seed)
    params = model.init(rng, jnp.zeros((1, 120, 120, 3)), rng)
    if a.init:
        c0 = pickle.load(open(a.init, "rb"))
        same = (jax.tree_util.tree_structure(c0["params"]) == jax.tree_util.tree_structure(params)
                and all(x.shape == y.shape for x, y in zip(jax.tree_util.tree_leaves(c0["params"]),
                                                          jax.tree_util.tree_leaves(params))))
        if same:
            params = jax.tree_util.tree_map(jnp.asarray, c0["params"])
            print(f"[init] warm start (all) from {a.init} (step {c0['step']})", flush=True)
        else:
            # DECODER CHANGED SHAPE: keep the new decoder, load the encoder only
            from flax.core import unfreeze
            params = unfreeze(params)
            params["params"]["encoder"] = jax.tree_util.tree_map(
                jnp.asarray, c0["params"]["params"]["encoder"])
            print(f"[init] warm start (ENCODER only; decoder is new) from {a.init} "
                  f"(step {c0['step']})", flush=True)
    sched = optax.warmup_cosine_decay_schedule(0.0, a.lr, min(1000, a.steps // 10), a.steps, a.lr * 0.05)
    opt = optax.adamw(sched, weight_decay=1e-4)
    opt_state = opt.init(params)

    @jax.jit
    def train_step(params, opt_state, x_u8, y, key):
        k1, k2, k3 = jax.random.split(key, 3)
        x = x_u8.astype(jnp.float32) / 255.0
        t = sv.soft_targets(y)
        x, t = sv.augment(k1, x, t)
        x = sv.patch_mask(k2, x)

        def lf(p):
            lg, mu, lv = model.apply(p, x, k3)
            loss, parts = sv.seg_vae_loss(lg, t, mu, lv, class_w, a.beta)
            if a.dice_w > 0:
                d = sv.soft_dice_loss(lg, t, present_j)
                loss = loss + a.dice_w * d
                parts = {**parts, "dice": d}
            return loss, parts
        (loss, parts), g = jax.value_and_grad(lf, has_aux=True)(params)
        upd, opt_state = opt.update(g, opt_state, params)
        return optax.apply_updates(params, upd), opt_state, loss, parts

    @jax.jit
    def val_step(params, x_u8, y):
        x = x_u8.astype(jnp.float32) / 255.0
        t = sv.soft_targets(y)
        mu, _ = model.apply(params, x, method=lambda m, x: m.encoder(x))
        lg = model.apply(params, mu, method=lambda m, z: m.decoder(z))
        return sv.class_iou(lg, t)

    os.makedirs(a.out, exist_ok=True)
    nprng = np.random.default_rng(a.seed)
    t0, log = time.time(), []
    for step in range(1, a.steps + 1):
        idx = nprng.integers(0, len(trx), a.batch)
        params, opt_state, loss, parts = train_step(
            params, opt_state, jnp.asarray(trx[idx]), jnp.asarray(try_[idx]),
            jax.random.fold_in(rng, step))
        if step % 1000 == 0 or step == a.steps or (a.steps < 100 and step % 10 == 0):
            ious = []
            for i in range(0, len(vax), 256):
                ious.append(np.asarray(val_step(params, jnp.asarray(vax[i:i + 256]),
                                                jnp.asarray(vay[i:i + 256]))))
            iou = np.mean(ious, 0)
            rec = dict(step=step, loss=float(loss), ce=float(parts["ce"]),
                       dice=float(parts.get("dice", 0.0)),
                       kl=float(parts["kl"]), val_iou=iou.round(3).tolist(),
                       val_miou=float(iou[present].mean()), sec=round(time.time() - t0))
            log.append(rec)
            print(f"[{step}] loss {rec['loss']:.3f} ce {rec['ce']:.3f} dice {rec['dice']:.3f} kl {rec['kl']:.3f} "
                  f"| val mIoU {rec['val_miou']:.3f} lane {iou[2]:.3f} road {iou[1]:.3f} "
                  f"| {rec['sec']} s", flush=True)
            with open(os.path.join(a.out, "seg_vae.pkl"), "wb") as f:
                pickle.dump({"params": jax.device_get(params), "step": step,
                             "latent_dim": sv.LATENT_DIM, "classes": sv.N_CLASSES,
                             "dec_start": sv.DEC_START,
                             "class_w": w.tolist(), "mask_ratio": sv.MASK_RATIO}, f)
            json.dump(log, open(os.path.join(a.out, "train_log.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
