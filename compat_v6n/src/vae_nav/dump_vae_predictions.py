"""Write VAE predictions as inspectable images.

The trained VAE's outputs previously existed in only two places: inside the
checkpoint as weights, and as a live panel in eval videos. Neither lets you
LOOK at what the model predicts on a given town -- and "val_iou = 0.57" is a
single number averaged over a corpus, which can hide a channel that works
everywhere except the map you are about to train on.

Produces a contact sheet per town: input frame, ground-truth mask, predicted
mask, and the error map, so a failure is visible rather than inferred.

    python src/vae_nav/dump_vae_predictions.py --town Town02
    python src/vae_nav/dump_vae_predictions.py           # every town in the corpus
"""
import argparse
import glob
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vae_nav.lane_mask import CHANNEL_NAMES  # noqa: E402

# Colour per channel, matching the eval videos so the two read the same way.
LANE_RGB = np.array([60, 200, 120], np.float32)      # green
OBST_RGB = np.array([215, 90, 60], np.float32)       # red


def colourise(mask2, threshold=None):
    """(H, W, 2) -> RGB. Lane green, obstacle red, overlap blended."""
    m = np.asarray(mask2, np.float32)
    if threshold is not None:
        m = (m > threshold).astype(np.float32)
    out = np.zeros((*m.shape[:2], 3), np.float32)
    out += m[..., 0:1] * LANE_RGB
    out += m[..., 1:2] * OBST_RGB
    return np.clip(out, 0, 255).astype(np.uint8)


def load_model(ckpt_path):
    import jax
    with open(ckpt_path, "rb") as f:
        ckpt = pickle.load(f)
    from vae_nav.lane_vae import LaneVAE
    model = LaneVAE(latent_dim=int(ckpt.get("latent_dim", 128)),
                    image_size=int(ckpt.get("image_size", 120)))
    params = ckpt["params"]

    @jax.jit
    def predict(x):
        logits, mu, _ = model.apply({"params": params}, x, deterministic=True)
        return jax.nn.sigmoid(logits), mu

    return predict, ckpt


def contact_sheet(rgb, truth, pred, n_cols, pad=2):
    """Stack rows of [input | truth | prediction | error] tiles."""
    h, w = rgb.shape[1:3]
    rows = []
    for i in range(len(rgb)):
        # Error map: red where the model missed a positive, blue where it
        # invented one. Silence is agreement.
        t = (truth[i] > 0.25).astype(np.float32)
        p = (pred[i] > 0.5).astype(np.float32)
        err = np.zeros((h, w, 3), np.float32)
        miss = np.clip(t - p, 0, 1).max(axis=-1)
        false = np.clip(p - t, 0, 1).max(axis=-1)
        err[..., 0] = miss * 230
        err[..., 2] = false * 230
        tiles = [rgb[i], colourise(truth[i], 0.25),
                 colourise(pred[i], 0.5), err.astype(np.uint8)]
        row = np.full((h, len(tiles) * (w + pad) - pad, 3), 18, np.uint8)
        for j, t_img in enumerate(tiles):
            row[:, j * (w + pad):j * (w + pad) + w] = t_img
        rows.append(row)

    sheet_rows = []
    for r in range(0, len(rows), n_cols):
        block = rows[r:r + n_cols]
        width = max(b.shape[1] for b in block)
        stacked = np.full((len(block) * (h + pad) - pad, width, 3), 18, np.uint8)
        for k, b in enumerate(block):
            stacked[k * (h + pad):k * (h + pad) + h, :b.shape[1]] = b
        sheet_rows.append(stacked)
    width = max(s.shape[1] for s in sheet_rows)
    total_h = sum(s.shape[0] for s in sheet_rows) + pad * (len(sheet_rows) - 1)
    sheet = np.full((total_h, width, 3), 18, np.uint8)
    y = 0
    for s in sheet_rows:
        sheet[y:y + s.shape[0], :s.shape[1]] = s
        y += s.shape[0] + pad
    return sheet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/lane_vae/best.pkl")
    ap.add_argument("--data", default="data/lane_vae")
    ap.add_argument("--out", default="results/vae_predictions")
    ap.add_argument("--town", default=None, help="one town, or all shards")
    ap.add_argument("--n", type=int, default=12, help="frames per town")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import cv2
    predict, ckpt = load_model(args.ckpt)
    print(f"[VAE] {args.ckpt}  val_iou={ckpt.get('val_iou', float('nan')):.4f} "
          f"channels={ckpt.get('channels')}")
    os.makedirs(args.out, exist_ok=True)

    pattern = (f"shard_{args.town}_*.npz" if args.town else "shard_*.npz")
    shards = sorted(glob.glob(os.path.join(args.data, pattern)))
    if not shards:
        raise SystemExit(f"no shards matching {pattern} in {args.data}")

    rng = np.random.default_rng(args.seed)
    summary = []
    for sp in shards:
        name = os.path.basename(sp)[len("shard_"):-len(".npz")]
        d = np.load(sp, allow_pickle=True)
        rgb_u8, mask = d["rgb"], d["mask"]
        # Spread the sample across the drive rather than taking the opening
        # frames, which are all one weather preset and one stretch of road.
        idx = np.linspace(0, len(rgb_u8) - 1, args.n).astype(int)
        x = (rgb_u8[idx].astype(np.float32) / 255.0)
        pred, mu = predict(x)
        pred = np.asarray(pred)

        # Per-channel IoU on THIS town, which the corpus-wide val_iou hides.
        ious = []
        for c in range(mask.shape[-1]):
            t = (mask[idx][..., c] > 0.25).astype(np.float32)
            p = (pred[..., c] > 0.5).astype(np.float32)
            inter = (t * p).sum()
            union = np.maximum(t, p).sum()
            ious.append(float(inter / max(union, 1.0)))
        lat = np.asarray(mu)
        summary.append((name, ious, float(lat.std()), float(np.abs(lat).max())))

        sheet = contact_sheet(rgb_u8[idx], mask[idx], pred, n_cols=4)
        path = os.path.join(args.out, f"{name}.png")
        cv2.imwrite(path, sheet[:, :, ::-1])
        print(f"  {name:18s} " +
              "  ".join(f"{CHANNEL_NAMES[c]}_iou={ious[c]:.3f}"
                        for c in range(len(ious))) +
              f"   latent_std={lat.std():.2f}  -> {path}")

    print("\n[VAE] columns per tile: input | ground truth | prediction | error")
    print("      error map: red = missed positive, blue = false positive")
    worst = min(summary, key=lambda r: min(r[1]))
    print(f"[VAE] weakest shard: {worst[0]} (min channel IoU {min(worst[1]):.3f})")


if __name__ == "__main__":
    main()
