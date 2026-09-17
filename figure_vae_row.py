"""One publication-quality row of VAE input / truth / prediction / error.

    python figure_vae_row.py --seed 3

Emits PDF (vector text, image data at native resolution) and 600-dpi PNG at
IEEE double-column width. The `--single` flag re-emits at 3.5 in.

WHY NOT JUST UPSCALE dump_vae_predictions.py's OUTPUT. That script writes the
panels at their native 120x120 with no text, so a row is a 31 KB PNG about 480
pixels wide. Enlarging it for print resamples 120 px across ~1.7 in, and any
smoothing interpolation invents lane detail the model never predicted -- which
in a figure whose whole subject is lane-mask fidelity would be a fabrication.
Here the arrays are re-rendered from the checkpoint with interpolation="nearest"
so every displayed pixel is one model pixel, and the text is vector rather than
part of the bitmap.

COLOURS MATCH THE EVAL VIDEOS AND dump_vae_predictions.py -- lane green,
obstacle red -- so the same scene reads the same way wherever it appears. The
error map keeps that script's convention: RED is a missed positive, BLUE a false
positive. Those are not interchangeable. A missed lane marking is the model
failing to see something the policy needs; a false positive is it hallucinating
one, which for this VAE is overwhelmingly the boundary of the drivable wedge and
is far less costly.
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

LANE_RGB = np.array([60, 200, 120], np.float32)     # green
OBST_RGB = np.array([215, 90, 60], np.float32)      # red
INK, INK2, MUTED = "#000000", "#3d3d3d", "#8a8a8a"


def colourise(mask2, threshold=None):
    m = np.asarray(mask2, np.float32)
    if threshold is not None:
        m = (m > threshold).astype(np.float32)
    out = np.zeros((*m.shape[:2], 3), np.float32)
    out += m[..., 0:1] * LANE_RGB
    out += m[..., 1:2] * OBST_RGB
    return np.clip(out, 0, 255).astype(np.uint8)


def error_map(pred, truth, threshold=0.5):
    """Red = missed positive, blue = false positive, over both channels."""
    p = (np.asarray(pred, np.float32) > threshold)
    t = (np.asarray(truth, np.float32) > 0.25)
    miss = (t & ~p).any(axis=-1)
    false = (p & ~t).any(axis=-1)
    out = np.zeros((*miss.shape, 3), np.uint8)
    out[miss] = (220, 40, 40)
    out[false] = (40, 60, 230)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/lane_vae_120/best.pkl")
    ap.add_argument("--data", default="data/lane_vae_120/lane_dataset.npz")
    ap.add_argument("--town", default="Town02")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--out", default="results/fig_vae_row")
    ap.add_argument("--single", action="store_true")
    ap.add_argument("--bare", action="store_true",
                    help="no titles, no caption -- just the panels")
    ap.add_argument("--no-truth", action="store_true",
                    help="drop the ground-truth panel, which is near-identical "
                         "to the prediction on a good frame")
    args = ap.parse_args()

    import jax
    from vae_nav.lane_vae import LaneVAE

    with open(args.ckpt, "rb") as f:
        ck = pickle.load(f)
    size = int(ck.get("image_size", 120))
    model = LaneVAE(latent_dim=int(ck.get("latent_dim", 128)), image_size=size)
    params = ck["params"]

    @jax.jit
    def predict(x):
        logits, mu, _ = model.apply({"params": params}, x, deterministic=True)
        return jax.nn.sigmoid(logits), mu

    d = np.load(args.data, allow_pickle=True)
    n_frames = len(d["rgb"])
    # `towns` is ONE ENTRY PER SHARD (4), not per frame (12000) -- indexing it as
    # if it were per-frame silently selects frame 1, which is Town01, while the
    # caption claims whatever --town asked for. The frame ranges come from
    # `shards`, whose insertion order matches the concatenation order.
    shards = json.loads(str(d["shards"])) if not isinstance(d["shards"].item(), dict) \
        else d["shards"].item()
    lo, idx = 0, None
    for name, count in shards.items():
        if name.split("_")[0] == args.town:
            idx = np.arange(lo, lo + int(count))
            break
        lo += int(count)
    if idx is None:
        raise SystemExit(f"{args.town} not in shards {list(shards)}")
    # A frame is chosen by SEED, not hand-picked. Cherry-picking the tile that
    # flatters the model is the obvious way to make this figure dishonest.
    i = int(np.random.default_rng(args.seed).choice(idx))
    rgb = d["rgb"][i].astype(np.float32) / 255.0
    truth = d["mask"][i].astype(np.float32)
    pred, mu = predict(rgb[None])
    pred = np.asarray(pred)[0]

    inter = (truth > 0.25) & (pred > 0.5)
    union = (truth > 0.25) | (pred > 0.5)
    iou = [float(inter[..., c].sum()) / max(float(union[..., c].sum()), 1.0)
           for c in range(2)]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
        "font.size": 7.5, "axes.titlesize": 8, "pdf.fonttype": 42,
        "ps.fonttype": 42, "savefig.facecolor": "white",
    })

    tiles = [((rgb * 255).astype(np.uint8), "Input frame"),
             (colourise(truth, 0.25), "Ground truth"),
             (colourise(pred, 0.5), "VAE prediction"),
             (error_map(pred, truth), "Error")]
    if args.no_truth:
        # The truth panel is dropped for LAYOUT, not from the metrics: the IoU
        # in the caption and the error map are both still computed against it,
        # so nothing here becomes unfalsifiable. On a frame the model gets
        # right the two masks are near-identical and the pair reads as a
        # duplicate; the error panel is what carries the disagreement.
        tiles.pop(1)

    width = 3.5 if args.single else 7.16
    panel = width / len(tiles)
    # Bare: the row is exactly 4 panels tall-by-wide with no text furniture, so
    # the figure height loses the title and caption allowance entirely rather
    # than keeping empty bands where they used to be.
    fig, axes = plt.subplots(1, len(tiles),
                             figsize=(width, panel if args.bare else panel + 0.42))
    for ax, (img, title) in zip(axes, tiles):
        # nearest: one displayed pixel per model pixel, no invented detail.
        ax.imshow(img, interpolation="nearest")
        if not args.bare:
            ax.set_title(title, fontsize=7.8, color=INK, pad=3)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(not args.bare)
            sp.set_color(MUTED); sp.set_linewidth(0.5)

    if not args.bare:
        fig.text(0.5, 0.015,
                 f"{args.town}, {size}$\\times${size} px   ·   lane IoU {iou[0]:.3f}, "
                 f"obstacle IoU {iou[1]:.3f}   ·   green = lane, red = obstacle   ·   "
                 f"error: red = missed, blue = false positive",
                 ha="center", fontsize=6.4, color=INK2)

    if args.bare:
        fig.subplots_adjust(0, 0, 1, 1, wspace=0.02, hspace=0)
    else:
        fig.tight_layout(rect=(0, 0.055, 1, 1), pad=0.25, w_pad=0.6)
    for ext, kw in ((".pdf", {}), (".png", {"dpi": 600})):
        fig.savefig(args.out + ext, bbox_inches="tight", **kw)
        print(f"[Fig] -> {args.out}{ext}")
    print(f"[Fig] frame {i} of {n_frames} ({args.town} range "
          f"{idx[0]}-{idx[-1]})  lane IoU {iou[0]:.3f}  "
          f"obstacle IoU {iou[1]:.3f}  latent std {float(np.std(np.asarray(mu))):.2f}")


if __name__ == "__main__":
    main()
