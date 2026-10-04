"""DINOv2 features for the RGB bank (user 2026-09-19: "replace localization with dinov2").

Runs in dependencies/vc1-venv (torch 2.5.1+cu121; the system torch cannot see this driver). Weights and
the hub repo are already in ~/.cache/torch/hub, so nothing is downloaded.

Stores BOTH pooling choices -- CLS token and mean of the patch tokens -- because place recognition
usually prefers the patch mean, and the offline report picks the winner before any sim time is spent.
"""
import argparse, pickle, sys
import numpy as np

HUB = "/home/robotlab/.cache/torch/hub/facebookresearch_dinov2_main"
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", default="results/episodes/town02_speed10_s7_rgb_bank.pkl")
    ap.add_argument("--out", default="results/episodes/town02_speed10_s7_dinov2.npz")
    ap.add_argument("--model", default="dinov2_vits14")
    ap.add_argument("--size", type=int, default=224, help="square resize before the patch grid (multiple of 14)")
    ap.add_argument("--batch", type=int, default=64)
    a = ap.parse_args()
    import torch
    import torch.nn.functional as F

    bank = pickle.load(open(a.bank, "rb"))
    rows = bank["rows"]
    X = np.stack([r["rgb"] for r in rows]).astype(np.float32) / 255.0
    print(f"[dinov2] {len(rows)} frames {X.shape[1:]} -> {a.size}x{a.size}, model {a.model}", flush=True)
    model = torch.hub.load(HUB, a.model, source="local", pretrained=True).eval().cuda()

    cls, pat = [], []
    with torch.no_grad():
        for i in range(0, len(X), a.batch):
            t = torch.from_numpy(X[i:i + a.batch]).permute(0, 3, 1, 2).cuda()
            t = F.interpolate(t, size=(a.size, a.size), mode="bicubic", align_corners=False).clamp(0, 1)
            t = (t - torch.tensor(MEAN).view(1, 3, 1, 1).cuda()) / torch.tensor(STD).view(1, 3, 1, 1).cuda()
            o = model.forward_features(t)
            cls.append(o["x_norm_clstoken"].float().cpu().numpy())
            pat.append(o["x_norm_patchtokens"].mean(1).float().cpu().numpy())
            if (i // a.batch) % 5 == 0:
                print(f"[dinov2] {min(i + a.batch, len(X))}/{len(X)}", flush=True)
    cls = np.concatenate(cls); pat = np.concatenate(pat)
    np.savez(a.out, cls=cls, patchmean=pat,
             kind=np.array([r["kind"] for r in rows]), index=np.array([r["index"] for r in rows]),
             role=np.array([r["role"] if r["role"] else "" for r in rows]),
             gt_xy=np.array([r["gt_xy"] for r in rows], np.float64),
             latent=np.stack([r["latent"] for r in rows]).astype(np.float32),
             meta=np.array([f"{a.model} size={a.size} bank={a.bank}"]))
    print(f"[dinov2] wrote {a.out}: cls {cls.shape}, patchmean {pat.shape}", flush=True)


if __name__ == "__main__":
    main()
