#!/usr/bin/env python3
"""Does the control stream care which tick its speed input belongs to?

scripts/pretrain_control.py fed frame t with the shard's speed[t], which the recorder wrote AFTER
action t (torch_rl/expert_data.py: "frame t+1 was recorded with speed[t]"), so the stream read the
speed one tick AHEAD of its frame. Online, and in the expert rows built for learner C
(torch_rl/frl/carla.ControlTrunk), the row carries the speed at the frame itself. This scores a
trained stream on the SAME held-out chunks under both pairings: the lane probes (ridge R^2 fitted
and scored per pairing, and fitted on the pretraining pairing then scored on the deployment one),
the cloning loss, and how far the GRU state moves.

Result 2026-11-14 for control_pre_n2_k8/step_6000 (repeat 8, 34 validation episodes, 8,644 rows):
the probe reads the same under both pairings (offset 0.759 vs 0.757, heading 0.663 vs 0.663, fitted
on one and scored on the other 0.750 / 0.658) and the GRU state moves by 2.2% of its sd; mean
one-tick speed change 0.16 km/h against a speed sd of 5.6. So the PAIRING changes little.

READ THE R^2 HERE ONLY AS A COMPARISON BETWEEN THE TWO PAIRINGS. They are a ridge probe FITTED AND
SCORED ON THE SAME ROWS, which says nothing about what the stream carries: held out by episode the
same stream gives heading about 0.1 (scripts/control_stream_heldout.py). This script's first
version quoted them as the stream's held-out figures; that was wrong.

usage: /home/robotlab/carla-sac-local/venv/bin/python -I scripts/control_speed_pairing_check.py CKPT_DIR
"""
import glob
import os
import sys
from concurrent.futures import ThreadPoolExecutor

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [REPO, os.path.join(REPO, "src"), os.path.join(REPO, "scripts")]
os.environ.setdefault("MONO_CHW", "1")
os.environ.setdefault("SPEED_OBS", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402

import pretrain_control as pc  # noqa: E402
from torch_rl.control_cnn import ControlGRU  # noqa: E402


def at_frame(e):
    """The speed WHEN THE FRAME WAS TAKEN: frame 0 at rest, frame t = the speed after tick t-1."""
    d = dict(e)
    d["speed"] = np.concatenate([[0.0], e["speed"]]).astype(np.float32)[:len(e["speed"])]
    return d


def ridge_fit(H, Y, lam=1.0):
    X = np.concatenate([H, np.ones((len(H), 1), np.float32)], 1)
    return np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1], dtype=np.float32), X.T @ Y)


def r2(H, Y, W):
    X = np.concatenate([H, np.ones((len(H), 1), np.float32)], 1)
    P = X @ W
    return [float(1 - ((Y[:, k] - P[:, k]) ** 2).sum() / ((Y[:, k] - Y[:, k].mean()) ** 2).sum())
            for k in range(Y.shape[1])]


def main():
    ckpt = sys.argv[1]
    ck = torch.load(os.path.join(ckpt, "params.pt"), map_location="cpu")
    a = ck["args"]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = ControlGRU(latent_dim=a["latent"], hidden_dim=a["hidden"], px=a["px"],
                     n_frames=a["n_frames"]).to(dev)
    net.load_state_dict(ck["net"])
    net.eval().requires_grad_(False)

    # the run's own validation shards: the same seeded shuffle as pretrain_control.main
    rng = np.random.default_rng(0)
    shards = []
    for root in a["data"].split(","):
        for t in a["towns"].split(","):
            shards += sorted(glob.glob(os.path.join(root, t, "*.npz")))
    rng.shuffle(shards)
    lanes = {}
    with ThreadPoolExecutor(8) as ex:
        val = [e for e in ex.map(lambda p: pc.load_shard(p, lanes, a["xodr"]),
                                 shards[:a["val_shards"]])
               if e["n_act"] > (a["chunk"] - 1) * a["repeat"]]
    val_frame = [at_frame(e) for e in val]
    lead = np.concatenate([np.abs(e["speed"] - f["speed"]) for e, f in zip(val, val_frame)])
    print(f"=== {ckpt}\n=== repeat {a['repeat']} n_frames {a['n_frames']} | {len(val)} held-out "
          f"episodes | |speed(t) - speed(t-1)| mean {lead.mean():.3f} km/h, p95 "
          f"{np.percentile(lead, 95):.3f}, max {lead.max():.2f}; speed sd "
          f"{np.concatenate([e['speed'] for e in val]).std():.2f}", flush=True)

    def collect(pool, seed, n=32):
        r = np.random.default_rng(seed)             # same seed -> the same chunks from either pool
        H, Y, bc, allh = [], [], [], []
        for _ in range(n):
            bv = pc.batch(pool, r, a["batch"], a["chunk"], a["n_frames"], dev, a["px"], a["repeat"])
            with torch.no_grad():
                b_, _, hs = pc.run_chunk(net, bv, a["burn_in"], 0)
            bc.append(float(b_))
            m = bv["lane_valid"][:, a["burn_in"]:].reshape(-1).bool().cpu().numpy()
            H.append(hs[:, a["burn_in"]:].reshape(-1, hs.shape[-1]).cpu().numpy()[m])
            Y.append((bv["lane"][:, a["burn_in"]:].reshape(-1, 2).cpu().numpy() * pc.SD)[m])
            allh.append(hs.reshape(-1, hs.shape[-1]).cpu().numpy())
        return np.concatenate(H), np.concatenate(Y), float(np.mean(bc)), np.concatenate(allh)

    Hp, Yp, bcp, hp = collect(val, 7)
    Hd, Yd, bcd, hd = collect(val_frame, 7)
    assert np.allclose(Yp, Yd), "the two pools did not draw the same chunks"
    Wp, Wd = ridge_fit(Hp, Yp), ridge_fit(Hd, Yd)
    print(f"PRETRAINING pairing (speed one tick ahead): R2 offset {r2(Hp, Yp, Wp)[0]:.3f} heading "
          f"{r2(Hp, Yp, Wp)[1]:.3f} | val bc {bcp:.4f} | rows {len(Hp)}", flush=True)
    print(f"DEPLOYMENT pairing (speed at the frame):    R2 offset {r2(Hd, Yd, Wd)[0]:.3f} heading "
          f"{r2(Hd, Yd, Wd)[1]:.3f} | val bc {bcd:.4f}", flush=True)
    print(f"probe FITTED on pretraining, SCORED on deployment: R2 offset {r2(Hd, Yd, Wp)[0]:.3f} "
          f"heading {r2(Hd, Yd, Wp)[1]:.3f}", flush=True)
    d = np.abs(hp - hd)
    print(f"GRU state: mean |difference| {d.mean():.4f} against the state's own sd {hp.std():.4f} "
          f"({100 * d.mean() / hp.std():.1f}%)", flush=True)


if __name__ == "__main__":
    main()
