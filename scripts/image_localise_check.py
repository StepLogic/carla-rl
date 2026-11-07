#!/usr/bin/env python3
"""Pose-free localisation, measured without driving (user 2026-11-07 rule: image
navigation for a paper -- true pose and goal coordinates are privileged).

For benchmark episodes that saved a video (mapless runs, every 10th episode), the
first frame holds the car's START view (left half) and the GOAL image (right
half). Each is matched top-1 to the map's node images with the AnyLoc descriptor
(scripts/anyloc_similarity.py, vocabulary rebuilt from the map's own trace as
there). Correct = the matched node is within 8 m of the true start / goal (and,
for the start, heading-aligned) -- the true pose is used ONLY to score.

CAVEAT, stated: the video's left half carries a text overlay in its top rows; it
is cropped (top 40 px) from both halves before matching.

    image_localise_check.py --map checkpoints/maps_expert_jx_img/uniform/Town02.json \
        --trace checkpoints/trace_expert_jx/trace_expert_Town02 --town Town02
"""
import argparse, glob, json, os, sys, importlib.util
import numpy as np
import cv2
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
from vae_nav import image_map as im  # noqa: E402
_spec = importlib.util.spec_from_file_location("anyloc", os.path.join(HERE, "anyloc_similarity.py"))
anyloc = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(anyloc)


def wrapd(a, b):
    return np.degrees(np.abs((np.asarray(a) - b + np.pi) % (2 * np.pi) - np.pi))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", required=True); ap.add_argument("--trace", required=True)
    ap.add_argument("--town", required=True); ap.add_argument("--crop-top", type=int, default=40)
    ap.add_argument("--radius", type=float, default=8.0)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(0)
    tr = im.load_trace(a.trace)
    g = im.Graph.load(a.map)
    N = np.array([n["pos"] for n in g.nodes]); Y = np.array([n["yaw"] for n in g.nodes])
    nframes = im.node_frames(tr, [n["idx"] for n in g.nodes])
    enc = anyloc.ValueFacet(dev)
    # vocabulary from the map's own trace, as anyloc_similarity / the map builder
    S = im.place_uniform(tr, 0.5); fr = im.node_frames(tr, S[rng.choice(len(S), min(2000, len(S)), replace=False)])
    pool = torch.cat([enc.patches(fr[i:i + 128]).reshape(-1, 768) for i in range(0, len(fr), 128)])
    pool = pool[torch.randperm(len(pool), generator=torch.Generator().manual_seed(0))[:200000].to(pool.device)]
    C = anyloc.spherical_kmeans(pool, 32, 30, 0); del pool

    def desc(frames):
        frames = np.asarray(frames)[:, a.crop_top:]
        return torch.cat([anyloc.vlad(enc.patches(frames[i:i + 128]), C) for i in range(0, len(frames), 128)])
    DN = desc(nframes)
    rows = []
    for f in sorted(glob.glob(f"checkpoints/bench_c0_true_{a.town}_*/eval_step_15000/eval.json")):
        for e in json.load(open(f))["episodes_detail"]:
            v = e.get("video")
            if not v or not os.path.exists(v):
                continue
            ok, fr0 = cv2.VideoCapture(v).read()
            if not ok:
                continue
            h, w = fr0.shape[:2]
            cur = cv2.resize(fr0[:, : w // 2][:, :, ::-1], (120, 120))
            goal = cv2.resize(fr0[:, w // 2:][:, :, ::-1], (120, 120))
            D = desc(np.stack([cur, goal]))
            s = (D @ DN.T).cpu().numpy()
            sn, gn = int(s[0].argmax()), int(s[1].argmax())
            ds = float(np.hypot(*(N[sn] - e["start_xy"]))); dg = float(np.hypot(*(N[gn] - e["goal_xy"])))
            rows.append({"band": e.get("band"), "seed": e["seed"],
                         "start_ok": ds <= a.radius and wrapd(Y[sn], e["start_yaw"]) <= 60, "start_err_m": round(ds, 1),
                         "goal_ok": dg <= a.radius, "goal_err_m": round(dg, 1),
                         "start_score": round(float(s[0].max()), 3), "goal_score": round(float(s[1].max()), 3)})
    n = len(rows)
    so = sum(r["start_ok"] for r in rows); go = sum(r["goal_ok"] for r in rows); both = sum(r["start_ok"] and r["goal_ok"] for r in rows)
    print(json.dumps({"town": a.town, "map": a.map, "episodes": n,
                      "start_correct": f"{so}/{n}", "goal_correct": f"{go}/{n}", "both_correct": f"{both}/{n}",
                      "start_err_m_median": float(np.median([r["start_err_m"] for r in rows])) if n else None,
                      "goal_err_m_median": float(np.median([r["goal_err_m"] for r in rows])) if n else None}, indent=1))
    for r in rows:
        if not (r["start_ok"] and r["goal_ok"]):
            print("MISS", r)


if __name__ == "__main__":
    main()
