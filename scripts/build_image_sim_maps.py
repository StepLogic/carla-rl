#!/usr/bin/env python3
"""Single-trace evaluation maps with IMAGE SIMILARITY in the map building (user
2026-11-07: "apply panorama similarity" -> "Map building only"). The Habitat
peer's three image rules (rig/rigbig.py, relayed 2026-11-06), applied EQUALLY to
every placement, on our single forward camera; the benchmark still localises by
true pose, so a change in results comes from the map alone.

  candidates   trace frames 0.5 m apart (directional thinning), c0-or-expert
               driven alike (an expert tour is the map, not the policy)
  descriptor   scripts/anyloc_similarity.py's AnyLoc (DINOv2 ViT-B/14 value facet,
               32-cluster per-trace VLAD); THRESHOLD = this town's 2% far-false
               point, calibrated here exactly as anyloc_similarity does
  1 key nodes  entropy: online MAD on the candidates (z 1, 15 m window), then
               IMAGE-DISTINCT in trace order (a key scoring >= --dedupe, 0.55, to
               an accepted key is dropped); uniform / random matched to that count
  2 gap nodes  greedily add the worst-covered candidate until every candidate
               scores >= THRESHOLD to some node; then the metric --gap-m (8 m,
               aligned) the peer advised for roads
  3 edges      TRANSITION edges: each candidate is labelled with its best node
               (score >= --label-floor, 0.40); an edge wherever consecutive
               candidates of one drive change label, cost = metres driven.
               Guards, counted: no edge between headings > 90 deg apart (the
               opposite-lane rule), none longer than --max-edge-m (25) in a line.

    build_image_sim_maps.py --trace checkpoints/trace_expert_jx/trace_expert_Town02 \
        --town Town02 --out checkpoints/maps_expert_jx_img
"""
import argparse
import json
import os
import sys
import importlib.util

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
sys.path.insert(0, os.path.dirname(HERE))                   # torch_rl lives at the repo root
sys.path.insert(0, HERE)
from vae_nav import image_map as im  # noqa: E402
from vae_nav import online_maps  # noqa: E402
from build_image_maps import largest_scc  # noqa: E402

_spec = importlib.util.spec_from_file_location("anyloc", os.path.join(HERE, "anyloc_similarity.py"))
anyloc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(anyloc)


def wrapd(a, b):
    return np.degrees(np.abs((np.asarray(a) - b + np.pi) % (2 * np.pi) - np.pi))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--town", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cand-m", type=float, default=0.5)
    # Habitat's thresholds were set against THEIR same-place median (~0.61, true
    # arrivals): spacing 0.55, coverage 0.50, label floor 0.40. Taken as-is on our
    # camera (same-place median 0.73) they placed 120 key nodes in Town02 and the
    # graph fell apart (largest SCC 3-29%, job 464471). So each is scaled by OUR
    # same-place median / theirs unless given explicitly (--scale-to-habitat 0).
    ap.add_argument("--dedupe", type=float, default=None)
    ap.add_argument("--label-floor", type=float, default=None)
    ap.add_argument("--cover", type=float, default=None,
                    help="coverage threshold (default: Habitat 0.50, scaled)")
    ap.add_argument("--habitat-same-median", type=float, default=0.61)
    ap.add_argument("--label-radius-m", type=float, default=25.0,
                    help="a frame's label is the best node among those within this many metres and "
                         "heading-aligned (90 deg): image chooses, pose only rules out far lookalikes")
    ap.add_argument("--gap-m", type=float, default=8.0)
    ap.add_argument("--max-edge-m", type=float, default=25.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=128)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(0)

    tr = im.load_trace(a.trace)
    with open(os.path.join(a.trace, "index.json")) as fh:
        index = json.load(fh)
    from torch_rl.expert_trace import drive_ids
    drive = drive_ids(tr, index)
    S = im.place_uniform(tr, a.cand_m)                       # candidate trace indices
    pos, yaw, ent, drv = tr["pos"][S], tr["yaw"][S], tr["entropy"][S], drive[S]
    frames = im.node_frames(tr, S)
    print(f"[ImgMap] {a.town}: {len(tr['pos'])} frames -> {len(S)} candidates at {a.cand_m:g} m", flush=True)

    enc = anyloc.ValueFacet(dev)
    vf = rng.choice(len(frames), size=min(2000, len(frames)), replace=False)
    pool = torch.cat([enc.patches(frames[vf[i:i + a.batch]]).reshape(-1, 768)
                      for i in range(0, len(vf), a.batch)])
    pool = pool[torch.randperm(len(pool), generator=torch.Generator().manual_seed(0))[:200000].to(pool.device)]
    C = anyloc.spherical_kmeans(pool, 32, 30, 0)
    del pool
    D = torch.cat([anyloc.vlad(enc.patches(frames[i:i + a.batch]), C).half()
                   for i in range(0, len(frames), a.batch)])
    print(f"[ImgMap] descriptors {tuple(D.shape)}", flush=True)

    # THRESHOLD: this town's 2% far-false point (pairs > 10 m apart)
    i = rng.integers(0, len(pos), 400000); j = rng.integers(0, len(pos), 400000)
    far = np.hypot(*(pos[i] - pos[j]).T) > 10
    i, j = i[far][:20000], j[far][:20000]
    thr_far = float(np.percentile((D[i].float() * D[j].float()).sum(1).cpu().numpy(), 98))
    # same-place median: pairs < 2 m apart, heading within 30 deg
    i = rng.integers(0, len(pos), 2000000); j = rng.integers(0, len(pos), 2000000)
    same = (np.hypot(*(pos[i] - pos[j]).T) < 2) & (wrapd(yaw[i], yaw[j]) < 30) & (i != j)
    i, j = i[same][:20000], j[same][:20000]
    same_med = float(np.median((D[i].float() * D[j].float()).sum(1).cpu().numpy()))
    scale = same_med / a.habitat_same_median
    a.dedupe = a.dedupe if a.dedupe is not None else round(0.55 * scale, 3)
    a.label_floor = a.label_floor if a.label_floor is not None else round(0.40 * scale, 3)
    thr = a.cover if a.cover is not None else round(0.50 * scale, 3)
    print(f"[ImgMap] 2% far-false {thr_far:.3f}; same-place median {same_med:.3f} (Habitat {a.habitat_same_median}) "
          f"-> scale {scale:.3f}: coverage {thr}, dedupe {a.dedupe}, label floor {a.label_floor}", flush=True)

    def sims_to(nodes):
        """(len(S), len(nodes)) cosine, chunked, on the GPU."""
        Dn = D[np.asarray(nodes)].float()
        return torch.cat([D[s:s + 4096].float() @ Dn.T for s in range(0, len(D), 4096)])

    # 1. KEY NODES
    mad, _st = online_maps.online_mad(ent, pos, drv, window_m=15.0, z=1.0)
    keys = []
    if len(mad):
        Dm = D[np.asarray(mad)].float()
        for r, k in enumerate(mad):
            if not keys or float((Dm[r:r + 1] @ D[np.asarray(keys)].float().T).max()) < a.dedupe:
                keys.append(int(k))
    n = len(keys)
    view = {"pos": pos, "yaw": yaw}
    _sp, uni = im.match_uniform(view, n)
    rnd = im.place_random(view, n, seed=a.seed)
    print(f"[ImgMap] key nodes: entropy MAD {len(mad)} -> {n} image-distinct; uniform {len(uni)} "
          f"(spacing {_sp:.2f} m); random {len(rnd)}", flush=True)

    rows = []
    for name, key in (("entropy", keys), ("uniform", list(map(int, uni))), ("random", list(map(int, rnd)))):
        nodes = list(dict.fromkeys(key))
        # 2a. IMAGE-COVERAGE GAP NODES
        cover = sims_to(nodes).max(1).values
        img_gap = 0
        while True:
            w = int(torch.argmin(cover))
            if float(cover[w]) >= thr:
                break
            nodes.append(w); img_gap += 1
            cover = torch.maximum(cover, (D @ D[w]).float())     # half matmul: no fp32 copy of D
        # 2b. METRIC GAP (aligned, gap_m)
        before = len(nodes)
        nodes = list(map(int, im.place_uniform(view, a.gap_m, seed_nodes=nodes)))
        met_gap = len(nodes) - before
        # 3. TRANSITION EDGES
        Sm = sims_to(nodes)
        npos_, nyaw_ = pos[np.asarray(nodes)], yaw[np.asarray(nodes)]
        val = np.full(len(S), -1.0); lab = np.zeros(len(S), int)
        for c0 in range(0, len(S), 4096):
            c1 = min(c0 + 4096, len(S))
            dd = np.hypot(pos[c0:c1, None, 0] - npos_[None, :, 0], pos[c0:c1, None, 1] - npos_[None, :, 1])
            ok = (dd <= a.label_radius_m) & (wrapd(yaw[c0:c1, None], nyaw_[None, :]) <= 90)
            sm = Sm[c0:c1].cpu().numpy(); sm[~ok] = -1.0
            lab[c0:c1] = sm.argmax(1); val[c0:c1] = sm.max(1)
        step = np.r_[0.0, np.hypot(*np.diff(pos, axis=0).T)]
        edges, rej_turn, rej_len, last, since = {}, 0, 0, None, 0.0
        for t in range(len(S)):
            if t and drv[t] != drv[t - 1]:
                last, since = None, 0.0
            since += step[t] if t else 0.0
            if val[t] < a.label_floor:
                continue
            k = int(lab[t])
            if last is not None and k != last:
                pa, pb = pos[nodes[last]], pos[nodes[k]]
                if wrapd(yaw[nodes[last]], yaw[nodes[k]]) > 90:
                    rej_turn += 1
                elif np.hypot(*(pa - pb)) > a.max_edge_m:
                    rej_len += 1
                else:
                    e = (last, k)
                    edges[e] = min(edges.get(e, np.inf), max(since, 1e-3))
            if k != last:
                last, since = k, 0.0
        gnodes = [{"pos": (float(pos[s][0]), float(pos[s][1])), "yaw": float(yaw[s]),
                   "idx": int(S[s])} for s in nodes]
        meta = {"placement": name, "town": a.town, "trace": a.trace, "image_rules": True,
                "threshold": round(thr, 3), "far_false_2pct": round(thr_far, 3), "same_place_median": round(same_med, 3),
                "dedupe": a.dedupe, "label_floor": a.label_floor, "label_radius_m": a.label_radius_m,
                "key_nodes": len(key), "image_gap_nodes": img_gap, "metric_gap_nodes": met_gap,
                "gap_m": a.gap_m, "edges_rejected_turn": rej_turn, "edges_rejected_length": rej_len,
                "max_edge_m": a.max_edge_m, "cand_m": a.cand_m}
        g = im.Graph(gnodes, {k: float(v) for k, v in edges.items()}, meta)
        d = os.path.join(a.out, name); os.makedirs(d, exist_ok=True)
        g.save(os.path.join(d, f"{a.town}.json"))
        np.save(os.path.join(d, f"{a.town}_frames.npy"), im.node_frames(tr, [x["idx"] for x in gnodes]))
        scc = largest_scc(g)
        row = {"town": a.town, "placement": name, "nodes": len(gnodes), "edges": len(edges),
               "scc_frac": round(scc / max(len(gnodes), 1), 3), **meta}
        print(json.dumps(row), flush=True)
        rows.append(row)
    with open(os.path.join(a.out, f"summary_{a.town}.json"), "w") as fh:
        json.dump(rows, fh, indent=1)


if __name__ == "__main__":
    main()
