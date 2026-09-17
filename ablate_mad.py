"""Ablation of the MAD spike detector's parameters, on a saved trace.

    python ablate_mad.py --in evidence/.../data/entropy_nogoal_500cap.json

WHAT THIS IS FOR. The headline placement result used one setting: z>=3, median
and MAD, 9-step smoothing, 8 m separation, no NMS. Those were chosen on
reasoning, not measured, so the obvious objection is that they were tuned --
knowingly or not -- to the answer. This sweeps each one and reports whether the
conclusion survives.

FOUR AXES, each isolated with the others held at the headline setting:

  z          1.5 .. 4.0   how far above the local baseline a step must stand
  centre     median/MAD vs mean/std -- ROBUST vs not
  nms_w      strict local-maximum window, 0 = off
  smooth     moving-average window before scoring
  min_sep    metres between accepted nodes

WHAT IS REPORTED PER CONFIG, at the MAP level (nodes merged across episodes,
which is the level the headline claim is made at):

  nodes      merged node count -- the detector's own budget, which FLOATS
  recall     fraction of junctions with a node within 10 m
  precision  fraction of nodes within 5 m of a junction
  uni_rec    recall of a uniform arm bisected to the SAME merged node count

The comparison that matters is `recall - uni_rec` at equal node count. A setting
that raises recall by raising the node count has proved nothing; the uniform arm
is re-matched for every config precisely so that cannot happen.

BASE RATE is printed once. Precision below it means the detector is worse than
dropping nodes at random.
"""
import argparse
import json

import numpy as np

import map_nodes_entropy as M
import node_placement_entropy as NP

MERGE_R, WARM = 6.0, 25


def collect(episodes, z, centre, nms_w, smooth_w, min_sep):
    """Per-episode nodes, pooled. Mirrors map_nodes_entropy.collect but exposes
    every detector parameter rather than only z."""
    pts, wts, traj, junc = [], [], [], []
    for e in episodes:
        st = e["steps"]
        if any(s.get("x") is None for s in st) or len(st) < WARM + 30:
            continue
        xy, h, inj, s = M.episode_arrays(e)
        if s[-1] < 2 * min_sep:
            continue
        traj.append(xy)
        junc.append(NP.junction_events(inj, xy))
        idx = [i + WARM for i in NP.mad_spikes(h[WARM:], s[WARM:], min_sep,
                                               smooth_w, z, centre=centre,
                                               nms_w=nms_w)]
        if idx:
            pts.append(xy[idx]); wts.append(NP.smooth(h, smooth_w)[idx])
    pts = np.vstack(pts) if pts else np.empty((0, 2))
    wts = np.concatenate(wts) if wts else np.empty(0)
    junc = np.vstack([j for j in junc if len(j)]) if junc else np.empty((0, 2))
    return pts, wts, traj, junc


def evaluate(episodes, z=3.0, centre="median", nms_w=0, smooth_w=9,
             min_sep=8.0, match_uniform=True):
    raw, w, traj, junc_raw = collect(episodes, z, centre, nms_w, smooth_w, min_sep)
    J = M.greedy_merge(junc_raw, MERGE_R)
    ne = M.greedy_merge(raw, MERGE_R, w)
    if len(ne) == 0 or len(J) == 0:
        return None
    pair = np.linalg.norm(J[:, None, :] - ne[None, :, :], axis=2)
    rec = float((pair.min(axis=1) <= 10).mean())
    prec = float((pair.min(axis=0) <= 5).mean())
    uni_rec = float("nan")
    if match_uniform:
        nu = M.uniform_to_count(traj, len(ne), MERGE_R)
        if len(nu):
            duj = np.linalg.norm(J[:, None, :] - nu[None, :, :], axis=2).min(axis=1)
            uni_rec = float((duj <= 10).mean())
    return {"raw": len(raw), "nodes": len(ne), "recall": rec,
            "precision": prec, "uni_recall": uni_rec, "junctions": len(J)}


def row(label, r):
    if r is None:
        return f"  {label:<26}      --   (detector found nothing)"
    d = (r["recall"] - r["uni_recall"]) * 100
    return (f"  {label:<26}{r['raw']:>5}{r['nodes']:>7}{r['recall']*100:>9.1f}%"
            f"{r['precision']*100:>11.1f}%{r['uni_recall']*100:>10.1f}%{d:>+9.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp",
                    default="evidence/2026-09-09_entropy_at_junctions/data/"
                            "entropy_nogoal_500cap.json")
    ap.add_argument("--out", default="results/ablation_mad.json")
    args = ap.parse_args()

    eps = json.load(open(args.inp))["episodes"]
    hdr = (f"  {'config':<26}{'raw':>5}{'nodes':>7}{'recall':>10}"
           f"{'precision':>11}{'uni_rec':>10}{'delta':>9}")

    # base rate, computed once
    _, _, traj, junc_raw = collect(eps, 3.0, "median", 0, 9, 8.0)
    J = M.greedy_merge(junc_raw, MERGE_R)
    allp = np.vstack(traj)
    base5 = float((np.linalg.norm(allp[:, None, :] - J[None, :, :],
                                  axis=2).min(axis=1) <= 5).mean())
    print(f"{len(eps)} episodes, {len(J)} distinct junctions, "
          f"base-rate precision <5 m = {base5*100:.1f}%")
    print("HEADLINE SETTING: z=3.0, median/MAD, nms_w=0, smooth=9, sep=8 m\n")

    out = {}
    for title, key, values, kw in (
            ("z threshold", "z", [1.5, 2.0, 2.5, 3.0, 3.5, 4.0], {}),
            ("centre/scale", "centre", ["median", "mean"], {}),
            ("NMS window (steps)", "nms_w", [0, 5, 15, 31], {}),
            ("smoothing (steps)", "smooth_w", [1, 5, 9, 15, 25], {}),
            ("min separation (m)", "min_sep", [4.0, 8.0, 16.0, 32.0], {})):
        print(f"{title}\n{hdr}")
        for v in values:
            r = evaluate(eps, **{key: v}, **kw)
            lab = f"{key}={v}"
            print(row(lab, r))
            out[lab] = r
        print()

    # The two the user asked for, crossed, since z and the centre statistic
    # interact: a non-robust scale is inflated by the spikes, so the SAME z
    # means a different thing under mean/std than under median/MAD.
    print("z x centre (crossed)\n" + hdr)
    for c in ("median", "mean"):
        for z in (2.0, 3.0):
            r = evaluate(eps, z=z, centre=c)
            lab = f"z={z}, {c}"
            print(row(lab, r)); out[lab] = r
    json.dump({"base_rate_precision_5m": base5, "configs": out},
              open(args.out, "w"), indent=2, default=float)
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
