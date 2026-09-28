#!/usr/bin/env python3
"""v25 sweep report: success, rule faults, and the two goal critics side by side.

For every <town>_checkpoint_<k>_goal.json in the sweep dir (and its .qtrace.jsonl):
  success / clean arrival / off-road / illegal crossing / collision rates
  per critic -- image-only `q` (the map-edge distance) and privileged `q_p`:
    ordering     Spearman rank correlation of Q with -route_remaining_m, over
                 all steps: does the value still rank states by distance to go?
    anticipation AUROC of -Q for a fault within 5 steps, within distance bins
    calibration  corr and mean bias of Q against the discounted (0.99) return
                 to go, on episodes that ended untruncated
usage: v25_critic_report.py SWEEP_DIR
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from vae_nav.q_fault_analysis import analyse  # noqa: E402


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


out = []
for f in sorted(glob.glob(os.path.join(sys.argv[1], "*_goal.json"))):
    j = json.load(open(f))
    agg = j.get("faults") or {}
    town = os.path.basename(f).split("_")[0]
    ck = os.path.basename(f).split("_")[2]
    row = {"town": town, "ckpt": int(ck), "n": j.get("n_episodes", j.get("episodes")),
           "success": j.get("success"),
           "clean": agg.get("clean_arrival_rate"),
           "off_road": agg.get("frac_any_off_road"),
           "illegal": agg.get("frac_any_illegal_crossing"),
           "lane_change": agg.get("frac_any_lane_change"),
           "collision": agg.get("frac_any_collision")}
    tr = f + ".qtrace.jsonl"
    if os.path.exists(tr):
        rows = [json.loads(l) for l in open(tr)]
        rem = np.array([np.nan if r["remaining_m"] is None else r["remaining_m"] for r in rows])
        for key in ("q", "q_p"):
            if any(r.get(key) is None for r in rows):
                continue
            qs = np.array([r[key] for r in rows], float)
            ok = ~np.isnan(rem)
            a = analyse([dict(r, q=r[key]) for r in rows], gamma=0.99)
            c = a["calibration"] or {}
            row[key] = {
                "mean": round(float(qs.mean()), 1),
                "order_rho": round(spearman(qs[ok], -rem[ok]), 3),
                "fault5_auroc_bins": a["anticipation"]["fault_within_5"].get(
                    "auroc_within_distance_bins"),
                "calib_corr": c.get("corr_q_vs_return_to_go"),
                "calib_bias": c.get("bias_mean_q_minus_g"),
            }
    out.append(row)

json.dump(out, open(os.path.join(sys.argv[1], "v25_critic_report.json"), "w"), indent=2)
hdr = (f"{'town':7s} {'ckpt':>6s} {'SR':>5s} {'clean':>5s} {'offrd':>5s} {'illeg':>5s} "
       f"{'lnchg':>5s} {'coll':>5s} | critic  mean   rho_dist  AUROC5  cal_r  cal_bias")
print(hdr)
for r in sorted(out, key=lambda r: (r["town"], r["ckpt"])):
    base = (f"{r['town']:7s} {r['ckpt']:6d} {r['success']:5.2f} {r['clean'] or 0:5.2f} "
            f"{r['off_road'] or 0:5.2f} {r['illegal'] or 0:5.2f} {r['lane_change'] or 0:5.2f} "
            f"{r['collision'] or 0:5.2f}")
    for i, key in enumerate(("q", "q_p")):
        if key not in r:
            continue
        v = r[key]
        name = "image" if key == "q" else "priv"
        print(f"{base if i == 0 else ' ' * len(base)} | {name:6s} {v['mean']:7.1f} "
              f"{v['order_rho']:8.3f} {v['fault5_auroc_bins']!s:>7} {v['calib_corr']!s:>6} "
              f"{v['calib_bias']!s:>8}")
