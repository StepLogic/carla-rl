#!/usr/bin/env python3
"""v25 explore-mode sweep report (user 2026-09-28: "evaluate explore mode",
"check only on town05").

Per <town>_checkpoint_<k>_explore.json (+ .qtrace.jsonl, the EXPLORE critic):
  reached    share of episodes that drove EXPLORE_GOAL_DIST_M (100 m) of path
             without a terminal fault
  ends       what ended the episodes (crossing / off_road / collision / reached)
  path       mean and median path driven (m)
  faults     share of episodes with any off-road / illegal crossing / lane change
  critic     explore-critic AUROC for a fault within 5 / 10 steps (raw: explore
             has no goal, so no distance bins), and calibration against the
             discounted (0.99) return on untruncated episodes
usage: v25_explore_report.py SWEEP_DIR
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from vae_nav.q_fault_analysis import analyse  # noqa: E402

out = []
for f in sorted(glob.glob(os.path.join(sys.argv[1], "*_explore.json"))):
    j = json.load(open(f))
    agg = j.get("faults") or {}
    name = os.path.basename(f).split("_")
    row = {"town": name[0], "ckpt": int(name[2]), "n": j.get("episodes"),
           "reached": j.get("explore_reached_rate"), "ends": j.get("explore_ends"),
           "path_mean": agg.get("mean_path_m"), "path_median": agg.get("median_path_m"),
           "off_road": agg.get("frac_any_off_road"),
           "illegal": agg.get("frac_any_illegal_crossing"),
           "lane_change": agg.get("frac_any_lane_change"),
           "collision": agg.get("frac_any_collision")}
    tr = f + ".qtrace.jsonl"
    if os.path.exists(tr):
        rows = [json.loads(l) for l in open(tr)]
        a = analyse(rows, gamma=0.99)
        c = a["calibration"] or {}
        row["critic"] = {
            "auroc5": a["anticipation"]["fault_within_5"]["auroc_raw"],
            "auroc10": a["anticipation"]["fault_within_10"]["auroc_raw"],
            "calib_corr": c.get("corr_q_vs_return_to_go"),
            "calib_bias": c.get("bias_mean_q_minus_g"),
            "q_mean": c.get("q_mean"), "g_mean": c.get("g_mean")}
    out.append(row)

json.dump(out, open(os.path.join(sys.argv[1], "v25_explore_report.json"), "w"), indent=2)
for r in sorted(out, key=lambda r: (r["town"], r["ckpt"])):
    c = r.get("critic", {})
    print(f"{r['town']} {r['ckpt']:>6}  reached={r['reached']}  path mean/median="
          f"{r['path_mean']}/{r['path_median']} m  ends={r['ends']}")
    print(f"         faults: off-road={r['off_road']} illegal={r['illegal']} "
          f"lane-change={r['lane_change']} collision={r['collision']}")
    if c:
        print(f"         explore critic: AUROC fault<5={c['auroc5']} <10={c['auroc10']}  "
              f"calib r={c['calib_corr']} bias={c['calib_bias']} "
              f"(Q {c['q_mean']} vs G {c['g_mean']})")
