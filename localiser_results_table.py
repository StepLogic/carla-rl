"""SR, SPL and localisation error by localiser (user 2026-09-19 "replace localization with dinov2").

Three rows per map, all on the SAME 40 episodes and the SAME maps -- only the features that node
matching uses differ:
  mobilenet_trace  shipped: MobileNet features recorded while driving past each node (the paper's rows)
  mobilenet_bank   same encoder, node features from the teleported RGB bank
  dinov2_bank      DINOv2 ViT-S/14 patch-mean over those same bank frames, live frame embedded on the fly
The middle row is the control: without it, a DINOv2 gain cannot be separated from the capture change.
Run from the icra-2027 folder.
"""
import argparse, collections, glob, json
from math import comb
import numpy as np

ARMS = [("Spacing", ["uniform15_trace", "uniform8_trace", "uniform6_trace", "uniform2_trace"]),
        ("SGM", ["sgm_g24_e55_c55", "sgm_g10_e55_c55", "sgm_g7_e55_c55", "sgm_g2_e55_c55"]),
        ("Entropy + SGM", ["hybrid_sgm_g44_entropy_thin4", "hybrid_sgm_g14_entropy_thin4",
                           "hybrid_sgm_g9_entropy_thin4", "hybrid_sgm_g2_entropy_thin4"])]
BASE = ["results/drive_plan_c/*.jsonl", "results/drive_plan_c/repeat/*.jsonl",
        "results/drive_plan_c/equal_points/*.jsonl"]


def sign_test(a, b):
    wa = sum(1 for k in a if a[k] > b[k]); wb = sum(1 for k in a if a[k] < b[k])
    n = wa + wb
    if n == 0:
        return wa, wb, 1.0
    k = min(wa, wb)
    return wa, wb, min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def load(globs, want_loc=None):
    per = collections.defaultdict(dict)
    for f in sorted({f for g in globs for f in glob.glob(g)}):
        if any(s in f for s in ("smoke", "debug")):
            continue
        for line in open(f):
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("localiser", "mobilenet_trace").split(":")[0] != want_loc:
                continue
            per[r["map"]][(r["set"], r["index"])] = r
    return {m: d for m, d in per.items() if len(d) == 40}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/drive_plan_c/localiser")
    ap.add_argument("--out", default="tables/localiser_sweep_town02.md")
    a = ap.parse_args()
    runs = {"mobilenet_trace": load(BASE, "mobilenet_trace"),
            "mobilenet_bank": load([f"{a.dir}/mobilenet_bank_*.jsonl"], "mobilenet_bank"),
            "dinov2_bank": load([f"{a.dir}/dinov2_bank_*.jsonl"], "dinov2_bank")}

    md = ["# Localiser sweep, CARLA Town02 (40 episodes, one run each)\n",
          "`mobilenet_trace` is the shipped row. `mobilenet_bank` changes only WHERE node features come "
          "from (teleported bank); `dinov2_bank` changes the encoder on top of that, so the two together "
          "separate capture from encoder.\n",
          "| Arm | Map | Nodes | Localiser | SR | SPL | Start err (med) | Goal err (med) | vs shipped |",
          "|---|---|---|---|---|---|---|---|---|"]
    for arm, maps in ARMS:
        for m in maps:
            base = runs["mobilenet_trace"].get(m)
            n = None
            for mode in ("mobilenet_trace", "mobilenet_bank", "dinov2_bank"):
                d = runs[mode].get(m)
                if not d:
                    md.append(f"| {arm} | {m} | | {mode} | not driven | | | | |"); continue
                R = list(d.values())
                sr = np.mean([r["success"] for r in R]); spl = np.mean([r["spl"] or 0 for r in R])
                se = np.median([r["start_node_err_m"] for r in R])
                ge = np.median([r["goal_node_err_m"] for r in R])
                cmp = ""
                if base and mode != "mobilenet_trace":
                    aa = {k: d[k]["success"] for k in d if k in base}
                    bb = {k: base[k]["success"] for k in aa}
                    wa, wb, p = sign_test(aa, bb)
                    cmp = f"{wa}-{wb}, p={p:.2f}"
                md.append(f"| {arm} | {m} | {len(R)} eps | {mode} | {sr:.3f} | {spl:.3f} | {se:.1f} m | "
                          f"{ge:.1f} m | {cmp} |")
    open(a.out, "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
