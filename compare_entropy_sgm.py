"""Entropy + SGM against the other arms at matched node counts (user 2026-09-19).

Pairs the SAME 40 episodes, per block. The Entropy + SGM arm has one run, so the paired sign test
uses each competitor's RUN 1; the competitor's 3-run mean is printed beside it for context.
Run from the icra-2027 folder (or anywhere the results/ symlinks resolve).
"""
import collections, glob, json, sys
import numpy as np
from math import comb

BLOCKS = [
    ("~60",  "hybrid_sgm_g44_entropy_thin4", ["uniform15_trace", "hybrid_uniform15_entropy_thin4", "sgm_g24_e55_c55"]),
    ("~115", "hybrid_sgm_g14_entropy_thin4", ["uniform8_trace", "hybrid_uniform8_entropy_thin4", "sgm_g10_e55_c55"]),
    ("~150", "hybrid_sgm_g9_entropy_thin4",  ["uniform6_trace", "hybrid_uniform6_entropy_thin4", "sgm_g7_e55_c55"]),
    ("~550", "hybrid_sgm_g2_entropy_thin4",  ["uniform2_trace", "hybrid_uniform2_entropy_thin4", "sgm_g2_e55_c55"]),
]
RUN2_3 = ("equal_points/run2", "equal_points/run3", "repeat/run2", "repeat/run3", "uniform8_run2")
GLOBS = ["results/drive_plan_c/*.jsonl", "results/drive_plan_c/repeat/*.jsonl",
         "results/drive_plan_c/equal_points/*.jsonl"]


def load():
    """-> per_run[map] = list of {(set,index): row}, and which file each run came from."""
    runs = collections.defaultdict(list)
    for f in sorted({f for g in GLOBS for f in glob.glob(g)}):
        if any(s in f for s in ("smoke", "debug")):
            continue
        by_map = collections.defaultdict(dict)
        for line in open(f):
            if line.strip():
                r = json.loads(line); by_map[r["map"]][(r["set"], r["index"])] = r
        for m, d in by_map.items():
            if len(d) == 40:
                runs[m].append((f, d))
    return runs


def sign_test(a, b):
    """Two-sided sign test over paired episodes: wins for a, wins for b, p."""
    wa = sum(1 for k in a if a[k] > b[k]); wb = sum(1 for k in a if a[k] < b[k])
    n = wa + wb
    if n == 0:
        return wa, wb, 1.0
    k = min(wa, wb)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)
    return wa, wb, p


def main():
    runs = load()
    for label, ours, others in BLOCKS:
        if ours not in runs:
            print(f"[{label}] {ours}: not driven yet"); continue
        _, A = runs[ours][0]
        sr_a = np.mean([r["success"] for r in A.values()])
        spl_a = np.mean([r["spl"] or 0 for r in A.values()])
        print(f"\n=== {label} nodes: Entropy + SGM  SR {sr_a:.3f}  SPL {spl_a:.3f} (1 run) ===")
        for o in others:
            if o not in runs:
                print(f"  {o:32s} missing"); continue
            run1 = next((d for f, d in runs[o] if not any(t in f for t in RUN2_3)), runs[o][0][1])
            sr_all = [np.mean([r["success"] for r in d.values()]) for _, d in runs[o]]
            a = {k: A[k]["success"] for k in A if k in run1}
            b = {k: run1[k]["success"] for k in a}
            wa, wb, p = sign_test(a, b)
            print(f"  vs {o:30s} run1 SR {np.mean(list(b.values())):.3f} | "
                  f"{len(sr_all)}-run mean {np.mean(sr_all):.3f} | paired {wa}-{wb}, p={p:.2f}")


if __name__ == "__main__":
    main()
