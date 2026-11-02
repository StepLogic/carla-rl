#!/usr/bin/env python3
"""Map arms vs mapless on the v30 long-distance benchmark.

For every placement found under checkpoints/bench_map_<pl>_<town>_<band>/, and
the mapless true-goal arm under checkpoints/bench_long_true_<town>_<band>/:

  * ALL       -- all 1,200 episodes, unroutable = failure (user's choice)
  * ROUTABLE  -- the episodes that placement could route
  * COMMON    -- the episodes EVERY placement could route (equal footing)

SR at 1 m (the run radius) and 2 m, each paired against mapless on the same
episodes (exact McNemar, Holm across placements within a block).

    bench_map_report.py [--root checkpoints] [--placements entropy uniform ...]
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from vae_nav.carla_benchmark import holm, mcnemar_p  # noqa: E402

TOWNS = ("Town02", "Town03", "Town04", "Town10HD")
BANDS = ("easy", "medium", "hard")


def load(pattern):
    rows = {}
    for f in glob.glob(pattern):
        for r in json.load(open(f))["episodes_detail"]:
            rows[(r["town"], r["seed"])] = r
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="checkpoints")
    ap.add_argument("--placements", nargs="+",
                    default=["entropy", "uniform", "random", "sgm", "sgm_matched"])
    a = ap.parse_args()
    base = load(os.path.join(a.root, "bench_long_true_*_*", "eval_step_300000", "eval.json"))
    arms = {}
    for pl in a.placements:
        r = load(os.path.join(a.root, f"bench_map_{pl}_*_*", f"eval_step_300000_map-{pl}", "eval.json"))
        if r:
            arms[pl] = r
    keys = sorted(set(base).intersection(*[set(r) for r in arms.values()])) if arms else []
    print(f"mapless {len(base)} episodes; arms {[(p, len(r)) for p, r in arms.items()]}; "
          f"{len(keys)} episodes present in every arm")
    routable = {p: {k for k in keys if r[k]["end"] != "unroutable"} for p, r in arms.items()}
    common = set(keys).intersection(*routable.values()) if routable else set()

    def block(name, subset_of):
        print(f"\n### {name}")
        print("| Placement | n | routable | SR@1m map | SR@1m mapless | map-only / mapless-only | p (Holm) | SR@2m map | SR@2m mapless |")
        print("|---|---|---|---|---|---|---|---|---|")
        lines, ps = [], []
        for p, r in arms.items():
            ks = sorted(subset_of(p))
            n = len(ks)
            if not n:
                continue
            m1 = sum(r[k]["reached"] for k in ks)
            b1 = sum(base[k]["reached"] for k in ks)
            om = sum(r[k]["reached"] and not base[k]["reached"] for k in ks)
            ob = sum(base[k]["reached"] and not r[k]["reached"] for k in ks)
            m2 = sum(r[k]["reached_2m"] for k in ks)
            b2 = sum(base[k]["reached_2m"] for k in ks)
            nr = sum(k in routable[p] for k in ks)
            ps.append(mcnemar_p(om, ob))
            lines.append((p, n, nr, m1, b1, om, ob, m2, b2))
        for (p, n, nr, m1, b1, om, ob, m2, b2), q in zip(lines, holm(ps) if ps else []):
            qs = "<0.0001" if q < 1e-4 else f"{q:.4f}"
            print(f"| {p} | {n} | {nr / n:.3f} | {m1 / n:.3f} | {b1 / n:.3f} | {om} / {ob} | {qs} "
                  f"| {m2 / n:.3f} | {b2 / n:.3f} |")

    block("ALL episodes (unroutable = failure)", lambda p: keys)
    block("ROUTABLE by that placement", lambda p: routable[p])
    block(f"COMMON: routable by every placement (n={len(common)})", lambda p: common)
    for band in BANDS:
        block(f"COMMON, {band}", lambda p, b=band: {k for k in common if base[k].get("band") == b})


if __name__ == "__main__":
    main()
