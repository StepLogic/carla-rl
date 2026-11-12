"""Pooled DINOv2-vs-shipped test across both towns (user 2026-09-20).

Every map is driven twice on the SAME 40 episodes -- once with the shipped MobileNet localiser, once with
DINOv2 -- so each episode is a matched pair and the pooled statistic is a sign test over all of them.
Reported per town, then pooled, with the dense maps split out: they are saturated (localisation already
~0 m), so including them only dilutes an effect that can only exist where matching was failing.
Run from the icra-2027 folder.
"""
import argparse, collections, glob, json
from math import comb
import numpy as np

TOWNS = {
    "Town02": dict(ship=["results/drive_plan_c/*.jsonl", "results/drive_plan_c/repeat/*.jsonl",
                         "results/drive_plan_c/equal_points/*.jsonl"],
                   dino=["results/drive_plan_c/localiser/dinov2_bank_*.jsonl"],
                   dense=["uniform2_trace", "sgm_g2_e55_c55", "hybrid_sgm_g2_entropy_thin4"]),
    "Town07": dict(ship=["results/town07/*.jsonl", "results/town07/repeat/*.jsonl",
                         "results/town07/equal_points/*.jsonl", "results/town07/localiser/mobilenet_trace_*.jsonl"],
                   dino=["results/town07/localiser/dinov2_bank_*.jsonl"],
                   dense=["uniform2_trace", "sgm_g2_e55_c55", "hybrid_sgm_g2_entropy_thin4"]),
}


def load(globs, want):
    per = collections.defaultdict(dict)
    for f in sorted({f for g in globs for f in glob.glob(g)}):
        if any(s in f for s in ("smoke", "debug")):
            continue
        by = collections.defaultdict(dict)
        for line in open(f):
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("localiser", "mobilenet_trace").split(":")[0] != want:
                continue
            by[r["map"]][(r["set"], r["index"])] = r
        for m, d in by.items():
            if len(d) == 40 and m not in per:
                per[m] = d
    return per


def sign_p(wa, wb):
    n = wa + wb
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(wa, wb) + 1)) / 2 ** n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="tables/localiser_pooled_test.md")
    a = ap.parse_args()
    md = ["# DINOv2 vs the shipped MobileNet localiser: pooled paired test\n",
          "Matched pairs: the same map, the same 40 episodes, the same policy -- only the features used for "
          "node matching differ. Sign test over episodes.\n",
          "| Scope | Maps | Paired episodes | DINOv2 wins | Shipped wins | p | Mean SR change |",
          "|---|---|---|---|---|---|---|"]
    tot = [0, 0, 0, 0]          # wins, losses, episodes, maps
    sparse = [0, 0, 0, 0]
    deltas, sdeltas = [], []
    for town, cfg in TOWNS.items():
        ship, dino = load(cfg["ship"], "mobilenet_trace"), load(cfg["dino"], "dinov2_bank")
        wa = wb = eps = nmap = 0
        d_town = []
        for m in sorted(set(ship) & set(dino)):
            A, B = dino[m], ship[m]
            pairs = [k for k in A if k in B]
            if len(pairs) != 40:
                continue
            nmap += 1; eps += len(pairs)
            w = sum(1 for k in pairs if A[k]["success"] > B[k]["success"])
            l = sum(1 for k in pairs if A[k]["success"] < B[k]["success"])
            wa += w; wb += l
            d = np.mean([A[k]["success"] for k in pairs]) - np.mean([B[k]["success"] for k in pairs])
            d_town.append(d); deltas.append(d)
            if m not in cfg["dense"]:
                sparse[0] += w; sparse[1] += l; sparse[2] += len(pairs); sparse[3] += 1
                sdeltas.append(d)
        if nmap:
            md.append(f"| {town} | {nmap} | {eps} | {wa} | {wb} | {sign_p(wa, wb):.4f} | "
                      f"{np.mean(d_town):+.3f} |")
        tot[0] += wa; tot[1] += wb; tot[2] += eps; tot[3] += nmap
    if tot[3]:
        md.append(f"| **Both towns** | {tot[3]} | {tot[2]} | {tot[0]} | {tot[1]} | "
                  f"{sign_p(tot[0], tot[1]):.4f} | {np.mean(deltas):+.3f} |")
        md.append(f"| Both towns, excluding dense maps | {sparse[3]} | {sparse[2]} | {sparse[0]} | {sparse[1]} | "
                  f"{sign_p(sparse[0], sparse[1]):.4f} | {np.mean(sdeltas):+.3f} |")
    open(a.out, "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
