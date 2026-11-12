"""SR ablation over the placement grid: uncertainty signal x filler x budget (user 2026-09-20).

Rows are the six grid arms (policy entropy / epistemic / aleatoric, each + Uniform and + SGM) against the
baselines they have to beat (Spacing, SGM, and the two unsizeable references Dense and Entropy-only).
Everything here is driven under ONE localiser -- DINOv2 -- so the grid is internally comparable; the
shipped-localiser column is carried alongside where it exists, never mixed into the same comparison.

Seeds are count-matched at 30 per signal. Paired sign tests are against the SAME budget's plain filler
(Spacing for the uniform arms, SGM for the SGM arms): the question is whether seeding helps AT ALL, and
the honest control for "+ Uniform" is uniform alone, not the other signal.
Run from the icra-2027 folder.
"""
import argparse, collections, glob, json
from math import comb
import numpy as np

SIGNALS = [("entropy", "Policy entropy"), ("epistemic", "Epistemic"), ("aleatoric", "Aleatoric")]
# budget -> (Spacing map, SGM map, per-signal uniform gate, per-signal SGM gate)
BUDGETS = [("~60", "uniform15_trace", "sgm_g24_e55_c55", "15", {"entropy": "44", "epistemic": "44", "aleatoric": "44"}),
           ("~115", "uniform8_trace", "sgm_g10_e55_c55", "8", {"entropy": "14", "epistemic": "13", "aleatoric": "13"}),
           ("~150", "uniform6_trace", "sgm_g7_e55_c55", "6", {"entropy": "9", "epistemic": "9", "aleatoric": "9"}),
           ("~550", "uniform2_trace", "sgm_g2_e55_c55", "2", {"entropy": "2", "epistemic": "2", "aleatoric": "2"})]
REFS = [("raw_trace", "Dense"), ("entropy_nosmooth_high_thin4", "Entropy only (30)")]
DINO = ["results/drive_plan_c/localiser/dinov2_bank_*.jsonl", "results/drive_plan_c/signal_grid/*.jsonl"]
SHIPPED = ["results/drive_plan_c/*.jsonl", "results/drive_plan_c/repeat/*.jsonl",
           "results/drive_plan_c/equal_points/*.jsonl"]
SEQ = "results/maps/sequence"


def load(globs, want):
    per = collections.defaultdict(lambda: collections.defaultdict(dict))
    for f in sorted({f for g in globs for f in glob.glob(g)}):
        if any(s in f for s in ("smoke", "debug")):
            continue
        for line in open(f):
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("localiser", "mobilenet_trace").split(":")[0] != want:
                continue
            per[r["map"]][f][(r["set"], r["index"])] = r
    out = {}
    for m, files in per.items():
        eps = [d for d in files.values() if len(d) == 40]
        if eps:
            out[m] = eps
    return out


def sr(runs, m):
    if m not in runs:
        return None
    return float(np.mean([np.mean([r["success"] for r in d.values()]) for d in runs[m]])), len(runs[m])


def sign_test(A, B):
    wa = sum(1 for k in A if k in B and A[k]["success"] > B[k]["success"])
    wb = sum(1 for k in A if k in B and A[k]["success"] < B[k]["success"])
    n = wa + wb
    if n == 0:
        return wa, wb, 1.0
    k = min(wa, wb)
    return wa, wb, min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="tables/signal_grid_sr_town02.md")
    a = ap.parse_args()
    dino, ship = load(DINO, "dinov2_bank"), load(SHIPPED, "mobilenet_trace")
    nodes = lambda m: len(json.load(open(f"{SEQ}/{m}_sequence.json"))["nodes"])
    fmt = lambda v: "—" if v is None else (f"{v[0]:.3f}" + ("" if v[1] == 1 else f" ({v[1]} runs)"))

    md = ["# SR ablation: uncertainty signal x filler, Town02, DINOv2 localiser\n",
          "Seeds count-matched at 30 per signal (the three sets share 0 of 30 trace samples). Each arm is "
          "compared against the SAME budget's unseeded filler -- Spacing for the + Uniform arms, SGM for the "
          "+ SGM arms -- so the test asks whether seeding helps at all.\n",
          "`shipped` is the MobileNet-localiser number where one exists, for reference only; it is never "
          "compared against a DINOv2 row.\n",
          "| Budget | Arm | Nodes | SR (DINOv2) | vs unseeded filler | SR (shipped) |",
          "|---|---|---|---|---|---|"]
    for label, umap, smap, ug, sg in BUDGETS:
        for base, nm in ((umap, "Spacing"), (smap, "SGM")):
            md.append(f"| {label} | {nm} (unseeded) | {nodes(base)} | {fmt(sr(dino, base))} | — | "
                      f"{fmt(sr(ship, base))} |")
        for key, name in SIGNALS:
            for filler, base, mp in (("Uniform", umap, f"hybrid_uniform{ug}_{key}_thin4"),
                                     ("SGM", smap, f"hybrid_sgm_g{sg[key]}_{key}_thin4")):
                mp = mp.replace("_entropy_thin4", "_entropy_thin4")
                if mp not in dino:
                    md.append(f"| {label} | {name} + {filler} | | not driven | | |"); continue
                cmp = ""
                if base in dino:
                    wa, wb, p = sign_test(dino[mp][0], dino[base][0])
                    cmp = f"{wa}-{wb}, p={p:.2f}"
                md.append(f"| {label} | {name} + {filler} | {nodes(mp)} | {fmt(sr(dino, mp))} | {cmp} | "
                          f"{fmt(sr(ship, mp))} |")
    md.append("")
    for m, nm in REFS:
        md.append(f"| fixed | {nm} | {nodes(m)} | {fmt(sr(dino, m))} | — | {fmt(sr(ship, m))} |")
    open(a.out, "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
