"""SGM threshold ablation (user 2026-09-14: "ablate SGM and find best threshold") from drive_plan_c.py JSONL.

One row per SGM map (gate m, edge cutoff steps, consistency steps): nodes, SR (all/short/long), off-graph goals,
SPL, collisions, sorted by gate then consistency. BEST = highest SR, ties broken by SPL, then fewer nodes.
The selection uses the same 40 evaluation episodes it reports on (no held-out split), so the best row's SR is an
optimistic estimate; the table says so. Writes <out>.md, <out>.csv and <out>.png (SR vs consistency per gate at
edge 55, calibration-chosen points marked, uniform[6 m] / uniform[2 m] SR as reference lines).
"""
import argparse, csv, json, re
import numpy as np


def load(paths):
    return [json.loads(l) for p in paths for l in open(p) if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", default="results/drive_plan_c/sgm_ablation.jsonl")
    ap.add_argument("--reference", default="results/drive_plan_c/sequence_undirected.jsonl")
    ap.add_argument("--out", default="results/drive_plan_c/sgm_ablation")
    a = ap.parse_args()
    R = load([a.jsonl])
    rows = []
    for m in dict.fromkeys(r["map"] for r in R):
        g = re.match(r"sgm_g([\d.]+)_e([\d.]+)_c([\d.]+)$", m)
        if not g:
            continue
        M = [r for r in R if r["map"] == m]
        sr = lambda L: float(np.mean([r["success"] for r in L])) if L else float("nan")
        nodes = len(json.load(open(f"results/maps/sequence/{m}_sequence.json"))["nodes"])
        off = [r for r in M if r["goal_kind"] == "off_graph"]
        rows.append({"map": m, "gate_m": float(g[1]), "edge": float(g[2]), "consistency": float(g[3]), "nodes": nodes,
                     "episodes": len(M), "sr": sr(M), "short": sr([r for r in M if r["set"] == "short"]),
                     "long": sr([r for r in M if r["set"] == "long"]),
                     "off_graph": f"{sum(r['success'] for r in off)} of {len(off)}",
                     "spl": float(np.mean([r["spl"] for r in M])), "collisions": sum(r["end"] == "collision" for r in M)})
    rows.sort(key=lambda r: (r["gate_m"], r["edge"] != 55.0, r["consistency"]))
    full = [r for r in rows if r["episodes"] == 40]
    best = max(full, key=lambda r: (r["sr"], r["spl"], -r["nodes"])) if full else None

    ref = {}
    for m in ("uniform6_trace", "uniform2_trace"):
        L = [r for r in load([a.reference]) if r["map"] == m]
        ref[m] = float(np.mean([r["success"] for r in L])) if L else None

    hdr = ["Gate (m)", "Edge cutoff", "Consistency", "Nodes", "Episodes", "SR", "Short", "Long", "Off-graph goals",
           "SPL", "Collisions"]
    fmt = lambda r: [f"{r['gate_m']:g}", f"{r['edge']:g}", f"{r['consistency']:g}", str(r["nodes"]), str(r["episodes"]),
                     f"{r['sr']:.2f}", f"{r['short']:.2f}", f"{r['long']:.2f}", r["off_graph"], f"{r['spl']:.2f}",
                     str(r["collisions"])]
    with open(a.out + ".csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["map"] + hdr + ["best"])
        for r in rows:
            w.writerow([r["map"]] + fmt(r) + [int(r is best)])
    md = ["# SGM threshold ablation, Town02 speed10 lane-off trace, option C, 40 episodes\n",
          "| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    md += ["| " + " | ".join(("**" + c + "**") if r is best else c for c in fmt(r)) + " |" for r in rows]
    md += ["", f"Reference (same episodes): uniform[6 m] SR {ref['uniform6_trace']}, uniform[2 m] SR {ref['uniform2_trace']}.",
           "Edge cutoff 55 = 1.5 x the 20-50 m band median critic distance (the uniform maps' rule); edge 45.8 rows are "
           "calibrate_critic.py's chosen setting per gate (smallest edge cutoff with learned directed reach >= 0.99, "
           "then fewest nodes). Driven maps use sequence edges (two-way), like every other table row.",
           (f"Best (bold): {best['map']} -- highest SR, then SPL, then fewer nodes. Chosen on the same 40 evaluation "
            "episodes it is scored on (no held-out split), so its SR is optimistic." if best else "No complete map yet.")]
    open(a.out + ".md", "w").write("\n".join(md) + "\n")
    print("\n".join(md))

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for gate, col in zip((2.0, 4.0, 6.0), ("#1f77b4", "#ff7f0e", "#2ca02c")):
        L = [r for r in rows if r["gate_m"] == gate and r["edge"] == 55.0]
        if L:
            ax.plot([r["consistency"] for r in L], [r["sr"] for r in L], "o-", color=col, label=f"Gate {gate:g} m, edge 55")
            for r in L:
                ax.annotate(str(r["nodes"]), (r["consistency"], r["sr"]), textcoords="offset points", xytext=(3, 4),
                            fontsize=7, color=col)
        C = [r for r in rows if r["gate_m"] == gate and r["edge"] != 55.0]
        if C:
            ax.plot([r["consistency"] for r in C], [r["sr"] for r in C], "*", ms=12, color=col,
                    label=f"Gate {gate:g} m, calibrated (edge {C[0]['edge']:g})")
    for m, ls, lab in (("uniform6_trace", "--", "Uniform[6 m]"), ("uniform2_trace", ":", "Uniform[2 m]")):
        if ref[m] is not None:
            ax.axhline(ref[m], color="grey", ls=ls, lw=1, label=lab)
    ax.set_xscale("log"); ax.set_xlabel("Consistency threshold (steps)"); ax.set_ylabel("Success rate")
    ax.legend(fontsize=7, loc="best", framealpha=0.9)
    fig.savefig(a.out + ".png", dpi=160, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
