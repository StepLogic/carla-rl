"""SGM success vs node count across consistency thresholds, with the Pareto
front (fewest nodes for a given success; a point is dominated if another has at
least as many successes with no more nodes, and is strictly better in one).

Reads <root>/c<consistency>/v6n__sgm__20-50m__s<seed>.json, one SGM row per
threshold, all on the same trace, edge cutoff and 60 episodes.
"""
import argparse, glob, json, os, re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--town", default="Town02")
    a = ap.parse_args()

    pts = []
    for f in glob.glob(os.path.join(a.root, "c*", f"v6n__sgm__20-50m__s{a.seed}.json")):
        d = json.load(open(f))
        c = float(re.search(r"/c([0-9.]+)/", f).group(1))
        k = int(round((d["success"] or 0.0) * d["n_routed"]))
        pts.append({"consistency": c, "nodes": int(d["n_nodes"]), "successes": k,
                    "routed": int(d["n_routed"]), "episodes": int(d["episodes"]),
                    "sr_routed": d["success"], "spl": d.get("spl"),
                    "edge_cutoff": d.get("sgm_edge_cutoff"), "reach": d.get("directed_reachability"),
                    "leg_timeouts": d.get("leg_timeouts")})
    if not pts:
        raise SystemExit(f"no SGM rows under {a.root}")
    pts.sort(key=lambda p: p["nodes"])
    for p in pts:
        p["pareto"] = not any((q["successes"] >= p["successes"] and q["nodes"] <= p["nodes"]) and
                              (q["successes"] > p["successes"] or q["nodes"] < p["nodes"])
                              for q in pts if q is not p)

    print(f"{'cons':>6} {'nodes':>6} {'succ/routed':>12} {'SR':>6} {'reach':>6} {'timeouts':>8}  pareto")
    for p in pts:
        print(f"{p['consistency']:>6g} {p['nodes']:>6} {p['successes']:>5}/{p['routed']:<6} "
              f"{p['sr_routed']:>6.3f} {p['reach']:>6.3f} {p['leg_timeouts']:>8}  {'YES' if p['pareto'] else ''}")

    fig, ax = plt.subplots(figsize=(7.5, 5))
    x = np.array([p["nodes"] for p in pts]); y = np.array([p["successes"] for p in pts])
    ax.plot(x, y, "o", color="#7f7f7f", label="SGM setting")
    front = [p for p in pts if p["pareto"]]
    ax.plot([p["nodes"] for p in front], [p["successes"] for p in front], "-o", color="#d62728",
            lw=2, ms=8, label="Pareto-optimal")
    for p in pts:
        ax.annotate(f"c={p['consistency']:g}", (p["nodes"], p["successes"]), textcoords="offset points",
                    xytext=(5, 6), fontsize=8)
    edge = pts[0]["edge_cutoff"]
    ax.set_xlabel("SGM node count")
    ax.set_ylabel(f"Successes (of {pts[0]['episodes']} episodes)")
    ax.set_title(f"SGM Pareto front, {a.town} 10k continued trace s{a.seed}\n"
                 f"Edge cutoff {edge:g} steps; consistency threshold sets the node count")
    ax.grid(alpha=0.3); ax.legend()
    out = os.path.join(a.root, f"sgm_pareto_{a.town.lower()}_s{a.seed}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    json.dump(pts, open(out.replace(".png", ".json"), "w"), indent=1)
    print(f"[Render] {out} (+ .json)")


if __name__ == "__main__":
    main()
