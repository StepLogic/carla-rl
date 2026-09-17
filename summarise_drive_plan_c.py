"""Success-rate comparison across maps for drive_plan_c.py JSONL output.

Writes <out>_summary.csv / .json and <out>_success.png (success per map, short vs
long, off-graph marked) -- short legends, no title.
"""
import argparse, csv, json, os
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", default="results/drive_plan_c/sequence_undirected.jsonl")
    a = ap.parse_args()
    recs = [json.loads(l) for l in open(a.jsonl) if l.strip()]
    base = a.jsonl[:-len(".jsonl")]
    maps = list(dict.fromkeys(r["map"] for r in recs))
    rows = []
    for m in maps:
        R = [r for r in recs if r["map"] == m]
        sub = lambda f: [r for r in R if f(r)]
        rate = lambda L: (float(np.mean([r["success"] for r in L])) if L else None)
        ends = {}
        for r in R:
            ends[r["end"]] = ends.get(r["end"], 0) + 1
        rows.append({
            "map": m, "nodes": len(json.load(open(f"results/maps/sequence/{m}_sequence.json"))["nodes"]),
            "episodes": len(R), "success": rate(R),
            "success_short": rate(sub(lambda r: r["set"] == "short")),
            "success_long": rate(sub(lambda r: r["set"] == "long")),
            "success_off_graph": rate(sub(lambda r: r["goal_kind"] == "off_graph")),
            "spl": float(np.mean([r["spl"] for r in R])),
            "success_before_explore": float(np.mean([r["success"] and not r["explore_steps"] for r in R])),
            "end_success": ends.get("success", 0), "end_collision": ends.get("collision", 0), "end_cap": ends.get("cap", 0),
            "reached_plan_end": float(np.mean([r["reached_plan_end"] for r in R])),
            "final_dist_median_m": float(np.median([r["final_dist_m"] for r in R])),
            "relocs_median": float(np.median([r["relocalisations"] for r in R])),
            "reloc_changed_frac": (sum(r["reloc_changed"] for r in R) / max(sum(r["relocalisations"] for r in R), 1)),
            "replans_median": float(np.median([r["replans"] for r in R])),
            "leg_timeouts_median": float(np.median([r["leg_timeouts"] for r in R])),
            "reversed_legs_driven_total": int(sum(r["reversed_legs_driven"] for r in R)),
            "lane_step_frac": float(sum(r["lane_steps"] for r in R) / max(sum(r["steps"] for r in R), 1)),
            "off_road_step_frac": float(sum(r["off_road_steps"] for r in R) / max(sum(r["steps"] for r in R), 1)),
            "start_node_err_median_m": float(np.median([r["start_node_err_m"] for r in R])),
            "goal_node_err_median_m": float(np.median([r["goal_node_err_m"] for r in R])),
            "steps_total": int(sum(r["steps"] for r in R)),
        })
    with open(base + "_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    json.dump(rows, open(base + "_summary.json", "w"), indent=1)
    for r in rows:
        f = lambda v: "  n/a" if v is None else f"{v:5.2f}"
        print(f"{r['map']:22} n={r['episodes']:2} SR {f(r['success'])} (short {f(r['success_short'])}, long {f(r['success_long'])}, "
              f"off-graph {f(r['success_off_graph'])}) SPL {r['spl']:.2f} | ends S/C/cap {r['end_success']}/{r['end_collision']}/"
              f"{r['end_cap']} | final median {r['final_dist_median_m']:.0f} m | relocs changed {r['reloc_changed_frac']:.2f}")

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from results_table import NAMES
    x = np.arange(len(rows)); w = 0.27
    fig, ax = plt.subplots(figsize=(13, 4.6))
    for k, (key, lab, col) in enumerate([("success", "All", "#2ca02c"), ("success_short", "Short", "#1f77b4"),
                                         ("success_long", "Long", "#ff7f0e")]):
        ax.bar(x + (k - 1) * w, [r[key] or 0 for r in rows], w, color=col, label=lab)
    wrap = lambda t: t.replace(" + ", " +\n").replace(", thinned", ",\nthinned").replace("entropy ", "entropy\n", 1)
    ax.set_xticks(x); ax.set_xticklabels([wrap(NAMES.get(r["map"], r["map"])) for r in rows], fontsize=8)
    ax.set_ylabel("Success rate"); ax.set_ylim(0, 1)
    ax.legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=3, framealpha=0.95)
    fig.savefig(base + "_success.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"[summary] {base}_summary.csv, _summary.json, _success.png")


if __name__ == "__main__":
    main()
