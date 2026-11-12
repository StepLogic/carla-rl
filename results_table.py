"""The driving results table (user-approved layout, 2026-09-13) from drive_plan_c.py JSONL.

Columns: Map, Nodes, Success rate, Short, Long, Off-graph goals, Successes that followed a multi-hop
plan, Collisions. "Followed a multi-hop plan" = success with plan_hops > 0, at least one
re-localisation, and no explore steps (excludes 0-hop plans, passing the goal before the first node,
and explore-fallback successes). Writes <out>.md, <out>.csv and <out>.png; rows sorted by success rate.
"""
import argparse, csv, json

NAMES = {"uniform6_trace": "uniform[6 m]", "uniform2_trace": "uniform[2 m]",
         "entropy_smooth9_high": "entropy smooth9 high", "entropy_nosmooth_high": "entropy nosmooth high",
         "entropy_nosmooth_high_thin4": "entropy nosmooth high, thinned 4 m", "hybrid_uniform6_entropy_thin4": "uniform[6 m] + thinned entropy high", "nomad_uniform2_trace": "NoMaD on uniform[2 m] trace map", "nomad_uniform2_trace_v6ncap_20kmh": "NoMaD, v6n step caps, 20 km/h", "junction": "junction",
         "entropy_smooth9_low": "entropy smooth9 low", "entropy_nosmooth_low": "entropy nosmooth low"}
SEQUENCE_OF = {"nomad_uniform2_trace": "uniform2_trace", "nomad_uniform2_trace_v6ncap_20kmh": "uniform2_trace"}
NAMES.update({"raw_trace": "dense (raw trace)", "epistemic_high": "epistemic uncertainty high",
              "aleatoric_high": "aleatoric uncertainty high"})
# user 2026-09-14: SGM ablation maps, named sgm_g<gate>_e<edge cutoff>_c<consistency> by build_sgm_map.py
import glob as _glob, os as _os, re as _re
for _p in _glob.glob("results/maps/sequence/sgm_*_sequence.json"):
    _m = _os.path.basename(_p)[:-len("_sequence.json")]
    _g = _re.match(r"sgm_g([\d.]+)_e([\d.]+)_c([\d.]+)$", _m)
    if _g:
        NAMES[_m] = f"SGM gate {_g[1]} m, edge {_g[2]}, consistency {_g[3]}"
HDR = ["Map", "Nodes", "Success rate", "Short", "Long", "Off-graph goals",
       "Successes that followed a multi-hop plan", "Collisions"]


def rows_from(jsonl, maps_dir="results/maps/sequence"):
    files = [jsonl] if isinstance(jsonl, str) else list(jsonl)
    R = [json.loads(l) for f in files for l in open(f) if l.strip()]
    out = []
    for m in dict.fromkeys(r["map"] for r in R):
        M = [r for r in R if r["map"] == m]
        S = [r for r in M if r["success"]]
        sr = lambda L: sum(r["success"] for r in L) / len(L) if L else float("nan")
        off = [r for r in M if r["goal_kind"] == "off_graph"]
        plan = sum(1 for r in S if r["plan_hops"] and r["relocalisations"] > 0 and r["explore_steps"] == 0)
        seq = SEQUENCE_OF.get(m, m)                     # NoMaD runs on an existing sequence map
        nodes = len(json.load(open(f"{maps_dir}/{seq}_sequence.json"))["nodes"])
        out.append({"key": m, "n": len(M), "sr": sr(M),
                    "row": [NAMES.get(m, m), nodes, f"{sr(M):.2f}", f"{sr([r for r in M if r['set'] == 'short']):.2f}",
                            f"{sr([r for r in M if r['set'] == 'long']):.2f}",
                            f"{sum(r['success'] for r in off)} of {len(off)}", f"{plan} of {len(S)}",
                            sum(r["end"] == "collision" for r in M)]})
    return sorted(out, key=lambda x: -x["sr"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", nargs="+", default=["results/drive_plan_c/sequence_undirected.jsonl"],
                    help="one or more drive JSONL files (e.g. add results/nomad/nomad_uniform2_trace.jsonl)")
    ap.add_argument("--out", default="results/drive_plan_c/results_table")
    a = ap.parse_args()
    rows = rows_from(a.jsonl)
    for x in rows:
        assert x["n"] == 40, f"{x['key']} has {x['n']} episodes, expected 40"
    with open(a.out + ".csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(HDR); w.writerows(x["row"] for x in rows)
    with open(a.out + ".md", "w") as f:
        f.write("# Driving results table — Town02, two-way sequence maps, option C, 40 episodes per map\n\n")
        f.write("| " + " | ".join(HDR) + " |\n|" + "---|" * len(HDR) + "\n")
        for x in rows:
            f.write("| " + " | ".join(map(str, x["row"])) + " |\n")
        f.write("\nSuccess = within 4 m of the true goal. Short = 20 episodes, long = 20 (incl. 5 off-graph goals). "
                "\"Followed a multi-hop plan\" excludes 0-hop plans, passing the goal before the first node, and "
                "explore-fallback successes.\n")
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    hdr = ["Map", "Nodes", "Success\nrate", "Short", "Long", "Off-graph\ngoals", "Successes that\nfollowed a\nmulti-hop plan",
           "Collisions"]
    fig, ax = plt.subplots(figsize=(13.5, 0.55 * len(rows) + 1.3)); ax.axis("off")
    t = ax.table(cellText=[[str(c) for c in x["row"]] for x in rows], colLabels=hdr, loc="center", cellLoc="center",
                 colWidths=[0.27, 0.07, 0.08, 0.07, 0.07, 0.10, 0.15, 0.09])
    t.auto_set_font_size(False); t.set_fontsize(10); t.scale(1, 1.7)
    for (i, j), c in t.get_celld().items():
        if i == 0:
            c.set_text_props(weight="bold"); c.set_facecolor("#e8e8e8"); c.set_height(c.get_height() * 1.9)
    fig.savefig(a.out + ".png", dpi=160, bbox_inches="tight")
    for x in rows:
        print(x["row"])
    print(f"[table] {a.out}.md/.csv/.png")


if __name__ == "__main__":
    main()
