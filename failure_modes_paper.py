"""Failure-mode bar plot for the paper's map table (user 2026-09-14: "show only" the tab:map-results maps, with the
table's labels). Same modes, colours and classification as failure_modes.py; maps in table order."""
import argparse, json
import numpy as np

from failure_modes import LOC_M, MODES, mode_of

MAPS = [("uniform6_trace", "Spacing[6 m]"), ("uniform2_trace", "Spacing[2 m]"), ("raw_trace", "Dense"),
        ("hybrid_uniform6_entropy_thin4", "Spacing[6 m] +\nClustered + Entropy"),
        ("entropy_nosmooth_high_thin4", "Entropy +\nClustered"), ("entropy_nosmooth_high", "Entropy"),
        ("junction", "Junction")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", nargs="+", default=["results/drive_plan_c/sequence_undirected.jsonl",
                                                   "results/drive_plan_c/raw_trace.jsonl"])
    ap.add_argument("--out", default="results/drive_plan_c/failure_modes_paper.png")
    ap.add_argument("--grouped", action="store_true", help="two bars segments only: all collisions, all step caps")
    ap.add_argument("--causal", action="store_true",
                    help="one cause per failure in pipeline order: start localisation, goal localisation, collision, timeout")
    ap.add_argument("--timeout-first", action="store_true",
                    help="with --causal: count every step-capped episode as Timeout before the localisation checks")
    ap.add_argument("--split", action="store_true",
                    help="three segments: planning failure (start/goal node > 10 m off, 0-hop, no plan), collision, step cap")
    a = ap.parse_args()
    R = [json.loads(l) for f in a.jsonl for l in open(f) if l.strip()]
    counts = []
    for key, label in MAPS:
        F = [r for r in R if r["map"] == key and not r["success"]]
        n = sum(r["map"] == key for r in R)
        assert n == 40, f"{key}: {n} episodes"
        counts.append({k: sum(mode_of(r) == k for r in F) for k, _, _ in MODES})
        print(f"{label.replace(chr(10), ' '):>32}: {len(F)} failed", {k: v for k, v in counts[-1].items() if v})

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = np.arange(len(MAPS)); b = np.zeros(len(MAPS))
    used = [m for m in MODES if any(c[m[0]] for c in counts)]
    if a.causal:    # user 2026-09-14: "group failures the best way possible"
        # One cause per failed episode, in pipeline order -- the first stage that went wrong:
        #   start localisation: image-localised start node > 10 m from the true start (the plan starts elsewhere)
        #   goal localisation:  start fine, goal node > 10 m from the true goal (the plan ends elsewhere)
        #   collision:          both localised, the car crashed while driving
        #   timeout:            both localised, the step cap ran out
        for (key, _), c in zip(MAPS, counts):
            F = [r for r in R if r["map"] == key and not r["success"]]
            # user 2026-09-14 "show timeout in plot": every episode that ran out the step cap is a Timeout, whatever
            # its localisation; the remaining (crashed) episodes are attributed in pipeline order.
            to = [r for r in F if r["end"] == "cap"] if a.timeout_first else []
            st = [r for r in F if r not in to and r["start_node_err_m"] > LOC_M]
            go = [r for r in F if r not in to and r not in st and r["goal_node_err_m"] > LOC_M]
            rest = [r for r in F if r not in to and r not in st and r not in go]
            c.update(start_loc=len(st), goal_loc=len(go), collision=sum(r["end"] == "collision" for r in rest),
                     cap=len(to) + sum(r["end"] == "cap" for r in rest))
        used = [("start_loc", "Start localisation (> 10 m)", "#1f77b4"), ("goal_loc", "Goal localisation (> 10 m)", "#9ecae1"),
                ("collision", "Collision", "#d62728"), ("cap", "Timeout (step cap)", "#7f7f7f")]
        used = [u for u in used if any(c[u[0]] for c in counts)]
        for (_, label), c in zip(MAPS, counts):
            print(f"{label.replace(chr(10), ' '):>32}: start {c['start_loc']}, goal {c['goal_loc']}, "
                  f"collision {c['collision']}, timeout {c['cap']}")
    elif a.split:   # user 2026-09-14: "split into planning failures, collision and step caps (timeouts)"
        # planning failure (user's choice): the plan could not lead to the goal -- localised start or goal node
        # > 10 m off, a 0-hop plan (start and goal on one node), or no plan -- whatever ended the episode.
        for (key, _), c in zip(MAPS, counts):
            F = [r for r in R if r["map"] == key and not r["success"]]
            plan = [r for r in F if (not r["plan_found"]) or r["plan_hops"] == 0
                    or r["start_node_err_m"] > LOC_M or r["goal_node_err_m"] > LOC_M]
            rest = [r for r in F if r not in plan]
            c.update(planning=len(plan), collision=sum(r["end"] == "collision" for r in rest),
                     cap=sum(r["end"] == "cap" for r in rest))
        used = [("planning", "Planning failure", "#1f77b4"), ("collision", "Collision", "#d62728"),
                ("cap", "Step cap (timeout)", "#7f7f7f")]
        for (_, label), c in zip(MAPS, counts):
            print(f"{label.replace(chr(10), ' '):>32}: planning {c['planning']}, collision {c['collision']}, cap {c['cap']}")
    elif a.grouped:  # user 2026-09-14: "group all caps - summarize all collisions into collisions"
        for c in counts:
            c["collision"] = sum(v for k, v in c.items() if "collision" in k)
            c["cap"] = sum(v for k, v in c.items() if k.endswith("_cap"))
        used = [("collision", "Collision", "#d62728"), ("cap", "Step cap", "#7f7f7f")]
        for (_, label), c in zip(MAPS, counts):
            print(f"{label.replace(chr(10), ' '):>32}: collision {c['collision']}, cap {c['cap']}")
    fig, ax = plt.subplots(figsize=(10, 4.6))
    for k, lab, colr in used:
        v = np.array([c[k] for c in counts], float)
        ax.bar(x, v, 0.62, bottom=b, color=colr, label=lab); b += v
    ax.set_xticks(x); ax.set_xticklabels([lab for _, lab in MAPS], fontsize=9)
    ax.set_ylabel("Failed episodes (of 40)"); ax.set_ylim(0, 40)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=4, framealpha=0.95)
    fig.savefig(a.out, dpi=200, bbox_inches="tight"); plt.close(fig)
    print("[wrote]", a.out)


if __name__ == "__main__":
    main()
