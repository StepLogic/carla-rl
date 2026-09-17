"""Failure-mode log for the option C driving runs (drive_plan_c.py JSONL).

Every FAILED episode gets exactly one mode, decided in this order from the episode record:

  first_leg_collision_mislocalised  collision before the first re-localisation, on a multi-hop plan, with the image-
                                    localised start node > 10 m from the true start
  first_leg_collision_localised     the same, start node within 10 m (the policy crashed on a sound first leg)
  zero_hop_collision                start and goal localised to one node, collision driving at the true goal
  mid_plan_collision                collision after >= 1 re-localisation, before the plan was finished
  final_leg_collision               plan finished, collision on the goal-mode leg to the true goal
  explore_collision                 collision in the explore fallback
  explore_cap                       plan finished short of the goal, explore fallback ran out the step cap
  plan_cap                          step cap reached while still following the plan

Flags per failed episode (not exclusive): goal node > 10 m from the true goal, any off-road steps, any lane
crossings, any leg driven against the trace's direction.
Writes <out>_episodes.csv (one row per failed episode), <out>_summary.csv/.md (counts per map), <out>.png.
"""
import argparse, csv, json
import numpy as np

MODES = [
    ("first_leg_collision_mislocalised", "Collision on first leg, start mislocalised", "#d62728"),
    ("first_leg_collision_localised", "Collision on first leg, start localised", "#ff9896"),
    ("zero_hop_collision", "Collision on 0-hop plan", "#9467bd"),
    ("mid_plan_collision", "Collision mid-plan", "#ff7f0e"),
    ("final_leg_collision", "Collision on final leg to goal", "#8c564b"),
    ("explore_collision", "Collision in explore fallback", "#e377c2"),
    ("explore_cap", "Cap in explore fallback", "#7f7f7f"),
    ("plan_cap", "Cap while following plan", "#bcbd22"),
]
LOC_M = 10.0


def mode_of(r):
    if r["end"] == "collision":
        if r["plan_hops"] == 0:
            return "zero_hop_collision"
        if r["explore_steps"] > 0:
            return "explore_collision"
        if r["reached_plan_end"]:
            return "final_leg_collision"
        if r["relocalisations"] == 0:
            return ("first_leg_collision_mislocalised" if r["start_node_err_m"] > LOC_M
                    else "first_leg_collision_localised")
        return "mid_plan_collision"
    return "explore_cap" if r["explore_steps"] > 0 else "plan_cap"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", default="results/drive_plan_c/sequence_undirected.jsonl")
    ap.add_argument("--out", default="results/drive_plan_c/failure_modes")
    a = ap.parse_args()
    from results_table import NAMES
    R = [json.loads(l) for l in open(a.jsonl) if l.strip()]
    maps = list(dict.fromkeys(r["map"] for r in R))
    F = [r for r in R if not r["success"]]

    with open(a.out + "_episodes.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["map", "set", "index", "goal_kind", "mode", "steps", "step_cap", "final_dist_m", "plan_hops", "plan_m",
                    "start_node_err_m", "goal_node_err_m", "relocalisations", "replans", "explore_steps", "off_road_steps",
                    "lane_steps", "reversed_legs_driven", "goal_mislocalised", "route_m"])
        for r in F:
            w.writerow([r["map"], r["set"], r["index"], r["goal_kind"], mode_of(r), r["steps"], r["step_cap"],
                        round(r["final_dist_m"], 1), r["plan_hops"], round(r["plan_m"], 1), round(r["start_node_err_m"], 1),
                        round(r["goal_node_err_m"], 1), r["relocalisations"], r["replans"], r["explore_steps"],
                        r["off_road_steps"], r["lane_steps"], r["reversed_legs_driven"],
                        int(r["goal_node_err_m"] > LOC_M), round(float(r["route_m"]), 1)])

    rows = []
    for m in maps:
        M = [r for r in F if r["map"] == m]
        cnt = {k: sum(mode_of(r) == k for r in M) for k, _, _ in MODES}
        col = [r for r in M if r["end"] == "collision"]
        rows.append({"map": m, "name": NAMES.get(m, m), "failures": len(M), **cnt,
                     "goal_mislocalised": sum(r["goal_node_err_m"] > LOC_M for r in M),
                     "off_road_any": sum(r["off_road_steps"] > 0 for r in M),
                     "reversed_leg_any": sum(r["reversed_legs_driven"] > 0 for r in M),
                     "collision_step_median": (float(np.median([r["steps"] for r in col])) if col else None)})
    rows.sort(key=lambda x: x["failures"])
    keys = ["name", "failures"] + [k for k, _, _ in MODES] + ["goal_mislocalised", "off_road_any", "reversed_leg_any",
                                                              "collision_step_median"]
    with open(a.out + "_summary.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(keys); w.writerows([[x[k] for k in keys] for x in rows])

    hdr = ["Map", "Failures (of 40)"] + [lab for _, lab, _ in MODES] + ["Goal node > 10 m off", "Any off-road",
                                                                        "Any reversed leg", "Collision step (median)"]
    with open(a.out + "_summary.md", "w") as fh:
        fh.write("# Failure modes per map — option C driving, 40 episodes per map\n\n")
        fh.write("| " + " | ".join(hdr) + " |\n|" + "---|" * len(hdr) + "\n")
        for x in rows:
            vals = [x["name"], x["failures"]] + [x[k] for k, _, _ in MODES] + [
                x["goal_mislocalised"], x["off_road_any"], x["reversed_leg_any"],
                "n/a" if x["collision_step_median"] is None else f"{x['collision_step_median']:.0f}"]
            fh.write("| " + " | ".join(map(str, vals)) + " |\n")
        fh.write("\nModes are exclusive and decided in the order listed in `failure_modes.py`; \"mislocalised\" = the "
                 "image-localised start node is more than 10 m from the true start. The last four columns are flags "
                 "counted over failed episodes.\n")

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = np.arange(len(rows)); b = np.zeros(len(rows))
    fig, ax = plt.subplots(figsize=(12, 4.8))
    for k, lab, colr in MODES:
        v = np.array([r[k] for r in rows], float)
        ax.bar(x, v, 0.62, bottom=b, color=colr, label=lab); b += v
    ax.set_xticks(x)
    ax.set_xticklabels([r["name"].replace(" + ", " +\n").replace(", thinned", ",\nthinned").replace("entropy ", "entropy\n", 1)
                        for r in rows], fontsize=8)
    ax.set_ylabel("Failed episodes (of 40)"); ax.set_ylim(0, 40)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=4, framealpha=0.95)
    fig.savefig(a.out + ".png", dpi=150, bbox_inches="tight"); plt.close(fig)
    for r in rows:
        print(r["name"], r["failures"], {k: r[k] for k, _, _ in MODES if r[k]},
              "goal_misloc", r["goal_mislocalised"], "offroad", r["off_road_any"], "rev", r["reversed_leg_any"])
    print(f"[failure modes] {a.out}_episodes.csv, _summary.csv, _summary.md, .png")


if __name__ == "__main__":
    main()
