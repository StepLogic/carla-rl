"""Paired comparison on the same 40 episodes: NoMaD vs the v6n option C drive on uniform[2 m] trace nodes.

Both runs use the uniform[2 m] trace map, the same episodes, success radius (4 m), step-cap formula and SPL
(route / max(driven, route) on success). They differ in the driver and localiser: NoMaD (distance head,
diffusion waypoints, SetPointAgent PID, baseline camera and vehicle) vs v6n (MobileNet latent localisation,
TQC policy). Writes <out>.md and <out>.csv; prints the table.
"""
import argparse, csv, json
import numpy as np


def load(path, map_name):
    return {(r["set"], r["index"]): r for r in (json.loads(l) for l in open(path) if l.strip()) if r["map"] == map_name}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nomad", default="results/nomad/nomad_uniform2_trace.jsonl")
    ap.add_argument("--v6n", default="results/drive_plan_c/sequence_undirected.jsonl")
    ap.add_argument("--out", default="results/nomad/nomad_vs_v6n_uniform2")
    ap.add_argument("--nomad-map", default="nomad_uniform2_trace")
    ap.add_argument("--v6n-map", default="uniform2_trace", help="map key of the v6n rows to pair with (raw_trace, ...)")
    a = ap.parse_args()
    A = load(a.nomad, a.nomad_map); B = load(a.v6n, a.v6n_map)
    keys = sorted(set(A) & set(B), key=lambda k: (k[0] != "short", k[1]))
    if len(keys) != 40:
        print(f"[compare] WARNING: {len(keys)} paired episodes (NoMaD {len(A)}, v6n {len(B)})")

    def stats(R, sel):
        L = [R[k] for k in keys if sel(k, R[k])]
        if not L:
            return None
        loc = lambda f: float(np.median([r[f] for r in L]))
        return {"n": len(L), "sr": np.mean([r["success"] for r in L]), "spl": np.mean([r["spl"] for r in L]),
                "collisions": sum(r["end"] == "collision" for r in L), "caps": sum(r["end"] == "cap" for r in L),
                "start_err": loc("start_node_err_m"), "goal_err": loc("goal_node_err_m"),
                "starts_10m": sum(r["start_node_err_m"] <= 10 for r in L), "final": loc("final_dist_m")}

    groups = [("All 40", lambda k, r: True), ("Short (20)", lambda k, r: k[0] == "short"),
              ("Long (20)", lambda k, r: k[0] == "long"), ("Off-graph goals (5)", lambda k, r: r["goal_kind"] == "off_graph")]
    rows = []
    for g, sel in groups:
        sa, sb = stats(A, sel), stats(B, sel)
        rows.append([g, f"{sa['sr']:.2f}", f"{sb['sr']:.2f}", f"{sa['spl']:.2f}", f"{sb['spl']:.2f}"])
    both = sum(A[k]["success"] and B[k]["success"] for k in keys)
    only_a = sum(A[k]["success"] and not B[k]["success"] for k in keys)
    only_b = sum(B[k]["success"] and not A[k]["success"] for k in keys)
    sa, sb = stats(A, groups[0][1]), stats(B, groups[0][1])
    detail = [
        ["Collisions", sa["collisions"], sb["collisions"]], ["Step caps", sa["caps"], sb["caps"]],
        ["Start node error, median (m)", f"{sa['start_err']:.1f}", f"{sb['start_err']:.1f}"],
        ["Goal node error, median (m)", f"{sa['goal_err']:.1f}", f"{sb['goal_err']:.1f}"],
        ["Starts localised within 10 m", sa["starts_10m"], sb["starts_10m"]],
        ["Final distance to goal, median (m)", f"{sa['final']:.1f}", f"{sb['final']:.1f}"],
    ]
    with open(a.out + ".csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["group", "nomad_sr", "v6n_sr", "nomad_spl", "v6n_spl"]); w.writerows(rows)
        w.writerow([]); w.writerow(["metric", "nomad", "v6n"]); w.writerows(detail)
        w.writerow([]); w.writerow(["paired", "both_succeed", "nomad_only", "v6n_only", "neither"])
        w.writerow(["episodes", both, only_a, only_b, len(keys) - both - only_a - only_b])
    md = [f"# NoMaD vs v6n on the {a.v6n_map} map — same 40 episodes\n",
          "| Episodes | NoMaD SR | v6n SR | NoMaD SPL | v6n SPL |", "|---|---|---|---|---|"]
    md += ["| " + " | ".join(r) + " |" for r in rows]
    md += ["", "| | NoMaD | v6n |", "|---|---|---|"] + ["| " + " | ".join(map(str, r)) + " |" for r in detail]
    md += ["", f"Paired: both succeed {both}, NoMaD only {only_a}, v6n only {only_b}, neither {len(keys) - both - only_a - only_b}.", "",
           "Same map nodes and edges, episodes, success radius (4 m), step-cap formula and SPL definition. NoMaD localises "
           "with its distance head and drives with diffusion waypoints through the baseline SetPointAgent PID (300x300 "
           "camera, baseline vehicle); v6n localises with MobileNet latents and drives with its TQC policy (option C)."]
    open(a.out + ".md", "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
