"""Raw-trace driving map (user 2026-09-14: "evaluate goal mode on raw trace, evaluate nomad on raw trace").

Every trace sample is a node (no merging, no thinning): the 1155 samples of the speed10 lane-off trace, each its
own member. Edges are left to sequence_link_map.py (trace order across episodes), like every other driven map.
"""
import argparse, json, pickle
import numpy as np

TRACE = "results/placement_v6n_town02_budget10k_laneoff_speed10/trace_s7.pkl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", default=TRACE)
    ap.add_argument("--out", default="results/maps/town02_speed10_s7_raw_trace_driving_map.json")
    ap.add_argument("--town", default="Town02")
    a = ap.parse_args()
    tr = pickle.load(open(a.trace, "rb"))
    P = np.asarray(tr["positions"], np.float64)[:, :2]; yaw = np.asarray(tr["yaws"], np.float64)
    nodes = [{"id": i, "trace_index": i, "x_m": float(P[i, 0]), "y_m": float(P[i, 1]), "yaw_rad": float(yaw[i]),
              "member_trace_indices": [i]} for i in range(len(P))]
    json.dump({"schema": "carla-rl raw trace map v1: every trace sample is a node", "town": a.town, "trace": a.trace,
               "nodes": nodes, "edges": [], "note": "edges are built by sequence_link_map.py (trace order across episodes)"},
              open(a.out, "w"), indent=1)
    print(f"[raw] {len(nodes)} nodes -> {a.out}")


if __name__ == "__main__":
    main()
