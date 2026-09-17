"""Recalibrate the critic-as-distance knobs on THIS checkpoint (protocol s3).

    python calibrate_critic.py --traces results/placement_v4/trace_s{7,8,9,10}.pkl \
        --horizon-m 50 --out results/placement_v4/calibration.json

RE-MEASURE, DO NOT INHERIT. Every cutoff below is a property of one critic.
H_STEPS=106 in value_distance was derived for a 30 m horizon at 0.283 m/step
under STEP_COST=0.1; the v4 gated critic trains on 20-50 m goals with
STEP_COST=1.0 and potential shaping, so its scale is different and the old
numbers must not be carried over (they would yield either a disconnected graph
or an unthinned trace, and both read as an SGM result).

TWO TABLES, then a choice justified against the TRAINED GOAL BAND:
  1. SEPARATION vs CRITIC DISTANCE: median predicted steps by true straight-line
     separation band, per trace. Shows the critic's scale, where it saturates,
     and how tightly it replicates across traces.
  2. CUTOFF SWEEP: for each (edge_cutoff, consistency) pair, per trace, the
     release-faithful SGM frame's node count, edge count, UNDIRECTED components
     and DIRECTED reachability (exhaustive BFS). The setting must give one
     component and full directed reachability on EVERY trace -- four seeds
     refuse settings one seed would accept.
  The edge cutoff is then justified against the band the critic was trained on:
  admitted edges' median/p90 true separation and the share beyond the horizon
  are reported so the caption can say how many edges carry extrapolated weight.
"""
import argparse
import json
import pickle

import numpy as np

from vae_nav.sgm_frame import (sgm_faithful, undirected_components,
                               directed_reachability, edge_separation_stats)

SEP_BANDS = ((0, 5), (5, 10), (10, 20), (20, 30), (30, 50), (50, 80), (80, 1e9))


def load(trace_path):
    with open(trace_path, "rb") as fh:
        tr = pickle.load(fh)
    dist = np.load(trace_path + ".dist.npy")
    P = np.asarray(tr["positions"], np.float64)[:, :2]
    assert dist.shape == (len(P), len(P)), (dist.shape, len(P))
    return P, dist


def separation_table(P, dist):
    n = len(P)
    sep = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1)
    off = ~np.eye(n, dtype=bool)
    rows = {}
    for lo, hi in SEP_BANDS:
        m = off & (sep >= lo) & (sep < hi)
        rows[f"{lo:g}-{hi:g}" if hi < 1e8 else f"{lo:g}+"] = (
            float(np.median(dist[m])) if m.any() else None, int(m.sum()))
    return rows


def sweep(P, dist, edge_cutoffs, consistencies, gate_m, horizon_m):
    out = []
    for e in edge_cutoffs:
        for c in consistencies:
            kept, edges = sgm_faithful(dist, P, consistency_cutoff=c,
                                       edge_cutoff=e, gate_m=gate_m)
            n = len(kept)
            rec = {"edge_cutoff": e, "consistency": c, "nodes": n,
                   "edges": len(edges),
                   "components_undirected": undirected_components(n, edges),
                   "reach_directed": directed_reachability(n, edges)}
            rec.update(edge_separation_stats(edges, P, kept, horizon_m))
            out.append(rec)
    return out


def choose_setting(results, min_reach):
    """results: {trace: [sweep rows]} -> ((edge_cutoff, consistency), recs).

    Smallest edge cutoff at which some consistency meets min_reach on EVERY
    trace (least extrapolation first), then the fewest nodes among those."""
    traces = list(results)
    keys = [(r["edge_cutoff"], r["consistency"]) for r in results[traces[0]]]
    ok = {}
    for e, c in keys:
        recs = [next(r for r in results[t] if r["edge_cutoff"] == e
                     and r["consistency"] == c) for t in traces]
        if all(r["reach_directed"] >= min_reach for r in recs):
            ok[(e, c)] = recs
    if not ok:
        return None, []
    e_min = min(e for e, _c in ok)
    cands = {k: v for k, v in ok.items() if k[0] == e_min}
    chosen = min(cands, key=lambda k: np.mean([r["nodes"] for r in cands[k]]))
    return chosen, ok[chosen]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", nargs="+", required=True)
    ap.add_argument("--horizon-m", type=float, default=50.0,
                    help="upper edge of the TRAINED goal band (v4: 20-50 m)")
    ap.add_argument("--gate-m", type=float, default=2.0,
                    help="metric gate; the env's goal radius, so two nodes "
                         "closer than an arrival are candidates to be one node")
    ap.add_argument("--edge-cutoffs", default="",
                    help="comma list in STEPS; default = quantiles of the "
                         "20-50 m band's critic distance")
    ap.add_argument("--consistencies", default="",
                    help="comma list in STEPS; default = fractions of each "
                         "edge cutoff")
    ap.add_argument("--out", required=True)
    # MEASURED 2026-09-11 (v5@200k, Town02, 4 traces): the best any setting
    # reaches is 0.995 directed, because a trace holds episode ENDS -- a node
    # where the car crashed or hit the cap has no forward edge unless a later
    # pass rejoined it. Reach 1.0 is therefore unattainable by construction
    # on a multi-episode trace, and demanding it chose NOTHING and failed
    # every downstream stage. The tolerance is reported with the choice.
    ap.add_argument("--min-reach", type=float, default=0.99,
                    help="directed reachability every trace must meet")
    ap.add_argument("--rechoose", action="store_true",
                    help="re-run only the choice on the sweep already in "
                         "--out (no recompute; ~40 min saved)")
    a = ap.parse_args()

    if a.rechoose:
        with open(a.out) as fh:
            prev = json.load(fh)
        results = prev["sweep"]
        chosen, recs = choose_setting(results, a.min_reach)
        prev["chosen"] = ({"edge_cutoff": chosen[0], "consistency": chosen[1],
                           "min_reach": a.min_reach,
                           "reach_directed": [r["reach_directed"] for r in recs]}
                          if chosen else None)
        prev["min_reach"] = a.min_reach
        with open(a.out, "w") as fh:
            json.dump(prev, fh, indent=2)
        print(f"[Calib] rechosen -> {a.out}")
        return

    data = [(t, *load(t)) for t in a.traces]
    print(f"[Calib] {len(data)} traces: " +
          ", ".join(f"{len(P)} cands" for _t, P, _d in data))

    # --- 1. separation vs critic distance ---------------------------------
    tables = {t: separation_table(P, d) for t, P, d in data}
    print("\nmedian critic distance (steps) by true separation, per trace")
    bands = list(next(iter(tables.values())).keys())
    print(f"{'band (m)':>10} " + " ".join(f"{t.split('/')[-1][:14]:>16}" for t in tables))
    for b in bands:
        print(f"{b:>10} " + " ".join(
            f"{(tables[t][b][0] if tables[t][b][0] is not None else float('nan')):>10.1f} n={tables[t][b][1]:<4d}"
            for t in tables))

    # --- 2. cutoff sweep ----------------------------------------------------
    if a.edge_cutoffs:
        E = [float(x) for x in a.edge_cutoffs.split(",")]
    else:
        # anchor on the trained band: the critic distance at 20-30 / 30-50 m
        meds = [tables[t][b][0] for t in tables for b in ("20-30", "30-50")
                if tables[t][b][0] is not None]
        base = float(np.median(meds))
        E = sorted({round(base * f, 1) for f in (0.5, 0.75, 1.0, 1.25, 1.5)})
    if a.consistencies:
        C = [float(x) for x in a.consistencies.split(",")]
    else:
        C = sorted({round(e * f, 1) for e in E for f in (0.25, 0.5, 0.75, 1.0)})
    print(f"\nsweep: edge cutoffs {E} x consistencies {C} (steps), gate {a.gate_m} m")
    results = {}
    print(f"{'trace':>14} {'edge':>7} {'cons':>7} {'nodes':>6} {'edges':>7} "
          f"{'comp(u)':>8} {'reach(d)':>9} {'sep_med':>8} {'sep_p90':>8} {'>H':>6}")
    for t, P, d in data:
        rows = sweep(P, d, E, C, a.gate_m, a.horizon_m)
        results[t] = rows
        for r in rows:
            print(f"{t.split('/')[-1][:14]:>14} {r['edge_cutoff']:>7.1f} "
                  f"{r['consistency']:>7.1f} {r['nodes']:>6d} {r['edges']:>7d} "
                  f"{r['components_undirected']:>8d} {r['reach_directed']:>9.3f} "
                  f"{r.get('sep_median_m', float('nan')):>8.1f} "
                  f"{r.get('sep_p90_m', float('nan')):>8.1f} "
                  f"{r.get('frac_beyond_horizon', float('nan')):>6.2f}")

    # --- 3. the setting: smallest edge cutoff giving reach >= min_reach on
    #        EVERY trace, then the consistency that thins most while keeping it
    chosen, recs = choose_setting(results, a.min_reach)
    if chosen:
        print(f"\nCHOSEN sgm[{chosen[1]:g}/{chosen[0]:g}]: reach " +
              "/".join(f"{r['reach_directed']:.3f}" for r in recs) +
              f" (>= {a.min_reach:g}) on all {len(recs)} traces; nodes " +
              "/".join(str(r["nodes"]) for r in recs) +
              f"; admitted-edge sep median " +
              "/".join(f"{r.get('sep_median_m', 0):.1f}" for r in recs) +
              f" m, beyond {a.horizon_m:g} m: " +
              "/".join(f"{100 * r.get('frac_beyond_horizon', 0):.0f}%" for r in recs))
    else:
        best = max(min(r["reach_directed"] for r in rs)
                   for rs in zip(*results.values()))
        print(f"\nNO setting reaches {a.min_reach:g} on every trace (best "
              f"min-reach {best:.4f}) -- widen the sweep; do not pick a "
              "single-trace winner.")
    with open(a.out, "w") as fh:
        json.dump({"traces": a.traces, "horizon_m": a.horizon_m,
                   "gate_m": a.gate_m, "separation_tables": tables,
                   "sweep": results,
                   "min_reach": a.min_reach,
                   "chosen": ({"edge_cutoff": chosen[0], "consistency": chosen[1],
                               "min_reach": a.min_reach,
                               "reach_directed": [r["reach_directed"] for r in recs]}
                              if chosen else None)}, fh, indent=2)
    print(f"[Calib] -> {a.out}")


if __name__ == "__main__":
    main()
