"""Re-bin value-vs-distance by ROUTE distance instead of Euclidean separation.

Question from the documentation session: calibrate_critic.separation_table bins
pairs by np.linalg.norm(P_i - P_j). Two points 5 m apart across a median can be
hundreds of metres apart by road, so a high critic estimate at "0-5 m" may be
correct. This recomputes every band on directed lane-graph route distance and
reports both.

Offline: carla.Map built from the town's .xodr, no simulator, no GPU.
Lane graph: waypoints every RES m, successor edges from wp.next(RES) (lanes are
driven in their own direction, junction connections included, no lane
changes). Source = the trace position snapped to a node within SNAP_M whose
heading matches the vehicle's yaw. Target = ANY node within GOAL_M of the goal
position (arrival is a Euclidean radius, so the nearest in-radius lane counts).
"""
import argparse, json, pickle, sys
import numpy as np
import carla
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

XODR = "/home/robotlab/Apps/CARLA_0.9.15/CarlaUE4/Content/Carla/Maps/OpenDrive/{}.xodr"
BANDS = [(0, 5), (5, 10), (10, 20), (20, 30), (30, 50), (50, 80), (80, 1e9)]
RES, SNAP_M, GOAL_M, YAW_TOL = 1.0, 3.0, 2.0, np.deg2rad(60)


def lane_graph(town):
    """Chains per topology segment (the map's own lane-to-lane connectivity,
    junction connectors included), joined exit -> entry where they coincide.
    A first version matched wp.next() to a KD grid and lost every junction
    branch (nodes == edges, 29% of pairs routable); the SCC check below is
    there so that cannot pass silently again."""
    m = carla.Map(town, open(XODR.format(town)).read())
    xy, yaw, rows, cols, lens = [], [], [], [], []
    first, last, entries, exits = [], [], [], []
    for entry, exit_ in m.get_topology():
        chain = [entry] + entry.next_until_lane_end(RES)
        base = len(xy)
        for w in chain:
            xy.append([w.transform.location.x, w.transform.location.y])
            yaw.append(np.deg2rad(w.transform.rotation.yaw))
        for k in range(len(chain) - 1):
            rows.append(base + k); cols.append(base + k + 1)
        first.append(base); last.append(base + len(chain) - 1)
        entries.append([entry.transform.location.x, entry.transform.location.y])
        exits.append([exit_.transform.location.x, exit_.transform.location.y])
    xy, yaw = np.asarray(xy), np.asarray(yaw)
    etree = cKDTree(np.asarray(entries))
    for s, ex in enumerate(exits):
        for t in etree.query_ball_point(ex, 1.0):
            if t != s:
                rows.append(last[s]); cols.append(first[t])
    lens = [max(np.linalg.norm(xy[c] - xy[r]), 1e-3) for r, c in zip(rows, cols)]
    G = csr_matrix((lens, (rows, cols)), shape=(len(xy), len(xy)))
    from scipy.sparse.csgraph import connected_components
    ncomp, lab = connected_components(G, directed=True, connection="strong")
    big = np.bincount(lab).max() / len(xy)
    print(f"[Graph] segments {len(first)}, strongly connected components {ncomp}, largest holds {big:.1%} of nodes")
    return xy, yaw, cKDTree(xy), G


def band_name(lo, hi):
    return f"{lo:g}-{hi:g}" if hi < 1e8 else f"{lo:g}+"


def table(sep, steps, mask):
    out = {}
    for lo, hi in BANDS:
        k = mask & (sep >= lo) & (sep < hi)
        out[band_name(lo, hi)] = (float(np.median(steps[k])) if k.any() else None, int(k.sum()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--town", default="Town02")
    ap.add_argument("--trace", action="append", required=True, help="label=path/to/trace.pkl (needs .dist.npy)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    xy, nyaw, tree, G = lane_graph(a.town)
    Grev = G.T.tocsr()
    print(f"[Graph] {a.town}: {len(xy)} nodes, {G.nnz} edges (res {RES} m)", flush=True)
    results = {}
    for spec in a.trace:
        label, path = spec.split("=", 1)
        tr = pickle.load(open(path, "rb"))
        P = np.asarray(tr["positions"], np.float64)[:, :2]
        Y = np.asarray(tr["yaws"], np.float64)
        if np.abs(Y).max() > 2 * np.pi + 0.1:
            Y = np.deg2rad(Y)
        steps = np.load(path + ".dist.npy")
        n = len(P)
        assert steps.shape == (n, n)

        # source snap: heading-compatible node within SNAP_M, else unsnapped
        src = np.full(n, -1)
        for i in range(n):
            cand = tree.query_ball_point(P[i], SNAP_M)
            cand = [c for c in cand if abs(np.angle(np.exp(1j * (nyaw[c] - Y[i])))) < YAW_TOL]
            if cand:
                src[i] = min(cand, key=lambda c: np.linalg.norm(xy[c] - P[i]))
        # route[i, j]: shortest directed path from src[i] to any node within GOAL_M of P[j]
        route = np.full((n, n), np.inf)
        ok = src >= 0
        for j in range(n):
            tgt = tree.query_ball_point(P[j], GOAL_M)
            if not tgt:
                continue
            d = dijkstra(Grev, directed=True, indices=tgt, min_only=True)  # node -> target set
            route[ok, j] = d[src[ok]]
        euc = np.linalg.norm(P[:, None] - P[None], axis=-1)
        off = ~np.eye(n, dtype=bool)
        fin = off & np.isfinite(route)
        rho_r = spearmanr(steps[fin], route[fin]).correlation
        rho_e = spearmanr(steps[off], euc[off]).correlation
        rho_e_same = spearmanr(steps[fin], euc[fin]).correlation
        ratio = route[fin] / np.maximum(euc[fin], 1e-3)
        r = {
            "n_positions": n, "snapped_frac": float(ok.mean()),
            "routable_pair_frac": float(fin.sum() / off.sum()),
            "spearman_steps_vs_route": float(rho_r),
            "spearman_steps_vs_euclid_all_pairs": float(rho_e),
            "spearman_steps_vs_euclid_routable_pairs": float(rho_e_same),
            "route_over_euclid_median": float(np.median(ratio)),
            "share_route_gt_2x_euclid": float((ratio > 2).mean()),
            "by_euclid": table(euc, steps, off),
            "by_route": table(route, steps, fin),
        }
        # the specific worry: short Euclidean pairs -- how far are they by road?
        short = fin & (euc < 5)
        if short.any():
            r["euclid_0_5m_route_median_m"] = float(np.median(route[short]))
            r["euclid_0_5m_share_route_gt_50m"] = float((route[short] > 50).mean())
        results[label] = r
        print(f"\n[{label}] {path}  n={n} snapped {ok.mean():.2f} routable pairs {r['routable_pair_frac']:.2f}")
        print(f"  spearman(steps, route) {rho_r:+.3f} | (steps, euclid) {rho_e:+.3f} all, {rho_e_same:+.3f} routable")
        print(f"  route/euclid median {r['route_over_euclid_median']:.2f}, >2x on {r['share_route_gt_2x_euclid']:.0%} of pairs")
        if short.any():
            print(f"  pairs <5 m Euclidean: route median {r['euclid_0_5m_route_median_m']:.1f} m, "
                  f"{r['euclid_0_5m_share_route_gt_50m']:.0%} are >50 m by road")
        print(f"  {'band':>7} | {'by euclid: med steps (pairs)':>28} | {'by route: med steps (pairs)':>28}")
        for lo, hi in BANDS:
            b = band_name(lo, hi)
            e, rr = r["by_euclid"][b], r["by_route"][b]
            f = lambda t: f"{t[0]:8.1f} ({t[1]:7d})" if t[0] is not None else f"{'-':>8} ({t[1]:7d})"
            print(f"  {b:>7} | {f(e):>28} | {f(rr):>28}")
        sys.stdout.flush()
    json.dump({"town": a.town, "res_m": RES, "snap_m": SNAP_M, "goal_m": GOAL_M,
               "note": "spearman positive = critic steps rise with distance (healthy)",
               "traces": results}, open(a.out, "w"), indent=1)
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
