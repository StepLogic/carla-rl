#!/usr/bin/env python3
"""ViNT on the frozen map benchmark (docs/map_benchmark_fresh_2026-11-07.md), in its
OWN block: the ViNT-native topomap arm (EVAL_VINT_TOPOMAP, slurm/vint_bench_map.sh
ARM=vinttopo) against ViNT mapless (ARM=mapless) on the same episodes. The two
arms differ only in the map; map, localisation and navigator all differ from the
c0 arm, so these rows are not mixed with c0's (the mapless rows are the clean
policy comparison).

Per town x band cell:
  * paired SR at 1 m (the run radius) and 2 m, SPL (bench_map_report.spl, as the
    c0 tables), the discordant pairs and an exact McNemar test (Holm over cells);
  * the topomap arm's fallbacks (no route / same node -> driven mapless in-run),
    and PRIVILEGED diagnostics (reporting only, never used to drive): start/goal
    node errors, the share of episodes that reached the final leg, and for failed
    routed episodes whether the car ended off its route (> OFF_ROUTE_M from every
    route node: lost on the map) and the gap between the navigator's own route
    position and the true one;
  * both arms' speed (pooled path over pooled time, 10 Hz steps) and end mix.

    vint_bench_report.py --root checkpoints --label vint_fresh
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "scripts"))
from bench_map_report import spl  # noqa: E402
from vae_nav.carla_benchmark import holm, mcnemar_p  # noqa: E402

STEP_S = 0.1          # one env step (CARLA fixed delta)
OFF_ROUTE_M = 5.0     # farther than this from every route node = off the route
LOC_OK_M = 8.0        # node "within 8 m" of the true start/goal, as the c0 table
BANDS = ("easy", "medium", "hard")
ENDS = ("reached", "crossing", "off_road", "collision")


def load(path):
    """eval.json -> {seed: row}; {} when the cell has not run."""
    if not os.path.isfile(path):
        return {}
    return {int(r["seed"]): r for r in json.load(open(path))["episodes_detail"]}


def pair_stats(topo, mapless):
    ks = sorted(set(topo) & set(mapless))
    n = len(ks)
    if not n:
        return {"n": 0}
    om = sum(bool(topo[k]["reached"]) and not mapless[k]["reached"] for k in ks)
    ob = sum(bool(mapless[k]["reached"]) and not topo[k]["reached"] for k in ks)
    return {"n": n, "keys": ks,
            "sr_topo": sum(bool(topo[k]["reached"]) for k in ks) / n,
            "sr_mapless": sum(bool(mapless[k]["reached"]) for k in ks) / n,
            "only_topo": om, "only_mapless": ob, "p": mcnemar_p(om, ob),
            "sr2_topo": sum(bool(topo[k]["reached_2m"]) for k in ks) / n,
            "sr2_mapless": sum(bool(mapless[k]["reached_2m"]) for k in ks) / n,
            "spl_topo": sum(spl(topo[k]) for k in ks) / n,
            "spl_mapless": sum(spl(mapless[k]) for k in ks) / n}


def _med(xs):
    return float(np.median(xs)) if xs else float("nan")


def topo_diag(rows):
    tv = [(r, r["vint_topo"]) for r in rows.values()]
    n = len(tv)
    s_err = [t["privileged_start_node_err_m"] for _, t in tv
             if t.get("privileged_start_node_err_m") is not None]
    g_err = [t["privileged_goal_node_err_m"] for _, t in tv
             if t.get("privileged_goal_node_err_m") is not None]
    fails = [t for r, t in tv if not r["reached"] and t.get("fallback") is None]
    off = [t for t in fails if t.get("privileged_end_route_err_m") is not None
           and t["privileged_end_route_err_m"] > OFF_ROUTE_M]
    gaps = [abs(t["route_idx_end"] - t["privileged_end_route_idx"]) for t in fails
            if t.get("route_idx_end") is not None
            and t.get("privileged_end_route_idx") is not None]
    return {"n": n,
            "no_route": sum(t.get("fallback") == "no route" for _, t in tv),
            "same_node": sum(t.get("fallback") == "same node" for _, t in tv),
            "start_err_med_m": _med(s_err),
            "start_within_8m": sum(e <= LOC_OK_M for e in s_err) / max(len(s_err), 1),
            "goal_err_med_m": _med(g_err),
            "goal_within_8m": sum(e <= LOC_OK_M for e in g_err) / max(len(g_err), 1),
            "final_leg": sum(t.get("final_at") is not None for _, t in tv) / max(n, 1),
            "routed_fails": len(fails), "off_route_fails": len(off),
            "belief_gap_med_nodes": _med(gaps)}


def speed_kmh(rows):
    steps = sum(int(r["steps"]) for r in rows.values())
    return 3.6 * sum(float(r["path_m"]) for r in rows.values()) / (STEP_S * steps) \
        if steps else float("nan")


def end_mix(rows):
    out = {e: 0 for e in ENDS + ("clock/other",)}
    for r in rows.values():
        e = "reached" if r["reached"] else r["end"]
        out[e if e in ENDS else "clock/other"] += 1
    return out


def cell_paths(root, label, town, band, tag, topomap_tag):
    return (os.path.join(root, f"bench_{label}_vinttopo_{town}_{band}",
                         f"eval_{tag}_vinttopo-{topomap_tag}", "eval.json"),
            os.path.join(root, f"bench_{label}_mapless_{town}_{band}", f"eval_{tag}",
                         "eval.json"))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="checkpoints")
    ap.add_argument("--label", default="vint_fresh")
    ap.add_argument("--towns", nargs="+", default=["Town02", "Town03"])
    ap.add_argument("--tag", default="epoch_1")
    ap.add_argument("--topomap-tag", default="epoch_1")
    a = ap.parse_args(argv)

    cells = []
    for town in a.towns:
        for band in BANDS:
            tp, mp = cell_paths(a.root, a.label, town, band, a.tag, a.topomap_tag)
            topo, mapless = load(tp), load(mp)
            if topo and mapless:
                cells.append((town, band, topo, mapless))
            else:
                print(f"(missing: {town} {band}: topomap {len(topo)} mapless {len(mapless)})")

    print("\n### ViNT-native topomap vs ViNT mapless, paired (same episodes)")
    print("| Town | Band | n | SR@1m Topomap | SR@1m Mapless | Topomap-only / Mapless-only "
          "| p (McNemar, Holm) | SR@2m Topomap | SR@2m Mapless | SPL Topomap | SPL Mapless |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    stats = [pair_stats(topo, mapless) for _, _, topo, mapless in cells]
    adj = holm([s["p"] for s in stats]) if stats else []
    for (town, band, _, _), s, q in zip(cells, stats, adj):
        print(f"| {town} | {band} | {s['n']} | {s['sr_topo']:.3f} | {s['sr_mapless']:.3f} | "
              f"{s['only_topo']} / {s['only_mapless']} | {q:.4f} | {s['sr2_topo']:.3f} | "
              f"{s['sr2_mapless']:.3f} | {s['spl_topo']:.3f} | {s['spl_mapless']:.3f} |")
    for town in a.towns + ["All"]:
        sel = [c for c in cells if town == "All" or c[0] == town]
        if not sel:
            continue
        topo = {(c[0], c[1], k): v for c in sel for k, v in c[2].items()}
        mapless = {(c[0], c[1], k): v for c in sel for k, v in c[3].items()}
        s = pair_stats(topo, mapless)
        print(f"| {town} | pooled | {s['n']} | {s['sr_topo']:.3f} | {s['sr_mapless']:.3f} | "
              f"{s['only_topo']} / {s['only_mapless']} | {s['p']:.4f} (raw) | "
              f"{s['sr2_topo']:.3f} | {s['sr2_mapless']:.3f} | {s['spl_topo']:.3f} | "
              f"{s['spl_mapless']:.3f} |")

    print("\n### Topomap arm: fallbacks and PRIVILEGED diagnostics (reporting only)")
    print("| Town | Band | n | No route | Same node | Start node err med (m) | Start <=8 m "
          "| Goal node err med (m) | Goal <=8 m | Final leg reached | Routed fails "
          f"| Ended >{OFF_ROUTE_M:g} m off route | Navigator-vs-true route gap med (nodes) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for town, band, topo, _ in cells:
        d = topo_diag(topo)
        print(f"| {town} | {band} | {d['n']} | {d['no_route']} | {d['same_node']} | "
              f"{d['start_err_med_m']:.1f} | {d['start_within_8m']:.2f} | "
              f"{d['goal_err_med_m']:.1f} | {d['goal_within_8m']:.2f} | {d['final_leg']:.2f} | "
              f"{d['routed_fails']} | {d['off_route_fails']} | {d['belief_gap_med_nodes']:.1f} |")

    print("\n### Speed and end mix")
    print("| Town | Band | Arm | km/h | Reached | Crossing | Off road | Collision | Clock/other |")
    print("|---|---|---|---|---|---|---|---|---|")
    for town, band, topo, mapless in cells:
        for arm, rows in (("Topomap", topo), ("Mapless", mapless)):
            m = end_mix(rows)
            print(f"| {town} | {band} | {arm} | {speed_kmh(rows):.1f} | {m['reached']} | "
                  f"{m['crossing']} | {m['off_road']} | {m['collision']} | {m['clock/other']} |")


if __name__ == "__main__":
    main()
