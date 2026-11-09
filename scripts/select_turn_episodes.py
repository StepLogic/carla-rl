#!/usr/bin/env python3
"""Hard-goal episode set (user 2026-11-09: "we need hard goals to show map works";
"we replace hard with this"): from EVAL_PROBE_TURNS probes, keep routes of
50-150 m with >= 2 junction turns, the first --per-town per town in seed order.

    select_turn_episodes.py logs/bench_turns_episodes.json probeA.json probeB.json
"""
import argparse, json, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from vae_nav.carla_benchmark import path_kind  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("out"); ap.add_argument("probes", nargs="+")
ap.add_argument("--per-town", type=int, default=100)
ap.add_argument("--min-m", type=float, default=50.0); ap.add_argument("--max-m", type=float, default=150.0)
ap.add_argument("--min-turns", type=int, default=2)
a = ap.parse_args()
out = []
for f in a.probes:
    ps = sorted(json.load(open(f)), key=lambda p: p["seed"])
    keep = [p for p in ps if p.get("route_m") is not None and p.get("turns") is not None
            and a.min_m <= p["route_m"] < a.max_m and p["turns"] >= a.min_turns]
    town = ps[0]["town"]
    print(f"{town}: {len(ps)} probed, {sum(1 for p in ps if p.get('route_m') is not None)} routed, "
          f"{len(keep)} qualify -> keep {min(len(keep), a.per_town)}")
    if len(keep) < a.per_town:
        sys.exit(f"{town}: only {len(keep)} qualifying episodes, need {a.per_town}")
    for p in keep[:a.per_town]:
        out.append({"town": town, "seed": p["seed"], "bin": "hard", "band": p["band"],
                    "route_m": p["route_m"], "euclid_m": p["euclid_m"],
                    "kind": path_kind(p["route_m"], p["euclid_m"]),
                    "turns": p["turns"], "junctions": p["junctions"]})
json.dump(out, open(a.out, "w"), indent=1)
print(f"{len(out)} episodes -> {a.out}")
