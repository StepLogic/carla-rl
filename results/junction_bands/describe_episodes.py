"""Describe the junction-band episode files (route properties only, no outcomes):
n, seeds, route length, share of routes with a left turn, turn-direction sequences."""
import collections
import glob
import json
import os
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
print("| Set | n | Seeds | Route m, median (min-max) | Contains a left turn | Turn sequences |")
print("|---|---|---|---|---|---|")
for f in sorted(glob.glob(os.path.join(HERE, "episodes", "*.json"))):
    rows = json.load(open(f))
    r = [x["route_m"] for x in rows]
    seeds = [x["seed"] for x in rows]
    left = sum(x["contains_left"] for x in rows)
    seq = collections.Counter(x["turn_dirs"] for x in rows)
    top = ", ".join(f"{k} {v}" for k, v in sorted(seq.items(), key=lambda kv: (-kv[1], kv[0]))[:6])
    more = "" if len(seq) <= 6 else f", +{len(seq) - 6} more"
    print(f"| {os.path.basename(f)[:-5]} | {len(rows)} | {min(seeds)}-{max(seeds)} | "
          f"{statistics.median(r):.0f} ({min(r):.0f}-{max(r):.0f}) | {left}/{len(rows)} | {top}{more} |")
