#!/usr/bin/env python3
"""Build the fixed CARLA benchmark episode set from probe files.

    python3 -I scripts/bench_select.py out/episodes.json probe1.json probe2.json ...

Each probe file is what `EVAL_PROBE=1` writes (one town). Keeps the first
--per-bin seeds per town per bin, in seed order (vae_nav.carla_benchmark), and
prints the counts per cell so a short bin is visible before anything runs.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from vae_nav.carla_benchmark import BINS, select_episodes  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("probes", nargs="+")
ap.add_argument("--per-bin", type=int, default=100)
a = ap.parse_args()

probes = []
for f in a.probes:
    probes += json.load(open(f))
eps = select_episodes(probes, per_bin=a.per_bin)
with open(a.out, "w") as fh:
    json.dump(eps, fh, indent=1)

towns = sorted({e["town"] for e in eps})
print(f"{len(eps)} episodes, {len(towns)} towns, {a.per_bin} per bin -> {a.out}")
print(f"{'town':>10} " + " ".join(f"{b[0]:>8}" for b in BINS) + f" {'straight':>9} {'curved':>7}")
for t in towns:
    te = [e for e in eps if e["town"] == t]
    print(f"{t:>10} " + " ".join(f"{sum(1 for e in te if e['bin'] == b[0]):>8}" for b in BINS)
          + f" {sum(1 for e in te if e['kind'] == 'straight'):>9}"
          f" {sum(1 for e in te if e['kind'] == 'curved'):>7}")
routed = sum(1 for p in probes if p.get("route_m") is not None)
print(f"probed {len(probes)} seeds, {routed} routed, "
      f"{len(probes) - routed} dropped as unroutable")
