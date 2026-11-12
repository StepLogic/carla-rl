#!/usr/bin/env python3
"""Report the CARLA replication of the Gibson image-goal benchmark.

    python3 -I scripts/bench_report.py <true eval.json ...> --wrong <wrong eval.json ...>

One eval.json per town per arm (EVAL_EPISODES_FILE runs). Prints, Gibson-style,
SR and SPL per difficulty bin and pooled, at the Gibson radius and at our 2 m
training radius, plus NRNS's straight/curved split and the per-town table --
each with the true-vs-wrong-goal control paired per episode (exact McNemar,
Holm across the rows of that table).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from vae_nav.carla_benchmark import paired_table  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("true", nargs="+")
ap.add_argument("--wrong", nargs="+", required=True)
a = ap.parse_args()


def rows(files):
    out, radius = [], set()
    for f in files:
        d = json.load(open(f))
        radius.add(d.get("arrival_radius_m"))
        out += d["episodes_detail"]
    return out, radius


t, rt = rows(a.true)
w, rw = rows(a.wrong)
if len(rt | rw) != 1:
    sys.exit(f"arms were run at different arrival radii: {rt | rw}")
radius = rt.pop()


def show(title, table, order):
    print(f"\n### {title}")
    print(f"| | n | SR@{radius:g}m true | wrong | SPL true | SR@2m true | wrong "
          f"| only-true | only-wrong | p (Holm) |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for k in [o for o in order if o in table] + ["all"]:
        c = table[k]
        p = c.get("p_holm", c["p"])
        print(f"| {k} | {c['n']} | {c['sr_true']:.3f} | {c['sr_wrong']:.3f} | "
              f"{c['spl_true']:.3f} | {c['sr_true_2m']:.3f} | {c['sr_wrong_2m']:.3f} | "
              f"{c['only_true']} | {c['only_wrong']} | "
              f"{'<0.0001' if p < 1e-4 else f'{p:.4f}'} |")


show("By difficulty (Mezghani-style)", paired_table(t, w, by="bin"),
     ["easy", "medium", "hard"])
show("By path type (NRNS-style)", paired_table(t, w, by="kind"), ["straight", "curved"])
show("By town", paired_table(t, w, by="town"), sorted({r["town"] for r in t}))
