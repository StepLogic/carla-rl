"""Build topological maps from exploration traces, and SWEEP node spacing.

The peer's headline finding is that node count acts through CERTIFICATION
COVERAGE, not through subgoal spacing: more nodes means more node pairs fall
within certification range, so more shortcuts are certified, so the graph has
fewer components -- and success tracks components. Their r=0.96 "denser is
better" curve is a connectivity curve wearing a density label.

So the numbers this prints are COMPONENTS and ROUTABLE FRACTION, not success.
Those are the axis that moves, and routable fraction is a hard ceiling on what
mapping can buy: a policy cannot navigate a route the graph cannot express.

The denominator for routable fraction is EVERY PAIR DRAWN. Scoring only pairs
that happened to route flatters exactly the fragmented configurations -- they
measured one arm at 54.2%->70.0% that way and 53.3%->23.3% once the lost routes
were counted.
"""
import argparse
import glob
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from vae_nav.habitat_map import (audit_shortcuts, build_graph,  # noqa: E402
                                 routable_fraction)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default="traces")
    ap.add_argument("--out", default="results/habitat_maps.json")
    ap.add_argument("--arrival", type=float, default=2.0,
                    help="policy's training success radius")
    ap.add_argument("--spacings", default="2,4,6,10")
    ap.add_argument("--max-shortcut", type=float, default=15.0)
    ap.add_argument("--pairs", type=int, default=300)
    ap.add_argument("--min-segment-nodes", default="0,3",
                    help="comma list; 3 reproduces the peer's filter, 0 keeps all")
    args = ap.parse_args()

    spacings = [float(s) for s in args.spacings.split(",")]
    results = []
    for tp in sorted(glob.glob(os.path.join(args.traces, "*_trace.pkl"))):
        with open(tp, "rb") as f:
            tr = pickle.load(f)
        town = tr["town"]
        pos, yaws = tr["positions"], tr["yaws"]
        rings, seg = tr["rings"], tr["segment_ids"]
        n_seg = len(np.unique(seg))
        print(f"\n=== {town}: {len(pos)} trace samples, {n_seg} segments "
              f"(ring {tr.get('ring_beams')} beams / {tr.get('ring_hfov')} deg) ===")
        print(f"{'spacing':>8s} {'nodes':>6s} {'trace_e':>8s} {'short_e':>8s} "
              f"{'comp':>5s} {'routable':>9s} {'med_hops':>9s} {'med_maxedge':>12s}")
        for msn in [int(x) for x in args.min_segment_nodes.split(",")]:
          if msn:
              print(f"  -- min_segment_nodes={msn} (their filter: drops segments "
                    f"under ~{msn * 0.8:.1f} m of travel) --")
          else:
              print(f"  -- min_segment_nodes=0 (keep every segment) --")
          for sp in spacings:
            G, keep, st = build_graph(pos, yaws, rings, seg, spacing=sp,
                                      arrival_radius=args.arrival,
                                      max_shortcut=args.max_shortcut,
                                      min_segment_nodes=msn, verbose=False)
            rt = routable_fraction(G, n_pairs=args.pairs)
            print(f"{sp:8.1f} {st['nodes']:6d} {st['trace_edges']:8d} "
                  f"{st['shortcut_edges']:8d} {st['components']:5d} "
                  f"{100*rt['routable']:8.1f}% {rt['median_hops']:9.1f} "
                  f"{rt['median_longest_edge_m']:11.1f}m"
                  + (f"  dropped {st['segments_dropped']} segs "
                     f"({100*st['steps_dropped_frac']:.1f}% of steps)" if msn else ""))
            results.append({"town": town, "spacing": sp, **st, **rt,
                            "shortcut_audit": audit_shortcuts(G),
                            "trace_segments": int(n_seg)})
        # Their invariant check: nodes - trace_edges should equal the segment
        # count and should NOT move with spacing. If it does, the trace changed.
        seg_implied = {r["spacing"]: r["segments_implied"]
                       for r in results if r["town"] == town}
        print(f"  segments_implied by spacing: {seg_implied}")
        print(f"  (their invariant: this equals the segment count, {n_seg}, "
              f"and is constant across spacings)")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Map] {len(results)} configurations -> {args.out}")


if __name__ == "__main__":
    main()
