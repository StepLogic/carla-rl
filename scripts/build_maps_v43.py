#!/usr/bin/env python3
"""Build the online-MAD map and its matched controls from a v43 trace.

The signal is the EXPLORE-MODE ACTOR ENTROPY recorded per kept sample
(`uncertainty["ent_total"]`: the policy's true action-space differential entropy,
exact Gaussian and rescale terms plus a Monte-Carlo tanh term). The placers and
every shared rule come from the Habitat peer's spec -- see vae_nav/online_maps.py
for what was ported and the one thing that had to be re-scaled for roads.

Three placers, all with MATCHED node counts, because a placement comparison that
does not match counts is a comparison of budgets:
  mad        online, causal, entropy-driven
  temporal   every Nth step, N searched to match           (the peer's control)
  uniform    every s metres, s searched to match           (the other control)

SGM is deliberately NOT here: it needs pairwise critic values over the
candidates, which is a GPU pass, and it belongs in its own script so a failure
there cannot invalidate these three.

usage: build_maps_v43.py OUT_DIR TRACE.pkl [TRACE2.pkl ...]
"""
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))
from vae_nav.online_maps import (build_edges, online_mad,  # noqa: E402
                                 temporal_nodes, uniform_nodes)

SIGNAL = "ent_total"


def load(path):
    with open(path, "rb") as fh:
        tr = pickle.load(fh)
    unc = tr.get("uncertainty")
    if not unc:
        raise SystemExit(f"{path}: no per-sample uncertainty, so there is no "
                         "entropy to place on.")
    missing = [i for i, r in enumerate(unc) if SIGNAL not in r]
    if missing:
        raise SystemExit(f"{path}: {len(missing)} samples have no {SIGNAL!r}; "
                         "an older collector wrote this trace.")
    tr["_h"] = np.array([float(r[SIGNAL]) for r in unc])
    tr["_pos"] = np.asarray(tr["positions"], np.float64)[:, :2]
    tr["_seg"] = np.asarray(tr["segments"], int)
    return tr


def build(tr, name):
    h, pos, seg = tr["_h"], tr["_pos"], tr["_seg"]
    mad_idx, st = online_mad(h, pos, seg)
    n = len(mad_idx)
    if n == 0:
        raise SystemExit(f"{name}: online MAD placed NO nodes "
                         f"(degenerate windows {st['degenerate_window_frac']:.0%}). "
                         "A pinned-entropy actor looks exactly like this; check "
                         "the entropy before changing z.")
    maps = {"mad": mad_idx,
            "temporal": temporal_nodes(pos, seg, n_target=n),
            "uniform": uniform_nodes(pos, seg, n_target=n)}

    out = {"trace": name, "town": tr.get("meta", {}).get("town", "?"),
           "signal": SIGNAL, "samples": len(h), "mad_stats": st, "maps": {}}
    print(f"\n{name} [{out['town']}]  {len(h)} samples, "
          f"spacing {st['sample_spacing_m']:.2f} m")
    print(f"  window {st['window_m']:.0f} m = {st['window_samples']} samples, "
          f"warmup {st['warmup_samples']}, z={st['z']}, "
          f"degenerate windows {st['degenerate_window_frac']:.1%}")
    for kind, idx in maps.items():
        edges, kinds = build_edges(idx, pos, seg)
        seglinks = kinds.count("seglink")
        xy = pos[idx]
        nn = _nearest_gap(xy)
        out["maps"][kind] = {
            "n_nodes": int(len(idx)),
            "indices": [int(i) for i in idx],
            "xy": xy.tolist(),
            "segments": [int(s) for s in seg[idx]],
            "n_edges": len(edges),
            "n_seglinks": seglinks,
            "edges": [[int(a), int(b)] for a, b in edges],
            "edge_kinds": kinds,
            "median_node_gap_m": float(np.median(nn)) if len(nn) else None,
        }
        print(f"  {kind:9s} {len(idx):4d} nodes  {len(edges):4d} edges "
              f"({seglinks} seglinks)  median nearest-node gap "
              f"{np.median(nn):6.1f} m" if len(nn) else "")
    return out


def _nearest_gap(xy):
    """Distance from each node to its nearest other node: the honest read of how
    a placer spreads, since a mean spacing hides clustering."""
    if len(xy) < 2:
        return np.empty(0)
    d = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    return d.min(axis=1)


def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    out_dir, paths = sys.argv[1], sys.argv[2:]
    os.makedirs(out_dir, exist_ok=True)
    for p in paths:
        tr = load(p)
        name = os.path.basename(p).replace(".pkl", "")
        res = build(tr, name)
        dest = os.path.join(out_dir, f"{name}_maps.json")
        with open(dest, "w") as fh:
            json.dump(res, fh, indent=1)
        print(f"  -> {dest}")


if __name__ == "__main__":
    main()
