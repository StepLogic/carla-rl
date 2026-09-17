"""Node-placement MAPS for Town02: what each criterion built, on the paths driven.

    python figure_maps_town02.py                      # both policies
    python figure_maps_town02.py --in results/entropy_nogoal.json

ONE FIGURE PER ARM, not a grid. Each map is meant to be looked at on its own,
so a grid that shrinks all four to thumbnails defeats the purpose (the same
choice plot_arm_maps.py makes in the Habitat tree).

WHAT IS DRAWN. The background of every map is the union of the paths the agent
actually drove -- that IS the region a topological map can cover, and drawing
the town's whole road network instead would imply coverage the data does not
have. Junction ground truth (the env's own `in_junction` flag) is marked so a
reader can see whether a criterion put nodes where the route is decidable.

EQUAL BUDGET, WHICH IS THE ONLY FAIR COMPARISON. The entropy arm picks its node
count from the data; uniform and random are then given EXACTLY that count on
EXACTLY the same path. A criterion with more nodes trivially covers more, so an
unmatched comparison measures spending, not placement.

ONE POLICY PER FIGURE. The goal-conditioned and goal-free traces come from
different checkpoints, so they are different agents and never share a figure;
each caption names its own checkpoint (experiments.txt s0a, s7).

SGM IS ABSENT FROM THESE FIGURES, DELIBERATELY AND VISIBLY. An SGM exists in
the gated-impl worktree (sgm_frame.py is the release-faithful frame), but it
needs an all-pairs critic matrix over the trace and knobs recalibrated on the
v4 critic; the entropy traces here carry no critic matrix. The manifest
records the SGM row as pending rather than quietly dropping it
(experiments.txt s7: "State a missing baseline. Never omit one silently.").
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import node_placement_entropy as NP

# Okabe-Ito, colour-blind safe. Arm colours match the ICRA placement figures.
C = {"entropy": "#1baf7a", "uniform": "#E69F00", "random": "#52514e"}
LBL = {"entropy": "Entropy Spikes (MAD)", "uniform": "Uniform Spacing",
       "random": "Random Placement"}
PATH, JUNC, INK = "#c9c7c3", "#d55e00", "#0b0b0b"


def episode_arrays(ep):
    """(xy, arclength, entropy, in_junction) for one episode."""
    st = ep["steps"]
    xy = np.array([[s["x"], s["y"]] for s in st], np.float64)
    h = np.array([s["entropy"] for s in st], np.float64)
    inj = np.array([bool(s.get("in_junction")) for s in st])
    return xy, NP.arclength(xy), h, inj


def draw_base(ax, eps, lo, hi):
    """Driven paths plus junction ground truth -- the background of every map."""
    for xy, _s, _h, inj in eps:
        ax.plot(xy[:, 0], xy[:, 1], "-", color=PATH, lw=1.0, zorder=1)
        if inj.any():
            ax.plot(xy[inj, 0], xy[inj, 1], ".", color=JUNC, ms=2.0,
                    alpha=0.55, zorder=2)
    ax.set_aspect("equal")
    ax.set_xlabel("Map X (m)")
    ax.set_ylabel("Map Y (m)")
    ax.grid(True, color="#eeecea", lw=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="",
                    help="one dataset; default runs both that exist")
    ap.add_argument("--out-dir", default="results/maps_town02")
    ap.add_argument("--min-sep-m", type=float, default=8.0)
    ap.add_argument("--smooth", type=int, default=9)
    ap.add_argument("--z", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    datasets = ([a.inp] if a.inp else
                [p for p in ("results/entropy_nogoal.json",
                             "results/entropy_nodes.json") if os.path.exists(p)])
    if not datasets:
        raise SystemExit("no entropy trace data found")

    manifest = {"missing_baselines": {
        "sgm": "No SGM implementation or cached critic distance matrix exists "
               "in carla-rl; the faithful port and its per-critic calibration "
               "live in the robot-learning-tools (Habitat) tree. Not run, "
               "not approximated."}}

    for inp in datasets:
        d = json.load(open(inp))
        tag = os.path.splitext(os.path.basename(inp))[0]
        out_dir = os.path.join(a.out_dir, tag)
        os.makedirs(out_dir, exist_ok=True)
        eps = [episode_arrays(e) for e in d["episodes"] if len(e["steps"]) > 20]
        lo, hi = d.get("goal_band", (None, None))
        rng = np.random.default_rng(a.seed)

        # Per-episode node sets. Entropy chooses the budget; the other two are
        # handed the same count on the same path.
        nodes = {k: [] for k in C}
        for xy, s, h, _inj in eps:
            idx_e = NP.mad_spikes(h, s, a.min_sep_m, a.smooth, a.z, nms_w=0)
            n = len(idx_e)
            nodes["entropy"].append(xy[idx_e] if n else xy[:0])
            iu = NP.place_uniform(s, n)
            nodes["uniform"].append(xy[iu] if iu else xy[:0])
            ir = NP.place_random(s, n, rng)
            nodes["random"].append(xy[ir] if ir else xy[:0])

        band = (f"{lo:g}-{hi:g} m goals" if lo is not None else "unknown band")
        cap = (f"{d.get('town', '?')}, {len(eps)} episodes, {band}\n"
               f"Checkpoint: {d.get('ckpt', '?')}")

        # --- the paths themselves, no nodes ---------------------------------
        fig, ax = plt.subplots(figsize=(7.2, 6.4))
        draw_base(ax, eps, lo, hi)
        ax.set_title(f"Paths Driven by the Agent\n{cap}", fontsize=10,
                     color=INK)
        ax.plot([], [], "-", color=PATH, lw=1.5, label="Driven Path")
        ax.plot([], [], ".", color=JUNC, ms=6, label="In Junction")
        ax.legend(loc="best", frameon=False, fontsize=8)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(out_dir, f"paths_driven.{ext}"), dpi=200)
        plt.close(fig)

        # --- one map per arm -------------------------------------------------
        counts = {}
        for arm in ("entropy", "uniform", "random"):
            alln = np.vstack([n for n in nodes[arm] if len(n)]) \
                if any(len(n) for n in nodes[arm]) else np.zeros((0, 2))
            counts[arm] = int(len(alln))
            fig, ax = plt.subplots(figsize=(7.2, 6.4))
            draw_base(ax, eps, lo, hi)
            if len(alln):
                ax.plot(alln[:, 0], alln[:, 1], "o", color=C[arm], ms=5.0,
                        mec="white", mew=0.6, zorder=3, label=LBL[arm])
            ax.set_title(f"Map Nodes: {LBL[arm]}\n{cap}", fontsize=10,
                         color=INK)
            ax.legend(loc="best", frameon=False, fontsize=8,
                      title=f"{len(alln)} Nodes", title_fontsize=8)
            fig.tight_layout()
            for ext in ("png", "pdf"):
                fig.savefig(os.path.join(out_dir, f"map_{arm}.{ext}"), dpi=200)
            plt.close(fig)

        manifest[tag] = {
            "source": inp, "checkpoint": d.get("ckpt"), "step": d.get("step"),
            "town": d.get("town"), "goal_band": d.get("goal_band"),
            "episodes_used": len(eps),
            "node_counts": counts,
            "budget_rule": "entropy chooses n per episode; uniform and random "
                           "are matched to it on the same path",
            "params": {"min_sep_m": a.min_sep_m, "smooth": a.smooth,
                       "mad_z": a.z, "endpoints": False},
            "figures": sorted(os.listdir(out_dir)),
        }
        print(f"[Maps] {tag}: {len(eps)} episodes, nodes {counts} -> {out_dir}",
              flush=True)

    with open(os.path.join(a.out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[Maps] manifest -> {os.path.join(a.out_dir, 'manifest.json')}")


if __name__ == "__main__":
    main()
