"""Plot policy entropy against time, with junction occupancy marked.

    python plot_entropy_junctions.py --in results/entropy_junctions.json \
        --out results/entropy_junctions.png

THREE PANELS, BECAUSE THE QUESTION HAS THREE PARTS.

  A  entropy vs time, per episode, junction spans shaded. This is the thing
     actually asked for and the only panel where a single junction traversal
     is visible as an event rather than as a summary statistic.
  B  per-EPISODE paired means, road vs junction. Panel A can be convincing for
     one lucky episode; this asks whether the same sign holds episode after
     episode. The pairing is what makes it a fair test -- episodes differ
     enormously in route and length, and an unpaired pooled comparison would
     mostly measure which episodes happened to contain junctions.
  C  per-dimension sigma. Entropy is sum_i log(sigma_i) + const, so a rise in H
     alone cannot say WHICH control became less certain. The hypothesis worth
     testing is specific: at a junction several exits are viable, so STEER
     should widen while throttle need not.

WHY NOT A P-VALUE OVER STEPS. Steps inside one junction traversal are the same
manoeuvre sampled tens of times; treating them as independent would inflate n by
the traversal length and make almost any difference "significant". The bootstrap
here resamples EPISODES, which is the unit that was actually randomised (one
seed per episode).

COLOUR. Okabe-Ito blue #0072B2 (road) and orange #E69F00 (junction) -- a
published colourblind-safe pair. The skill's validator is a node script and node
is not installed on this machine, so this pair is used precisely because its
CVD separation is established in the literature rather than asserted here.
Colour encodes road-vs-junction everywhere it appears and never anything else.
"""
import argparse
import json

import numpy as np

ROAD = "#0072B2"
JUNCTION = "#E69F00"
INK = "#1a1a1a"
INK_2 = "#555555"
MUTED = "#8a8a8a"
GRID = "#e4e4e4"


def bootstrap_paired(deltas, n_boot=10000, seed=0):
    """Mean of per-episode deltas, with a percentile CI resampling EPISODES."""
    rng = np.random.default_rng(seed)
    d = np.asarray(deltas, float)
    if len(d) == 0:
        return float("nan"), (float("nan"), float("nan"))
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    return float(d.mean()), (float(np.percentile(means, 2.5)),
                             float(np.percentile(means, 97.5)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="results/entropy_junctions.json")
    ap.add_argument("--out", default="results/entropy_junctions.png")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    with open(args.inp) as f:
        data = json.load(f)
    episodes = data["episodes"]

    # Per-episode paired means. An episode contributes to the comparison ONLY
    # if it has both kinds of step -- an episode that never entered a junction
    # has no within-episode contrast and would otherwise silently import a
    # between-episode difference into a "paired" number.
    paired, delta, sig_pairs = [], [], []
    for e in episodes:
        h_j = [s["entropy"] for s in e["steps"] if s["in_junction"]]
        h_r = [s["entropy"] for s in e["steps"] if not s["in_junction"]]
        if h_j and h_r:
            paired.append((e["episode"], float(np.mean(h_r)), float(np.mean(h_j)),
                           len(h_r), len(h_j)))
            delta.append(np.mean(h_j) - np.mean(h_r))
            s_j = np.array([s["sigma"] for s in e["steps"] if s["in_junction"]], float)
            s_r = np.array([s["sigma"] for s in e["steps"] if not s["in_junction"]], float)
            sig_pairs.append((s_r.mean(axis=0), s_j.mean(axis=0)))

    n_all = sum(len(e["steps"]) for e in episodes)
    n_j = sum(s["in_junction"] for e in episodes for s in e["steps"])
    print(f"[Plot] {len(episodes)} episodes, {n_all} steps, {n_j} in junction "
          f"({n_j / max(n_all, 1):.1%})")
    print(f"[Plot] {len(paired)} episodes have BOTH junction and road steps")

    # THREE is the floor, not one. A bootstrap that resamples a single episode
    # returns that episode's value every time, so the CI collapses to zero width
    # and prints as a precise result -- which is exactly what happened on the
    # checkpoint_175000 rollout: n=1 produced "-0.467 [-0.467, -0.467]", a
    # degenerate interval that reads as certainty and means nothing.
    if len(paired) < 3:
        print(f"[Plot] ABORT: only {len(paired)} episode(s) contain both junction "
              "and road steps; a bootstrap over that resamples the same episode "
              "and returns a zero-width CI that LOOKS certain. Nothing plotted.")
        return

    mean_d, (lo, hi) = bootstrap_paired(delta)
    verdict = ("HIGHER at junctions" if lo > 0 else
               "LOWER at junctions" if hi < 0 else
               "NO RELIABLE DIFFERENCE (CI spans zero)")
    print(f"[Plot] mean per-episode delta (junction - road) = {mean_d:+.4f} nats "
          f"[95% CI {lo:+.4f}, {hi:+.4f}]  -> {verdict}")

    # Episodes for panel A: the ones with the most junction steps, since an
    # episode with two junction steps shows nothing.
    top = sorted((e for e in episodes
                  if any(s["in_junction"] for s in e["steps"])),
                 key=lambda e: -sum(s["in_junction"] for s in e["steps"]))[:2]

    # ---- junction-entry-aligned average (the peri-event panel) --------------
    # A per-episode mean cannot tell "entropy is high THROUGHOUT the junction"
    # from "entropy spikes just BEFORE entering it" -- both raise the in-junction
    # mean. Aligning every entry at t=0 separates them, and the traces above
    # suggest the rise leads entry, which is a different claim than the one the
    # paired means test.
    PRE, POST = 40, 60
    aligned, onset_owner = [], []
    for e in episodes:
        h = np.array([s["entropy"] for s in e["steps"]], float)
        inj = np.array([s["in_junction"] for s in e["steps"]], bool)
        if not inj.any():
            continue
        d = np.diff(np.concatenate([[0], inj.view(np.int8)]))
        for onset in np.flatnonzero(d == 1):
            a, b = onset - PRE, onset + POST
            if a < 0 or b > len(h):
                continue          # partial windows would bias the edges
            aligned.append(h[a:b])
            onset_owner.append(e["episode"])
    aligned = np.array(aligned, float) if aligned else np.empty((0, PRE + POST))

    fig = plt.figure(figsize=(13, 8.5), facecolor="white")
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], hspace=0.42, wspace=0.28)

    # ---- A: entropy vs time -------------------------------------------------
    for i, e in enumerate(top):
        ax = fig.add_subplot(gs[0, i])
        h = [s["entropy"] for s in e["steps"]]
        t = [s["t"] for s in e["steps"]]
        inj = np.array([s["in_junction"] for s in e["steps"]], bool)

        # Contiguous junction runs as shaded spans, not per-point colouring:
        # a junction is an interval in time and drawing it as one says so.
        d = np.diff(np.concatenate([[0], inj.view(np.int8), [0]]))
        for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
            ax.axvspan(t[a], t[min(b, len(t) - 1)], color=JUNCTION,
                       alpha=0.20, linewidth=0)

        ax.plot(t, h, color=ROAD, linewidth=1.6, solid_capstyle="round")
        ax.set_title(f"Episode {e['episode']}  ({int(inj.sum())} junction steps)",
                     fontsize=10, color=INK, pad=8)
        ax.set_xlabel("Step", fontsize=9, color=INK_2)
        if i == 0:
            ax.set_ylabel("Policy entropy  (nats, base Gaussian)",
                          fontsize=9, color=INK_2)
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(MUTED)
        ax.tick_params(labelsize=8, colors=INK_2)

    # ---- A2: junction-entry-aligned average ---------------------------------
    ax = fig.add_subplot(gs[0, 2])
    if len(aligned):
        tt = np.arange(-PRE, POST)
        mu = aligned.mean(axis=0)
        # Spread over ONSETS, shown as a band. Onsets within one episode are not
        # independent, so this is a visual spread and is deliberately not
        # called a confidence interval -- the CI in the verdict panel is the
        # one clustered by episode.
        sd = aligned.std(axis=0)
        ax.axvspan(0, POST, color=JUNCTION, alpha=0.20, linewidth=0)
        ax.fill_between(tt, mu - sd, mu + sd, color=ROAD, alpha=0.16,
                        linewidth=0)
        ax.plot(tt, mu, color=ROAD, linewidth=2.0, solid_capstyle="round")
        ax.axvline(0, color=INK_2, linewidth=1.0, linestyle=(0, (4, 3)))
        ax.text(1, ax.get_ylim()[1], " junction entry", fontsize=8,
                color=INK_2, va="top")
        ax.set_title(f"Aligned on junction entry  (n={len(aligned)} entries)",
                     fontsize=10, color=INK, pad=8)
        ax.set_xlabel("Steps from entry", fontsize=9, color=INK_2)
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "no complete junction-entry windows",
                ha="center", va="center", fontsize=9, color=MUTED,
                transform=ax.transAxes)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(MUTED)
    ax.tick_params(labelsize=8, colors=INK_2)

    # ---- B: paired per-episode means ----------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    for (ep, r, j, nr, nj) in paired:
        ax.plot([0, 1], [r, j], color=MUTED, linewidth=1.0, zorder=1)
        ax.scatter([0], [r], s=42, color=ROAD, zorder=3, edgecolor="white",
                   linewidth=1.2)
        ax.scatter([1], [j], s=42, color=JUNCTION, zorder=3, edgecolor="white",
                   linewidth=1.2)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["On road", "In junction"], fontsize=9, color=INK_2)
    ax.set_xlim(-0.35, 1.35)
    ax.set_ylabel("Mean entropy (nats)", fontsize=9, color=INK_2)
    ax.set_title(f"Per-episode paired means  (n={len(paired)} episodes)",
                 fontsize=10, color=INK, pad=8)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(MUTED)
    ax.tick_params(labelsize=8, colors=INK_2)

    # ---- C: per-dimension sigma ---------------------------------------------
    names = ["steer", "throttle"]
    r_mat = np.array([p[0] for p in sig_pairs], float)
    j_mat = np.array([p[1] for p in sig_pairs], float)
    ax = fig.add_subplot(gs[1, 1])
    x = np.arange(len(names))
    w = 0.34
    # 2px surface gap between adjacent bars is the `w` gap plus the white edge.
    ax.bar(x - w / 2 - 0.01, r_mat.mean(axis=0), w, color=ROAD, label="On road",
           edgecolor="white", linewidth=1.5)
    ax.bar(x + w / 2 + 0.01, j_mat.mean(axis=0), w, color=JUNCTION,
           label="In junction", edgecolor="white", linewidth=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9, color=INK_2)
    ax.set_ylabel("Mean sigma (pre-tanh)", fontsize=9, color=INK_2)
    ax.set_title("Which control widens", fontsize=10, color=INK, pad=8)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(MUTED)
    ax.tick_params(labelsize=8, colors=INK_2)

    # ---- verdict panel ------------------------------------------------------
    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    ax.text(0, 0.92, "junction - road", fontsize=10, color=INK_2,
            transform=ax.transAxes)
    ax.text(0, 0.70, f"{mean_d:+.3f}", fontsize=30, color=INK, weight="bold",
            transform=ax.transAxes)
    ax.text(0, 0.58, "nats, mean of per-episode differences", fontsize=8.5,
            color=MUTED, transform=ax.transAxes)
    ax.text(0, 0.42, f"95% CI  [{lo:+.3f}, {hi:+.3f}]", fontsize=9.5,
            color=INK_2, transform=ax.transAxes)
    ax.text(0, 0.30, f"bootstrap over {len(paired)} episodes", fontsize=8.5,
            color=MUTED, transform=ax.transAxes)
    ax.text(0, 0.12, verdict, fontsize=10.5, color=INK, weight="bold",
            transform=ax.transAxes, wrap=True)

    handles = [Patch(facecolor=ROAD, label="On road"),
               Patch(facecolor=JUNCTION, label="In junction"),
               Line2D([0], [0], color=ROAD, linewidth=1.6, label="Entropy")]
    fig.legend(handles=handles, loc="upper right", frameon=False, fontsize=9,
               bbox_to_anchor=(0.995, 0.995), labelcolor=INK_2)

    step = data.get("step")
    band = data.get("goal_band", ["?", "?"])
    fig.suptitle(f"Policy entropy at junctions -- {data.get('town')}, "
                 f"checkpoint {step}, goals {band[0]:g}-{band[1]:g} m",
                 fontsize=13, color=INK, x=0.012, ha="left", y=0.985)

    fig.savefig(args.out, dpi=160, bbox_inches="tight", facecolor="white")
    print(f"[Plot] -> {args.out}")


if __name__ == "__main__":
    main()
