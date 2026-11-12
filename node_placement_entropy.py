"""Do entropy peaks mark good places to put topological-graph nodes?

    python node_placement_entropy.py --in results/entropy_nodes.json \
        --out-json results/node_placement.json --out-png results/node_placement.png
    python node_placement_entropy.py --self-test      # no CARLA, no data needed

THE CLAIM BEING TESTED. A topological map needs nodes where the route is
DECIDABLE -- junctions -- and wastes them on featureless straight road. If the
policy's own entropy rises where the world is ambiguous, then entropy peaks are a
free node-placement heuristic: no map, no junction oracle, just the policy.

THE COMPARISON IS AGAINST AN EQUAL-BUDGET BASELINE, WHICH IS THE ONLY FAIR ONE.
Entropy placement with more nodes would trivially cover more junctions, so every
arm places EXACTLY the same number of nodes on EXACTLY the same trajectory:

    peaks     local maxima of smoothed entropy, min-separated
    uniform   equal arc-length spacing            <- the honest baseline
    random    uniform random along the path       <- the floor

If `peaks` does not beat `uniform`, the heuristic is worthless however good the
entropy trace looks by eye, because uniform spacing needs no policy at all.

WHY MIN-SEPARATION IS NOT A FUDGE. Entropy is autocorrelated over tens of steps,
so raw local maxima arrive in clusters of near-duplicates around one event. Nodes
a metre apart are not two nodes in any useful sense. Enforcing a minimum spacing
is what makes the arm's node COUNT meaningful, and the uniform arm satisfies the
same constraint by construction -- so this equalises the arms rather than
favouring one.

GROUND TRUTH is the env's own `in_junction` flag, not a geometric heuristic. Each
contiguous run of in_junction steps is ONE junction event, located at the run's
centroid -- runs, not steps, or a slow traversal would count as dozens of
junctions and coverage would mostly measure how slowly the car drove.

THE STATISTIC IS CLUSTERED BY EPISODE. Junction events inside one episode share a
route and a rollout, so they are not independent; the bootstrap resamples
EPISODES. Same correction the entropy-vs-junction analysis needed, same reason.
"""
import argparse
import json

import numpy as np

PEAKS = "#0072B2"      # Okabe-Ito blue
UNIFORM = "#E69F00"    # Okabe-Ito orange
RANDOM = "#999999"
JUNC = "#009E73"       # Okabe-Ito bluish green
INK, INK_2, MUTED, GRID = "#1a1a1a", "#555555", "#8a8a8a", "#e4e4e4"


def smooth(x, w):
    """Centred moving average. Odd window; edges shrink rather than pad.

    Padding with the edge value would invent a plateau at both ends of every
    episode, and episode starts already carry a large entropy transient -- that
    plateau would become a spurious peak in exactly the place the policy is
    simply initialising.
    """
    x = np.asarray(x, float)
    if w <= 1 or len(x) < 3:
        return x.copy()
    w = int(w) | 1
    half, out = w // 2, np.empty_like(x)
    for i in range(len(x)):
        a, b = max(0, i - half), min(len(x), i + half + 1)
        out[i] = x[a:b].mean()
    return out


def arclength(xy):
    """Cumulative path length in metres, same length as xy."""
    d = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(d)])


def place_peaks(h, s, n_max, min_sep_m, smooth_w):
    """Local maxima of smoothed entropy, greedily taken highest-first.

    Greedy-by-height with a separation veto, rather than "every local max then
    thin": thinning left-to-right keeps whichever peak happens to come first,
    which is an arbitrary tiebreak that would make the arm depend on travel
    direction.
    """
    hs = smooth(h, smooth_w)
    order = np.argsort(-hs)
    picked = []
    for i in order:
        if len(picked) >= n_max:
            break
        if all(abs(s[i] - s[j]) >= min_sep_m for j in picked):
            picked.append(int(i))
    return sorted(picked)


def mad_spikes(h, s, min_sep_m, smooth_w, z_thresh, centre="median",
               nms_w=0):
    """Robust-z spike detection, then the same min-separation greedy pick.

    z = (h - median) / (1.4826 * MAD). Median and MAD rather than mean and std
    because BOTH of the latter are inflated by the very spikes being detected --
    and this trace has a large episode-start transient (~1.8 nats decaying to
    ~0.85 within 20 steps) that would raise a mean/std threshold for the whole
    episode. MAD has a 50% breakdown point, so those excursions do not move the
    baseline.

    1.4826 is the Gaussian consistency constant: it makes MAD estimate sigma for
    normal data, so z_thresh reads on the familiar sigma scale.

    UNLIKE the top-N criterion, this CAN return nothing. That is the point -- a
    detector that always fills a budget cannot say "no spike here". Episodes
    where it fires zero times are reported rather than silently dropped.
    """
    hs = smooth(h, smooth_w)
    if centre == "mean":
        # NON-ROBUST variant, kept only so the ablation can show what it costs.
        # Mean and std are both inflated by the very spikes being detected, and
        # by the episode-start transient, so the threshold rises with the signal.
        c, scale = float(np.mean(hs)), float(np.std(hs))
    else:
        c = float(np.median(hs))
        scale = 1.4826 * float(np.median(np.abs(hs - c)))
    if scale <= 1e-12:
        return []                      # degenerate: a flat trace has no spikes
    z = (hs - c) / scale
    cand = np.flatnonzero(z >= z_thresh)
    if len(cand) == 0:
        return []
    if nms_w > 0:
        # NON-MAXIMAL SUPPRESSION, in the strict sense: a candidate survives only
        # if it is the maximum of its own +/- nms_w window.
        #
        # WHY THIS IS NOT ALREADY DONE BY THE SEPARATION RULE. The greedy pick
        # below vetoes a candidate that is too CLOSE to an accepted node, but it
        # never asks whether the candidate is a peak at all. A point on the
        # rising flank of a spike clears the threshold, and if the spike's true
        # apex has already claimed its exclusion zone the flank point can still
        # be accepted once the walk moves past that zone -- so a single broad
        # spike can yield a node at its top and another partway down its side.
        # NMS removes flank points before the separation rule ever sees them.
        #
        # The two are complementary, not redundant: NMS is a LOCAL shape test in
        # steps, separation is a GLOBAL spacing test in metres.
        keep = []
        for i in cand:
            a, b = max(0, i - nms_w), min(len(hs), i + nms_w + 1)
            if hs[i] >= hs[a:b].max():
                keep.append(i)
        cand = np.array(keep, dtype=int)
        if len(cand) == 0:
            return []
    # Greedy by height among candidates only, same separation rule as top-N, so
    # the two detectors differ ONLY in which steps are eligible.
    picked = []
    for i in cand[np.argsort(-hs[cand])]:
        if all(abs(s[i] - s[j]) >= min_sep_m for j in picked):
            picked.append(int(i))
    return sorted(picked)


def place_uniform(s, n):
    """n nodes at equal arc-length spacing, at the interval midpoints.

    Midpoints, not endpoints: endpoints would pin a node at s=0 on every
    episode, which is the reset pose and not a place anyone would choose.
    """
    if n <= 0:
        return []
    edges = np.linspace(s[0], s[-1], n + 1)
    mids = (edges[:-1] + edges[1:]) / 2
    return [int(np.argmin(np.abs(s - m))) for m in mids]


def place_random(s, n, rng):
    if n <= 0:
        return []
    targets = rng.uniform(s[0], s[-1], size=n)
    return sorted({int(np.argmin(np.abs(s - t))) for t in targets})


def junction_events(inj, xy):
    """Centroid of each contiguous in_junction run -> (E, 2)."""
    inj = np.asarray(inj, bool)
    if not inj.any():
        return np.empty((0, 2))
    d = np.diff(np.concatenate([[0], inj.view(np.int8), [0]]))
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return np.array([xy[a:b].mean(axis=0) for a, b in zip(starts, ends)])


def coverage(nodes_xy, events_xy, radii):
    """Distance from each junction event to its nearest node."""
    if len(events_xy) == 0:
        return None
    if len(nodes_xy) == 0:
        d = np.full(len(events_xy), np.inf)
    else:
        d = np.linalg.norm(events_xy[:, None, :] - nodes_xy[None, :, :],
                           axis=2).min(axis=1)
    return {"dists": d,
            "median": float(np.median(d)),
            **{f"within_{r}m": float((d <= r).mean()) for r in radii}}


def analyse(episodes, n_nodes_per_100m, min_sep_m, smooth_w, radii, seed=0,
            detector="topn", z_thresh=3.0):
    rng = np.random.default_rng(seed)
    per_ep, arms = [], None
    n_no_spike = [0]

    for e in episodes:
        st = e["steps"]
        if any(s.get("x") is None for s in st) or len(st) < 10:
            continue
        xy = np.array([[s["x"], s["y"]] for s in st], float)
        h = np.array([s["entropy"] for s in st], float)
        inj = np.array([s["in_junction"] for s in st], bool)
        s = arclength(xy)
        if s[-1] < 2 * min_sep_m:
            continue

        events = junction_events(inj, xy)
        if len(events) == 0:
            continue

        # BUDGET IS SET BY PATH LENGTH, not by how many peaks exist -- so the
        # arms are equal-budget by construction and the budget cannot be tuned
        # per-arm after seeing the result.
        if detector == "mad":
            # MAD DECIDES THE COUNT, and the baselines are then matched TO it.
            # Equal-budget comparison is preserved -- the budget is just set by
            # the detector instead of by path length. Without this the entropy
            # arm would be compared against a different number of nodes, and
            # node count is known to dominate the placement criterion.
            peaks_idx = mad_spikes(h, s, min_sep_m, smooth_w, z_thresh)
            n = len(peaks_idx)
            if n == 0:
                n_no_spike[0] += 1
                continue
        else:
            n = max(1, int(round(s[-1] / 100.0 * n_nodes_per_100m)))
            peaks_idx = place_peaks(h, s, n, min_sep_m, smooth_w)

        idx = {
            "peaks": peaks_idx,
            "uniform": place_uniform(s, n),
            "random": place_random(s, n, rng),
        }
        if arms is None:
            arms = tuple(idx.keys())
        rec = {"episode": e["episode"], "path_m": float(s[-1]),
               "n_junctions": int(len(events)), "budget": n,
               "n_placed": {a: len(idx[a]) for a in arms}, "arms": {}}
        for a in arms:
            c = coverage(xy[idx[a]], events, radii)
            rec["arms"][a] = {k: v for k, v in c.items() if k != "dists"}
            rec["arms"][a]["nodes_xy"] = xy[idx[a]].tolist()
        rec["events_xy"] = events.tolist()
        rec["xy"] = xy[::4].tolist()          # decimated, for the map panel only
        per_ep.append(rec)

    if detector == "mad":
        print(f"[Nodes] MAD found no spike in {n_no_spike[0]} episode(s) "
              f"-- those are excluded, not counted as zero coverage")
    return per_ep, (arms or ("peaks", "uniform", "random"))


def pareto(episodes, sweep, min_sep_m, smooth_w, radii, seed=0):
    """Count-vs-quality frontier per arm.

    WHY A FRONTIER RATHER THAN A MATCHED POINT. If node COUNT dominates the
    placement criterion -- which is what the fixed-budget result and the peer's
    2-to-64 sweep both suggest -- then any single matched count is one arbitrary
    slice through the thing that actually drives the metric. Two arms can tie at
    n=8 and separate at n=30. The frontier asks the real question: at every
    budget, does one criterion buy more coverage per node than another?

    SEPARATION MUST SHRINK WITH THE BUDGET, or the comparison silently becomes
    unfair at the dense end. A fixed 8 m floor caps the peaks arm at roughly
    path/8 nodes however large the budget, while `uniform` keeps subdividing --
    so peaks would appear to plateau for a reason that is the constraint, not
    the criterion. Here the floor is min(min_sep_m, half the uniform spacing at
    that budget), which is the tightest rule that both arms can satisfy.
    """
    rng = np.random.default_rng(seed)
    arms = ("peaks", "uniform", "random")
    out = {a: {"n": [], "median": [], **{f"within_{r}m": [] for r in radii}}
           for a in arms}

    prepared = []
    for e in episodes:
        st = e["steps"]
        if any(q.get("x") is None for q in st) or len(st) < 10:
            continue
        xy = np.array([[q["x"], q["y"]] for q in st], float)
        inj = np.array([q["in_junction"] for q in st], bool)
        ev = junction_events(inj, xy)
        if len(ev) == 0:
            continue
        prepared.append((xy, np.array([q["entropy"] for q in st], float),
                         arclength(xy), ev))
    if not prepared:
        return arms, out, 0

    for dens in sweep:
        acc = {a: {"median": [], **{f"within_{r}m": [] for r in radii}}
               for a in arms}
        counts = []
        for xy, h, s_, ev in prepared:
            n = max(1, int(round(s_[-1] / 100.0 * dens)))
            sep = min(min_sep_m, 0.5 * s_[-1] / n)
            idx = {"peaks": place_peaks(h, s_, n, sep, smooth_w),
                   "uniform": place_uniform(s_, n),
                   "random": place_random(s_, n, rng)}
            counts.append(np.mean([len(idx[a]) for a in arms]))
            for a in arms:
                c = coverage(xy[idx[a]], ev, radii)
                acc[a]["median"].append(c["median"])
                for r in radii:
                    acc[a][f"within_{r}m"].append(c[f"within_{r}m"])
        for a in arms:
            out[a]["n"].append(float(np.mean(counts)))
            out[a]["median"].append(float(np.mean(acc[a]["median"])))
            for r in radii:
                out[a][f"within_{r}m"].append(float(np.mean(acc[a][f"within_{r}m"])))
    return arms, out, len(prepared)


def plot_pareto(arms, out, radii, out_png, title, n_eps):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"peaks": PEAKS, "uniform": UNIFORM, "random": RANDOM}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), facecolor="white")

    ax = axes[0]
    for a in arms:
        ax.plot(out[a]["n"], out[a]["median"], color=colors[a], linewidth=2.0,
                marker="o", markersize=4, label=a, markeredgecolor="white",
                markeredgewidth=0.8)
    ax.set_xlabel("Mean nodes per episode", fontsize=9, color=INK_2)
    ax.set_ylabel("Median junction->node distance (m)", fontsize=9, color=INK_2)
    ax.set_title("Lower is better", fontsize=10, color=INK, pad=8)
    ax.set_xscale("log")

    ax = axes[1]
    R = radii[-1]
    for a in arms:
        ax.plot(out[a]["n"], out[a][f"within_{R}m"], color=colors[a],
                linewidth=2.0, marker="o", markersize=4, label=a,
                markeredgecolor="white", markeredgewidth=0.8)
    ax.set_xlabel("Mean nodes per episode", fontsize=9, color=INK_2)
    ax.set_ylabel(f"Fraction of junctions within {R} m", fontsize=9, color=INK_2)
    ax.set_title("Higher is better", fontsize=10, color=INK, pad=8)
    ax.set_xscale("log")

    for ax in axes:
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(MUTED)
        ax.tick_params(labelsize=8, colors=INK_2)
        ax.legend(frameon=False, fontsize=9, labelcolor=INK_2)

    fig.suptitle(f"{title}  (n={n_eps} episodes)", fontsize=12, color=INK,
                 x=0.012, ha="left", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_png, dpi=160, bbox_inches="tight", facecolor="white")
    print(f"[Pareto] -> {out_png}")


def boot_paired(a_vals, b_vals, n_boot=10000, seed=0):
    """Mean of per-episode (a - b), CI resampling EPISODES."""
    rng = np.random.default_rng(seed)
    d = np.asarray(a_vals, float) - np.asarray(b_vals, float)
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return float("nan"), (float("nan"), float("nan"))
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    m = d[idx].mean(axis=1)
    return float(d.mean()), (float(np.percentile(m, 2.5)),
                             float(np.percentile(m, 97.5)))


def self_test():
    """Synthetic world with a KNOWN answer, so the machinery is falsifiable.

    A straight path with three junction patches. Entropy is flat plus a bump on
    each patch. If `peaks` does not beat `uniform` HERE, the implementation is
    broken -- this is the case the heuristic is supposed to win, by construction.
    """
    rng = np.random.default_rng(0)
    eps = []
    for k in range(12):
        n = 400
        x = np.linspace(0, 200, n)
        y = np.zeros(n)
        h = 0.85 + rng.normal(0, 0.02, n)
        inj = np.zeros(n, bool)
        for c in rng.choice(np.arange(40, n - 40), size=3, replace=False):
            inj[c - 12:c + 12] = True
            h[c - 18:c + 18] += 0.35 * np.hanning(36)
        eps.append({"episode": k, "steps": [
            {"x": float(x[i]), "y": float(y[i]), "entropy": float(h[i]),
             "in_junction": bool(inj[i])} for i in range(n)]})

    per_ep, arms = analyse(eps, 4.0, 8.0, 9, (5, 10))
    print(f"[self-test] {len(per_ep)} synthetic episodes")
    for a in arms:
        med = np.mean([r["arms"][a]["median"] for r in per_ep])
        w10 = np.mean([r["arms"][a]["within_10m"] for r in per_ep])
        print(f"  {a:8s} median dist {med:6.2f} m   within 10 m {w10:.1%}")
    d, (lo, hi) = boot_paired([r["arms"]["uniform"]["median"] for r in per_ep],
                              [r["arms"]["peaks"]["median"] for r in per_ep])
    print(f"  uniform - peaks = {d:+.2f} m [{lo:+.2f}, {hi:+.2f}]")
    ok = lo > 0
    print(f"[self-test] {'PASS' if ok else 'FAIL'} -- peaks "
          f"{'beat' if ok else 'did NOT beat'} uniform on the planted signal")
    return 0 if ok else 1


def plot(per_ep, arms, out_png, radii, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    colors = {"peaks": PEAKS, "uniform": UNIFORM, "random": RANDOM}
    fig = plt.figure(figsize=(13, 7.5), facecolor="white")
    gs = fig.add_gridspec(2, 3, hspace=0.38, wspace=0.30)

    # --- map panels: the two most junction-rich episodes ---------------------
    top = sorted(per_ep, key=lambda r: -r["n_junctions"])[:2]
    for i, r in enumerate(top):
        ax = fig.add_subplot(gs[0, i])
        xy = np.array(r["xy"])
        ax.plot(xy[:, 0], xy[:, 1], color=MUTED, linewidth=1.0, zorder=1)
        ev = np.array(r["events_xy"])
        ax.scatter(ev[:, 0], ev[:, 1], s=150, facecolor="none",
                   edgecolor=JUNC, linewidth=2.0, zorder=2, label="Junction")
        for a in ("peaks", "uniform"):
            nd = np.array(r["arms"][a]["nodes_xy"])
            if len(nd):
                ax.scatter(nd[:, 0], nd[:, 1], s=30, color=colors[a], zorder=3,
                           edgecolor="white", linewidth=0.8)
        ax.set_title(f"Episode {r['episode']} -- {r['n_junctions']} junctions, "
                     f"{r['budget']} nodes", fontsize=9.5, color=INK, pad=6)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("X (m)", fontsize=8.5, color=INK_2)
        if i == 0:
            ax.set_ylabel("Y (m)", fontsize=8.5, color=INK_2)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(labelsize=7.5, colors=INK_2)
        ax.grid(True, color=GRID, linewidth=0.7)
        ax.set_axisbelow(True)

    # --- median distance per arm ---------------------------------------------
    ax = fig.add_subplot(gs[0, 2])
    meds = {a: [r["arms"][a]["median"] for r in per_ep] for a in arms}
    for j, a in enumerate(arms):
        ax.scatter(np.full(len(meds[a]), j) + rng_jitter(len(meds[a])),
                   meds[a], s=22, color=colors[a], alpha=0.65,
                   edgecolor="white", linewidth=0.6)
        ax.plot([j - 0.25, j + 0.25], [np.mean(meds[a])] * 2, color=INK,
                linewidth=2.0, zorder=4)
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels(arms, fontsize=9, color=INK_2)
    ax.set_ylabel("Median junction->node distance (m)", fontsize=8.5, color=INK_2)
    ax.set_title("Lower is better", fontsize=9.5, color=INK, pad=6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=7.5, colors=INK_2)
    ax.grid(True, axis="y", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)

    # --- coverage within R ----------------------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    w = 0.26
    xs = np.arange(len(radii))
    for j, a in enumerate(arms):
        vals = [np.mean([r["arms"][a][f"within_{R}m"] for r in per_ep])
                for R in radii]
        ax.bar(xs + (j - 1) * (w + 0.02), vals, w, color=colors[a], label=a,
               edgecolor="white", linewidth=1.4)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"within {R} m" for R in radii], fontsize=9, color=INK_2)
    ax.set_ylabel("Fraction of junctions covered", fontsize=8.5, color=INK_2)
    ax.set_title("Junction coverage", fontsize=9.5, color=INK, pad=6)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_2)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=7.5, colors=INK_2)
    ax.grid(True, axis="y", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)

    # --- paired peaks vs uniform ---------------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    for r in per_ep:
        ax.plot([0, 1], [r["arms"]["uniform"]["median"],
                         r["arms"]["peaks"]["median"]],
                color=MUTED, linewidth=0.9, zorder=1)
    ax.scatter([0] * len(per_ep), [r["arms"]["uniform"]["median"] for r in per_ep],
               s=30, color=UNIFORM, zorder=3, edgecolor="white", linewidth=0.8)
    ax.scatter([1] * len(per_ep), [r["arms"]["peaks"]["median"] for r in per_ep],
               s=30, color=PEAKS, zorder=3, edgecolor="white", linewidth=0.8)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Uniform", "Peaks"], fontsize=9, color=INK_2)
    ax.set_xlim(-0.35, 1.35)
    ax.set_ylabel("Median distance (m)", fontsize=8.5, color=INK_2)
    ax.set_title(f"Paired per episode (n={len(per_ep)})", fontsize=9.5,
                 color=INK, pad=6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=7.5, colors=INK_2)
    ax.grid(True, axis="y", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)

    # --- verdict --------------------------------------------------------------
    d, (lo, hi) = boot_paired([r["arms"]["uniform"]["median"] for r in per_ep],
                              [r["arms"]["peaks"]["median"] for r in per_ep])
    verdict = ("ENTROPY BEATS UNIFORM" if lo > 0 else
               "UNIFORM BEATS ENTROPY" if hi < 0 else
               "NO RELIABLE DIFFERENCE")
    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    ax.text(0, 0.90, "uniform - peaks, median distance", fontsize=9, color=INK_2,
            transform=ax.transAxes)
    ax.text(0, 0.64, f"{d:+.2f} m", fontsize=28, color=INK, weight="bold",
            transform=ax.transAxes)
    ax.text(0, 0.50, f"95% CI  [{lo:+.2f}, {hi:+.2f}]", fontsize=9.5,
            color=INK_2, transform=ax.transAxes)
    ax.text(0, 0.40, f"bootstrap over {len(per_ep)} episodes", fontsize=8.5,
            color=MUTED, transform=ax.transAxes)
    ax.text(0, 0.20, verdict, fontsize=11, color=INK, weight="bold",
            transform=ax.transAxes)
    ax.text(0, 0.08, "positive = entropy nodes land closer to junctions",
            fontsize=8, color=MUTED, transform=ax.transAxes)

    handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=c,
                      markersize=8, label=a) for a, c in colors.items()]
    handles.append(Line2D([0], [0], marker="o", color="none",
                          markerfacecolor="none", markeredgecolor=JUNC,
                          markeredgewidth=2, markersize=11, label="Junction"))
    fig.legend(handles=handles, loc="upper right", frameon=False, fontsize=9,
               bbox_to_anchor=(0.995, 0.995), labelcolor=INK_2)
    fig.suptitle(title, fontsize=12.5, color=INK, x=0.012, ha="left", y=0.985)
    fig.savefig(out_png, dpi=160, bbox_inches="tight", facecolor="white")
    print(f"[Nodes] -> {out_png}")
    return d, lo, hi, verdict


_JIT = np.random.default_rng(7)


def rng_jitter(n):
    return _JIT.uniform(-0.06, 0.06, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="results/entropy_nodes.json")
    ap.add_argument("--out-json", default="results/node_placement.json")
    ap.add_argument("--out-png", default="results/node_placement.png")
    ap.add_argument("--nodes-per-100m", type=float, default=4.0)
    ap.add_argument("--min-sep-m", type=float, default=8.0)
    ap.add_argument("--smooth", type=int, default=9)
    ap.add_argument("--detector", choices=("topn", "mad"), default="topn",
                    help="topn: rank all steps, take budget. mad: robust-z "
                         "threshold, count floats and baselines match it.")
    ap.add_argument("--z", type=float, default=3.0, help="MAD z threshold")
    ap.add_argument("--pareto", action="store_true",
                    help="sweep node count and plot the frontier per arm")
    ap.add_argument("--sweep", default="0.5,1,2,4,8,16,32,64",
                    help="nodes per 100 m values for --pareto")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        raise SystemExit(self_test())

    with open(args.inp) as f:
        data = json.load(f)
    radii = (5, 10)

    if args.pareto:
        sweep = [float(v) for v in args.sweep.split(",")]
        arms, out, n_eps = pareto(data["episodes"], sweep, args.min_sep_m,
                                  args.smooth, radii)
        if n_eps == 0:
            print("[Pareto] ABORT: no usable episodes.")
            return
        hdr = "  ".join(f"{a:>9s}" for a in arms)
        print(f"[Pareto] {n_eps} episodes\n  nodes/ep   {hdr}   (median m)")
        for i in range(len(out[arms[0]]["n"])):
            row = "  ".join(f"{out[a]['median'][i]:9.2f}" for a in arms)
            print(f"  {out[arms[0]]['n'][i]:8.1f}   {row}")
        print(f"  nodes/ep   {hdr}   (within {radii[-1]} m)")
        for i in range(len(out[arms[0]]["n"])):
            row = "  ".join(f"{out[a][f'within_{radii[-1]}m'][i]:9.1%}" for a in arms)
            print(f"  {out[arms[0]]['n'][i]:8.1f}   {row}")
        plot_pareto(arms, out, radii,
                    args.out_png.replace(".png", "_pareto.png"),
                    f"Node-count frontier -- {data.get('town')}, "
                    f"checkpoint {data.get('step')}", n_eps)
        return
    per_ep, arms = analyse(data["episodes"], args.nodes_per_100m,
                           args.min_sep_m, args.smooth, radii,
                           detector=args.detector, z_thresh=args.z)
    print(f"[Nodes] {len(per_ep)} usable episodes "
          f"(need pose, >=1 junction, path >= {2 * args.min_sep_m:.0f} m)")
    if len(per_ep) < 3:
        print("[Nodes] ABORT: too few usable episodes to compare arms.")
        return
    for a in arms:
        med = np.mean([r["arms"][a]["median"] for r in per_ep])
        print(f"  {a:8s} median {med:6.2f} m  " + "  ".join(
            f"within{R}m={np.mean([r['arms'][a][f'within_{R}m'] for r in per_ep]):.1%}"
            for R in radii))

    title = (f"Node placement by policy entropy -- {data.get('town')}, "
             f"checkpoint {data.get('step')}")
    d, lo, hi, verdict = plot(per_ep, arms, args.out_png, radii, title)
    print(f"[Nodes] uniform - peaks = {d:+.2f} m [{lo:+.2f}, {hi:+.2f}] -> {verdict}")

    with open(args.out_json, "w") as f:
        json.dump({"source": args.inp, "verdict": verdict,
                   "delta_m": d, "ci": [lo, hi],
                   "nodes_per_100m": args.nodes_per_100m,
                   "min_sep_m": args.min_sep_m,
                   "episodes": [{k: v for k, v in r.items() if k != "xy"}
                                for r in per_ep]}, f)
    print(f"[Nodes] -> {args.out_json}")


if __name__ == "__main__":
    main()
