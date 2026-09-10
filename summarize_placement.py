#!/usr/bin/env python
"""Print the node-placement table from per-method result files.

A SEPARATE FILE RATHER THAN A HEREDOC in the driver, so it can be tested
without a simulator, a GPU or a checkpoint -- the same reason
summarize_four_way.py is one. The table is the deliverable; formatting it
somewhere nothing can exercise is how a comparison ends up with an n/a printed
as 0.000.
"""
import glob
import json
import os
import sys

# Presentation order: the two policy-free baselines first, then the two methods
# that consult the policy. Reading down the column is then reading "what does
# consulting the policy buy", which is the question.
ORDER = {"uniform": 0, "random": 1, "policy": 2, "sgm": 3}

HEADER = (f"{'method':<10} {'nodes':>6} {'edges':>7} {'routed':>8} "
          f"{'SR|rtd':>7} {'SPL|rtd':>8} {'SR|byp':>7} {'leg_TO':>7}")

FOOTNOTES = [
    "EQUAL BUDGET: every arm places the number of nodes SGM derived for",
    "  itself. An arm with more nodes trivially covers more of the world, so a",
    "  comparison at unequal counts measures sparsity rather than placement.",
    "  `nodes` below `budget` means snapping collided, not that the arm cheated.",
    "SPL is n/a, NOT 0.000, when no route planner was available: a zero reads",
    "  as 'took an infinite detour' rather than 'was never measured'.",
    "unroutable counts episodes the GRAPH could not offer a path for -- start",
    "  or goal off the map, or no connected route. Those episodes are still",
    "  DRIVEN, with the true goal only, because excluding them would score a",
    "  graph on the subset it happened to cover and flatter a sparse map.",
    "leg_TO counts subgoal legs abandoned on a step budget. A leg the policy",
    "  cannot drive is an edge the graph should not have offered, which is",
    "  itself a placement result.",
]

UNROUTABLE = ("no_path", "start_unlocalised", "goal_unlocalised",
              "node_missing", "empty_graph")


def load(results_dir):
    rows = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        with open(path) as fh:
            rows.append(json.load(fh))
    rows.sort(key=lambda r: ORDER.get(r.get("method", ""), 99))
    return rows


def _band_key(band):
    try:
        return float(str(band).split("-")[0])
    except (ValueError, IndexError):
        return float("inf")


def _band_upper(band):
    """The band's UPPER bound. Training is pinned to 5-20 m, so it is the top
    of the band, not the bottom, that decides whether a row is extrapolation."""
    try:
        return float(str(band).split("-")[1].rstrip("m"))
    except (ValueError, IndexError):
        return float("inf")


def format_row(r):
    """One sweep point. SR and SPL are ROUTED-ONLY.

    `routed` must be read alongside them, and is placed before them for that
    reason: a graph that routes 3 episodes of 20 is scored on those 3, so a
    sparse or disconnected map can post a strong SR while being useless. SR|byp
    is the unaided policy on the episodes this graph could NOT route -- a
    control, not a headline, and it should be broadly similar across methods.
    """
    spl = "n/a" if r.get("spl") is None else f"{r['spl']:.3f}"
    sr = "n/a" if r.get("success") is None else f"{r['success']:.3f}"
    byp = ("n/a" if r.get("success_bypassed") is None
           else f"{r['success_bypassed']:.3f}")
    routed = f"{r.get('n_routed', 0)}/{r.get('episodes', 0)}"
    return (f"{r.get('method', '?'):<10} {r.get('n_nodes', 0):>6} "
            f"{r.get('n_edges', 0):>7} {routed:>8} "
            f"{sr:>7} {spl:>8} {byp:>7} {r.get('leg_timeouts', 0):>7}")


def pareto_front(rows):
    """Indices of the non-dominated rows in (nodes DOWN, SR UP, SPL UP).

    THE QUESTION IS NOT "WHICH METHOD WINS AT n NODES" but what the frontier of
    cost against performance looks like, so a point matters when nothing else
    is at least as good on every axis while costing no more nodes. A method can
    own part of the frontier and be dominated everywhere else, which a
    fixed-budget table cannot express at all.

    SPL is compared only where BOTH points have it; a missing SPL never counts
    as better and never as worse, so an unmeasured arm cannot dominate.
    """
    front = []
    for i, a in enumerate(rows):
        dominated = False
        for j, b in enumerate(rows):
            if i == j:
                continue
            if b.get("n_nodes", 0) > a.get("n_nodes", 0):
                continue
            # An UNMEASURED point (nothing routed) can neither dominate nor be
            # dominated on success; it simply has no value on that axis.
            sra, srb = a.get("success"), b.get("success")
            if sra is not None and (srb is None or srb < sra):
                continue
            sa, sb = a.get("spl"), b.get("spl")
            if sa is not None and sb is not None and sb < sa:
                continue
            strictly = (b.get("n_nodes", 0) < a.get("n_nodes", 0)
                        or (sra is not None and srb is not None and srb > sra)
                        or (sa is not None and sb is not None and sb > sa))
            if strictly:
                dominated = True
                break
        if not dominated:
            front.append(i)
    return front


def render(rows):
    """One table per band. Pooling bands would average an in-distribution
    result with an extrapolation one and describe neither."""
    if not rows:
        return "no results found"

    bands = sorted({r.get("band", "default") for r in rows}, key=_band_key)
    lines = []
    for band in bands:
        here = sorted((r for r in rows if r.get("band", "default") == band),
                      key=lambda r: (ORDER.get(r.get("method", ""), 99),
                                     r.get("n_nodes", 0)))
        star = set(pareto_front(here))
        tag = ("in-distribution" if _band_upper(band) <= 20.0
               else "EXTRAPOLATION -- training is pinned to 5-20 m")
        lines += [f"band {band}  ({tag})", HEADER]
        lines += [format_row(r) for r in here]
        missing = [m for m in ORDER if m not in {r.get("method") for r in here}]
        if missing:
            lines.append("  MISSING: " + ", ".join(sorted(missing)))
        lines.append("")

    budgets = {r.get("budget") for r in rows if r.get("budget") is not None}
    if len(budgets) > 1:
        # LOUD, because it silently invalidates the comparison: the arms were
        # not run at one budget, so a difference between them may be sparsity.
        lines += [f"WARNING: rows carry DIFFERENT budgets {sorted(budgets)}. "
                  "The arms are not comparable at equal node count.", ""]
    lines += FOOTNOTES
    return "\n".join(lines)


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "results/placement"
    print(render(load(d)))


if __name__ == "__main__":
    main()
