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

HEADER = (f"{'method':<10} {'nodes':>6} {'budget':>7} {'edges':>7} "
          f"{'SR':>7} {'SPL':>7} {'unroutable':>11} {'leg_TO':>7}")

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
    spl = "n/a" if r.get("spl") is None else f"{r['spl']:.3f}"
    reasons = r.get("end_reasons") or {}
    unroutable = sum(v for k, v in reasons.items() if k in UNROUTABLE)
    return (f"{r.get('method', '?'):<10} {r.get('n_nodes', 0):>6} "
            f"{r.get('budget', 0):>7} {r.get('n_edges', 0):>7} "
            f"{r.get('success', 0.0):>7.3f} {spl:>7} "
            f"{unroutable:>11} {r.get('leg_timeouts', 0):>7}")


def render(rows):
    """One table per band. Pooling bands would average an in-distribution
    result with an extrapolation one and describe neither."""
    if not rows:
        return "no results found"

    bands = sorted({r.get("band", "default") for r in rows}, key=_band_key)
    lines = []
    for band in bands:
        here = sorted((r for r in rows if r.get("band", "default") == band),
                      key=lambda r: ORDER.get(r.get("method", ""), 99))
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
