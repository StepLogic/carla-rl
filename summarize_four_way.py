#!/usr/bin/env python
"""Print the four-way ablation table from per-arm result files.

A SEPARATE FILE RATHER THAN A HEREDOC inside the driver, so it can be tested
without a simulator, a GPU or a checkpoint. The table is the deliverable of the
whole exercise; formatting it in a place nothing can exercise is how a comparison
ends up with a column silently misaligned or an n/a printed as 0.000.
"""
import glob
import json
import os
import sys

# Presentation order: the two specialists first, then the gated arms they are
# there to be compared against.
#
# THE TWO goal_only_* ROWS ARE THE SAME FROZEN CHECKPOINT as goal_only, run
# under a different TERMINATION RULE, and they sit next to it for that reason.
# They exist because the recorded extrapolation gap is confounded: a lane
# crossing ended an episode for the baseline and was a no-op for the gated
# arms. `lanenoop` frees the crossing (off-road and collision stay fatal);
# `freerun` ends episodes only on arrival, which is exactly the gated goal
# arm's rule. Reading down those three rows is reading off how much of the gap
# was the rule rather than the gate.
ORDER = {"goal_only": 0, "goal_only_lanenoop": 1, "goal_only_freerun": 2,
         "goal_free": 3, "gated": 4, "gated_her": 5}

# WHICH ARMS A COMPLETE TABLE NEEDS. The deconfound rows are OPTIONAL: they
# answer a follow-up question and their absence is not a hole in the four-way
# claim, so reporting them as MISSING would cry wolf on every ordinary run.
REQUIRED = ("goal_only", "goal_free", "gated", "gated_her")

HEADER = (f"{'arm':<20} {'success':>10} {'steps(med)':>11} {'n_arr':>6} "
          f"{'ep_len':>8} {'timeout':>8} {'q_bias':>10} {'q_corr':>7}")

FOOTNOTES = [
    "success is n/a for goal_free BY CONSTRUCTION: it measures arrival at a",
    "  goal that policy cannot see and is not paid for. Reported as n/a rather",
    "  than 0.000, which would read as 'worse' instead of 'not applicable'.",
    "steps(med) is the median time to goal over ARRIVING episodes. Under the",
    "  SORB reward that is the negated return, i.e. exactly what the goal",
    "  critic is trained to predict.",
    "All arms drive the SAME episode set: identical towns and start states from",
    "  a fixed seed, so a difference in the table is not a difference in draw.",
    "q_bias/q_corr test the critic AS A DISTANCE, which is why the minimum-time",
    "  reward was chosen at all: under it Q(s0) should equal -(steps taken).",
    "  q_bias is the mean signed error IN STEPS, so -570 means the critic",
    "  believes the goal is 570 steps further than it is. q_corr is whether it",
    "  still ORDERS states correctly -- a badly biased critic can remain usable",
    "  to a map builder that compares distances rather than reading them.",
    "  Measured on ARRIVING episodes only: a timeout has no true cost-to-go.",
]


def load(results_dir):
    rows = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        with open(path) as fh:
            rows.append(json.load(fh))
    rows.sort(key=lambda r: ORDER.get(r.get("arm", ""), 99))
    return rows


def format_row(r):
    succ = ("n/a" if not r.get("goal_conditioned")
            else f"{r['success']:.3f}")
    med = ("n/a" if r.get("steps_median") is None
           else f"{r['steps_median']:.0f}")
    bias = ("n/a" if r.get("q_bias_steps") is None
            else f"{r['q_bias_steps']:+.0f}")
    corr = "n/a" if r.get("q_corr") is None else f"{r['q_corr']:.2f}"
    return (f"{r['arm']:<20} {succ:>10} {med:>11} {r.get('n_arrived', 0):>6} "
            f"{r.get('episode_len_mean', float('nan')):>8.1f} "
            f"{r.get('timeout_frac', float('nan')):>8.3f} "
            f"{bias:>10} {corr:>7}")


def _band_key(band):
    """Sort bands by their lower bound, so 5-20 precedes 20-40 precedes 40-70."""
    try:
        return float(str(band).split("-")[0])
    except (ValueError, IndexError):
        return float("inf")


def render(rows):
    """Format the tables. ORDERING IS IMPOSED HERE, not inherited.

    load() also sorts, but ordering is a presentation guarantee and belongs
    with the presenter: rows reaching this function from anywhere else -- a
    test, a notebook, a future caller that globs differently -- must come out
    specialists first, gated arms after, nearest band first.

    ONE TABLE PER DISTANCE BAND. Pooling bands would average an in-distribution
    result with an extrapolation one and produce a number describing neither.
    """
    if not rows:
        return "no results found"

    goal_rows = [r for r in rows if r.get("mode", "goal") == "goal"]
    explore_rows = [r for r in rows if r.get("mode") == "explore"]

    bands = sorted({r.get("band", "default") for r in goal_rows}, key=_band_key)
    lines = []
    for band in bands:
        here = sorted((r for r in goal_rows
                       if r.get("band", "default") == band),
                      key=lambda r: ORDER.get(r.get("arm", ""), 99))
        in_dist = all(r.get("in_distribution", True) for r in here)
        tag = ("in-distribution" if in_dist
               else "EXTRAPOLATION -- training is pinned to 5-20 m")
        lines += [f"band {band}  ({tag})", HEADER]
        lines += [format_row(r) for r in here]
        missing = [a for a in REQUIRED if a not in {r.get("arm") for r in here}]
        if missing:
            lines.append("  MISSING: " + ", ".join(sorted(missing))
                         + " -- no checkpoint")
        lines.append("")

    if explore_rows:
        # THE OTHER HALF OF A GATED POLICY. Reporting only the goal half can
        # show an arm as better while its explore behaviour has degraded, which
        # is exactly what the second training run did: goal episodes improved
        # while explore return fell from 62.5 to 40.1 over the same period.
        # Success is absent here by construction -- the goal block is zeroed in
        # explore mode, so arrival is measured against a goal the policy cannot
        # see. Read ep_len and the return-bearing columns instead.
        lines += ["EXPLORE half (gated arms only; no goal, so no bands "
                  "and no success)", HEADER]
        lines += [format_row(r) for r in
                  sorted(explore_rows,
                         key=lambda r: ORDER.get(r.get("arm", ""), 99))]
        lines.append("")

    lines += FOOTNOTES
    all_missing = [a for a in REQUIRED if a not in {r.get("arm") for r in rows}]
    if all_missing:
        lines += ["",
                  "MISSING ARMS: " + ", ".join(sorted(all_missing)) + ".",
                  "  A four-way claim cannot be made from a partial table; the",
                  "  absent arms were skipped for want of a checkpoint."]
    return "\n".join(lines)


def main():
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "results/four_way_eval"
    print(render(load(results_dir)))


if __name__ == "__main__":
    main()
