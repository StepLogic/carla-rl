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
ORDER = {"goal_only": 0, "goal_free": 1, "gated": 2, "gated_her": 3}

HEADER = (f"{'arm':<12} {'success':>10} {'steps(med)':>11} {'n_arr':>6} "
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
    return (f"{r['arm']:<12} {succ:>10} {med:>11} {r.get('n_arrived', 0):>6} "
            f"{r.get('episode_len_mean', float('nan')):>8.1f} "
            f"{r.get('timeout_frac', float('nan')):>8.3f} "
            f"{bias:>10} {corr:>7}")


def render(rows):
    """Format the table. ORDERING IS IMPOSED HERE, not inherited.

    load() also sorts, but ordering is a presentation guarantee and belongs
    with the presenter: rows reaching this function from anywhere else -- a
    test, a notebook, a future caller that globs differently -- must come out
    specialists first, gated arms after, so the table always reads the same way.
    """
    if not rows:
        return "no results found"
    rows = sorted(rows, key=lambda r: ORDER.get(r.get("arm", ""), 99))
    lines = [HEADER] + [format_row(r) for r in rows] + [""] + FOOTNOTES
    missing = [a for a in ORDER if a not in {r.get("arm") for r in rows}]
    if missing:
        lines += ["",
                  "MISSING ARMS: " + ", ".join(sorted(missing)) + ".",
                  "  A four-way claim cannot be made from a partial table; the",
                  "  absent arms were skipped for want of a checkpoint."]
    return "\n".join(lines)


def main():
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "results/four_way_eval"
    print(render(load(results_dir)))


if __name__ == "__main__":
    main()
