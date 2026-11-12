#!/usr/bin/env python3
"""Choose the cloning checkpoint from training-town validation evals
(imitation-first plan, Task 7).

    python3 -I scripts/select_checkpoint.py --out logs/il_bc_r0_selected.txt \
        checkpoints/il_bc_r0_val_Town05/eval_step_* checkpoints/il_bc_r0_val_Town01/eval_step_*

Each argument is an eval folder (its eval.json, as EVAL_FROM writes it); a
folder whose name ends in "_wrong" is a WRONG_GOAL_MODE=1 eval. Results are
pooled per checkpoint step over the towns, weighted by episodes. The rule:
the highest true-goal success among steps whose wrong-goal success is at most
--max-wrong (ties to the later step); if none qualifies, the largest
true - wrong gap. Never run it on Town02/Town03 evals: those are the test set.
"""
import argparse
import json
import os
import re
import sys

_STEP = re.compile(r"step_(\d+)$")


def select_checkpoint(results, max_wrong=0.10):
    """results: {step: (true_sr, wrong_sr)} -> the chosen step."""
    if not results:
        raise ValueError("no checkpoint results to choose from")
    ok = [s for s, (_, w) in results.items() if w <= max_wrong]
    if ok:
        return max(ok, key=lambda s: (results[s][0], s))
    return max(results, key=lambda s: (results[s][0] - results[s][1], s))


def pool_eval_dirs(dirs):
    """Eval folders -> {step: (true_sr, wrong_sr)}, pooled over towns by
    episode count. Every step must have both a true and a wrong-goal eval."""
    hits = {}                                   # (step, wrong) -> [reached, episodes]
    for d in dirs:
        with open(os.path.join(d, "eval.json")) as fh:
            summ = json.load(fh)
        m = _STEP.search(os.path.basename(os.path.normpath(summ["checkpoint"])))
        if m is None:
            raise ValueError(f"{d}: checkpoint {summ['checkpoint']!r} is not a step_<n> folder")
        key = (int(m.group(1)), os.path.basename(os.path.normpath(d)).endswith("_wrong"))
        n = int(summ["episodes"])
        acc = hits.setdefault(key, [0.0, 0])
        acc[0] += float(summ["success_rate"]) * n
        acc[1] += n
    steps = sorted({s for s, _ in hits})
    missing = [s for s in steps if (s, False) not in hits or (s, True) not in hits]
    if missing:
        raise ValueError(f"steps without both a true- and a wrong-goal eval: {missing}")
    return {s: (hits[(s, False)][0] / hits[(s, False)][1],
                hits[(s, True)][0] / hits[(s, True)][1]) for s in steps}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--max-wrong", type=float, default=0.10)
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    res = pool_eval_dirs(a.dirs)
    best = select_checkpoint(res, a.max_wrong)
    for s, (t, w) in sorted(res.items()):
        print(f"step {s:>6}: true {t:.3f}  wrong {w:.3f}  gap {t - w:+.3f}"
              + ("   <- chosen" if s == best else ""))
    if a.out:
        with open(a.out, "w") as fh:
            fh.write(f"{best}\n")
    print(best)
    return best


if __name__ == "__main__":
    sys.exit(0 if main() is not None else 1)
