#!/usr/bin/env python3
"""Junction-band analysis, exactly as pre-specified in docs/junction_bands_prespec_2026-11-12.md.

Checks before any test: every wrong-goal cell lists the same seeds, in the same order, as
its true-goal partner, and the seeds are the episode file's.
Per cell (reported regardless): n, successes, SR, SPL, ends, median route m, share of
routes with a left turn, median entry speed of the successes.
Test 1: goal use, true vs wrong goal on the same seeds, exact two-sided McNemar; headline
at any speed, secondary with slow arrival (entry <= 10 km/h); Holm within each family of 6.
Test 2 (H-left): true-goal success with vs without a left turn, Fisher exact two-sided
(1-turn band: left vs right); Holm within the family of 6.
Descriptive only: success by direction sequence; no-left stratum across bands.
"""
import collections
import json
import os
import statistics

from scipy.stats import binomtest, fisher_exact

HERE = os.path.dirname(os.path.abspath(__file__))
TOWNS = ("Town02", "Town07")
BANDS = ("turns1", "turns2", "turns3plus")
NAME = {"turns1": "1 turn", "turns2": "2 turns", "turns3plus": "3+ turns"}
SLOW = 10.0


def cell(town, label, wrong=False):
    """The trainer writes a wrong-goal run to eval_<tag>_wrong (WRONG_GOAL_MODE=1)."""
    p = os.path.join(HERE, "checkpoints", f"bench_{label}_mapless_{town}_turns",
                     "eval_step_15000" + ("_wrong" if wrong else ""), "eval.json")
    return json.load(open(p))


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    out, run = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        out[i] = run
    return out


def mcnemar(a, b):
    """a, b: lists of bools on the same seeds -> (only a, only b, exact two-sided p)."""
    x = sum(1 for u, v in zip(a, b) if u and not v)
    y = sum(1 for u, v in zip(a, b) if v and not u)
    return x, y, (binomtest(x, x + y, 0.5).pvalue if x + y else 1.0)


def slow(e):
    return bool(e["reached"]) and (e.get("entry_speed_kmh") or 0.0) <= SLOW


rows, t1_any, t1_slow, t2 = [], [], [], []
desc_seq, desc_noleft = [], []
for town in TOWNS:
    for band in BANDS:
        eps_file = json.load(open(os.path.join(HERE, "episodes", f"{town}_{band}.json")))
        meta = {e["seed"]: e for e in eps_file}
        tj, wj = cell(town, f"jb_{band}"), cell(town, f"jb_{band}wrong", wrong=True)
        te, we = tj["episodes_detail"], wj["episodes_detail"]
        # ---- checks before any test
        assert [e["seed"] for e in te] == [e["seed"] for e in we], f"{town} {band}: pairing broken"
        assert set(e["seed"] for e in te) == set(meta), f"{town} {band}: seeds differ from the file"
        assert tj["arrival_radius_m"] == wj["arrival_radius_m"] == 1.0
        for name, j, es in (("true", tj, te), ("wrong", wj, we)):
            succ = [e for e in es if e["reached"]]
            sp = sorted(e.get("entry_speed_kmh") or 0.0 for e in succ)
            rows.append(dict(
                town=town, band=band, goal=name, n=len(es), succ=len(succ),
                sr=len(succ) / len(es), spl=j["spl"], slow=sum(slow(e) for e in es),
                ends=collections.Counter(e["end"] for e in es),
                route=statistics.median(meta[e["seed"]]["route_m"] for e in es),
                left=sum(meta[e["seed"]]["contains_left"] for e in es),
                entry=(sp[len(sp) // 2] if sp else None)))
        # ---- test 1
        t1_any.append((town, band) + mcnemar([bool(e["reached"]) for e in te],
                                             [bool(e["reached"]) for e in we]))
        t1_slow.append((town, band) + mcnemar([slow(e) for e in te], [slow(e) for e in we]))
        # ---- test 2: H-left, true goal, any speed
        L = [bool(e["reached"]) for e in te if meta[e["seed"]]["contains_left"]]
        N = [bool(e["reached"]) for e in te if not meta[e["seed"]]["contains_left"]]
        table = [[sum(L), len(L) - sum(L)], [sum(N), len(N) - sum(N)]]
        p = fisher_exact(table, alternative="two-sided")[1] if L and N else 1.0
        t2.append((town, band, sum(L), len(L), sum(N), len(N), p))
        # ---- descriptive
        by = collections.defaultdict(lambda: [0, 0])
        for e in te:
            by[meta[e["seed"]]["turn_dirs"]][0] += bool(e["reached"])
            by[meta[e["seed"]]["turn_dirs"]][1] += 1
        desc_seq.append((town, band, dict(sorted(by.items()))))
        desc_noleft.append((town, band, sum(N), len(N)))

out = []
w = out.append
w("# Junction-turn bands: results (analysis as pre-specified)\n")
w("c0 il_bc_c0 step_15000, mapless, local CARLA 0.9.15, the map benchmark's harness. "
  "Success = within 1 m of the goal at any speed (headline); slow arrival = success with "
  f"entry into the 1 m circle at <= {SLOW:g} km/h (secondary). Checks passed: every wrong-goal "
  "cell ran the same seeds in the same order as its true-goal partner, and the seeds are the "
  "episode file's.\n")
w("## Every cell\n")
w("| Town | Band | Goal | n | Successes | SR | SPL | Slow arrivals | Median entry km/h | "
  "Median route m | Contains a left | Ends |")
w("|---|---|---|---|---|---|---|---|---|---|---|---|")
for r in rows:
    ends = ", ".join(f"{k} {v}" for k, v in r["ends"].most_common())
    entry = f"{r['entry']:.1f}" if r["entry"] is not None else "-"
    w(f"| {r['town']} | {NAME[r['band']]} | {r['goal']} | {r['n']} | {r['succ']} | {r['sr']:.2f} | "
      f"{r['spl']:.2f} | {r['slow']} | {entry} | {r['route']:.0f} | {r['left']}/{r['n']} | {ends} |")
for title, res in (("Test 1, headline: goal use at any speed (true vs wrong, McNemar)", t1_any),
                   ("Test 1, secondary: goal use with slow arrival (true vs wrong, McNemar)", t1_slow)):
    ph = holm([x[4] for x in res])
    w(f"\n## {title}\n")
    w("| Town | Band | True only | Wrong only | p | p (Holm, 6) |")
    w("|---|---|---|---|---|---|")
    for (town, band, a, b, p), q in zip(res, ph):
        w(f"| {town} | {NAME[band]} | {a} | {b} | {p:.3g} | {q:.3g} |")
ph = holm([x[6] for x in t2])
w("\n## Test 2, H-left: true-goal success with vs without a left turn (Fisher)\n")
w("1-turn band: left vs right. Hypothesis: lower with a left turn.\n")
w("| Town | Band | With a left: success / n | Without: success / n | p | p (Holm, 6) |")
w("|---|---|---|---|---|---|")
for (town, band, sl, nl, sn, nn, p), q in zip(t2, ph):
    w(f"| {town} | {NAME[band]} | {sl}/{nl} ({sl / nl if nl else 0:.2f}) | "
      f"{sn}/{nn} ({sn / nn if nn else 0:.2f}) | {p:.3g} | {q:.3g} |")
w("\n## Descriptive only (no test)\n")
w("No-left stratum across bands, true goal (n printed; too few for a test):\n")
for town in TOWNS:
    w(f"- {town}: " + ", ".join(f"{NAME[b]} {s}/{n}" for t, b, s, n in desc_noleft if t == town))
w("\nTrue-goal success by turn-direction sequence (successes / routes):\n")
for town, band, by in desc_seq:
    w(f"- {town} {NAME[band]}: " + ", ".join(f"{k} {v[0]}/{v[1]}" for k, v in by.items()))
w("\nPilot (Town07, 20 true-goal episodes per band, seeds after the evaluated 100; harness "
  "check only): see logs/eval_jbpilot_*.log.\n")
open(os.path.join(HERE, "RESULTS.md"), "w").write("\n".join(out) + "\n")
print("\n".join(out))
