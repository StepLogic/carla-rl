"""ViNT's Town07 main-table cells: per cell n, SR, SPL, SR@2m, ends, median entry speed of
the successes; and each map arm vs mapless on the same seeds (exact two-sided McNemar).
Asserts that every arm of a band ran the same seeds in the same order."""
import collections
import glob
import json
import os

from scipy.stats import binomtest

HERE = os.path.dirname(os.path.abspath(__file__))
ARMS = ("mapless", "uniform", "vinttopo")
BANDS = ("easy", "medium", "turns")
NAMES = {"easy": "easy", "medium": "medium (>= 1 turn)", "turns": "hard (>= 2 turns)"}


def load(band, arm):
    f = glob.glob(os.path.join(HERE, "checkpoints", f"bench_vintd3_t07_{arm}_Town07_{band}",
                               "eval_*", "eval.json"))
    assert len(f) == 1, (band, arm, f)
    return json.load(open(f[0])), f[0]


print("| Band | Arm | n | SR | SPL | SR@2m | Ends | Median entry km/h (successes) | vs mapless: b / c, p |")
print("|---|---|---|---|---|---|---|---|---|")
for band in BANDS:
    base, _ = load(band, "mapless")
    base_eps = base["episodes_detail"]
    for arm in ARMS:
        j, path = load(band, arm)
        eps = j["episodes_detail"]
        assert [e["seed"] for e in eps] == [e["seed"] for e in base_eps], f"{band} {arm}: seeds differ"
        assert j["arrival_radius_m"] == 1.0 and j["deterministic"] is True
        n = len(eps)
        ends = collections.Counter(e["end"] for e in eps)
        sp = sorted(e.get("entry_speed_kmh") or 0 for e in eps if e["reached"])
        med = f"{sp[len(sp) // 2]:.1f}" if sp else "-"
        test = "-"
        if arm != "mapless":
            b = sum(1 for e, m in zip(eps, base_eps) if e["reached"] and not m["reached"])
            c = sum(1 for e, m in zip(eps, base_eps) if m["reached"] and not e["reached"])
            p = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
            test = f"{b} / {c}, p = {p:.3g}"
        print(f"| {NAMES[band]} | {arm} | {n} | {j['success_rate']:.2f} | {j['spl']:.2f} | "
              f"{j['success_rate_2m']:.2f} | "
              + ", ".join(f"{k} {v}" for k, v in ends.most_common())
              + f" | {med} | {test} |")
    print(f"<!-- {band}: episodes file {os.path.basename(base.get('episodes_file') or '-')} -->")
