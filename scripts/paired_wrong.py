"""Paired true-vs-wrong-goal test on the SAME episodes. McNemar's exact test
on the discordant pairs: the episodes one goal image solved and the other did
not. Usage: python3 -I paired_wrong.py <true_dir> <wrong_dir>"""
import json
import sys
from math import comb


def by_seed(p):
    d = json.load(open(p + "/eval.json"))
    return {e["seed"]: bool(e["reached"]) for e in d["episodes_detail"]}, d["success_rate"]


t, sr_t = by_seed(sys.argv[1])
w, sr_w = by_seed(sys.argv[2])
seeds = sorted(set(t) & set(w))
only_true = sum(1 for s in seeds if t[s] and not w[s])
only_wrong = sum(1 for s in seeds if w[s] and not t[s])
both = sum(1 for s in seeds if t[s] and w[s])
neither = sum(1 for s in seeds if not t[s] and not w[s])
n = only_true + only_wrong
# two-sided exact binomial on the discordant pairs
p = min(1.0, 2 * sum(comb(n, k) for k in range(min(only_true, only_wrong) + 1)) / 2 ** n) if n else 1.0
print(f"paired on {len(seeds)} episodes")
print(f"  true {sr_t:.3f}   wrong {sr_w:.3f}   gap {sr_t - sr_w:+.3f}")
print(f"  both {both}  neither {neither}  only-true {only_true}  only-wrong {only_wrong}")
print(f"  McNemar exact two-sided p = {p:.4f}")
