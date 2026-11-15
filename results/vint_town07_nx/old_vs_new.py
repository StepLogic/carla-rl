"""Route-by-route: ViNT Town07 old cells (2026-11-12, old navigator, ViNT tree) against the
rerun (2026-11-15, next-node advance, merged code). Same seeds. Descriptive only: two things
changed at once (navigator rule and code version), plus run-to-run noise (~3/100)."""
import glob
import json
import os
import statistics

R = "/home/robotlab/scratch/carla-rl/.claude/worktrees/vint-carla/results"


def load(label, arm, band, base):
    f = glob.glob(f"{R}/{base}/checkpoints/bench_{label}_{arm}_Town07_{band}/eval_*/eval.json")
    assert len(f) == 1, (label, arm, band, f)
    return {e["seed"]: e for e in json.load(open(f[0]))["episodes_detail"]}


def med(xs):
    xs = [x for x in xs if x is not None]
    return f"{statistics.median(xs):.1f}" if xs else "-"


for band in ("easy", "medium", "turns"):
    for arm in ("mapless", "uniform", "vinttopo"):
        old = load("vintd3_t07", arm, band, "vint_town07")
        new = load("vintd3_nx_ar1", arm, band, "vint_town07_nx")
        assert set(old) == set(new)
        lost = [s for s in old if old[s]["reached"] and not new[s]["reached"]]
        gained = [s for s in old if new[s]["reached"] and not old[s]["reached"]]
        both = [s for s in old if old[s]["reached"] and new[s]["reached"]]
        print(f"{band:6s} {arm:8s} old {sum(e['reached'] for e in old.values()):3d} new "
              f"{sum(e['reached'] for e in new.values()):3d} | both {len(both):2d}, lost {len(lost):2d}, "
              f"gained {len(gained):2d} | entry km/h of LOST (old run) {med([old[s].get('entry_speed_kmh') for s in lost])}, "
              f"of GAINED (new run) {med([new[s].get('entry_speed_kmh') for s in gained])}, "
              f"of BOTH (old/new) {med([old[s].get('entry_speed_kmh') for s in both])}/"
              f"{med([new[s].get('entry_speed_kmh') for s in both])} | lost now end: "
              + ",".join(sorted({new[s]['end'] for s in lost})))
