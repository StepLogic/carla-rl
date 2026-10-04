"""One loader for the grid's LaTeX table and its Pareto plot, so the two can never disagree.

Every row is Town02 under the DINOv2 localiser unless stated. The shipped-localiser SR is carried
alongside for the arms that have one; it is a reference column, never a comparison within a row.
"""
import collections, glob, json
import numpy as np

SEQ = "results/maps/sequence"
DINO = ["results/drive_plan_c/localiser/dinov2_bank_*.jsonl", "results/drive_plan_c/signal_grid/*.jsonl"]
SHIP = ["results/drive_plan_c/*.jsonl", "results/drive_plan_c/repeat/*.jsonl",
        "results/drive_plan_c/equal_points/*.jsonl"]
NOMAD = ["results/nomad/nomad_*v6ncap*.jsonl", "results/nomad/repeat/*.jsonl", "results/nomad/equal_points/*.jsonl"]

SIGNALS = [("entropy", "Policy entropy"), ("epistemic", "Epistemic"), ("aleatoric", "Aleatoric")]
BUDGETS = [("$\\approx$60", "60", {"spacing": "uniform15_trace", "sgm": "sgm_g24_e55_c55",
                                  "nomad": "nomad_uniform15_trace_v6ncap_20kmh", "ug": "15",
                                  "sg": {"entropy": "44", "epistemic": "44", "aleatoric": "44"}}),
           ("$\\approx$115", "115", {"spacing": "uniform8_trace", "sgm": "sgm_g10_e55_c55",
                                     "nomad": "nomad_uniform8_trace_v6ncap_20kmh", "ug": "8",
                                     "sg": {"entropy": "14", "epistemic": "13", "aleatoric": "13"}}),
           ("$\\approx$150", "150", {"spacing": "uniform6_trace", "sgm": "sgm_g7_e55_c55",
                                     "nomad": "nomad_uniform6_trace_v6ncap_20kmh", "ug": "6",
                                     "sg": {"entropy": "9", "epistemic": "9", "aleatoric": "9"}}),
           ("$\\approx$550", "550", {"spacing": "uniform2_trace", "sgm": "sgm_g2_e55_c55",
                                     "nomad": "nomad_uniform2_trace_v6ncap_20kmh", "ug": "2",
                                     "sg": {"entropy": "2", "epistemic": "2", "aleatoric": "2"}})]
REFS = [("raw_trace", "Dense (every sample)", "dense"),
        ("entropy_nosmooth_high_thin4", "Policy entropy only", "seed"),
        ("epistemic_thin4_n30", "Epistemic only", "seed"),
        ("aleatoric_thin4_n30", "Aleatoric only", "seed")]


def load(globs, want):
    per = collections.defaultdict(list)
    for f in sorted({f for g in globs for f in glob.glob(g)}):
        if any(s in f for s in ("smoke", "debug")):
            continue
        by = collections.defaultdict(dict)
        for line in open(f):
            if not line.strip():
                continue
            r = json.loads(line)
            if want is not None and r.get("localiser", "mobilenet_trace").split(":")[0] != want:
                continue
            by[r["map"]][(r["set"], r["index"])] = r
        for m, d in by.items():
            if len(d) == 40:
                per[m].append(d)
    return per


def stat(per, m):
    """-> (SR, SPL, n_runs, SR sd) or None."""
    if m not in per:
        return None
    sr = [np.mean([r["success"] for r in d.values()]) for d in per[m]]
    spl = [np.mean([r["spl"] or 0 for r in d.values()]) for d in per[m]]
    sd = float(np.std(sr, ddof=1)) if len(sr) > 1 else 0.0
    return float(np.mean(sr)), float(np.mean(spl)), len(sr), sd


def nodes(m):
    import re
    base = re.sub(r"^nomad_(.*)_v6ncap_20kmh$", r"\1", m)
    return len(json.load(open(f"{SEQ}/{base}_sequence.json"))["nodes"])


def rows():
    """-> list of dicts, in table order."""
    dino, ship, nomad = load(DINO, "dinov2_bank"), load(SHIP, "mobilenet_trace"), load(NOMAD, None)
    out = []
    for label, key, cfg in BUDGETS:
        blk = []
        blk.append(dict(block=label, arm="Spacing", ours=False, map=cfg["spacing"],
                        d=stat(dino, cfg["spacing"]), s=stat(ship, cfg["spacing"])))
        blk.append(dict(block=label, arm="SGM$^{\\dagger}$", ours=False, map=cfg["sgm"],
                        d=stat(dino, cfg["sgm"]), s=stat(ship, cfg["sgm"])))
        blk.append(dict(block=label, arm="NoMaD$^{\\ddagger}$", ours=False, map=cfg["nomad"],
                        d=None, s=stat(nomad, cfg["nomad"])))
        for skey, sname in SIGNALS:
            for filler, m in (("Uniform", f"hybrid_uniform{cfg['ug']}_{skey}_thin4"),
                              ("SGM", f"hybrid_sgm_g{cfg['sg'][skey]}_{skey}_thin4")):
                blk.append(dict(block=label, arm=f"{sname} + {filler}", ours=True, map=m,
                                d=stat(dino, m), s=stat(ship, m), signal=skey, filler=filler))
        out += [b for b in blk if b["d"] or b["s"]]
    for m, name, kind in REFS:
        out.append(dict(block="Fixed size", arm=name, ours=(kind == "seed"), map=m,
                        d=stat(dino, m), s=stat(ship, m), fixed=True))
    return out
