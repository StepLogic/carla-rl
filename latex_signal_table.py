"""Policy entropy vs epistemic vs aleatoric as seeds, at four matched budgets (user 2026-09-20).

Town02, DINOv2 localiser throughout, same 40 trajectories everywhere. Column groups are node budgets,
mirroring the domain groups of the main table. Seeds are count-matched at 30 per signal -- the signals
natively return 71 / 93 / 202 candidates, and since a seeded map's damage scales with the seed SHARE of
its budget, an unmatched comparison would measure the count rather than the signal.
Run from the icra-2027 folder.
"""
import argparse, collections, glob, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
SEQ = "results/maps/sequence"
GLOBS = ["results/drive_plan_c/localiser/dinov2_bank_*.jsonl", "results/drive_plan_c/signal_grid/*.jsonl"]
BUDGETS = [("$\\approx$60", "uniform15_trace", "sgm_g24_e55_c55", "15",
            {"entropy": "44", "epistemic": "44", "aleatoric": "44"}),
           ("$\\approx$115", "uniform8_trace", "sgm_g10_e55_c55", "8",
            {"entropy": "14", "epistemic": "13", "aleatoric": "13"}),
           ("$\\approx$150", "uniform6_trace", "sgm_g7_e55_c55", "6",
            {"entropy": "9", "epistemic": "9", "aleatoric": "9"}),
           ("$\\approx$550", "uniform2_trace", "sgm_g2_e55_c55", "2",
            {"entropy": "2", "epistemic": "2", "aleatoric": "2"})]
SIGNALS = [("entropy", "Policy entropy"), ("epistemic", "Epistemic"), ("aleatoric", "Aleatoric")]
SEEDONLY = [("entropy_nosmooth_high_thin4", "policy entropy"), ("epistemic_thin4_n30", "epistemic"),
            ("aleatoric_thin4_n30", "aleatoric")]


def load():
    per = collections.defaultdict(list)
    for f in sorted({f for g in GLOBS for f in glob.glob(g)}):
        if any(s in f for s in ("smoke", "debug")):
            continue
        by = collections.defaultdict(dict)
        for line in open(f):
            if line.strip():
                r = json.loads(line)
                if r.get("localiser", "").split(":")[0] == "dinov2_bank":
                    by[r["map"]][(r["set"], r["index"])] = r
        for m, d in by.items():
            if len(d) == 40:
                per[m].append(d)
    return per


def stat(per, m):
    if m not in per:
        return None
    sr = [np.mean([r["success"] for r in d.values()]) for d in per[m]]
    spl = [np.mean([r["spl"] or 0 for r in d.values()]) for d in per[m]]
    f = lambda v: (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else None)
    return f(sr), f(spl), len(per[m])


def nodes(m):
    return len(json.load(open(f"{SEQ}/{m}_sequence.json"))["nodes"])


def pad(n):
    s = str(n)
    return "\\phantom{" + "0" * (4 - len(s)) + "}" + s if len(s) < 4 else s


def fmt(v, best):
    if v is None:
        return "-"
    mu, sd = v
    t = f"{mu:.2f}" + (f"\\,$\\pm$\\,{sd:.2f}" if sd is not None else "$^{\\ast}$")
    return "\\textbf{" + t + "}" if best else t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="tables/table_signals_town02.tex")
    a = ap.parse_args()
    per = load()
    ROWS = [("Spacing", False, {b[0]: b[1] for b in BUDGETS}),
            ("SGM$^{\\dagger}$\\cite{emmons2020sparse}", False, {b[0]: b[2] for b in BUDGETS}),
            ("MIDRULE", False, None)]
    for key, name in SIGNALS:
        ROWS.append((f"{name} + Spc.", True, {b[0]: f"hybrid_uniform{b[3]}_{key}_thin4" for b in BUDGETS}))
        ROWS.append((f"{name} + SGM", True, {b[0]: f"hybrid_sgm_g{b[4][key]}_{key}_thin4" for b in BUDGETS}))

    cells = {}
    for label, _o, maps in ROWS:
        if maps is None:
            continue
        for b in BUDGETS:
            s = stat(per, maps[b[0]])
            cells[(label, b[0])] = (nodes(maps[b[0]]) if s else None, s[0] if s else None, s[1] if s else None)
    bsr = {b[0]: max((c[1][0] for (l, bb), c in cells.items() if bb == b[0] and c[1]), default=None)
           for b in BUDGETS}
    bspl = {b[0]: max((c[2][0] for (l, bb), c in cells.items() if bb == b[0] and c[2]), default=None)
            for b in BUDGETS}
    so = ", ".join(f"{n} ({stat(per, m)[0][0]:.2f} SR)" for m, n in SEEDONLY if stat(per, m))

    L = ["% Town02, DINOv2 localiser, same 40 trajectories in every cell.",
         "% Seeds count-matched at 30 per signal; the three seed sets share no trace sample.",
         "% $\\ast$ = single run; a +/- is mean +/- sd over runs of the SAME trajectories.",
         "\\begin{table*}[t]",
         "\\centering",
         "\\footnotesize",
         "\\setlength{\\tabcolsep}{4pt}",
         "\\caption{Which uncertainty signal makes a better map seed, in CARLA Town02 under the DINOv2 "
         "localiser. Each block fixes a node budget; within a block every map has the same size, so the "
         "rows differ only in which 30 states were seeded and what filled the rest. The three signals are "
         "count-matched at 30 seeds and select disjoint states (no shared trace sample), because a seeded "
         "map's cost scales with the seed share of its budget. Seeding with 30 states alone, and no filler, "
         f"reaches only {so}. Shaded rows are ours. $^{{\\ast}}$single run. $^{{\\dagger}}$our implementation.}}",
         "\\label{tab:carla-signals}",
         "\\begin{tabular}{@{}l rcc rcc rcc rcc@{}}",
         "\\toprule",
         "& \\multicolumn{3}{c}{$\\approx$60 nodes} & \\multicolumn{3}{c}{$\\approx$115 nodes}",
         "& \\multicolumn{3}{c}{$\\approx$150 nodes} & \\multicolumn{3}{c}{$\\approx$550 nodes} \\\\",
         "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
         "\\cmidrule(lr){8-10}\\cmidrule(lr){11-13}",
         "Map & Nds & SR $\\uparrow$ & SPL $\\uparrow$ & Nds & SR $\\uparrow$ & SPL $\\uparrow$",
         "& Nds & SR $\\uparrow$ & SPL $\\uparrow$ & Nds & SR $\\uparrow$ & SPL $\\uparrow$ \\\\",
         "\\midrule"]
    for label, ours, maps in ROWS:
        if maps is None:
            L.append("\\midrule"); continue
        parts = []
        for b in BUDGETS:
            n, sr, spl = cells[(label, b[0])]
            if n is None:
                parts.append("- & - & -"); continue
            parts.append(f"{pad(n)} & {fmt(sr, sr and abs(sr[0]-bsr[b[0]])<1e-9)} "
                         f"& {fmt(spl, spl and abs(spl[0]-bspl[b[0]])<1e-9)}")
        if ours:
            L.append("\\rowcolor{ourrow}")
        L.append(f"{label} & {parts[0]} & {parts[1]}")
        L.append(f"& {parts[2]} & {parts[3]} \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
    open(a.out, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
