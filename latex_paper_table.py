"""The paper's four-domain table, emitted once per localiser (user 2026-09-20).

Same row set and column structure as the submitted table. CARLA/Town02 and CARLA/Town07 are filled from
this repo's runs; the two Habitat columns are NOT produced here (they come from the Habitat session) and
are carried verbatim from the submitted table in the MobileNet version, blank in the DINOv2 one -- Habitat
has never been re-run with a DINOv2 localiser, and printing its MobileNet numbers under a DINOv2 heading
would be a false claim.

  --localiser mobilenet   the shipped rows: exactly what the paper has today
  --localiser dinov2      the same maps and episodes, DINOv2 features for node matching
"""
import argparse, collections, glob, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

TOWNS = {
    "Town02": dict(seq="results/maps/sequence",
                   ship=["results/drive_plan_c/*.jsonl", "results/drive_plan_c/repeat/*.jsonl",
                         "results/drive_plan_c/equal_points/*.jsonl"],
                   dino=["results/drive_plan_c/localiser/dinov2_bank_*.jsonl",
                         "results/drive_plan_c/signal_grid/*.jsonl"],
                   nomad=["results/nomad/nomad_*v6ncap*.jsonl", "results/nomad/repeat/*.jsonl",
                          "results/nomad/equal_points/*.jsonl"]),
    "Town07": dict(seq="results/town07/maps/sequence",
                   ship=["results/town07/*.jsonl", "results/town07/repeat/*.jsonl",
                         "results/town07/equal_points/*.jsonl", "results/town07/localiser/mobilenet_trace_*.jsonl"],
                   dino=["results/town07/localiser/dinov2_bank_*.jsonl"],
                   nomad=["results/town07/nomad/*.jsonl", "results/town07/nomad/repeat/*.jsonl",
                          "results/town07/nomad/equal_points/*.jsonl"]),
}
# label, Town02 map, Town07 map, is-ours, source ("nomad" rows never have a DINOv2 entry)
ROWS = [
    ("SGM$^{\\dagger}$\\cite{emmons2020sparse}", "sgm_g2_e55_c55", "sgm_g2_e55_c55", False, "main"),
    ("NoMaD\\cite{sridhar2023nomad}+Spc.", "nomad_uniform2_trace_v6ncap_20kmh",
     "nomad_uniform2_trace_v6ncap_20kmh", False, "nomad"),
    ("MIDRULE", None, None, False, None),
    ("Dense", "raw_trace", "raw_trace", False, "main"),
    ("Spacing[2,0.8\\,m]", "uniform2_trace", "uniform2_trace", False, "main"),
    ("Spacing[6,1.5\\,m]", "uniform6_trace", "uniform6_trace", False, "main"),
    ("Ent. + Spc.[6,8\\,m]", "hybrid_uniform6_entropy_thin4", "hybrid_uniform6_entropy_thin4", True, "main"),
    # user 2026-09-20: the SGM-filled twin of the row above, at the same budget (metric gate 9 m in Town02,
    # 7 m in Town07 -- the gate is what SGM varies to hit a node count, so it differs per town).
    ("Ent. + SGM$^{\\dagger}$", "hybrid_sgm_g9_entropy_thin4", "hybrid_sgm_g7_entropy_thin4", True, "main"),
    ("Entropy (Ours)", "entropy_nosmooth_high_thin4", "entropy_nosmooth_high_thin4", True, "main"),
]
# Habitat, verbatim from the submitted table -- produced by the Habitat session, not by this repo.
HABITAT = {
    "SGM$^{\\dagger}$\\cite{emmons2020sparse}":
        "\\phantom{00}89 & 0.18\\,$\\pm$\\,0.01 & 0.11\\,$\\pm$\\,0.03 & \\textbf{\\phantom{00}44} & 0.18\\,$\\pm$\\,0.03 & 0.12\\,$\\pm$\\,0.01",
    "NoMaD\\cite{sridhar2023nomad}+Spc.": "- & - & - & - & - & -",
    "Dense": "5000 & 0.22\\,$\\pm$\\,0.08 & 0.13\\,$\\pm$\\,0.06 & 5000 & 0.25\\,$\\pm$\\,0.02 & 0.17\\,$\\pm$\\,0.02",
    "Spacing[2,0.8\\,m]":
        "\\phantom{0}317 & 0.23\\,$\\pm$\\,0.06 & 0.15\\,$\\pm$\\,0.05 & \\phantom{0}277 & 0.21\\,$\\pm$\\,0.05 & 0.14\\,$\\pm$\\,0.02",
    "Spacing[6,1.5\\,m]":
        "\\phantom{0}166 & 0.23\\,$\\pm$\\,0.06 & 0.14\\,$\\pm$\\,0.02 & \\phantom{0}153 & 0.20\\,$\\pm$\\,0.02 & 0.13\\,$\\pm$\\,0.02",
    "Ent. + Spc.[6,8\\,m]":
        "\\phantom{00}76 & 0.24\\,$\\pm$\\,0.08 & 0.16\\,$\\pm$\\,0.04 & \\phantom{00}64 & 0.22\\,$\\pm$\\,0.08 & 0.13\\,$\\pm$\\,0.03",
    # no Habitat run exists for the SGM-filled variant
    "Ent. + SGM$^{\\dagger}$": "- & - & - & - & - & -",
    "Entropy (Ours)":
        "\\textbf{\\phantom{00}30} & 0.23\\,$\\pm$\\,0.02 & 0.12\\,$\\pm$\\,0.01 & \\phantom{00}45 & 0.21\\,$\\pm$\\,0.06 & 0.13\\,$\\pm$\\,0.04",
}
BLANK_HABITAT = "- & - & - & - & - & -"


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
    if m not in per:
        return None
    sr = [np.mean([r["success"] for r in d.values()]) for d in per[m]]
    spl = [np.mean([r["spl"] or 0 for r in d.values()]) for d in per[m]]
    f = lambda v: (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else None)
    return f(sr), f(spl), len(sr)


def nodes(seq, m):
    import re
    base = re.sub(r"^nomad_(.*)_v6ncap_20kmh$", r"\1", m)
    return len(json.load(open(f"{seq}/{base}_sequence.json"))["nodes"])


def pad(n):
    s = str(n)
    return "\\phantom{" + "0" * (4 - len(s)) + "}" + s if len(s) < 4 else s


def val(v, best):
    if v is None:
        return "-"
    (mu, sd) = v
    txt = f"{mu:.2f}" + (f"\\,$\\pm$\\,{sd:.2f}" if sd is not None else "$^{\\ast}$")
    return "\\textbf{" + txt + "}" if best else txt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--localiser", choices=["mobilenet", "dinov2"], required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    want = "mobilenet_trace" if a.localiser == "mobilenet" else "dinov2_bank"
    data, nomad = {}, {}
    for t, cfg in TOWNS.items():
        data[t] = load(cfg["ship" if a.localiser == "mobilenet" else "dino"], want)
        nomad[t] = load(cfg["nomad"], None)

    cells = {}
    for label, m02, m07, _ours, src in ROWS:
        if label == "MIDRULE":
            continue
        for t, m in (("Town02", m02), ("Town07", m07)):
            per = nomad[t] if src == "nomad" else data[t]
            if src == "nomad" and a.localiser == "dinov2":
                cells[(label, t)] = (None, None, None)
                continue
            s = stat(per, m)
            cells[(label, t)] = (nodes(TOWNS[t]["seq"], m) if s else None, s[0] if s else None,
                                 s[1] if s else None)
    bestsr = {t: max((c[1][0] for (l, tt), c in cells.items() if tt == t and c[1]), default=None)
              for t in TOWNS}
    bestspl = {t: max((c[2][0] for (l, tt), c in cells.items() if tt == t and c[2]), default=None)
               for t in TOWNS}
    bestn = {t: min((c[0] for (l, tt), c in cells.items() if tt == t and c[0]), default=None) for t in TOWNS}

    head = ("the shipped MobileNet localiser" if a.localiser == "mobilenet"
            else "the DINOv2 localiser (CARLA only; Habitat has not been re-run)")
    CAP = {
        "mobilenet":
            ("Success rate and SPL across four environments, with node matching driven by the frozen "
             "ImageNet encoder used throughout the paper. CARLA maps are built from a single exploration "
             "trace per town and driven on the same 40 fixed trajectories; $\\pm$ is the spread over "
             "repeated runs of those trajectories, so it reports simulator variability rather than seed "
             "variance. Shaded rows seed a map with 30 actor-entropy nodes and fill the remainder with the "
             "named rule; Ent.~+~SGM uses SGM's own merge test as the filler, its metric gate set per town "
             "(9\\,m Town02, 7\\,m Town07) to match the node budget of the row above. Habitat columns come "
             "from the Habitat experiments, not from the CARLA pipeline. $^{\\ast}$single run. "
             "$^{\\dagger}$our implementation.",
             "tab:main-mobilenet"),
        "dinov2":
            ("The same maps, trajectories, policy and node budgets as Table~\\ref{tab:main-mobilenet}: only "
             "the features used to match an observation to a map node change, from the frozen ImageNet "
             "encoder to DINOv2~ViT-S/14 patch-mean descriptors. Pooled over both CARLA towns the swap wins "
             "112 of 161 decided trajectories ($p<10^{-4}$, $+0.066$ SR; $+0.076$ once the saturated dense "
             "maps are excluded), and the gain concentrates on sparse maps, where the older encoder matched "
             "visually similar but distant places. Shaded rows are as in Table~\\ref{tab:main-mobilenet}, "
             "with Ent.~+~SGM at gates of 9\\,m (Town02) and 7\\,m (Town07). NoMaD carries its own encoder "
             "and localiser, and Habitat was not re-run, so those groups are empty by construction rather "
             "than by omission. $^{\\ast}$single run. $^{\\dagger}$our implementation.",
             "tab:main-dinov2"),
    }[a.localiser]
    L = [f"% CARLA columns: {head}. Same maps, same 40 episodes per town.",
         "% Habitat columns are NOT produced in this repo." +
         ("  Carried verbatim from the submitted table." if a.localiser == "mobilenet"
          else "  Blank: no DINOv2 run exists for Habitat."),
         "% $\\ast$ = single run; a +/- is mean +/- sd over runs of the SAME episodes (simulator variability).",
         "\\begin{table*}[t]",
         "\\centering",
         "\\footnotesize",
         "\\setlength{\\tabcolsep}{4pt}",
         f"\\caption{{{CAP[0]}}}",
         f"\\label{{{CAP[1]}}}",
         "\\begin{tabular}{@{}l rcc rcc rcc rcc@{}}",
         "\\toprule",
         "& \\multicolumn{3}{c}{CARLA/Town02} & \\multicolumn{3}{c}{CARLA/Town07}",
         "& \\multicolumn{3}{c}{Habitat/Pleasant} & \\multicolumn{3}{c}{Habitat/Artois} \\\\",
         "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
         "\\cmidrule(lr){8-10}\\cmidrule(lr){11-13}",
         "Map & Nds $\\downarrow$ & SR $\\uparrow$ & SPL $\\uparrow$ & Nds $\\downarrow$ & SR $\\uparrow$ & SPL $\\uparrow$",
         "& Nds $\\downarrow$ & SR $\\uparrow$ & SPL $\\uparrow$ & Nds $\\downarrow$ & SR $\\uparrow$ & SPL $\\uparrow$ \\\\",
         "\\midrule"]
    for label, _m02, _m07, ours, _src in ROWS:
        if label == "MIDRULE":
            L.append("\\midrule"); continue
        parts = []
        for t in ("Town02", "Town07"):
            n, sr, spl = cells[(label, t)]
            if n is None:
                parts.append("- & - & -"); continue
            ntxt = pad(n)
            if bestn[t] is not None and n == bestn[t]:
                ntxt = "\\textbf{" + ntxt + "}"
            parts.append(f"{ntxt} & {val(sr, sr and bestsr[t] is not None and abs(sr[0]-bestsr[t])<1e-9)} "
                         f"& {val(spl, spl and bestspl[t] is not None and abs(spl[0]-bestspl[t])<1e-9)}")
        hab = HABITAT[label] if a.localiser == "mobilenet" else BLANK_HABITAT
        if ours:
            L.append("\\rowcolor{ourrow}")
        L.append(f"{label} & {parts[0]} & {parts[1]}")
        L.append(f"& {hab} \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
    out = a.out or f"tables/table_paper_{a.localiser}.tex"
    open(out, "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\n-> {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
