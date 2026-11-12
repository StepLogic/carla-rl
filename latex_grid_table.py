"""The full matched-node table for the placement grid (user 2026-09-20 "create full latex table").

House style follows tables/table_matched_nodes_town02.tex: booktabs, one block per node budget, our arms
shaded. Every DINOv2 column is the same 40 episodes and the same policy; the shipped-localiser column is
carried for reference and is NEVER a within-row comparison -- the two localisers are different systems.
Bold marks the best SR in each block among the DINOv2-driven rows.
Run from the icra-2027 folder.
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from grid_rows import rows, nodes  # noqa: E402


def cell(v, digits=3):
    if v is None:
        return "---"
    sr, _spl, n, sd = v
    return f"{sr:.{digits}f}" + (f"\\,$\\pm$\\,{sd:.3f}" if n > 1 else "$^{\\ast}$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="tables/table_signal_grid_town02.tex")
    a = ap.parse_args()
    R = rows()
    L = [
        "% Town02 placement grid, all maps from one trace, matched node budgets.",
        "% DINOv2 columns: one run unless a +/- is shown (then mean +/- sd over runs of the SAME 40 episodes).",
        "% The MobileNet column is the shipped localiser, for reference only -- it is a different system, so it is",
        "% never compared within a row. NoMaD carries its own encoder and localiser and so has no DINOv2 entry.",
        "\\begin{table}[t]",
        "\\caption{Success rate and SPL in CARLA Town02 at matched node budgets, over the same 40 trajectories.",
        "Within a block every map has the same number of nodes. Seeded maps place 30 nodes by an uncertainty",
        "signal and fill the rest with the named filler; the three seed sets are count-matched and share no",
        "trace sample. Our arms are shaded; the best SR per block among the DINOv2 rows is bold.",
        "$^{\\ast}$single run. $^{\\dagger}$our implementation. $^{\\ddagger}$NoMaD uses its own encoder, so it",
        "appears only under its own localiser.}",
        "\\label{tab:carla-signal-grid}",
        "\\centering",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{@{}l r cc c@{}}",
        "\\toprule",
        " & & \\multicolumn{2}{c}{DINOv2 localiser} & MobileNet \\\\",
        "\\cmidrule(lr){3-4}\\cmidrule(lr){5-5}",
        "Map & Nodes & SR $\\uparrow$ & SPL $\\uparrow$ & SR \\\\",
        "\\midrule",
    ]
    blocks = []
    for r in R:
        if not blocks or blocks[-1][0] != r["block"]:
            blocks.append((r["block"], []))
        blocks[-1][1].append(r)
    for bname, brows in blocks:
        head = bname if bname == "Fixed size" else f"\\emph{{{bname} nodes}}"
        L.append(f"\\multicolumn{{5}}{{@{{}}l}}{{{head}}} \\\\")
        best = max((x["d"][0] for x in brows if x["d"]), default=None)
        for r in brows:
            pre = "\\rowcolor{ourrow} " if r["ours"] else ""
            n = nodes(r["map"])
            sr = cell(r["d"])
            if r["d"] and best is not None and abs(r["d"][0] - best) < 1e-9:
                sr = "\\textbf{" + sr + "}"
            spl = "---" if not r["d"] else (f"{r['d'][1]:.3f}" + ("" if r["d"][2] > 1 else "$^{\\ast}$"))
            ship = "---" if not r["s"] else f"{r['s'][0]:.3f}"
            L.append(f"{pre}{r['arm']:<32} & {n:4d} & {sr} & {spl} & {ship} \\\\")
        if (bname, brows) != blocks[-1]:
            L.append("\\midrule")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    open(a.out, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
