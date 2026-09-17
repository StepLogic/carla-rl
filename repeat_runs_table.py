"""Table of the 7 map-results rows over 3 runs of the same 40 episodes (user 2026-09-15: "repeat driving experiments
for 3 seeds and update table"; answers: same 40 episodes, 3 runs). Run 1 = the original drives; runs 2-3 =
results/drive_plan_c/repeat/run{2,3}.jsonl and results/nomad/repeat/nomad_uniform2_trace_v6ncap_20kmh_run{2,3}.jsonl.

Reports per row: nodes, SR and SPL per run, and mean +/- sd over the runs (sd with ddof=1). Refuses a row whose run
does not hold exactly 40 episodes. Writes <out>.md / .csv and prints a LaTeX body.
"""
import argparse, csv, json
import numpy as np

ROWS = [  # (label, map key in the JSONL, run-1 file, repeat file pattern, sequence map for node count)
    ("Dense", "raw_trace", "results/drive_plan_c/raw_trace.jsonl", "results/drive_plan_c/repeat/run{r}.jsonl", "raw_trace"),
    ("Spacing[2 m]", "uniform2_trace", "results/drive_plan_c/sequence_undirected.jsonl", "results/drive_plan_c/repeat/run{r}.jsonl", "uniform2_trace"),
    ("Spacing[6 m]", "uniform6_trace", "results/drive_plan_c/sequence_undirected.jsonl", "results/drive_plan_c/repeat/run{r}.jsonl", "uniform6_trace"),
    ("Entropy + Spacing[6 m] (Ours)", "hybrid_uniform6_entropy_thin4", "results/drive_plan_c/sequence_undirected.jsonl", "results/drive_plan_c/repeat/run{r}.jsonl", "hybrid_uniform6_entropy_thin4"),
    ("Entropy (Ours)", "entropy_nosmooth_high_thin4", "results/drive_plan_c/sequence_undirected.jsonl", "results/drive_plan_c/repeat/run{r}.jsonl", "entropy_nosmooth_high_thin4"),
    ("SGM", "sgm_g2_e55_c55", "results/drive_plan_c/sgm_ablation.jsonl", "results/drive_plan_c/repeat/run{r}.jsonl", "sgm_g2_e55_c55"),
    ("NoMaD", "nomad_uniform2_trace_v6ncap_20kmh", "results/nomad/nomad_uniform2_trace_v6ncap_20kmh.jsonl", "results/nomad/repeat/nomad_uniform2_trace_v6ncap_20kmh_run{r}.jsonl", "uniform2_trace"),
]


def load(path, key):
    try:
        return [r for r in (json.loads(l) for l in open(path) if l.strip()) if r["map"] == key]
    except FileNotFoundError:
        return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--out", default="results/drive_plan_c/repeat/map_results_3runs")
    a = ap.parse_args()
    rows = []
    for label, key, first, pattern, seq in ROWS:
        srs, spls, ns = [], [], []
        for r in range(1, a.runs + 1):
            L = load(first if r == 1 else pattern.format(r=r), key)
            ns.append(len(L))
            if len(L) != 40:
                continue
            srs.append(np.mean([x["success"] for x in L])); spls.append(np.mean([x["spl"] for x in L]))
        nodes = len(json.load(open(f"results/maps/sequence/{seq}_sequence.json"))["nodes"])
        sd = lambda v: float(np.std(v, ddof=1)) if len(v) > 1 else float("nan")
        rows.append({"label": label, "nodes": nodes, "episodes_per_run": ns, "sr_runs": [round(float(x), 3) for x in srs],
                     "spl_runs": [round(float(x), 3) for x in spls], "sr_mean": float(np.mean(srs)) if srs else float("nan"),
                     "sr_sd": sd(srs), "spl_mean": float(np.mean(spls)) if spls else float("nan"), "spl_sd": sd(spls),
                     "complete": len(srs) == a.runs})
    with open(a.out + ".csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["map", "nodes", "episodes_per_run", "sr_runs", "sr_mean", "sr_sd", "spl_runs", "spl_mean", "spl_sd"])
        for x in rows:
            w.writerow([x["label"], x["nodes"], x["episodes_per_run"], x["sr_runs"], f"{x['sr_mean']:.3f}", f"{x['sr_sd']:.3f}",
                        x["spl_runs"], f"{x['spl_mean']:.3f}", f"{x['spl_sd']:.3f}"])
    md = [f"# Map results over {a.runs} runs of the same 40 episodes\n", "| Map | Nodes | SR per run | SR mean ± sd | SPL per run | SPL mean ± sd |",
          "|---|---|---|---|---|---|"]
    md += [f"| {x['label']} | {x['nodes']} | {x['sr_runs']} | {x['sr_mean']:.2f} ± {x['sr_sd']:.2f} | {x['spl_runs']} | "
           f"{x['spl_mean']:.2f} ± {x['spl_sd']:.2f} |" + ("" if x["complete"] else f" INCOMPLETE {x['episodes_per_run']}") for x in rows]
    open(a.out + ".md", "w").write("\n".join(md) + "\n")
    print("\n".join(md))
    print("\n% LaTeX rows: SR and SPL as mean $\\pm$ sd")
    for x in rows:
        print(f"{x['label']} & {x['nodes']} & {x['sr_mean']:.2f}$\\pm${x['sr_sd']:.2f} & {x['spl_mean']:.2f}$\\pm${x['spl_sd']:.2f} \\\\")


if __name__ == "__main__":
    main()
