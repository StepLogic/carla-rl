#!/usr/bin/env python3
"""Print and save the Q-vs-fault analysis for EVAL_Q_TRACE files.
usage: q_fault_report.py GAMMA TRACE.jsonl [TRACE.jsonl ...]"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from vae_nav.q_fault_analysis import analyse, load  # noqa: E402

gamma = float(sys.argv[1])
for path in sys.argv[2:]:
    rows = load(path)
    res = {"trace": path, "gamma": gamma,
           "any_fault": analyse(rows, gamma), "serious_fault": analyse(rows, gamma, serious=True)}
    with open(path.replace(".qtrace.jsonl", "") + ".qfault.json", "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"\n##### {os.path.basename(path)}  steps={res['any_fault']['n_steps']} "
          f"episodes={res['any_fault']['n_episodes']}")
    for kind in ("any_fault", "serious_fault"):
        a = res[kind]
        print(f"  [{kind}] fault-step rate {a['fault_step_rate']}")
        for k, e in a["anticipation"].items():
            print(f"    {k:18s} pos={e['pos_rate']:.3f} AUROC raw={e['auroc_raw']} "
                  f"within-distance-bins={e.get('auroc_within_distance_bins')}")
        sh = a["shape"]
        pick = [d for d in ("20", "10", "5", "3", "2", "1", "0", "none_within_30") if d in sh]
        print("    Q by steps-to-fault: " + "  ".join(
            f"{d}:{sh[d]['q_mean']}" + (f"(res {sh[d]['q_minus_distance_bin_mean']})"
                                         if 'q_minus_distance_bin_mean' in sh[d] else "")
            for d in pick))
    c = res["any_fault"]["calibration"]
    print(f"  [calibration, terminated episodes] {c}")
