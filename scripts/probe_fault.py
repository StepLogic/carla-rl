#!/usr/bin/env python3
"""Can a fault a few steps ahead be read from what the critic sees?

usage: probe_fault.py TRACE.qtrace.jsonl   (needs TRACE's .probe.npz beside it)

WHY (user 2026-09-27: "can the critic learn from the image latent?" -> "do it").
The goal critic's Q carries no warning before a fault (AUROC 0.50). Either the
observation lacks the information, or the critic fails to learn it. A logistic
probe fitted with supervision separates the two:
  latent+state   what the critic's input layers receive (VC-1 is frozen)
  lane geometry  the true lateral offset / heading error / lane width /
                 junction flag from the CARLA map -- an upper bound on what
                 lane position alone can tell
  critic Q       the critic itself, for reference
Scored as out-of-sample AUROC with 5-fold CV GROUPED BY EPISODE, so no episode
is in both the fit and the score.
"""
import json
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from vae_nav.q_fault_analysis import load, per_step_arrays  # noqa: E402

trace = sys.argv[1]
rows = load(trace)
pz = np.load(trace.replace(".qtrace.jsonl", ".probe.npz"))
assert len(pz["lane"]) == len(rows), (len(pz["lane"]), len(rows))

# per_step_arrays sorts by (ep, t); the trace is written in that order already
order = sorted(range(len(rows)), key=lambda i: (rows[i]["ep"], rows[i]["t"]))
assert order == list(range(len(rows)))

lane = pz["lane"].astype(np.float64)
lat, herr, width, junc = lane.T
lane_feats = np.column_stack([lat, np.abs(lat), width / 2 - np.abs(lat),
                              herr, np.abs(herr), junc])
feature_sets = {
    "latent+state (critic input)": np.hstack([pz["latent"].astype(np.float32),
                                              pz["state"]]),
    "state only": pz["state"],
    "lane geometry (true, from map)": lane_feats,
    "latent+state+lane": np.hstack([pz["latent"].astype(np.float32), pz["state"],
                                    lane_feats]),
}
out = {"trace": trace, "n": len(rows)}
for serious in (False, True):
    a = per_step_arrays(rows, gamma=0.99, serious=serious)
    groups = a["ep"]
    ok = ~np.isnan(lane_feats).any(axis=1)          # t=0 rows may lack a pose
    kind = "serious_fault" if serious else "any_fault"
    out[kind] = {}
    print(f"\n[{kind}]  rows={int(ok.sum())}")
    for k in (5, 10):
        y = (a["ttf"] < k)[ok]
        g = groups[ok]
        res = {"pos_rate": round(float(y.mean()), 4),
               "critic_q": round(float(roc_auc_score(y, -a["q"][ok])), 3)}
        for name, X in feature_sets.items():
            X = X[ok]
            pred = np.zeros(len(y))
            for tr, te in GroupKFold(n_splits=5).split(X, y, g):
                clf = make_pipeline(StandardScaler(),
                                    LogisticRegression(C=0.05, max_iter=2000,
                                                       class_weight="balanced"))
                clf.fit(X[tr], y[tr])
                pred[te] = clf.predict_proba(X[te])[:, 1]
            res[name] = round(float(roc_auc_score(y, pred)), 3)
        out[kind][f"within_{k}"] = res
        print(f"  fault within {k:2d} (pos {res['pos_rate']:.3f}): " +
              "  ".join(f"{n}={v}" for n, v in res.items() if n != "pos_rate"))
with open(trace.replace(".qtrace.jsonl", ".probe.json"), "w") as fh:
    json.dump(out, fh, indent=2)
