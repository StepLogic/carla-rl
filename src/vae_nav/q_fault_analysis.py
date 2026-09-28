"""Does the critic see faults coming? Offline analysis of an EVAL_Q_TRACE file.

WHY (user 2026-09-27: "does q function predict fatal states correctly?"). Every
fault costs a fine (goal mode) or ends the episode (explore mode), so a critic
that has learned the task must hold a LOWER value in the steps before a fault
than in comparable steps that are not followed by one. Three questions, each
answered separately because they fail independently (three-checks memory):

  1. ANTICIPATION  AUROC of -Q for "a fault within k steps" (k = 5, 10, 20).
     0.5 = Q carries no warning. In goal mode Q also falls with distance to the
     goal, and far starts see more faults, so the raw AUROC is confounded; the
     distance-controlled one is the average AUROC inside route-remaining bins.
  2. SHAPE         mean Q (and Q minus its distance-bin mean) against steps
     until the next fault: a warning shows up as a slope over the last steps.
  3. CALIBRATION   Q against the realised discounted return-to-go, on episodes
     that TERMINATED (arrival, or a terminal fault). A truncated or capped
     episode's return is censored -- training bootstraps its tail -- so it is
     left out rather than compared against a number the critic never targets.

A "fault" step is off-road, collision, or any lane crossing (v17's fine rule
and explore mode's terminal rule); "serious" drops legal crossings (junction,
road edge, broken-line lane change), keeping off-road, collision and illegal.
"""
import json

import numpy as np

KS = (5, 10, 20)
DIST_BINS = (0.0, 25.0, 50.0, 100.0, 200.0, np.inf)


def load(path):
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def is_fault(row, serious=False):
    if row["off_road"] or row["collision"]:
        return True
    c = row.get("crossing")
    if c is None:
        return False
    return c == "illegal" if serious else True


def auroc(score, label):
    """P(score of a positive > score of a negative), ties counted half.
    None when either class is empty."""
    score = np.asarray(score, float)
    label = np.asarray(label, bool)
    npos, nneg = int(label.sum()), int((~label).sum())
    if npos == 0 or nneg == 0:
        return None
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score))
    s = score[order]
    i = 0
    while i < len(s):                       # average ranks over ties
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[label].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def per_step_arrays(rows, gamma, serious=False):
    """Per row: q, steps to the next fault in the same episode (inf if none,
    0 = this step's action produced one), discounted return-to-go, whether
    the episode terminated, and route remaining (nan when absent)."""
    by_ep = {}
    for r in rows:
        by_ep.setdefault(r["ep"], []).append(r)
    q, ttf, g, term, rem, ep_id = [], [], [], [], [], []
    for e, ep in sorted(by_ep.items()):
        ep.sort(key=lambda r: r["t"])
        n = len(ep)
        nxt, acc = np.inf, 0.0
        ttf_e, g_e = [0.0] * n, [0.0] * n
        for i in range(n - 1, -1, -1):
            if is_fault(ep[i], serious):
                nxt = i
            ttf_e[i] = nxt - i
            acc = ep[i]["r"] + gamma * acc
            g_e[i] = acc
        # the step cap and the fault-count truncation both set trunc (the
        # env raises term as well at the cap); only an untruncated end is a
        # true terminal whose return-to-go is uncensored
        ended_term = bool(ep[-1]["term"]) and not bool(ep[-1]["trunc"])
        for i, r in enumerate(ep):
            q.append(np.nan if r["q"] is None else r["q"])
            ttf.append(ttf_e[i])
            g.append(g_e[i])
            term.append(ended_term)
            rem.append(np.nan if r.get("remaining_m") is None else r["remaining_m"])
            ep_id.append(e)
    return dict(q=np.array(q), ttf=np.array(ttf), g=np.array(g),
                term=np.array(term), rem=np.array(rem), ep=np.array(ep_id))


def bin_residual(q, rem):
    """Q minus the mean Q of its route-remaining bin (rows without a distance
    keep their raw deviation from the global mean)."""
    out = q - np.nanmean(q)
    ok = ~np.isnan(rem)
    idx = np.digitize(rem, DIST_BINS) if ok.any() else None
    if idx is not None:
        for b in np.unique(idx[ok]):
            m = ok & (idx == b)
            out[m] = q[m] - q[m].mean()
    return out


def analyse(rows, gamma, serious=False):
    a = per_step_arrays(rows, gamma, serious)
    keep = ~np.isnan(a["q"])
    q, ttf, rem = a["q"][keep], a["ttf"][keep], a["rem"][keep]
    g, term = a["g"][keep], a["term"][keep]
    have_dist = bool((~np.isnan(rem)).any())
    resid = bin_residual(q, rem) if have_dist else None
    out = {"n_steps": int(keep.sum()),
           "n_episodes": int(len(np.unique(a["ep"]))),
           "fault_step_rate": round(float(np.mean(ttf == 0)), 4),
           "anticipation": {}, "shape": {}, "calibration": None}
    for k in KS:
        lab = ttf < k
        entry = {"pos_rate": round(float(lab.mean()), 4),
                 "auroc_raw": _r(auroc(-q, lab))}
        if have_dist:
            # within-bin AUROC averaged by bin size: no cross-bin comparisons,
            # so distance cannot manufacture the separation
            idx = np.digitize(rem, DIST_BINS)
            vals, wts = [], []
            for b in np.unique(idx[~np.isnan(rem)]):
                m = (idx == b) & ~np.isnan(rem)
                v = auroc(-q[m], lab[m])
                if v is not None:
                    vals.append(v)
                    wts.append(m.sum())
            entry["auroc_within_distance_bins"] = (
                _r(float(np.average(vals, weights=wts))) if vals else None)
        out["anticipation"][f"fault_within_{k}"] = entry
    for d in list(range(0, 21)) + ["none_within_30"]:
        m = (ttf > 30) if d == "none_within_30" else (ttf == d)
        if m.sum() == 0:
            continue
        row = {"n": int(m.sum()), "q_mean": _r(float(q[m].mean()), 2)}
        if resid is not None:
            row["q_minus_distance_bin_mean"] = _r(float(resid[m].mean()), 2)
        out["shape"][str(d)] = row
    t = term
    if t.sum() >= 10:
        err = q[t] - g[t]
        out["calibration"] = {
            "n_steps_on_terminated_episodes": int(t.sum()),
            "corr_q_vs_return_to_go": _r(float(np.corrcoef(q[t], g[t])[0, 1])),
            "bias_mean_q_minus_g": _r(float(err.mean()), 2),
            "mae": _r(float(np.abs(err).mean()), 2),
            "q_mean": _r(float(q[t].mean()), 2),
            "g_mean": _r(float(g[t].mean()), 2)}
    return out


def _r(x, nd=3):
    return None if x is None else round(float(x), nd)
