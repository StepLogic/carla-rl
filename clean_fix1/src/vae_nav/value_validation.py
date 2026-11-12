"""Does V actually predict cost-to-go?

This has been asserted repeatedly in this project and never measured. The
gates below are the measurement.

RANK CORRELATION IS PRIMARY. Dijkstra depends only on the relative ordering of
edge costs, so a uniformly-scaled V still routes correctly; absolute error
affects only whether the reported cost means anything.
"""
import numpy as np
from scipy.stats import spearmanr

# Bins chosen to match the training band. Graph edges after thinning are mostly
# SHORT, so the 2-5 m bin is the one most likely to be quietly broken.
DISTANCE_BINS_M = [(2.0, 5.0), (5.0, 10.0), (10.0, 20.0), (20.0, 30.0)]

# PROVISIONAL. These are estimates, not derived values -- there is no prior
# measurement of V calibration in this project to set them from. Revisit once
# the first report exists.
GATE_SPEARMAN = 0.8
GATE_MEDIAN_REL_ERROR = 0.25
GATE_PER_BIN_SPEARMAN = 0.5


def spearman(pred, true):
    """Rank correlation. Returns 0.0 rather than nan for degenerate input."""
    pred, true = np.asarray(pred, np.float64), np.asarray(true, np.float64)
    if pred.size < 3 or np.all(pred == pred[0]) or np.all(true == true[0]):
        return 0.0
    rho = spearmanr(pred, true).statistic
    return 0.0 if np.isnan(rho) else float(rho)


def median_abs_rel_error(pred_steps, true_steps):
    """Median |pred - true| / true. Pairs with true == 0 are dropped."""
    pred = np.asarray(pred_steps, np.float64)
    true = np.asarray(true_steps, np.float64)
    ok = true > 0
    if not ok.any():
        return float("nan")
    return float(np.median(np.abs(pred[ok] - true[ok]) / true[ok]))


def per_bin_spearman(pred_steps, true_steps, dist_m, bins=DISTANCE_BINS_M):
    """Rank correlation within each distance band.

    Pooled rho can look healthy while an entire band is garbage -- and the
    short band is exactly where most graph edges live.
    """
    pred = np.asarray(pred_steps, np.float64)
    true = np.asarray(true_steps, np.float64)
    dist = np.asarray(dist_m, np.float64)
    out = {}
    for lo, hi in bins:
        sel = (dist >= lo) & (dist < hi)
        if sel.sum() < 3:
            continue
        out[f"{lo}-{hi}m"] = {"n": int(sel.sum()),
                              "rho": spearman(pred[sel], true[sel])}
    return out


def gate_report(pred_steps, true_steps, dist_m, v_values,
                v_min=-30.0, v_max=0.0):
    """Every acceptance gate, with the numbers behind each verdict."""
    v = np.asarray(v_values, np.float64)
    rho = spearman(pred_steps, true_steps)
    rel = median_abs_rel_error(pred_steps, true_steps)
    bins = per_bin_spearman(pred_steps, true_steps, dist_m)
    n_out = int(((v < v_min) | (v > v_max)).sum())
    worst_bin = min((b["rho"] for b in bins.values()), default=float("nan"))

    gates = {
        "spearman": {"value": rho, "threshold": GATE_SPEARMAN,
                     "passed": rho > GATE_SPEARMAN},
        "median_rel_error": {"value": rel, "threshold": GATE_MEDIAN_REL_ERROR,
                             "passed": bool(rel < GATE_MEDIAN_REL_ERROR)},
        "v_in_bounds": {"value": n_out, "threshold": 0,
                        "passed": n_out == 0,
                        "range": [v_min, v_max]},
        "per_bin_spearman": {"value": worst_bin,
                             "threshold": GATE_PER_BIN_SPEARMAN,
                             "passed": bool(bins) and worst_bin > GATE_PER_BIN_SPEARMAN},
    }
    return {"n_pairs": int(np.asarray(true_steps).size),
            "bins": bins,
            "gates": gates,
            "passed": all(g["passed"] for g in gates.values())}
