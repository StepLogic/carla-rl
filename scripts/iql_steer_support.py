#!/usr/bin/env python3
"""Does the IQL critic rank steering WITHIN THE DATA'S SUPPORT?

WHY THIS EXISTS. The gate inherited from the 2026-11-09 offline critic study scans steering over the
whole action range [-1, 1] and asks where Q peaks. For an offline IQL critic that measure is
confounded: IQL deliberately never queries out-of-distribution actions, and the expert's steering is
tiny (mean |steer| 0.03-0.07), so Q at full lock is pure extrapolation -- and an MLP extrapolates
monotonically, which puts the argmax on the boundary whatever the critic has learned. The first IQL
run shows exactly that shape: span over the full grid rose to 6.5% of |Q| (c0's critic ~1%) while
the full-lock share stayed at 89-99% and the correlation with the expert sat near zero.

So this reports BOTH:
  * FULL RANGE, as before, comparable with the 11-09 numbers;
  * SUPPORT-RESTRICTED: the grid confined to the expert's action +/- `band`, where the critic has
    data. Inside that band, does Q peak near the expert's steering?
The second is the question that matters for SAC, because the actor's gradient is local: it moves the
action a little at a time from wherever the policy currently is, and it never jumps to full lock in
one step. A critic that ranks correctly near the data but extrapolates badly far away is usable with
a trust region; one that cannot rank locally is not usable at all.

usage:
  /home/robotlab/carla-sac-local/venv/bin/python -I scripts/iql_steer_support.py \
    --iql <iql_pre dir>/step_100000/params.pt \
    --control <control_pre dir>/step_5800/params.pt \
    --data <shard roots> [--band 0.1,0.25,0.5] [--val-shards 80]
"""
import argparse
import glob
import json
import os
import sys

os.environ.setdefault("MONO_CHW", "1")
os.environ.setdefault("SPEED_OBS", "1")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [REPO, os.path.join(REPO, "src"), os.path.join(REPO, "scripts")]

import numpy as np  # noqa: E402
import torch  # noqa: E402

from pretrain_critic_iql import transitions  # noqa: E402
from torch_rl.control_cnn import ControlGRU  # noqa: E402
from torch_rl.frl.iql import IQLLearner  # noqa: E402


def measure(lr, data, dev, band, grid=21, n_states=512, seed=0):
    """Q over a steering grid of half-width `band` around the expert's steering (band >= 1 is the
    full range). Reports the span against |Q|, how often the peak is at an edge of the grid, the
    correlation of the peak with the expert's steering, and -- the local test -- the mean distance
    from the peak to the expert's steering, against the 'blind critic' baseline of band/2 that a Q
    flat in steering would give."""
    rng = np.random.default_rng(seed)
    n = data["observations"].shape[0]
    idx = rng.choice(n, min(n_states, n), replace=False)
    spans, qmags, peaks, experts, edge = [], [], [], [], 0
    for i in idx:
        o = data["observations"][i].to(dev)
        a_exp = data["actions"][i].to(dev)
        lo, hi = (float(a_exp[0]) - band, float(a_exp[0]) + band) if band < 1 else (-1.0, 1.0)
        lo, hi = max(lo, -1.0), min(hi, 1.0)
        steers = torch.linspace(lo, hi, grid, device=dev)
        q = lr.q_over_actions(o, torch.stack([steers, a_exp[1].expand(grid)], dim=-1))
        j = int(q.argmax())
        spans.append(float(q.max() - q.min()))
        qmags.append(float(q.abs().mean()))
        peaks.append(float(steers[j]))
        experts.append(float(a_exp[0]))
        edge += int(j == 0 or j == grid - 1)
    peaks, experts = np.array(peaks), np.array(experts)
    span, qm = float(np.mean(spans)), float(np.mean(qmags))
    corr = (float(np.corrcoef(peaks, experts)[0, 1])
            if peaks.std() > 1e-9 and experts.std() > 1e-9 else float("nan"))
    err = float(np.abs(peaks - experts).mean())
    blind = band / 2.0 if band < 1 else 0.5      # a Q flat in steering peaks uniformly at random
    return {"band": band, "span": round(span, 4), "q_mag": round(qm, 3),
            "span_frac_pct": round(100 * span / qm, 2) if qm else None,
            "corr_peak_vs_expert": round(corr, 4), "edge_share": round(edge / len(idx), 3),
            "mean_abs_peak_minus_expert": round(err, 4),
            "blind_baseline": round(blind, 4), "beats_blind": bool(err < blind),
            "mean_abs_expert": round(float(np.abs(experts).mean()), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iql", required=True)
    ap.add_argument("--control", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--towns", default="Town01,Town04,Town05,Town10HD")
    ap.add_argument("--val-shards", type=int, default=80)
    ap.add_argument("--bands", default="0.1,0.25,0.5,1.0")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    dev = torch.device(a.device)

    ck = torch.load(a.control, map_location="cpu")
    ca = ck["args"]
    net = ControlGRU(latent_dim=ca["latent"], hidden_dim=ca["hidden"], px=ca["px"],
                     n_frames=ca["n_frames"]).to(dev)
    net.load_state_dict(ck["net"])
    net.eval().requires_grad_(False)

    shards = []
    for root in a.data.split(","):
        for t in a.towns.split(","):
            shards += sorted(glob.glob(os.path.join(root, t, "*.npz")))
    np.random.default_rng(0).shuffle(shards)
    val = transitions(shards[:a.val_shards], net, dev, ca["px"], ca["n_frames"], {}, None,
                      a.threads)

    st = torch.load(a.iql, map_location="cpu")
    lr = IQLLearner(obs_dim=int(st["obs_dim"]), action_dim=2, num_qs=st["args"]["num_qs"],
                    device=str(dev), seed=0)
    lr.load_state_dict(st["iql"])
    print(f"=== IQL {a.iql} (its gate row: {st['row']})", flush=True)
    print(f"=== control {a.control} heading R2 {ck['row']['r2_heading']:.3f} "
          f"offset R2 {ck['row']['r2_offset']:.3f}", flush=True)
    rows = [measure(lr, val, dev, float(b)) for b in a.bands.split(",")]
    for r in rows:
        print(f"[band {r['band']:.2f}] span={r['span_frac_pct']}% of |Q| "
              f"corr={r['corr_peak_vs_expert']:+.3f} edge={r['edge_share']*100:.0f}% "
              f"|peak-expert|={r['mean_abs_peak_minus_expert']:.3f} vs blind "
              f"{r['blind_baseline']:.3f} -> {'BEATS' if r['beats_blind'] else 'no better than'} "
              f"a steering-blind critic", flush=True)
    out = os.path.join(os.path.dirname(a.iql), "steer_support.json")
    with open(out, "w") as fh:
        json.dump(rows, fh, indent=1)
    print(f"=== wrote {out}", flush=True)


if __name__ == "__main__":
    main()
