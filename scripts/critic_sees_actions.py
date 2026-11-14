#!/usr/bin/env python3
"""CAN THE CRITIC SEE THE ACTION AT ALL? (user 2026-11-14: "focus in RL - lets see if critic can see
actions".)

Everything measured so far says Q barely depends on steering: span 0.30% of |Q| over the band the
data lives in, 2.9% over the whole range, peak at the edge of any grid, no better than a
steering-blind critic -- with the executed action, heading at R^2 0.81 and 227k expert transitions.
Two very different explanations remain, and they imply opposite next steps:

  A. THE RETURN GENUINELY BARELY RESPONDS to one steering choice (short step, self-correcting
     behaviour policy). Then the fix is the action's consequence: hold it longer, or make it a
     sub-goal. The k-sweep tests that.
  B. THE CRITIC CANNOT SEE THE ACTION, for a mundane reason: it is 2 numbers among 3,267 inputs,
     3,136 of them DEBiT features, so the action is 0.06% of the input vector and may simply be
     drowned. Learner C compresses to a 50-d latent first, where the action is 2 of 52; this
     pretraining dropped that "for more capacity". Then the fix is the critic's input layout, and
     no amount of action repeat would help.

This separates them with three measurements:
  1. INPUT SCALES per block (control GRU state / DEBiT binocular / speed / action), so dilution can
     be seen rather than guessed.
  2. PER-AXIS Q SENSITIVITY: span across a steering grid and across a THROTTLE grid. The policy
     learned throttle (it drives, it brakes to arrive) and never learned steering, so if the critic
     ranks throttle and not steering, it does see actions and explanation A survives.
  3. A PERMUTATION TEST, the cleanest form of the question: replace each row's action with another
     row's and measure |dQ|, against permuting an equal-sized random slice of the OBSERVATION. If
     shuffling the action moves Q far less than shuffling 2 arbitrary observation features, the
     action is being ignored and explanation B holds.

usage:
  /home/robotlab/carla-sac-local/venv/bin/python -I scripts/critic_sees_actions.py \
    --iql <dir>/step_60000/params.pt --control <dir>/step_5800/params.pt --data <roots>
"""
import argparse
import glob
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


@torch.no_grad()
def axis_span(lr, data, dev, axis, grid=21, n=512, seed=0):
    """Q across a grid on ONE action axis, the other axis held at the dataset's value."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(data["observations"].shape[0], min(n, data["observations"].shape[0]), False)
    vals = torch.linspace(-1, 1, grid, device=dev)
    spans, mags, peaks, datums = [], [], [], []
    for i in idx:
        o = data["observations"][i].to(dev)
        a = data["actions"][i].to(dev)
        acts = (torch.stack([vals, a[1].expand(grid)], -1) if axis == 0
                else torch.stack([a[0].expand(grid), vals], -1))
        q = lr.q_over_actions(o, acts)
        spans.append(float(q.max() - q.min())); mags.append(float(q.abs().mean()))
        peaks.append(float(vals[int(q.argmax())])); datums.append(float(a[axis]))
    span, mag = float(np.mean(spans)), float(np.mean(mags))
    peaks, datums = np.array(peaks), np.array(datums)
    corr = (float(np.corrcoef(peaks, datums)[0, 1])
            if peaks.std() > 1e-9 and datums.std() > 1e-9 else float("nan"))
    return {"span": span, "span_pct_of_Q": 100 * span / mag if mag else None,
            "corr_peak_vs_data": corr, "edge_share": float(np.mean(np.abs(peaks) >= 0.999)),
            "data_abs_mean": float(np.abs(datums).mean())}


@torch.no_grad()
def permutation_test(lr, data, dev, n=2048, seed=0, obs_block=2):
    """|dQ| when the ACTION is replaced by another row's, against |dQ| when an equal-sized random
    slice of the OBSERVATION is permuted the same way. The ratio says whether the critic uses the
    action more or less than two arbitrary observation features."""
    rng = np.random.default_rng(seed)
    N = data["observations"].shape[0]
    idx = rng.choice(N, min(n, N), False)
    o = data["observations"][idx].to(dev)
    a = data["actions"][idx].to(dev)
    perm = torch.from_numpy(rng.permutation(len(idx))).to(dev)
    q0 = lr.critic(o, a).min(dim=0).values
    q_act = lr.critic(o, a[perm]).min(dim=0).values
    cols = rng.choice(o.shape[1], obs_block, replace=False)
    o2 = o.clone()
    o2[:, torch.from_numpy(cols).to(dev)] = o[perm][:, torch.from_numpy(cols).to(dev)]
    q_obs = lr.critic(o2, a).min(dim=0).values
    d_act, d_obs = float((q_act - q0).abs().mean()), float((q_obs - q0).abs().mean())
    return {"q_abs_mean": float(q0.abs().mean()), "dQ_shuffle_action": d_act,
            "dQ_shuffle_2_obs_features": d_obs,
            "action_vs_obs_ratio": d_act / d_obs if d_obs else None,
            "dQ_action_pct_of_Q": 100 * d_act / float(q0.abs().mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iql", required=True)
    ap.add_argument("--control", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--towns", default="Town01,Town04,Town05,Town10HD")
    ap.add_argument("--val-shards", type=int, default=40)
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
    val = transitions(shards[:a.val_shards], net, dev, ca["px"], ca["n_frames"], {}, None, a.threads)

    st = torch.load(a.iql, map_location="cpu")
    lr = IQLLearner(obs_dim=int(st["obs_dim"]), action_dim=2, num_qs=st["args"]["num_qs"],
                    device=str(dev), seed=0)
    lr.load_state_dict(st["iql"])

    h = int(ca["hidden"])
    o = val["observations"]
    blocks = {"control GRU state": o[:, :h], "DEBiT binocular": o[:, h:-1], "speed": o[:, -1:],
              "action (steer, throttle)": val["actions"]}
    print(f"=== {a.iql}\n=== obs dim {o.shape[1]} = {h} control + {o.shape[1]-h-1} binoc + 1 speed, "
          f"action 2 -> the action is {100*2/(o.shape[1]+2):.2f}% of the critic's input vector",
          flush=True)
    print("--- INPUT SCALES (is the action drowned?)", flush=True)
    for name, blk in blocks.items():
        b = blk.float()
        print(f"    {name:26s} dims {b.shape[1]:5d}  |x| mean {b.abs().mean():8.4f}  "
              f"sd {b.std():8.4f}  block L2 {b.norm(dim=1).mean():9.3f}", flush=True)
    print("--- PER-AXIS Q SENSITIVITY", flush=True)
    for axis, name in ((0, "STEERING"), (1, "THROTTLE")):
        r = axis_span(lr, val, dev, axis)
        print(f"    {name:9s} span {r['span']:8.4f} = {r['span_pct_of_Q']:5.2f}% of |Q|  "
              f"corr(peak, data) {r['corr_peak_vs_data']:+.3f}  edge {100*r['edge_share']:3.0f}%  "
              f"|data| mean {r['data_abs_mean']:.3f}", flush=True)
    print("--- PERMUTATION TEST (does Q use the action at all?)", flush=True)
    p = permutation_test(lr, val, dev)
    print(f"    |Q| {p['q_abs_mean']:.3f} | shuffling the ACTION moves Q by "
          f"{p['dQ_shuffle_action']:.4f} ({p['dQ_action_pct_of_Q']:.2f}% of |Q|)", flush=True)
    print(f"    shuffling 2 random OBS features moves Q by {p['dQ_shuffle_2_obs_features']:.4f} "
          f"-> action/obs ratio {p['action_vs_obs_ratio']:.2f}", flush=True)
    print("    ratio << 1 means the action is ignored relative to arbitrary observation features "
          "(a critic-input problem); ratio >> 1 means the critic does see the action.", flush=True)


if __name__ == "__main__":
    main()
