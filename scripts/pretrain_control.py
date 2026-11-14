#!/usr/bin/env python3
"""Pretrain the CONTROL STREAM on the recorded expert data (user 2026-11-14: "ok lets start
pretainig"; "pretrained with already collected expert data from c0 and c1").

WHAT AND WHY. A small CNN (torch_rl.control_cnn) reads the last n frames, a GRU integrates them and
the speed, and two heads are trained:
  * ACTION: the expert's label (behaviour cloning) -- the job the network must end up doing;
  * LANE: lateral offset and heading error from torch_rl.lane_labels (positions + the town's
    OpenDRIVE map), TRAINING-TIME ONLY, dropped before RL.
The lane head is the point. The 2026-11-09 probe showed c0's features carry offset (R^2 0.70-0.81)
but not heading error (camera latent 0.046; GRU state 0.31-0.36), and heading is what a steering
action changes -- with image features a trained critic's Q varies ~1% across the whole steering grid,
and ascending it destroyed c0's own actor in 10k steps. Supervising heading directly is the cheapest
way to force the missing variable into the latent instead of hoping imitation keeps it.

NOT DEBiT-STYLE. DEBiT's RPEV pose pretext task only pays in its two-view transformer: the paper's
Table 3 gives a Siamese half-width ResNet-18 8.0 SR / 7.7 SPL WITH RPEV against 10.1 / 9.6 without,
and says late-fusion CNNs "cannot exploit this signal". What is borrowed is the loss form (L1 on a
geometric target) and nothing else.

THE GATE, declared before the first run: heading R^2 >= 0.5 from the GRU state on HELD-OUT episodes,
with offset >= 0.7. c0 scores 0.046 (camera) and 0.31-0.36 (GRU). Below that, the control stream has
not fixed the measured deficit and there is no point fine-tuning it with SAC.

THE FIRST STREAMS DID NOT PASS IT, AND WERE REPORTED AS PASSING (corrected 2026-11-14). This script
judged them with a ridge probe fitted and scored on the SAME validation rows ("heading 0.81"), while
c0's figures come from scripts/probe_c0_features.py, which holds whole episodes out. Scored fairly on
123 episodes it never loaded, control_pre_n2_k8/step_6000 gives: its own lane head offset R^2 0.38,
heading 0.10; an episode-held-out probe on its state 0.40 / 0.08; its own action head's steering
R^2 -0.60 -- so it does not carry heading, which is what it was built for, and it is overfit (val
lane loss 0.44 against train 0.12 in its own log). It is NOT shown to be worse than c0: on those same
episodes c0's GRU state probes at offset 0.04, heading 0.03 (scripts/control_stream_vs_c0.py); c0's
0.31-0.36 came from other, more numerous episodes. The validation line below now prints, in the order to read them: the heads' own
R^2 (nothing fitted), the probe with episodes held out (torch_rl.control_data.ridge_r2_episode_cv,
the same procedure as c0's probe), and the old in-sample probe, labelled as such, to show the gap.
THE GATE IS READ ON THE FIRST TWO.

usage (local GPU; this venv has torch+cu121 AND the carla module for the lane labels):
  /home/robotlab/carla-sac-local/venv/bin/python -I scripts/pretrain_control.py \
      --data /home/robotlab/carla-sac-local/checkpoints/il_data,/home/robotlab/carla-sac-local/checkpoints/il_data_dagger_chain1 \
      --out /home/robotlab/carla-sac-local/checkpoints/control_pre_n2 --n-frames 2 --updates 6000
"""
import argparse
import glob
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

os.environ.setdefault("MONO_CHW", "1")
os.environ.setdefault("SPEED_OBS", "1")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [REPO, os.path.join(REPO, "src")]

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from torch_rl import lane_labels as ll  # noqa: E402
from torch_rl.control_cnn import ControlGRU, random_shift  # noqa: E402
from torch_rl.control_data import (chunk_from_episode, r2_score, ridge_r2_episode_cv,  # noqa: E402
                                   ridge_r2_in_sample)

# Label scales from results/step0_encoder_2026-11-09/probe.json (label_sd): offset 0.21 m,
# heading_err 3.457 deg. The aux targets are divided by these so neither term dominates the other.
SD = np.array([0.21, 3.457], np.float32)
MIN_KMH = 5.0            # below this the heading label is undefined (the probe used the same bar)
# THE LANE LABEL IS ONLY MEANINGFUL WHILE THE CAR IS IN A LANE. The DAgger data is c0 driving, so it
# contains genuinely off-lane and mid-fault frames whose offset runs to metres; at the probe's SD of
# 0.21 m a 3 m offset becomes a target of 14, and a squared error on that swamped every other term
# (the first run's lane loss grew 4.6 -> 31.3 while R^2 fell). Those frames are masked out, and the
# loss is the L1/Huber form DEBiT's RPEV head uses rather than MSE.
MAX_OFFSET_M = 3.0
MAX_HEADING_DEG = 90.0


def load_shard(path, lanes_cache, xodr_dir, labels=True):
    """One shard -> the arrays chunk_from_episode wants. labels=False skips the OpenDRIVE queries
    (the IQL stage needs frames, actions, speed and rewards but no lane geometry) and returns zeroed
    label columns with an all-False valid mask, so nothing downstream can mistake them for labels."""
    z = np.load(path)
    town = str(z["town"]) if "town" in z.files else os.path.basename(os.path.dirname(path))
    if not labels:
        n_act = int(z["act_label"].shape[0])
        zeros = np.zeros(n_act + 1, np.float32)
        return {"frames": z["frames"], "act_label": z["act_label"],
                "speed": np.asarray(z["speed"], np.float32), "offset": zeros,
                "heading_err": zeros, "valid": np.zeros(n_act, bool), "town": town,
                "n_act": n_act}
    if town not in lanes_cache:
        lanes_cache[town] = ll.OpenDriveLanes(town, xodr_dir=xodr_dir)
    lab = ll.episode_labels(z["pos"], lanes_cache[town])
    n_act = int(z["act_label"].shape[0])
    valid = np.asarray(lab["valid"], bool)[:n_act].copy()
    speed = np.asarray(z["speed"], np.float32)
    valid &= speed[:n_act] >= MIN_KMH                    # heading is undefined at rest
    off, herr = np.asarray(lab["offset"], float), np.asarray(lab["heading_err"], float)
    with np.errstate(invalid="ignore"):                  # nan compares false, which is what we want
        valid &= np.abs(off[:n_act]) <= MAX_OFFSET_M
        valid &= np.abs(herr[:n_act]) <= MAX_HEADING_DEG
    return {"frames": z["frames"], "act_label": z["act_label"], "speed": speed,
            "offset": lab["offset"], "heading_err": lab["heading_err"], "valid": valid,
            "town": town, "n_act": n_act}


def batch(pool, rng, b, length, n_frames, dev, px, repeat=1):
    """b random chunks from the pool -> device tensors. Episodes too short for a chunk are skipped."""
    picks, which, tries = [], [], 0
    while len(picks) < b and tries < 20 * b:
        tries += 1
        i = int(rng.integers(len(pool)))
        ep = pool[i]
        span = (length - 1) * repeat         # ticks covered: GRU steps are `repeat` ticks apart
        if ep["n_act"] <= span:
            continue
        picks.append(chunk_from_episode(ep, int(rng.integers(0, ep["n_act"] - span)),
                                        length, n_frames, stride=repeat, step=repeat))
        which.append(i)                      # the chunk's EPISODE: a probe must hold episodes out
    f = torch.from_numpy(np.stack([p["frames"] for p in picks])).to(dev, non_blocking=True)
    B, L = f.shape[0], f.shape[1]
    f = f.reshape(B * L, *f.shape[2:]).float()
    if f.shape[-1] != px:
        f = F.interpolate(f, size=(px, px), mode="area")
    out = {"frames": f.reshape(B, L, *f.shape[1:])}
    for k, dt in (("actions", np.float32), ("speed", np.float32), ("lane", np.float32),
                  ("lane_valid", np.float32)):
        out[k] = torch.from_numpy(np.stack([p[k] for p in picks]).astype(dt)).to(dev)
    out["lane"] = out["lane"] / torch.tensor(SD, device=dev)
    out["episode"] = np.asarray(which, np.int64)
    return out


def run_chunk(net, bt, burn_in, aug_pad):
    """The GRU over a chunk; losses on the steps after the burn-in. Returns (bc, lane, hiddens)."""
    B, L = bt["frames"].shape[0], bt["frames"].shape[1]
    h = net.reset_hidden(B, bt["frames"].device)
    bc = lane = n_lane = 0.0
    hs = []
    for t in range(L):
        x = bt["frames"][:, t]
        if aug_pad:
            x = random_shift(x, aug_pad)
        out, h = net(x, bt["speed"][:, t:t + 1], h)
        hs.append(out.detach())
        if t < burn_in:
            continue
        bc = bc + F.mse_loss(net.action(out), bt["actions"][:, t])
        # Boolean INDEXING, not multiply-by-mask: `nan * 0` is nan, and Huber (RPEV's L1 form) keeps
        # the few large-offset frames that survive the mask from dominating the gradient.
        m = bt["lane_valid"][:, t] > 0
        if bool(m.any()):
            lane = lane + F.smooth_l1_loss(net.lane(out)[m], bt["lane"][:, t][m])
            n_lane += 1
    k = max(1, L - burn_in)
    return bc / k, (lane / n_lane if n_lane else torch.zeros((), device=x.device)), torch.stack(hs, 1)


def ridge_r2(H, Y, lam=1.0):
    """IN-SAMPLE ridge probe: fitted and scored on the SAME rows. NOT what the 11-09 probe reports
    (that one holds episodes out) and NOT evidence that the state carries anything -- it gave
    "heading 0.81" for a stream whose held-out figure is about 0.1. Kept, under its old name, only
    so the gap to the honest figures stays visible in the log."""
    return ridge_r2_in_sample(np.asarray(H), np.asarray(Y), lam)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="comma-separated shard roots")
    ap.add_argument("--out", required=True)
    ap.add_argument("--towns", default="Town01,Town04,Town05,Town10HD")
    ap.add_argument("--n-frames", type=int, default=2)
    ap.add_argument("--px", type=int, default=84)
    ap.add_argument("--repeat", type=int, default=1,
                    help="the decision rate the stream will run at (ACTION_REPEAT): the GRU "
                         "steps once every `repeat` ticks and the frame stack is `repeat` apart")
    ap.add_argument("--latent", type=int, default=50)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--updates", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--chunk", type=int, default=24)
    ap.add_argument("--burn-in", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--aux-lane", type=float, default=1.0, help="0 = behaviour cloning only")
    ap.add_argument("--aug-pad", type=int, default=4, help="DrQ random shift; 0 = off")
    ap.add_argument("--pool", type=int, default=300, help="shards held in RAM")
    ap.add_argument("--val-shards", type=int, default=40)
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--threads", type=int, default=10)
    ap.add_argument("--xodr", default=ll.DEFAULT_XODR_DIR)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(0)
    towns = a.towns.split(",")

    shards = []
    for root in a.data.split(","):
        for t in towns:
            shards += sorted(glob.glob(os.path.join(root, t, "*.npz")))
    rng.shuffle(shards)
    val_paths, train_paths = shards[:a.val_shards], shards[a.val_shards:]
    print(f"=== {len(train_paths)} train shards, {len(val_paths)} val shards, towns {towns}",
          flush=True)

    lanes, pool_paths = {}, list(rng.choice(train_paths, min(a.pool, len(train_paths)), replace=False))
    with ThreadPoolExecutor(a.threads) as ex:
        t0 = time.time()
        pool = [e for e in ex.map(lambda p: load_shard(p, lanes, a.xodr), pool_paths) if e["n_act"] > (a.chunk - 1) * a.repeat]
        val = [e for e in ex.map(lambda p: load_shard(p, lanes, a.xodr), val_paths) if e["n_act"] > (a.chunk - 1) * a.repeat]
    print(f"=== pool {len(pool)} episodes, val {len(val)} episodes, loaded in {time.time()-t0:.0f}s",
          flush=True)

    dev = torch.device(a.device)
    net = ControlGRU(latent_dim=a.latent, hidden_dim=a.hidden, px=a.px,
                     n_frames=a.n_frames).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    print(f"=== params {sum(p.numel() for p in net.parameters())/1e6:.2f}M  n_frames={a.n_frames} "
          f"px={a.px} hidden={a.hidden} aux_lane={a.aux_lane} aug_pad={a.aug_pad} repeat={a.repeat}", flush=True)

    log, t0 = [], time.time()
    for u in range(1, a.updates + 1):
        bt = batch(pool, rng, a.batch, a.chunk, a.n_frames, dev, a.px, a.repeat)
        bc, lane, _ = run_chunk(net, bt, a.burn_in, a.aug_pad)
        loss = bc + a.aux_lane * lane
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if u % a.log_every and u != a.updates:
            continue
        net.eval()
        with torch.no_grad():
            vb, vl, H, Y, P, G, SP, SY = [], [], [], [], [], [], [], []
            for _ in range(8):
                bv = batch(val, rng, a.batch, a.chunk, a.n_frames, dev, a.px, a.repeat)
                b_, l_, hs = run_chunk(net, bv, a.burn_in, 0)
                vb.append(float(b_)); vl.append(float(l_))
                steps = hs.shape[1] - a.burn_in
                m = bv["lane_valid"][:, a.burn_in:].reshape(-1).bool().cpu().numpy()
                H.append(hs[:, a.burn_in:].reshape(-1, hs.shape[-1]).cpu().numpy()[m])
                Y.append((bv["lane"][:, a.burn_in:].reshape(-1, 2).cpu().numpy()
                          * SD)[m])                       # un-standardised: metres and degrees
                # the stream's OWN heads on the same rows, and each row's episode
                P.append((net.lane(hs[:, a.burn_in:]).reshape(-1, 2).cpu().numpy() * SD)[m])
                G.append(np.repeat(bv["episode"], steps)[m])
                SP.append(net.action(hs[:, a.burn_in:])[..., 0].reshape(-1).cpu().numpy())
                SY.append(bv["actions"][:, a.burn_in:, 0].reshape(-1).cpu().numpy())
            H, Y, P, G = (np.concatenate(v) for v in (H, Y, P, G))
            r2 = ridge_r2(H, Y)                           # in-sample: NOT the gate
            head = [r2_score(Y[:, k], P[:, k]) for k in range(2)]
            steer = r2_score(np.concatenate(SY), np.concatenate(SP))
            cv = (ridge_r2_episode_cv(H, Y, G) if len(np.unique(G)) >= 5
                  else [float("nan"), float("nan")])
        net.train()
        row = {"update": u, "train_bc": float(bc), "train_lane": float(lane),
               "val_bc": float(np.mean(vb)), "val_lane": float(np.mean(vl)),
               "head_r2_offset": head[0], "head_r2_heading": head[1], "head_r2_steer": steer,
               "cv_r2_offset": cv[0], "cv_r2_heading": cv[1], "val_episodes": int(len(np.unique(G))),
               "r2_offset": r2[0], "r2_heading": r2[1],     # IN-SAMPLE ridge (the old, wrong gate)
               "secs": round(time.time() - t0)}
        log.append(row)
        print(f"[pre] u={u} bc={row['train_bc']:.4f} lane={row['train_lane']:.4f} | "
              f"val bc={row['val_bc']:.4f} lane={row['val_lane']:.4f} | HELD-OUT own heads: offset="
              f"{head[0]:+.3f} HEADING={head[1]:+.3f} steer={steer:+.3f} | probe, episodes held out "
              f"({row['val_episodes']}): offset={cv[0]:+.3f} heading={cv[1]:+.3f} | (in-sample "
              f"probe {r2[0]:.3f}/{r2[1]:.3f}) | {row['secs']}s", flush=True)
        with open(os.path.join(a.out, "log.jsonl"), "a") as fh:
            fh.write(json.dumps(row) + "\n")
        d = os.path.join(a.out, f"step_{u}")
        os.makedirs(d, exist_ok=True)
        torch.save({"net": net.state_dict(), "args": vars(a), "row": row},
                   os.path.join(d, "params.pt"))
    print("=== PRETRAIN DONE", flush=True)


if __name__ == "__main__":
    main()
