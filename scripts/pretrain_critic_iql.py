#!/usr/bin/env python3
"""Pretrain learner C's critic with IQL on the expert shards, on top of the pretrained control
stream (user 2026-11-14: "also pretrain critic with IQL first"; "ensure you borrow IQL from
fastrlap's and use the new proven learner").

THE QUESTION THIS ANSWERS. With c0's features a trained SAC critic's Q varied by about 1% across
the WHOLE steering grid (offline study 2026-11-09: span 0.5-1.8 on Q of 77-163, best steering at
full lock in 82-96% of states, correlation with the expert -0.09 to -0.26), and ascending that
gradient destroyed c0's own actor in 10k steps. The control stream now predicts heading at R^2 0.81
where c0's camera latent gave 0.046. So: does a critic on THESE features rank steering?

THE GATE, declared before the first run, against the 11-09 numbers on the same 21-point grid:
  * Q span across the steering grid >= 5% of |Q| (c0's critic: ~1%), and
  * correlation between the grid's best steering and the expert's > 0.3 (c0's critic: -0.09..-0.26),
  * full-lock share < 50% (c0's critic: 82-96%).
Fail -> the representation was not the binding constraint either, and the next lever is the action
itself (a much longer action repeat, or RL choosing a sub-goal that c0 executes), not more encoder.

WHAT IS FED TO THE CRITIC: obs = [control GRU state (frozen, pretrained) | DEBiT binocular features
(frozen, from the shard) | speed]. The control stream is FROZEN here -- it was just trained, and
freezing keeps this stage a clean test of the critic. DEBiT stays frozen throughout (it is what
gives the policy its wrong-goal gap). No Multiplexer compression: the raw vector goes to the critic
MLPs, which is MORE capacity than learner C's 50-d bottleneck, so a failure here is not a capacity
failure.

REWARDS are the recorded ones with ONE correction: the shards were recorded under the any-speed
rule, so their +100 arrival bonus is withheld when the final speed exceeds the current rule's
10 km/h (in a 60-episode sample, 54 of 60 arrivals were already slow, median 6.4 km/h). masks are
FastRLAP's 1 - terminal: 0 on reached/crossing/off_road/collision, 1 on a clock truncation.

usage:
  /home/robotlab/carla-sac-local/venv/bin/python -I scripts/pretrain_critic_iql.py \
    --control /home/robotlab/carla-sac-local/checkpoints/control_pre_n2/step_5800/params.pt \
    --data /home/robotlab/carla-sac-local/checkpoints/il_data,/home/robotlab/carla-sac-local/checkpoints/il_data_dagger_chain1 \
    --out /home/robotlab/carla-sac-local/checkpoints/iql_pre_n2 --updates 20000
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
sys.path[:0] = [REPO, os.path.join(REPO, "src"), os.path.join(REPO, "scripts")]

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from pretrain_control import load_shard  # noqa: E402  (same shard reader and lane labels)
from torch_rl.control_cnn import ControlGRU  # noqa: E402
from torch_rl.control_data import (nstep_rows, stack_frames,  # noqa: E402
                                   transition_actions)
from torch_rl.frl.iql import IQLLearner  # noqa: E402

ARRIVAL_BONUS = 100.0
ARRIVAL_MAX_KMH = 10.0          # the current rule (v52_reward.env), live in every eval since 11-13
FAULT_ENDS = ("reached", "crossing", "off_road", "collision")   # terminal; "clock" truncates
SPEED_SCALE = 30.0              # speed enters the critic's obs divided by this


@torch.no_grad()
def embed_episode(net, ep, z, dev, px, n_frames, repeat=1):
    """Run the FROZEN control stream over one episode -> (T, hidden) GRU states, plus the shard's
    frozen DEBiT features and the speed column, as the critic's observation rows."""
    n = int(ep["n_act"])
    # the GRU steps at the stream's own decision rate: one step every `repeat` ticks, with the
    # frame stack `repeat` ticks apart, exactly as it was pretrained and as it will be deployed
    ticks = list(range(0, n + 1, max(1, int(repeat))))
    frames = np.stack([stack_frames(ep["frames"], t, n_frames, repeat) for t in ticks])
    x = torch.from_numpy(frames).to(dev).float()
    if x.shape[-1] != px:
        x = F.interpolate(x, size=(px, px), mode="area")
    sp = torch.from_numpy(np.asarray(ep["speed"], np.float32)).to(dev)
    sp = torch.cat([sp, sp[-1:]])[: n + 1]                     # one more row than the actions
    sp = sp[ticks]
    h = net.reset_hidden(1, dev)
    hs = []
    for t in range(len(ticks)):
        out, h = net(x[t:t + 1], sp[t:t + 1].reshape(1, 1), h)
        hs.append(out[0])
    hs = torch.stack(hs)                                        # (n+1, hidden)
    binoc = torch.from_numpy(np.asarray(z["binoc"], np.float32)).to(dev)[: n + 1][ticks]
    obs = torch.cat([hs, binoc, (sp / SPEED_SCALE).reshape(-1, 1)], dim=-1)
    return obs.cpu()


def transitions(paths, net, dev, px, n_frames, lanes, xodr, threads=8, log=print, nstep=1,
                gamma=0.99, repeat=1, lane_cost=0.0, priv=False):
    """Every shard -> (obs, action, reward, next_obs, mask) rows for IQL.

    lane_cost > 0 (OFFLINE TEST ONLY, 2026-11-14): a per-tick cost lane_cost * (|offset m| +
    |heading error deg| / 10) is subtracted from the recorded reward, from the shard's positions and
    the town's OpenDRIVE map. The recorded reward charges the lane only at the terminal crossing, so
    over one decision a worse steering action costs nothing; this asks whether a critic ranks
    steering once it does. It is NOT potential-based on purpose: a potential leaves the action
    ranking unchanged. No run's reward is changed by this flag.

    nstep > 1 builds the ACTION-REPEAT MDP from the same data (control_data.nstep_rows): one row per
    non-overlapping window of k steps, the action averaged over it, the reward discounted inside it,
    and the next state k steps later. The learner's discount must then be gamma**k.
    """
    repeat = max(1, int(repeat))
    if repeat > 1 and int(nstep) != repeat:
        raise ValueError(f"the control stream runs at one decision per {repeat} ticks, so the "
                         f"critic's transitions must be {repeat}-step ones (got nstep={nstep})")
    O, A, R, O2, M = [], [], [], [], []
    with ThreadPoolExecutor(threads) as ex:
        want = float(lane_cost) > 0 or bool(priv)
        for path, ep in zip(paths, ex.map(lambda p: load_shard(p, lanes, xodr, labels=want), paths)):
            z = np.load(path)
            n = int(ep["n_act"])
            if n < 2 * int(nstep):
                continue
            obs = embed_episode(net, ep, z, dev, px, n_frames, repeat)   # one row per DECISION
            if priv:
                # PRIVILEGED CRITIC INPUT (offline test, user 2026-11-14: "lets try priveledge critic
                # with per steering and lane error"): the TRUE lateral offset and heading error at each
                # decision, standardised as in pretraining, appended to the observation. Training-only
                # information: at run time it would feed the critic alone, never the actor.
                dec = np.arange(obs.shape[0]) * max(1, int(repeat))
                po = np.nan_to_num(np.asarray(ep["offset"], np.float32)[dec], nan=0.0)
                ph = np.nan_to_num(np.asarray(ep["heading_err"], np.float32)[dec], nan=0.0)
                pv = np.stack([np.clip(po, -3, 3) / 0.21, np.clip(ph, -90, 90) / 3.457], -1)
                if int(priv) >= 2:
                    # THE FULL PRIVILEGED STATE (user 2026-11-14: "add goal and current location etc
                    # to critic input so it learns to drive lane following"): lane curvature ahead,
                    # in-junction flag, speed, and the GOAL'S POSITION IN THE CAR'S FRAME. The shards
                    # store no goal coordinates, so the goal is taken as the last position of a
                    # goal-mode episode that ended "reached" (within 1 m of it); other episodes get
                    # zeros and a goal-known flag of 0.
                    from torch_rl import lane_labels as _ll2
                    lab = _ll2.episode_labels(z["pos"], lanes[ep["town"]])
                    pos = np.asarray(z["pos"], np.float32)
                    T = len(pos)
                    di = np.minimum(dec, T - 1)
                    curv = np.clip(np.nan_to_num(np.asarray(lab["curv"], np.float32)[dec]), -90, 90) / 30.0
                    junc = np.asarray(lab["junction"], bool)[dec].astype(np.float32)
                    kmh = np.concatenate([[0.0], np.asarray(z["speed"], np.float32)])[dec] / 20.0
                    nxt = pos[np.minimum(di + 1, T - 1)] - pos[np.maximum(di - 1, 0)]
                    yaw = np.arctan2(nxt[:, 1], nxt[:, 0])
                    known = float(int(z["mode"]) == 1 and str(z["end"]) == "reached")
                    d = (pos[-1][None] - pos[di]) * known
                    gx = (np.cos(yaw) * d[:, 0] + np.sin(yaw) * d[:, 1]) / 10.0
                    gy = (-np.sin(yaw) * d[:, 0] + np.cos(yaw) * d[:, 1]) / 10.0
                    pv = np.concatenate([pv, np.stack([curv, junc, kmh, gx, gy,
                                                       np.full(len(dec), known, np.float32)], -1)], -1)
                if int(priv) >= 3:
                    # CRITIC ON THE PRIVILEGED STATE ALONE (user 2026-11-14: "images go to only
                    # actor"): no image features in the critic's input at all, only the true state
                    # and the mode bit.
                    md = np.full((len(dec), 1), float(int(z["mode"])), np.float32)
                    obs = torch.from_numpy(np.concatenate([pv.astype(np.float32), md], -1))
                else:
                    obs = torch.cat([obs, torch.from_numpy(pv.astype(np.float32))], dim=-1)
            r = np.asarray(z["reward"], np.float32)[:n].copy()
            end = str(z["end"])
            # the one reward correction: the recording paid arrival at ANY speed
            if end == "reached" and float(ep["speed"][n - 1]) > ARRIVAL_MAX_KMH:
                r[n - 1] -= ARRIVAL_BONUS
            if float(lane_cost) > 0:
                off = np.nan_to_num(np.abs(np.asarray(ep["offset"], np.float32)[1:n + 1]), nan=0.0)
                head = np.nan_to_num(np.abs(np.asarray(ep["heading_err"], np.float32)[1:n + 1]), nan=0.0)
                # the state AFTER tick t is what action t produced; capped like the pretraining mask
                r -= float(lane_cost) * (np.minimum(off, 3.0) + np.minimum(head, 90.0) / 10.0)
            acts = transition_actions(z, n)           # EXECUTED action: see control_data
            rows = nstep_rows(acts, r, end in FAULT_ENDS, int(nstep), float(gamma))
            if len(rows["starts"]) == 0:
                continue
            i0, i1 = rows["starts"] // repeat, rows["nexts"] // repeat   # tick -> decision index
            assert int(i1[-1]) < obs.shape[0], (int(i1[-1]), obs.shape[0], n, repeat)
            O.append(obs[i0]); O2.append(obs[i1])
            A.append(rows["actions"]); R.append(rows["rewards"]); M.append(rows["masks"])
    out = {"observations": torch.cat(O), "next_observations": torch.cat(O2),
           "actions": torch.from_numpy(np.concatenate(A)),
           "rewards": torch.from_numpy(np.concatenate(R)),
           "masks": torch.from_numpy(np.concatenate(M))}
    log(f"    {out['observations'].shape[0]} transitions, obs dim {out['observations'].shape[1]}, "
        f"reward mean {float(out['rewards'].mean()):.3f} max {float(out['rewards'].max()):.1f}, "
        f"terminal rows {int((out['masks'] == 0).sum())}")
    return out


def steering_gate(lr, data, dev, n_states=256, grid=21, seed=0):
    """THE GATE. For n_states held-out rows, Q over a 21-point steering grid at the expert's
    throttle: the span as a fraction of |Q|, the correlation of the grid's best steering with the
    expert's, and the full-lock share. Directly comparable to the 2026-11-09 offline critic study."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(data["observations"].shape[0], min(n_states, data["observations"].shape[0]),
                     replace=False)
    steers = torch.linspace(-1, 1, grid, device=dev)
    spans, qmags, best, expert = [], [], [], []
    for i in idx:
        o = data["observations"][i].to(dev)
        a_exp = data["actions"][i].to(dev)
        acts = torch.stack([steers, a_exp[1].expand(grid)], dim=-1)
        q = lr.q_over_actions(o, acts)
        spans.append(float(q.max() - q.min()))
        qmags.append(float(q.abs().mean()))
        best.append(float(steers[int(q.argmax())]))
        expert.append(float(a_exp[0]))
    best, expert = np.array(best), np.array(expert)
    corr = float(np.corrcoef(best, expert)[0, 1]) if best.std() > 0 and expert.std() > 0 else 0.0
    span, qm = float(np.mean(spans)), float(np.mean(qmags))
    return {"q_span": span, "q_mag": qm, "span_frac": span / qm if qm else 0.0,
            "corr_best_vs_expert": corr, "full_lock_share": float(np.mean(np.abs(best) >= 0.999)),
            "mean_abs_best": float(np.abs(best).mean()), "mean_abs_expert": float(np.abs(expert).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--control", required=True, help="pretrained control stream params.pt")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--towns", default="Town01,Town04,Town05,Town10HD")
    ap.add_argument("--train-shards", type=int, default=400)
    ap.add_argument("--val-shards", type=int, default=40)
    ap.add_argument("--updates", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--num-qs", type=int, default=10)
    ap.add_argument("--latent", type=int, default=0,
                    help="compress the observation to N dims first (learner C uses 50); "
                         "0 = raw, which drowns the action at 2 of 3,267 inputs")
    ap.add_argument("--latent-hidden", type=int, default=0,
                    help="hidden width of an MLP compression (obs -> hidden -> latent); "
                         "0 = one layer")
    ap.add_argument("--obs", choices=("all", "nobinoc"), default="all",
                    help="nobinoc drops the 3,136 frozen DEBiT dims from the CRITIC "
                         "(training-only network; the actor keeps them)")
    ap.add_argument("--expectile", type=float, default=0.8)
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--discount", type=float, default=0.99)
    ap.add_argument("--nstep", type=int, default=1,
                    help="k-step (action-repeat) transitions; the learner discount becomes gamma**k")
    ap.add_argument("--priv", type=int, default=0,
                    help="OFFLINE TEST: append the true offset and heading error to the observation")
    ap.add_argument("--lane-cost", type=float, default=0.0,
                    help="OFFLINE TEST: per-tick cost K * (|offset m| + |heading deg| / 10)")
    ap.add_argument("--log-every", type=int, default=1000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device(a.device)
    rng = np.random.default_rng(0)

    ck = torch.load(a.control, map_location="cpu")
    ca = ck["args"]
    net = ControlGRU(latent_dim=ca["latent"], hidden_dim=ca["hidden"], px=ca["px"],
                     n_frames=ca["n_frames"]).to(dev)
    net.load_state_dict(ck["net"])
    net.eval().requires_grad_(False)
    print(f"=== control stream {a.control}: n_frames={ca['n_frames']} px={ca['px']} "
          f"hidden={ca['hidden']} | its probe row {ck['row']}", flush=True)

    shards = []
    for root in a.data.split(","):
        for t in a.towns.split(","):
            shards += sorted(glob.glob(os.path.join(root, t, "*.npz")))
    rng.shuffle(shards)
    val_paths = shards[:a.val_shards]
    train_paths = shards[a.val_shards:a.val_shards + a.train_shards]
    lanes = {}
    from torch_rl import lane_labels as _ll
    ll_dir = _ll.DEFAULT_XODR_DIR
    t0 = time.time()
    print(f"=== embedding {len(train_paths)} train shards | lane_cost {a.lane_cost}", flush=True)
    train = transitions(train_paths, net, dev, ca["px"], ca["n_frames"], lanes, ll_dir,
                        a.threads, nstep=a.nstep, gamma=a.discount,
                        repeat=int(ca.get("repeat", 1)), lane_cost=a.lane_cost, priv=int(a.priv))
    print(f"=== embedding {len(val_paths)} val shards", flush=True)
    val = transitions(val_paths, net, dev, ca["px"], ca["n_frames"], lanes, ll_dir,
                      a.threads, nstep=a.nstep, gamma=a.discount,
                      repeat=int(ca.get("repeat", 1)), lane_cost=a.lane_cost, priv=int(a.priv))
    print(f"=== embedded in {time.time()-t0:.0f}s", flush=True)

    if a.obs == "nobinoc":
        h = int(ca["hidden"])
        for dd in (train, val):
            for k in ("observations", "next_observations"):
                dd[k] = torch.cat([dd[k][:, :h], dd[k][:, -1:]], dim=1)
        print(f"=== dropped the DEBiT block from the critic: obs dim now "
              f"{train['observations'].shape[1]} ({h} control + 1 speed)", flush=True)
    obs_dim = int(train["observations"].shape[1])
    lr = IQLLearner(obs_dim=obs_dim, action_dim=2, num_qs=a.num_qs, expectile=a.expectile,
                    temperature=a.temperature, discount=a.discount ** a.nstep,
                    latent=a.latent, latent_hidden=a.latent_hidden, device=str(dev),
                    seed=0)
    print(f"=== IQL obs_dim={obs_dim} num_qs={a.num_qs} expectile={a.expectile} "
          f"temperature={a.temperature} nstep={a.nstep} latent={a.latent} mlp_hidden={a.latent_hidden} obs={a.obs}"
          f" discount={a.discount ** a.nstep:.4f}"
          f" | gate: span>=5% of |Q|, corr>0.3, full-lock<50%", flush=True)

    n = train["observations"].shape[0]
    t0 = time.time()
    for u in range(1, a.updates + 1):
        i = torch.from_numpy(rng.choice(n, a.batch, replace=False))
        info = lr.update({k: v[i] for k, v in train.items()})
        if u % a.log_every and u != a.updates:
            continue
        g = steering_gate(lr, val, dev)
        row = {"update": u, **{k: round(float(v), 4) for k, v in info.items()},
               **{k: round(float(v), 4) for k, v in g.items()}, "secs": round(time.time() - t0)}
        print(f"[iql] u={u} v={row['value_loss']:.2f} c={row['critic_loss']:.1f} "
              f"a={row['actor_loss']:.2f} q={row['q_mean']:.1f} adv={row['adv_mean']:.2f} | "
              f"GATE span={row['span_frac']*100:.1f}% of |Q| corr={row['corr_best_vs_expert']:+.3f} "
              f"lock={row['full_lock_share']*100:.0f}% | {row['secs']}s", flush=True)
        with open(os.path.join(a.out, "log.jsonl"), "a") as fh:
            fh.write(json.dumps(row) + "\n")
        d = os.path.join(a.out, f"step_{u}")
        os.makedirs(d, exist_ok=True)
        torch.save({"iql": lr.state_dict(), "args": vars(a), "row": row, "obs_dim": obs_dim},
                   os.path.join(d, "params.pt"))
    print("=== IQL PRETRAIN DONE", flush=True)


if __name__ == "__main__":
    main()
