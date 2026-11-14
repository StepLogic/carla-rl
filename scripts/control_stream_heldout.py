#!/usr/bin/env python3
"""Does a pretrained control stream carry lane offset and heading error OUT OF SAMPLE?

Written 2026-11-14 because the answer first given was wrong. scripts/pretrain_control.py judged its
streams with a ridge probe fitted and scored on the SAME validation rows ("heading R^2 0.81"), and
that was compared with c0's 0.046, which scripts/probe_c0_features.py measures with whole episodes
held out. An independent review of the commit that put the stream into learner C caught it.

This scores a stream on whole episodes run the way deployment runs them (GRU from a zero state at
the episode start, one step per decision, frame pair `repeat` ticks apart), three ways:
  1. its OWN lane head and action head against the labels -- nothing is fitted here at all;
  2. a probe on the GRU state with whole EPISODES held out (torch_rl.control_data
     .ridge_r2_episode_cv: the procedure of c0's probe, so the two are comparable);
  3. the old in-sample probe, to show the gap.
on three sets: the pretraining run's validation shards, shards the run never loaded, and a sample of
the shards it trained on. Each is repeated with the GRU reset every `chunk` decisions and the
burn-in dropped -- the only regime pretraining supervised.

RESULT for control_pre_n2_k8/step_6000 (repeat 8, chunk 12, burn-in 4), whole episodes:
                              own lane head         probe, episodes held out   own action head
                              offset   heading      offset   heading           steering
  123 never-loaded episodes   +0.38    +0.10        +0.40    +0.08             -0.60
   56 validation episodes     +0.27    +0.09        -0.11    -0.02             -0.18
   90 training-pool episodes  +0.56    +0.14        +0.51    +0.11             +0.38
c0 by the same kind of probe (2026-11-09): camera latent offset 0.70-0.81, heading 0.046; GRU state
heading 0.31-0.36. So this stream is below c0 on both lane variables, does not fit heading even on
its own training pool, and its cloning head does not transfer to unseen episodes. It is overfit
(600-shard pool of 5,400, 6,000 updates; val lane loss 0.44 against train 0.12 in its own log).

usage: /home/robotlab/carla-sac-local/venv/bin/python -I scripts/control_stream_heldout.py CKPT_DIR \
           [n_never_loaded=150] [n_pool=100]
"""
import glob
import os
import sys
from concurrent.futures import ThreadPoolExecutor

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [REPO, os.path.join(REPO, "src"), os.path.join(REPO, "scripts")]
os.environ.setdefault("MONO_CHW", "1")
os.environ.setdefault("SPEED_OBS", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import pretrain_control as pc  # noqa: E402
from torch_rl.control_cnn import ControlGRU  # noqa: E402
from torch_rl.control_data import (r2_score, ridge_r2_episode_cv, ridge_r2_in_sample,  # noqa: E402
                                   stack_frames)


def main():
    ckpt = sys.argv[1]
    n_unseen = int(sys.argv[2]) if len(sys.argv) > 2 else 150
    n_pool = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    ck = torch.load(os.path.join(ckpt, "params.pt"), map_location="cpu")
    a = ck["args"]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = ControlGRU(latent_dim=a["latent"], hidden_dim=a["hidden"], px=a["px"],
                     n_frames=a["n_frames"]).to(dev)
    net.load_state_dict(ck["net"])
    net.eval().requires_grad_(False)
    k, chunk, burn = int(a.get("repeat", 1)), int(a["chunk"]), int(a["burn_in"])
    sd = np.asarray(pc.SD, np.float32)

    # the run's own split, reproduced: shuffle, validation shards, then the pool it held in RAM
    rng = np.random.default_rng(0)
    shards = []
    for root in a["data"].split(","):
        for t in a["towns"].split(","):
            shards += sorted(glob.glob(os.path.join(root, t, "*.npz")))
    rng.shuffle(shards)
    val_paths, train_paths = shards[:a["val_shards"]], shards[a["val_shards"]:]
    pool_paths = list(rng.choice(train_paths, min(a["pool"], len(train_paths)), replace=False))
    pool_set = set(pool_paths)
    pick = np.random.default_rng(1)
    never = [p for p in train_paths if p not in pool_set]
    never = list(pick.choice(never, min(n_unseen, len(never)), replace=False))
    pool_s = list(pick.choice(pool_paths, min(n_pool, len(pool_paths)), replace=False))
    print(f"=== {ckpt} | repeat {k} chunk {chunk} burn-in {burn} | shards: {len(val_paths)} "
          f"validation, {len(pool_paths)} in the training pool, "
          f"{len(train_paths) - len(pool_paths)} never loaded", flush=True)

    @torch.no_grad()
    def run_episode(e, reset_every):
        ticks = list(range(0, int(e["n_act"]), k))
        h = net.reset_hidden(1, dev)
        H, lane, act, keep = [], [], [], []
        for j, t in enumerate(ticks):
            if reset_every and j % reset_every == 0:
                h = net.reset_hidden(1, dev)
            x = torch.from_numpy(stack_frames(e["frames"], t, a["n_frames"], k)[None]).to(dev).float()
            if x.shape[-1] != a["px"]:
                x = F.interpolate(x, size=(a["px"], a["px"]), mode="area")
            out, h = net(x, torch.tensor([[float(e["speed"][t])]], device=dev), h)
            H.append(out[0].cpu().numpy())
            lane.append(net.lane(out)[0].cpu().numpy() * sd)
            act.append(net.action(out)[0].cpu().numpy())
            keep.append(not reset_every or (j % reset_every) >= burn)
        idx = np.array(ticks)
        y = np.stack([np.asarray(e["offset"], np.float32)[idx],
                      np.asarray(e["heading_err"], np.float32)[idx]], -1)
        v = np.asarray(e["valid"], bool)[idx] & np.isfinite(y).all(-1) & np.array(keep)
        return (np.stack(H)[v], np.stack(lane)[v], np.nan_to_num(y)[v], np.stack(act)[v],
                np.asarray(e["act_label"], np.float32)[idx][v])

    def report(name, eps, reset_every=0):
        parts = [p for p in (run_episode(e, reset_every) for e in eps) if len(p[0])]
        H, L, Y, A, AL = (np.concatenate([p[i] for p in parts]) for i in range(5))
        ep = np.concatenate([np.full(len(p[0]), i) for i, p in enumerate(parts)])
        cv, ins = ridge_r2_episode_cv(H, Y, ep), ridge_r2_in_sample(H, Y)
        var_ep = np.array([((Y[ep == i, 1] - Y[:, 1].mean()) ** 2).sum() for i in range(len(parts))])
        print(f"--- {name}: {len(parts)} episodes, {len(Y)} labelled decisions | heading label sd "
              f"{Y[:, 1].std():.2f} deg, offset sd {Y[:, 0].std():.2f} m | 3 episodes hold "
              f"{100 * float(np.sort(var_ep)[-3:].sum() / var_ep.sum()):.0f}% of the heading variance\n"
              f"    OWN heads (nothing fitted): offset R2 {r2_score(Y[:, 0], L[:, 0]):+.3f} | heading "
              f"{r2_score(Y[:, 1], L[:, 1]):+.3f} | steering {r2_score(AL[:, 0], A[:, 0]):+.3f}\n"
              f"    probe, EPISODES HELD OUT: offset {cv[0]:+.3f} | heading {cv[1]:+.3f}\n"
              f"    probe fitted and scored on the same rows (not evidence): offset {ins[0]:+.3f} | "
              f"heading {ins[1]:+.3f}", flush=True)

    def load(paths):
        lanes = {}
        with ThreadPoolExecutor(8) as ex:
            return [e for e in ex.map(lambda p: pc.load_shard(p, lanes, a["xodr"]), paths)
                    if e["n_act"] > k]

    for name, paths in (("VALIDATION shards", val_paths), ("NEVER-LOADED shards", never),
                        ("TRAINING-POOL shards", pool_s)):
        eps = load(paths)
        report(f"{name}, whole episodes from a zero state (deployment)", eps)
        report(f"{name}, GRU reset every {chunk} decisions, burn-in {burn} dropped (pretraining)",
               eps, reset_every=chunk)


if __name__ == "__main__":
    main()
