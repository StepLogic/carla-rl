#!/usr/bin/env python3
"""c0's trunk against a control stream ON THE SAME ROWS, with the same episode-held-out probe.

The stream's held-out figures (scripts/control_stream_heldout.py) must not be set beside c0's
2026-11-09 figures: those come from other episodes. Here every feature set is computed for the same
episodes the stream never loaded, at the same decision ticks:
  * the control stream's GRU state (one step per decision from zero, as deployed);
  * c0's GRU state (one step per TICK from zero, as c0 runs), read at those ticks;
  * DEBiT's binocular vector at those ticks (what both arms pass through to the learner);
and each is probed, with whole episodes held out (torch_rl.control_data.ridge_r2_episode_cv), for
lateral offset, heading error and the expert's steering label.

RESULT 2026-11-14, control_pre_n2_k8/step_6000 against il_bc_c0/step_15000, 123 episodes, 2,077
labelled decisions (heading sd 6.8 deg, offset sd 0.31 m):
                                   offset   heading   steering label   (in-sample offset / heading)
  control stream GRU state (128)   +0.40    +0.08     -0.03            (0.57 / 0.27)
  c0 GRU state (512)               +0.04    +0.03     +0.08            (0.76 / 0.53)
  DEBiT binocular (3136)           +0.19    +0.01     +0.07            (0.97 / 0.92)
  c0 GRU + binocular               +0.23    +0.02     +0.12
  stream GRU + binocular           +0.33    +0.03     +0.04
HEADING IS IN NONE OF THEM on these episodes, and no feature set predicts the expert's steering.

CAVEAT that bounds the reading: 123 episodes starve a linear probe on 512 or 3,136 dimensions (it
fits them in-sample to 0.5-0.97 and transfers nothing), so this table cannot rank c0 against the
stream; a held-out probe is a LOWER bound that depends on how many episodes it is given. What it
does settle: the stream was built to add heading to c0's features, and it does not.

usage: /home/robotlab/carla-sac-local/venv/bin/python -I scripts/control_stream_vs_c0.py \
           STREAM_CKPT_DIR C0_DIR [n_episodes=150]
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
from torch_rl import pixel_rows as pxr  # noqa: E402
from torch_rl.control_cnn import ControlGRU  # noqa: E402
from torch_rl.control_data import ridge_r2_episode_cv, ridge_r2_in_sample, stack_frames  # noqa: E402
from torch_rl.frl.carla import C0Trunk  # noqa: E402


def main():
    ckpt, c0_dir = sys.argv[1], sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 150
    ck = torch.load(os.path.join(ckpt, "params.pt"), map_location="cpu")
    a = ck["args"]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = ControlGRU(latent_dim=a["latent"], hidden_dim=a["hidden"], px=a["px"],
                     n_frames=a["n_frames"]).to(dev)
    net.load_state_dict(ck["net"])
    net.eval().requires_grad_(False)
    k = int(a.get("repeat", 1))
    c0 = C0Trunk(c0_dir, str(dev))

    # the stream's own split, reproduced, to find shards it never loaded
    rng = np.random.default_rng(0)
    shards = []
    for root in a["data"].split(","):
        for t in a["towns"].split(","):
            shards += sorted(glob.glob(os.path.join(root, t, "*.npz")))
    rng.shuffle(shards)
    train_paths = shards[a["val_shards"]:]
    pool = set(rng.choice(train_paths, min(a["pool"], len(train_paths)), replace=False))
    never = [p for p in train_paths if p not in pool]
    never = list(np.random.default_rng(1).choice(never, min(n, len(never)), replace=False))
    lanes = {}
    with ThreadPoolExecutor(8) as ex:
        eps = list(ex.map(lambda p: (p, pc.load_shard(p, lanes, a["xodr"])), never))

    Hs, Hc, Hb, Y, S, G = [], [], [], [], [], []
    with torch.no_grad():
        for gi, (p, e) in enumerate(eps):
            if e["n_act"] <= k:
                continue
            z = np.load(p, allow_pickle=False)
            mode, binoc = float(int(z["mode"])), z["binoc"]
            kmh = np.concatenate([[0.0], np.asarray(z["speed"], np.float32)])
            ticks = list(range(0, int(e["n_act"]), k))
            h, hs = net.reset_hidden(1, dev), []
            for t in ticks:                                   # the stream: one step per decision
                x = torch.from_numpy(stack_frames(e["frames"], t, a["n_frames"], k)[None])
                x = F.interpolate(x.to(dev).float(), size=(a["px"], a["px"]), mode="area")
                out, h = net(x, torch.tensor([[float(e["speed"][t])]], device=dev), h)
                hs.append(out[0].cpu().numpy())
            hid, hc, want = c0.zero_hidden(), [], set(ticks)
            for t in range(ticks[-1] + 1):                    # c0: one step per tick
                row = pxr.mono_row(e["frames"][t], mode, np.asarray(binoc[t], np.float32),
                                   speed_kmh=float(kmh[t]))
                o, hid = c0.step(row, hid)
                if t in want:
                    hc.append(o["embeddings"][0, :c0.hidden_dim].float().cpu().numpy())
            idx = np.array(ticks)
            y = np.stack([np.asarray(e["offset"], np.float32)[idx],
                          np.asarray(e["heading_err"], np.float32)[idx]], -1)
            v = np.asarray(e["valid"], bool)[idx] & np.isfinite(y).all(-1)
            Hs.append(np.stack(hs)[v])
            Hc.append(np.stack(hc)[v])
            Hb.append(np.asarray(binoc, np.float32)[idx][v])
            Y.append(np.nan_to_num(y)[v])
            S.append(np.asarray(e["act_label"], np.float32)[idx, :1][v])
            G.append(np.full(int(v.sum()), gi))
    Hs, Hc, Hb, Y, S, G = (np.concatenate(v) for v in (Hs, Hc, Hb, Y, S, G))
    print(f"=== {len(np.unique(G))} never-loaded episodes, {len(Y)} labelled decisions | heading sd "
          f"{Y[:, 1].std():.2f} deg, offset sd {Y[:, 0].std():.2f} m, steering label sd "
          f"{S.std():.3f}", flush=True)
    for name, H in ((f"control stream GRU state ({Hs.shape[1]})", Hs),
                    (f"c0 GRU state ({Hc.shape[1]})", Hc),
                    (f"DEBiT binocular ({Hb.shape[1]})", Hb),
                    ("c0 GRU + binocular", np.concatenate([Hc, Hb], 1)),
                    ("stream GRU + binocular", np.concatenate([Hs, Hb], 1))):
        cv, st = ridge_r2_episode_cv(H, Y, G), ridge_r2_episode_cv(H, S, G)
        ins = ridge_r2_in_sample(H, Y)
        print(f"--- {name}: episodes held out: offset {cv[0]:+.3f} heading {cv[1]:+.3f} steering "
              f"{st[0]:+.3f} | (in-sample offset {ins[0]:+.3f} heading {ins[1]:+.3f})", flush=True)


if __name__ == "__main__":
    main()
