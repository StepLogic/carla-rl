#!/usr/bin/env python3
"""Pre-flight for the ViNT fine-tune, on CPU, before a GPU is spent:

  1. every module train.py imports, imports;
  2. the released vint.pth loads into the configured architecture with NO missing
     and NO unexpected keys (train.py's load_model uses strict=False, which would
     otherwise drop a mismatched layer without a word);
  3. a ViNT_Dataset over a few converted trajectories yields a sample whose
     images and labels have the expected shapes, and the model runs on it.

usage: probe_vint_setup.py CONFIG.yaml N_TRAJ   (cwd = visualnav train/)
"""
import os
import sys
import tempfile
import time

t0 = time.time()
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
import diffusers  # noqa: E402,F401
import efficientnet_pytorch  # noqa: E402,F401
import lmdb  # noqa: E402,F401
import prettytable  # noqa: E402,F401
import warmup_scheduler  # noqa: E402,F401
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D  # noqa: E402,F401
from vint_train.data.vint_dataset import ViNT_Dataset  # noqa: E402
from vint_train.models.vint.vint import ViNT  # noqa: E402
print(f"[Probe] imports ok ({time.time() - t0:.0f} s)", flush=True)

cfg = yaml.safe_load(open(sys.argv[1]))
n_traj = int(sys.argv[2]) if len(sys.argv) > 2 else 4
model = ViNT(context_size=cfg["context_size"], len_traj_pred=cfg["len_traj_pred"],
             learn_angle=cfg["learn_angle"], obs_encoder=cfg["obs_encoder"],
             obs_encoding_size=cfg["obs_encoding_size"], late_fusion=cfg["late_fusion"],
             mha_num_attention_heads=cfg["mha_num_attention_heads"],
             mha_num_attention_layers=cfg["mha_num_attention_layers"],
             mha_ff_dim_factor=cfg["mha_ff_dim_factor"])
ck = torch.load(cfg["pretrained_checkpoint"], map_location="cpu", weights_only=False)
src = ck["model"]
src = getattr(src, "module", src)
res = model.load_state_dict(src.state_dict(), strict=False)
n_par = sum(p.numel() for p in model.parameters())
print(f"[Probe] released weights: {n_par / 1e6:.1f} M params, missing={list(res.missing_keys)} "
      f"unexpected={list(res.unexpected_keys)}", flush=True)
if res.missing_keys or res.unexpected_keys:
    sys.exit("[Probe] FAIL: the architecture does not match the released checkpoint")

(name, dc), = cfg["datasets"].items()
names = open(os.path.join(dc["train"], "traj_names.txt")).read().split()[:n_traj]
with tempfile.TemporaryDirectory() as split:
    with open(os.path.join(split, "traj_names.txt"), "w") as fh:
        fh.write("\n".join(names) + "\n")
    ds = ViNT_Dataset(data_folder=dc["data_folder"], data_split_folder=split, dataset_name=name,
                      image_size=cfg["image_size"], waypoint_spacing=dc["waypoint_spacing"],
                      min_dist_cat=cfg["distance"]["min_dist_cat"],
                      max_dist_cat=cfg["distance"]["max_dist_cat"],
                      min_action_distance=cfg["action"]["min_dist_cat"],
                      max_action_distance=cfg["action"]["max_dist_cat"],
                      negative_mining=dc["negative_mining"], len_traj_pred=cfg["len_traj_pred"],
                      learn_angle=cfg["learn_angle"], context_size=cfg["context_size"],
                      context_type=cfg["context_type"], end_slack=dc["end_slack"],
                      goals_per_obs=dc["goals_per_obs"], normalize=cfg["normalize"],
                      goal_type=cfg["goal_type"])
    print(f"[Probe] {len(names)} trajectories -> {len(ds)} samples", flush=True)
    # vint_dataset.__getitem__: (obs, goal, actions, distance, goal_pos, dataset_index, action_mask)
    obs, goal, action, dist, goal_pos, _idx, action_mask = ds[len(ds) // 2]
    print(f"[Probe] sample: obs {tuple(obs.shape)} goal {tuple(goal.shape)} "
          f"dist {int(dist)} action {tuple(action.shape)} action_mask {float(action_mask):.0f}",
          flush=True)
    acts = np.stack([ds[i][2].numpy() for i in range(0, len(ds), max(1, len(ds) // 40))])
    print(f"[Probe] normalised waypoints over {len(acts)} samples: first step x "
          f"{acts[:, 0, 0].mean():.2f}+-{acts[:, 0, 0].std():.2f}, y "
          f"{acts[:, 0, 1].mean():+.2f}+-{acts[:, 0, 1].std():.2f}; last step x "
          f"{acts[:, -1, 0].mean():.2f}", flush=True)
    model.eval()
    with torch.no_grad():
        d_pred, a_pred = model(obs[None], goal[None])
    print(f"[Probe] forward ok: dist_pred {tuple(d_pred.shape)} action_pred "
          f"{tuple(a_pred.shape)}; released model says dist {float(d_pred):.1f} vs label "
          f"{float(dist):.0f} ({time.time() - t0:.0f} s)", flush=True)
print("[Probe] PASS", flush=True)
