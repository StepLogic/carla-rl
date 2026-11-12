#!/usr/bin/env python3
"""Package one ViNT training checkpoint for EVAL_POLICY=vint (torch_rl.vint_driver).

    export_vint_ckpt.py --ckpt <run>/<epoch>.pth --config scripts/vint/vint_carla_expert.yaml \
        --summary checkpoints/vint_data/carla_expert/summary.json \
        --out checkpoints/vint_carla_expert/epoch_<N>

Writes <out>/model.pth (a copy: the eval dir must not depend on the training
logs) and <out>/vint.json: the architecture, image size, the dataset's waypoint
stride and the measured metres per frame, exactly as trained, so the eval reads
ViNT's normalised waypoints and distances at the scale they were learned at.
"""
import argparse
import json
import os
import shutil

import yaml

ARCH_KEYS = ("context_size", "len_traj_pred", "learn_angle", "obs_encoder",
             "obs_encoding_size", "late_fusion", "mha_num_attention_heads",
             "mha_num_attention_layers", "mha_ff_dim_factor")


def vint_meta(cfg, summary):
    """The eval's view of the training run. Several datasets (a DAgger round:
    carla_expert + carla_dagger1) must share one waypoint stride -- the label
    scale the driver un-normalises with; their names are joined with '+'."""
    strides = {int(dc["waypoint_spacing"]) for dc in cfg["datasets"].values()}
    if len(strides) != 1:
        raise SystemExit(f"datasets with different waypoint strides {sorted(strides)}: the "
                         "driver un-normalises with one")
    name = "+".join(cfg["datasets"])
    meta = {k: cfg[k] for k in ARCH_KEYS}
    meta.update({"image_size": list(cfg["image_size"]), "stride": strides.pop(),
                 "metric_step_m": float(summary["metric_waypoint_spacing_m"]),
                 # the shards hold one frame per env step, 0.1 s (CARLA's fixed delta)
                 "frame_dt_s": 0.1,
                 "dataset": name, "normalize": bool(cfg.get("normalize", True))})
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--link", action="store_true",
                    help="symlink model.pth to the training checkpoint instead of copying "
                         "(a 430 MB copy takes minutes on CephFS)")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    meta = vint_meta(cfg, json.load(open(a.summary)))
    if not meta["normalize"]:
        raise SystemExit("the driver un-normalises waypoints; a normalize=False model needs another path")
    meta["source_checkpoint"] = a.ckpt
    os.makedirs(a.out, exist_ok=True)
    dst = os.path.join(a.out, "model.pth")
    if a.link:
        if os.path.lexists(dst):
            os.remove(dst)
        os.symlink(os.path.abspath(a.ckpt), dst)
    else:
        shutil.copyfile(a.ckpt, dst)
    with open(os.path.join(a.out, "vint.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    print(f"exported {a.ckpt} -> {a.out}: stride {meta['stride']}, "
          f"{meta['metric_step_m']} m/frame", flush=True)


if __name__ == "__main__":
    main()
