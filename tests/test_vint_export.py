"""Packaging a ViNT training checkpoint for EVAL_POLICY=vint: model.pth plus a
vint.json that records how the model was trained -- its architecture, image
size, waypoint stride and the measured metres per frame -- so the eval reads
the waypoints at the scale they were learned at."""
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts", "vint"))


def test_vint_meta_comes_from_the_training_config_and_the_data_summary():
    import yaml
    from export_vint_ckpt import vint_meta
    cfg = yaml.safe_load(open(os.path.join(ROOT, "scripts", "vint", "vint_carla_expert.yaml")))
    meta = vint_meta(cfg, {"metric_waypoint_spacing_m": 0.3712})
    assert meta["stride"] == 3 and meta["metric_step_m"] == 0.3712
    assert meta["frame_dt_s"] == 0.1          # the shards are 10 Hz env steps
    assert meta["context_size"] == 5 and meta["len_traj_pred"] == 5
    assert meta["image_size"] == [85, 64] and meta["dataset"] == "carla_expert"
    assert meta["mha_num_attention_layers"] == 4 and meta["obs_encoder"] == "efficientnet-b0"


def test_the_export_writes_model_and_meta(tmp_path):
    ck = tmp_path / "7.pth"
    ck.write_bytes(b"weights")
    summ = tmp_path / "summary.json"
    summ.write_text(json.dumps({"metric_waypoint_spacing_m": 0.37}))
    out = tmp_path / "epoch_7"
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "vint", "export_vint_ckpt.py"),
                        "--ckpt", str(ck), "--summary", str(summ), "--out", str(out),
                        "--config", os.path.join(ROOT, "scripts", "vint", "vint_carla_expert.yaml")],
                       capture_output=True, text=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
    assert r.returncode == 0, r.stderr
    assert (out / "model.pth").read_bytes() == b"weights"
    meta = json.loads((out / "vint.json").read_text())
    assert meta["metric_step_m"] == 0.37 and meta["source_checkpoint"] == str(ck)
