"""Expert episodes -> ViNT trajectories (user 2026-11-07: train ViNT on c0's expert
data and evaluate it like c0).

A shard holds T+1 frames and T positions: frame 0 is the reset view, frame k+1
the view after step k, taken at pos[k]. ViNT wants one (x, y, yaw) per frame in a
RIGHT-HANDED frame (its local frames are x forward, y LEFT); CARLA's y points
south, so y is flipped. The shards carry no yaw: heading comes from the motion.
"""
import os
import pickle
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from vae_nav.vint_data import (episode_to_traj, frame_positions,  # noqa: E402
                               headings_from_motion, local_coords, to_right_handed,
                               write_traj)


def test_frame_k_plus_1_sits_at_pos_k_and_frame_0_at_the_start():
    pos = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    xy = frame_positions(pos)
    assert xy.shape == (4, 2)
    np.testing.assert_allclose(xy, [[1, 0], [1, 0], [2, 0], [3, 0]])


def test_carla_y_south_becomes_y_left():
    np.testing.assert_allclose(to_right_handed([[3.0, 2.0]]), [[3.0, -2.0]])


def test_heading_follows_the_motion_in_the_right_handed_frame():
    east = to_right_handed(np.stack([np.arange(6.0), np.zeros(6)], 1))
    np.testing.assert_allclose(headings_from_motion(east), 0.0, atol=1e-9)
    south = to_right_handed(np.stack([np.zeros(6), np.arange(6.0)], 1))   # CARLA +y
    np.testing.assert_allclose(headings_from_motion(south), -np.pi / 2, atol=1e-9)


def test_a_stationary_start_takes_the_heading_of_the_first_real_move():
    xy = np.array([[0, 0], [0, 0], [0.01, 0], [0.5, 0.5], [1.0, 1.0]], float)
    yaw = headings_from_motion(xy, min_step_m=0.3)
    np.testing.assert_allclose(yaw[:3], np.pi / 4, atol=0.05)


def test_the_stopped_tail_keeps_the_last_heading_and_no_motion_gives_zero():
    xy = np.array([[0, 0], [1, 0], [2, 0], [2, 0], [2, 0]], float)
    np.testing.assert_allclose(headings_from_motion(xy), 0.0, atol=1e-9)
    xy[:, :] = [[0, 0], [0, 1], [0, 2], [0, 2], [0, 2]]
    np.testing.assert_allclose(headings_from_motion(xy), np.pi / 2, atol=1e-9)
    np.testing.assert_allclose(headings_from_motion(np.zeros((4, 2))), 0.0)


def test_a_carla_left_turn_is_positive_y_in_vint_local_coords():
    # Driving EAST (+x) in CARLA, a left turn heads NORTH, i.e. CARLA -y.
    carla = np.array([[0, 0], [1, 0], [2, 0], [3, -0.5], [3.5, -1.5]], float)
    xy = to_right_handed(carla)
    yaw = headings_from_motion(xy)
    loc = local_coords(xy[3:], xy[0], yaw[0])
    assert np.all(loc[:, 0] > 0) and np.all(loc[:, 1] > 0)


def test_local_coords_matches_vints_own_transform():
    # vint_train.data.data_utils.to_local_coords: (p - c) . R(yaw)[:2, :2]
    c, yaw = np.array([1.0, 2.0]), 0.7
    rot = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    p = np.array([[3.0, -1.0], [0.5, 4.0]])
    np.testing.assert_allclose(local_coords(p, c, yaw), (p - c).dot(rot))


def _shard(T, end):
    frames = np.random.default_rng(0).integers(0, 255, (T + 1, 8, 8, 3), dtype=np.uint8)
    pos = np.stack([np.arange(1, T + 1, dtype=float) * 0.4, np.zeros(T)], 1)
    return frames, pos, end


def test_a_fault_episode_loses_its_last_frames_and_a_clean_one_keeps_all():
    f, xy, yaw = episode_to_traj(*_shard(50, "off_road"), trim_frames=20)
    assert len(f) == len(xy) == len(yaw) == 31
    f, xy, yaw = episode_to_traj(*_shard(50, "reached"), trim_frames=20)
    assert len(f) == 51


def test_write_traj_writes_numbered_jpegs_and_the_pickle(tmp_path):
    from PIL import Image
    frames, xy, yaw = episode_to_traj(*_shard(5, "reached"))
    d = tmp_path / "Town01_100000"
    write_traj(str(d), frames, xy, yaw)
    assert sorted(os.listdir(d)) == sorted([f"{k}.jpg" for k in range(6)] + ["traj_data.pkl"])
    assert Image.open(d / "0.jpg").size == (8, 8)
    with open(d / "traj_data.pkl", "rb") as fh:
        td = pickle.load(fh)
    assert td["position"].shape == (6, 2) and td["yaw"].shape == (6,)


def test_the_converter_writes_c0s_split_and_the_metric_spacing(tmp_path):
    import json
    import subprocess
    paths = []
    for seed, end, T in ((100000, "reached", 40), (100001, "off_road", 60), (100002, "reached", 10)):
        frames, pos, _ = _shard(T, end)
        p = tmp_path / f"ep_{seed}.npz"
        np.savez(p, frames=frames, pos=pos, end=np.array(end), town=np.array("Town01"),
                 seed=np.array(seed), mode=np.array(1, np.int8))
        paths.append(str(p))
    split = tmp_path / "bc_split.json"
    split.write_text(json.dumps({"train": paths[:2] + paths[2:], "val": paths[:1]}))
    out, splits = tmp_path / "data", tmp_path / "splits"
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "convert_il_to_vint.py"),
                        "--split", str(split), "--out", str(out), "--splits", str(splits),
                        "--workers", "2"], capture_output=True, text=True,
                       env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
    assert r.returncode == 0, r.stderr
    train = (splits / "train" / "traj_names.txt").read_text().split()
    assert train == ["Town01_100000", "Town01_100001"]          # the 11-frame one is too short
    assert (splits / "test" / "traj_names.txt").read_text().split() == ["Town01_100000"]
    assert len(list((out / "Town01_100001").glob("*.jpg"))) == 41  # 61 frames - 20 trimmed
    s = json.loads((out / "summary.json").read_text())
    assert s["train"]["skipped_short"] == 1 and s["train"]["fault_trimmed"] == 1
    assert abs(s["metric_waypoint_spacing_m"] - 0.4) < 1e-6


def test_encode_traj_gives_the_same_files_write_traj_writes(tmp_path):
    from vae_nav.vint_data import encode_traj
    frames, xy, yaw = episode_to_traj(*_shard(5, "reached"))
    files = dict(encode_traj(frames, xy, yaw))
    assert sorted(files) == sorted([f"{k}.jpg" for k in range(6)] + ["traj_data.pkl"])
    write_traj(str(tmp_path / "t"), frames, xy, yaw)
    for name, data in files.items():
        assert (tmp_path / "t" / name).read_bytes() == data


def test_the_converter_can_write_one_tar_instead_of_small_files(tmp_path):
    """CephFS reads ~85 ms per small file, so 290k JPEGs could not be tarred
    afterwards in useful time: --tar streams the same layout into one file."""
    import json
    import subprocess
    import tarfile
    paths = []
    for seed, end, T in ((100000, "reached", 40), (100001, "off_road", 60)):
        frames, pos, _ = _shard(T, end)
        p = tmp_path / f"ep_{seed}.npz"
        np.savez(p, frames=frames, pos=pos, end=np.array(end), town=np.array("Town01"),
                 seed=np.array(seed), mode=np.array(1, np.int8))
        paths.append(str(p))
    split = tmp_path / "bc_split.json"
    split.write_text(json.dumps({"train": paths, "val": paths[:1]}))
    tar = tmp_path / "carla_expert.tar"
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "convert_il_to_vint.py"),
                        "--split", str(split), "--tar", str(tar), "--workers", "2"],
                       capture_output=True, text=True,
                       env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
    assert r.returncode == 0, r.stderr
    with tarfile.open(tar) as tf:
        names = set(tf.getnames())
        assert "carla_expert/Town01_100001/40.jpg" in names
        assert "carla_expert/Town01_100001/41.jpg" not in names      # 61 frames - 20 trimmed
        assert "carla_expert/Town01_100000/traj_data.pkl" in names
        assert tf.extractfile("splits/carla_expert/train/traj_names.txt").read().split() == [
            b"Town01_100000", b"Town01_100001"]
        assert tf.extractfile("splits/carla_expert/test/traj_names.txt").read().split() == [
            b"Town01_100000"]
        s = json.loads(tf.extractfile("carla_expert/summary.json").read())
    assert abs(s["metric_waypoint_spacing_m"] - 0.4) < 1e-6


def test_frames_and_positions_must_agree():
    frames, pos, end = _shard(5, "reached")
    with pytest.raises(ValueError):
        episode_to_traj(frames[:-2], pos, end)
