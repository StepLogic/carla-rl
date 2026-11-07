"""The lane-centre ViNT dataset differs from the first one ONLY in its labels:
scripts/vint/add_lane_labels_to_tar.py copies every image and split file and adds
cf_position / cf_yaw to each traj_data.pkl; scripts/vint/patch_vint_cf_labels.py
makes ViNT's loader read them when present (and nothing else changes)."""
import io
import os
import pickle
import sys
import tarfile

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts", "vint"))

import add_lane_labels_to_tar as addl  # noqa: E402
import patch_vint_cf_labels as patch  # noqa: E402


def _add(tf, name, data):
    info = tarfile.TarInfo(name)
    info.size = len(data)
    tf.addfile(info, io.BytesIO(data))


def _tar(path, n=4):
    pos = np.stack([np.arange(n, dtype=np.float32), np.zeros(n, np.float32)], 1)
    with tarfile.open(path, "w") as tf:
        _add(tf, "carla_expert/Town01_7/0.jpg", b"jpeg0")
        _add(tf, "carla_expert/Town01_7/traj_data.pkl",
             pickle.dumps({"position": pos, "yaw": np.zeros(n, np.float32)}))
        _add(tf, "splits/carla_expert/train/traj_names.txt", b"Town01_7\n")
        _add(tf, "carla_expert/summary.json", b"{}")
    return pos


def _labels(path, pos, extra=2, shift=0.0):
    n = len(pos) + extra                       # labels cover the untrimmed episode
    P = np.zeros((n, 6, 2), np.float32)
    P[:len(pos), 0] = pos + shift
    np.savez(path, names=np.array(["Town01_7"]), lens=np.array([n]), position=P,
             yaw=np.zeros((n, 6), np.float32), ok=np.ones(n, bool))


def test_only_the_labels_change(tmp_path):
    src, lab, out = tmp_path / "a.tar", tmp_path / "l.npz", tmp_path / "b.tar"
    pos = _tar(src)
    _labels(lab, pos)
    addl.main(["--tar-in", str(src), "--labels", str(lab), "--tar-out", str(out)])
    with tarfile.open(src) as a, tarfile.open(out) as b:
        assert a.getnames() == b.getnames()
        for m in a.getmembers():
            if m.name.endswith("traj_data.pkl"):
                old = pickle.loads(a.extractfile(m).read())
                new = pickle.loads(b.extractfile(m.name).read())
                assert np.array_equal(new["position"], old["position"])
                assert new["cf_position"].shape == (len(pos), 6, 2)   # trimmed to the kept frames
                assert new["cf_yaw"].shape == (len(pos), 6)
            elif not m.name.endswith("summary.json"):
                assert a.extractfile(m).read() == b.extractfile(m.name).read()


def test_misaligned_labels_are_refused(tmp_path):
    src, lab, out = tmp_path / "a.tar", tmp_path / "l.npz", tmp_path / "b.tar"
    pos = _tar(src)
    _labels(lab, pos, shift=0.5)
    with pytest.raises(SystemExit):
        addl.main(["--tar-in", str(src), "--labels", str(lab), "--tar-out", str(out)])


SRC = '''
def _compute_actions(self, traj_data, curr_time, goal_time):
        start_index = curr_time
        end_index = curr_time + self.len_traj_pred * self.waypoint_spacing + 1
        yaw = traj_data["yaw"][start_index:end_index:self.waypoint_spacing]
        positions = traj_data["position"][start_index:end_index:self.waypoint_spacing]
        return positions, yaw
'''


def test_the_loader_reads_lane_labels_when_present_and_is_unchanged_otherwise():
    new = patch.apply(SRC)
    assert patch.apply(new) == new                               # idempotent
    ns = {}
    exec(new, ns)

    class S:
        len_traj_pred, waypoint_spacing = 5, 3
    plain = {"yaw": np.arange(20.0), "position": np.arange(40.0).reshape(20, 2)}
    p, y = ns["_compute_actions"](S(), plain, 2, 10)
    assert np.array_equal(y, np.arange(20.0)[2:18:3])            # the original path
    cf = dict(plain, cf_position=np.ones((20, 6, 2)) * 7, cf_yaw=np.ones((20, 6)) * 3)
    p, y = ns["_compute_actions"](S(), cf, 2, 10)
    assert np.array_equal(p, np.ones((6, 2)) * 7) and np.array_equal(y, np.ones(6) * 3)


def test_the_patch_refuses_unknown_code():
    with pytest.raises(SystemExit):
        patch.apply("def f():\n    pass\n")
