"""The driving-video renderer draws a frame and writes a playable mp4 without a simulator."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drive_video import PANEL, TopDown, open_writer, render_frame  # noqa: E402


def _td():
    lanes = [np.array([[0.0, 0.0], [100.0, 0.0]]), np.array([[100.0, 0.0], [100.0, 100.0]])]
    nodes = np.array([[0.0, 0.0], [50.0, 0.0], [100.0, 0.0], [100.0, 60.0]])
    return TopDown(lanes, nodes, [(0, 1), (1, 2), (2, 3)])


def test_pixels_stay_inside_the_panel():
    td = _td()
    p = td.px(np.array([[-1000.0, -1000.0], [1000.0, 1000.0], [50.0, 50.0]]))
    assert p.min() >= 0 and p.max() <= PANEL - 1


def test_frame_shape_and_writer(tmp_path):
    td = _td()
    rgb = np.random.default_rng(0).random((84, 84, 3)).astype(np.float32)
    f = render_frame(td, rgb, [[0.0, 0.0], [10.0, 0.0]], [0, 1, 2, 3], 1, td.node_xy[1], np.array([100.0, 60.0]),
                     0, 3, ["Map x", "Step 1"])
    assert f.shape == (PANEL, 2 * PANEL, 3) and f.dtype == np.uint8
    out = str(tmp_path / "t.mp4")
    w = open_writer(out)
    for _ in range(5):
        w.write(f)
    w.release()
    assert os.path.getsize(out) > 1000
