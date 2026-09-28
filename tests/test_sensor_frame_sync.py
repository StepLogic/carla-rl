"""SENSOR_FRAME_SYNC: the camera must be the tick's own frame (2026-09-28).

Measured before the fix: every reset's bare world.tick() left one unread frame
per sensor, so the image the policy saw fell one frame further behind the world
per episode (0 -> -25 frames over 20 eval episodes)."""
import sys

import pytest

sys.path.insert(0, "/home/robotlab/scratch/carla-rl/dependencies/rlib_integration")
si = pytest.importorskip("rlib_integration.sensors.sensor_interface")


class _S:
    def __init__(self, event=False):
        self._e = event

    def is_event_sensor(self):
        return self._e


def _iface():
    it = si.SensorInterface()
    it.register("rgb", _S())
    it.register("goal", _S())
    it.register("lane_invasion", _S(event=True))
    it.register("collision", _S(event=True))
    return it


def test_stale_frames_and_events_are_dropped(monkeypatch):
    monkeypatch.setattr(si, "FRAME_SYNC", True)
    it = _iface()
    for f in (10, 11, 12):                       # 10 and 11 were never read
        it._data_buffers.put(("rgb", f, f"img{f}"))
        it._data_buffers.put(("goal", f, f"g{f}"))
    it._event_data_buffers.put(("lane_invasion", 11, "old"))
    it._event_data_buffers.put(("collision", 12, "hit"))
    d = it.get_data(12)
    assert d["rgb"] == (12, "img12") and d["goal"] == (12, "g12")
    assert d["collision"] == (12, "hit") and "lane_invasion" not in d
    assert it._data_buffers.empty() and it._event_data_buffers.empty()


def test_default_path_unchanged(monkeypatch):
    monkeypatch.setattr(si, "FRAME_SYNC", False)
    it = _iface()
    for f in (10, 11):
        it._data_buffers.put(("rgb", f, f"img{f}"))
        it._data_buffers.put(("goal", f, f"g{f}"))
    d = it.get_data(11)
    assert d["rgb"] == (10, "img10")             # the old, oldest-first behaviour
