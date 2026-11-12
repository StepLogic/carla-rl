"""LOCAL CARLA (user 2026-10-27, "Local v18, Town05 only").

On the cluster CarlaServer runs CARLA inside apptainer, offscreen. Locally there
is a CARLA install (CARLA_LOCAL_ROOT), and the user's rule is that local
training shows the render (windowed) -- Town05 also segfaults offscreen here.
CARLA_LOCAL_ROOT switches CarlaServer to that binary, windowed unless
CARLA_OFFSCREEN=1. Unset, nothing changes.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from torch_rl.carla_server import CarlaServer  # noqa: E402


def test_cluster_command_is_unchanged(monkeypatch, tmp_path):
    monkeypatch.delenv("CARLA_LOCAL_ROOT", raising=False)
    s = CarlaServer(2100, sif="/x/carla.sif", quality="Epic", log_dir=str(tmp_path))
    cmd = s._cmd()
    assert cmd[:5] == ["apptainer", "run", "--nv", "-e", "/x/carla.sif"]
    assert "-RenderOffScreen" in cmd and "-carla-rpc-port=2100" in cmd


def test_local_command_is_windowed_by_default(monkeypatch, tmp_path):
    monkeypatch.setenv("CARLA_LOCAL_ROOT", "/opt/CARLA_0.9.15")
    monkeypatch.delenv("CARLA_OFFSCREEN", raising=False)
    s = CarlaServer(2100, quality="Epic", log_dir=str(tmp_path))
    cmd = s._cmd()
    assert cmd[0] == "/opt/CARLA_0.9.15/CarlaUE4.sh"
    assert "apptainer" not in cmd and "-RenderOffScreen" not in cmd
    assert "-quality-level=Epic" in cmd and "-carla-rpc-port=2100" in cmd
    assert "CARLA_0.9.15" in s.describe()


def test_local_offscreen_on_request(monkeypatch, tmp_path):
    monkeypatch.setenv("CARLA_LOCAL_ROOT", "/opt/CARLA_0.9.15")
    monkeypatch.setenv("CARLA_OFFSCREEN", "1")
    cmd = CarlaServer(2100, log_dir=str(tmp_path))._cmd()
    assert "-RenderOffScreen" in cmd
