"""THE NVIDIA SHADER CACHE MUST BE NODE-LOCAL (2026-11-05).

Every CARLA start-up "segfault" on the cluster was UE4's own watchdog --
"GameThread timed out waiting for RenderThread after 60.00 secs" -- firing while
the town loaded. The NVIDIA driver keeps its shader disk cache under $HOME
(~/.cache/nvidia/GLCache), which on the ACG cluster is CephFS shared by every
node; a server with new shaders to compile stalled on it (its RHI and render
threads made no progress for 60 s while engine threads sat in file I/O).
Start-up hangs: 7 of 13 starts on RTX 2080 Ti, 1 of 2 on A100, 1 of 1 on A30,
1 in 130 on the L40s (whose cache was already complete). With the cache in
node-local /tmp, one directory per server: 0 of 6 (jobs 463306-463311), and the
first episode came ~45 s sooner.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from torch_rl import carla_server as cs  # noqa: E402

# `apptainer run -e` (cleanenv) passes only APPTAINERENV_-prefixed variables in.
KEY = "APPTAINERENV___GL_SHADER_DISK_CACHE_PATH"


class _Popen:
    calls = []

    def __init__(self, cmd, **kw):
        _Popen.calls.append((cmd, kw))
        self.pid = 12345

    def poll(self):
        return None


def _start(monkeypatch, tmp_path, port=2100, server=None):
    monkeypatch.delenv("CARLA_LOCAL_ROOT", raising=False)
    monkeypatch.setenv("CARLA_SHADER_CACHE_ROOT", str(tmp_path / "local"))
    sif = tmp_path / "carla.sif"
    sif.write_text("")
    monkeypatch.setattr(cs.subprocess, "Popen", _Popen)
    s = server or cs.CarlaServer(port, sif=str(sif), log_dir=str(tmp_path / "logs"))
    s.start()
    return s, (_Popen.calls[-1][1].get("env") or {})


def test_the_cluster_server_is_given_a_node_local_shader_cache(monkeypatch, tmp_path):
    _, env = _start(monkeypatch, tmp_path)
    d = env.get(KEY)
    assert d, "the image's driver would fall back to ~/.cache/nvidia on Ceph"
    assert d.startswith(str(tmp_path / "local"))
    assert os.path.isdir(d), "the driver does not create the directory itself"


def test_the_rest_of_the_environment_still_reaches_the_server(monkeypatch, tmp_path):
    """The cache path is ADDED to the trainer's environment, not swapped for it:
    apptainer needs PATH, HOME and the rest to run at all."""
    monkeypatch.setenv("SHADER_CACHE_TEST_VAR", "kept")
    _, env = _start(monkeypatch, tmp_path)
    assert env.get("SHADER_CACHE_TEST_VAR") == "kept"


def test_each_server_gets_its_own_cache(monkeypatch, tmp_path):
    """One CARLA per directory is the configuration that ran clean; two servers
    in one job (parallel towns) must not share one."""
    _, e1 = _start(monkeypatch, tmp_path, port=2100)
    _, e2 = _start(monkeypatch, tmp_path, port=2150)
    assert e1[KEY] != e2[KEY]


def test_a_restart_reuses_the_servers_own_cache(monkeypatch, tmp_path):
    """A town change restarts the same server; its shaders are already there."""
    s, e1 = _start(monkeypatch, tmp_path)
    _, e2 = _start(monkeypatch, tmp_path, server=s)
    assert e1[KEY] == e2[KEY]


def test_the_default_root_is_node_local_tmp(monkeypatch):
    monkeypatch.delenv("CARLA_SHADER_CACHE_ROOT", raising=False)
    assert cs.shader_cache_root() == "/tmp"


def test_the_cache_is_removed_when_the_trainer_exits(monkeypatch, tmp_path):
    """/tmp is shared by every job on a node and Slurm does not clean it
    (job_container/none), so the server removes its own directory -- once, at
    exit, not on stop(), which a restart calls."""
    registered = []
    monkeypatch.setattr(cs.atexit, "register",
                        lambda fn, *a, **k: registered.append((fn, a, k)))
    s, env = _start(monkeypatch, tmp_path)
    _start(monkeypatch, tmp_path, server=s)
    assert len(registered) == 1
    fn, a, k = registered[0]
    fn(*a, **k)
    assert not os.path.exists(env[KEY])


def test_a_local_carla_keeps_the_driver_default(monkeypatch, tmp_path):
    """Locally the home directory is a local disk and there is no apptainer."""
    root = tmp_path / "CARLA"
    root.mkdir()
    (root / "CarlaUE4.sh").write_text("")
    monkeypatch.setenv("CARLA_LOCAL_ROOT", str(root))
    monkeypatch.setattr(cs.subprocess, "Popen", _Popen)
    cs.CarlaServer(2100, log_dir=str(tmp_path / "logs")).start()
    env = _Popen.calls[-1][1].get("env")
    assert env is None or KEY not in env
