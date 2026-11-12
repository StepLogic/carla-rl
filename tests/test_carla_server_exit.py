"""A CRASHED EVAL MUST NOT LEAVE ITS CARLA SERVER RUNNING (2026-11-09).

The server is started in its own session (start_new_session=True) so stop() can
sweep its process group -- which also means nothing stops it when the Python that
started it dies. In-job evals whose checkpoint failed to load (strict load,
critic-ensemble size mismatch) exited with a traceback and left their servers on
4600 / 4650 / 4700; together they held 11.8 GB and the next process ran out of
memory. Fix: every started server stops itself at interpreter exit (atexit), and
il_eval_pair.sh sweeps each half's exact port after the half ends (covers a
SIGKILLed Python, which runs no atexit hook).
"""
import os
import subprocess
import sys
import textwrap
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from torch_rl import carla_server as cs  # noqa: E402


def _alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # a zombie still answers kill(0); it is dead for our purpose
    try:
        with open(f"/proc/{pid}/stat") as fh:
            return fh.read().split()[2] != "Z"
    except FileNotFoundError:
        return False


def test_a_crashed_python_stops_the_server_it_started(tmp_path):
    root = tmp_path / "carla"
    root.mkdir()
    (root / "CarlaUE4.sh").write_text("#!/bin/sh\n")
    script = textwrap.dedent(f"""
        import os, sys
        sys.path.insert(0, {ROOT!r})
        os.environ["CARLA_LOCAL_ROOT"] = {str(root)!r}
        from torch_rl import carla_server as cs
        class Dummy(cs.CarlaServer):
            def _cmd(self):
                return ["sleep", "300"]
        s = Dummy(29999, log_dir={str(tmp_path / "logs")!r})
        s.start()
        print(s.proc.pid, flush=True)
        raise RuntimeError("checkpoint failed to load")   # the eval crashes
    """)
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                         timeout=120, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
    assert out.returncode != 0 and "checkpoint failed to load" in out.stderr
    pid = int(out.stdout.strip().splitlines()[0])
    t0 = time.time()
    while _alive(pid) and time.time() - t0 < 10:
        time.sleep(0.2)
    survived = _alive(pid)
    if survived:                          # never leave the dummy behind, even on failure
        os.kill(pid, 9)
    assert not survived, "the server outlived the Python that started it"


class _Popen:
    def __init__(self, cmd, **kw):
        self.pid = 4242

    def poll(self):
        return 0


def test_one_exit_hook_per_server_however_often_it_restarts(monkeypatch, tmp_path):
    root = tmp_path / "carla"
    root.mkdir()
    (root / "CarlaUE4.sh").write_text("")
    monkeypatch.setenv("CARLA_LOCAL_ROOT", str(root))
    monkeypatch.setattr(cs.subprocess, "Popen", _Popen)
    hooks = []
    monkeypatch.setattr(cs.atexit, "register", lambda fn, *a, **k: hooks.append(fn))
    s = cs.CarlaServer(29998, log_dir=str(tmp_path / "logs"))
    s.start()
    s.start()
    assert len([h for h in hooks if getattr(h, "__self__", None) is s]) == 1


def test_the_exit_hook_never_raises(monkeypatch, tmp_path):
    s = cs.CarlaServer(29997, log_dir=str(tmp_path / "logs"))

    def boom(*a, **k):
        raise OSError("already gone")
    monkeypatch.setattr(s, "stop", boom)
    s._stop_at_exit()                     # must not raise during interpreter shutdown


def test_the_eval_pair_sweeps_each_halfs_exact_port_after_it_ends():
    src = open(os.path.join(ROOT, "slurm", "il_eval_pair.sh")).read()
    run = src.index("bash slurm/train_sac.sbatch")
    sweep = src.index("pkill -9 -f --", run)
    line = src[sweep:src.index("\n", sweep)]
    # exact port (no prefix match: 4600 must not catch 46000), bracketed so the
    # pattern cannot match the shell that runs it, and never fatal
    assert '"-carla-rpc-por[t]=$p( |\\$)"' in line
    assert line.rstrip().endswith("|| true")
    assert "p=$((PORT + 50 * wg))" in src[:sweep]
