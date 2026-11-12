#!/usr/bin/env python3
"""Own the CARLA server from Python, so a town change can restart it.

WHY THIS EXISTS. Training on six towns needs the map to change during a run, and
the cheap way -- close the env and rebuild it against the same live server --
does not work: the first change succeeds, the next rebuild hangs retrying
"time-out of 50000ms while waiting for the simulator", and the server meanwhile
floods "Invalid session: no stream available with id N". env.close() leaves the
old sensors' streams dangling and the next client cannot attach (measured, job
453315). Restarting the simulator is the one approach that cannot inherit that
state, and at ~60 s a restart against 25 episodes of driving it costs a few per
cent of wall clock.

The local rule "start CARLA from Python" holds here too, but the mechanism
differs: on the cluster the simulator lives in an apptainer image, so CarlaCore
cannot spawn it (there is no host binary) and this does it instead.

THREE LESSONS ARE BUILT IN. `apptainer run` FORKS, so the UE4 binary's pid is not
the wrapper's and killing the wrapper leaves the simulator running and holding
GPU memory -- that orphan once had its memory credited to a different image. So
the process is started in its own session and the whole group is signalled, then
the binary is swept by name. Readiness is polled WHILE the server runs,
never by wrapping it in a timeout and judging the exit code, which is how a
healthy 44-second start was once scored as a failure. And the NVIDIA driver's
shader disk cache is moved off the home directory: there it is
~/.cache/nvidia/GLCache on CephFS, shared by every node, and a server with new
shaders to compile stalled on it while the town loaded until UE4's watchdog
killed it ("GameThread timed out waiting for RenderThread after 60.00 secs",
reported as Signal 11). That was 7 of 13 starts on the RTX 2080 Ti nodes; with
one node-local directory per server, 0 of 6 (2026-11-05, jobs 463306-463311).
"""
import atexit
import os
import shutil
import signal
import socket
import subprocess
import time
from typing import Optional

DEFAULT_SIF = os.path.expanduser("~/carla-cluster/carla0916nv.sif")
ENTRY = "/workspace/CarlaUE4.sh"
BINARY = "CarlaUE4-Linux-Shipping"
# `apptainer run -e` (cleanenv) lets in only APPTAINERENV_-prefixed variables.
SHADER_CACHE_KEY = "APPTAINERENV___GL_SHADER_DISK_CACHE_PATH"


def shader_cache_root() -> str:
    """Where the per-server shader caches live: node-local /tmp unless
    CARLA_SHADER_CACHE_ROOT says otherwise. Never the (Ceph) home directory."""
    return os.environ.get("CARLA_SHADER_CACHE_ROOT", "/tmp")


def shader_cache_dir(port: int) -> str:
    """One directory per server: the Slurm job and the RPC port (a job can run
    several servers, one per port)."""
    job = os.environ.get("SLURM_JOB_ID") or f"pid{os.getpid()}"
    return os.path.join(shader_cache_root(), f"carla_nvcache_{job}_{int(port)}")


def connect_with_restarts(build, server, tries: int = 3, log=print):
    """build() connects a new env to `server`; on CarlaCore's "Cannot connect to
    server" the server is restarted and build() tried again, up to `tries`
    builds in all. Even with the node-local shader cache about one start in
    sixteen hung in the client's first load_world until UE4's watchdog killed
    the server (2026-11-05); a restart costs about a minute where waiting the
    hang out cost the job. Any other error is raised at once."""
    for attempt in range(1, tries + 1):
        try:
            return build()
        except Exception as e:      # CarlaCore raises a bare Exception
            if (server is None or attempt == tries
                    or "Cannot connect to server" not in str(e)):
                raise
            log(f"  carla[{server.port}]: no answer after start ({e}); "
                f"restarting the server, build {attempt + 1} of {tries}", flush=True)
            took = server.restart()
            log(f"  carla[{server.port}]: restarted, ready in {took:.0f}s", flush=True)


def port_open(port: int, host: str = "localhost", timeout: float = 1.0) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        return s.connect_ex((host, int(port))) == 0


class CarlaServer:
    """Start, wait for, and stop one containerised CARLA."""

    def __init__(self, port: int, sif: str = None, quality: str = "Epic",
                 log_dir: str = ".", extra_args=()) -> None:
        self.port = int(port)
        self.sif = sif or os.environ.get("CARLA_SIF", DEFAULT_SIF)
        self.quality = quality
        self.log_dir = log_dir
        self.extra_args = list(extra_args)
        self.proc: Optional[subprocess.Popen] = None
        self.n_starts = 0
        self._cache_dir: Optional[str] = None
        self._exit_hook = False
        os.makedirs(log_dir, exist_ok=True)

    # ------------------------------------------------------------- lifecycle --
    def _cmd(self) -> list:
        """The launch command. CARLA_LOCAL_ROOT (user 2026-10-27, local v18):
        a local CARLA install instead of the apptainer image, WINDOWED unless
        CARLA_OFFSCREEN=1 -- the user's rule for local training, and Town05
        segfaults offscreen here. Unset, the cluster command is unchanged."""
        root = os.environ.get("CARLA_LOCAL_ROOT")
        if root:
            off = os.environ.get("CARLA_OFFSCREEN", "0") == "1"
            return ([os.path.join(root, "CarlaUE4.sh")]
                    + (["-RenderOffScreen"] if off else [])
                    + ["-nosound", f"-quality-level={self.quality}",
                       f"-carla-rpc-port={self.port}"] + self.extra_args)
        return ["apptainer", "run", "--nv", "-e", self.sif, ENTRY,
                "-RenderOffScreen", "-nosound",
                f"-quality-level={self.quality}",
                f"-carla-rpc-port={self.port}"] + self.extra_args

    def start(self) -> None:
        root = os.environ.get("CARLA_LOCAL_ROOT")
        target = os.path.join(root, "CarlaUE4.sh") if root else self.sif
        if not os.path.exists(target):
            raise SystemExit(f"no CARLA at {target}")
        log = os.path.join(self.log_dir, f"carla_{self.port}_{self.n_starts}.log")
        cmd = self._cmd()
        self.proc = subprocess.Popen(
            cmd, stdout=open(log, "w"), stderr=subprocess.STDOUT,
            start_new_session=True,          # its own group, so it can be swept
            env=self._launch_env())
        self.n_starts += 1
        # Its own session also means nothing stops it when THIS Python dies: an
        # eval that crashed (strict load failure) left its servers on 4600/4650/
        # 4700 holding 11.8 GB until the next process ran out of memory
        # (2026-11-09). Stop it at interpreter exit -- registered once per server,
        # after the shader-cache hook, so it runs first (atexit is LIFO).
        if not self._exit_hook:
            atexit.register(self._stop_at_exit)
            self._exit_hook = True

    def _stop_at_exit(self) -> None:
        """atexit: stop this server as the Python that started it exits. Never
        raises (interpreter shutdown)."""
        try:
            self.stop(grace=3.0)
        except Exception:
            pass

    def _launch_env(self) -> Optional[dict]:
        """The server's environment. A local CARLA inherits the trainer's as is
        (None). The apptainer image gets it plus a node-local shader cache,
        created here, kept across restarts (a town change restarts the same
        server) and removed when the trainer exits: /tmp is shared by every job
        on a node and nothing else cleans it."""
        if os.environ.get("CARLA_LOCAL_ROOT"):
            return None
        d = shader_cache_dir(self.port)
        os.makedirs(d, exist_ok=True)
        if self._cache_dir is None:
            self._cache_dir = d
            atexit.register(shutil.rmtree, d, True)
        return dict(os.environ, **{SHADER_CACHE_KEY: d})

    def wait_ready(self, timeout: float = 240.0, poll: float = 1.0) -> float:
        """Block until the RPC port answers. Returns the seconds it took.

        Polls WHILE the server runs and checks liveness each pass, so a crashed
        start is reported as a crash rather than as a timeout.
        """
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.proc is not None and self.proc.poll() is not None:
                raise SystemExit(
                    f"CARLA exited with {self.proc.returncode} after "
                    f"{time.time() - t0:.0f}s; see {self.log_dir}")
            if port_open(self.port):
                return time.time() - t0
            time.sleep(poll)
        raise SystemExit(f"CARLA did not open port {self.port} in {timeout:.0f}s")

    def stop(self, grace: float = 8.0) -> None:
        """Signal the whole process group, then sweep the binary by name."""
        if self.proc is not None and self.proc.poll() is None:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
            t0 = time.time()
            while time.time() - t0 < grace and self.proc.poll() is None:
                time.sleep(0.2)
            if self.proc.poll() is None:
                try:
                    os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
        # The forked UE4 binary can outlive the wrapper; sweep it by name, but
        # only ours -- the pattern is the binary, and the port is checked after.
        subprocess.run(["pkill", "-9", "-f", f"{BINARY}.*carla-rpc-port={self.port}"],
                       check=False)
        for _ in range(20):
            if not port_open(self.port):
                return
            time.sleep(0.5)

    def restart(self, timeout: float = 240.0) -> float:
        self.stop()
        self.start()
        return self.wait_ready(timeout=timeout)

    def describe(self) -> str:
        return (f"CarlaServer(port={self.port}, quality={self.quality}, "
                f"starts={self.n_starts}, image="
                f"{os.environ.get('CARLA_LOCAL_ROOT') or os.path.basename(self.sif)})")
