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

TWO LESSONS ARE BUILT IN. `apptainer run` FORKS, so the UE4 binary's pid is not
the wrapper's and killing the wrapper leaves the simulator running and holding
GPU memory -- that orphan once had its memory credited to a different image. So
the process is started in its own session and the whole group is signalled, then
the binary is swept by name. And readiness is polled WHILE the server runs,
never by wrapping it in a timeout and judging the exit code, which is how a
healthy 44-second start was once scored as a failure.
"""
import os
import signal
import socket
import subprocess
import time
from typing import Optional

DEFAULT_SIF = os.path.expanduser("~/carla-cluster/carla0916nv.sif")
ENTRY = "/workspace/CarlaUE4.sh"
BINARY = "CarlaUE4-Linux-Shipping"


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
            start_new_session=True)          # its own group, so it can be swept
        self.n_starts += 1

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
