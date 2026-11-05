"""A CARLA start that hangs is restarted, not waited out (2026-11-05).

Even with the node-local shader cache, about one start in sixteen still hung in
the client's first load_world: the render thread stalls, UE4's 60 s watchdog
kills the server, and CarlaCore went on retrying the dead port 25 x 50 s before
raising "Cannot connect to server" -- 21 minutes, then the job failed and the
eval half was lost. A healthy start never needed a second attempt. So the
client now gives up after a few attempts and the trainer restarts the server
and builds the env again.
"""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from torch_rl.carla_server import connect_with_restarts  # noqa: E402

HANG = Exception("Cannot connect to server. Try increasing 'timeout' or "
                 "'retries_on_error' at the carla configuration")


class _Server:
    port = 4150

    def __init__(self):
        self.restarts = 0

    def restart(self):
        self.restarts += 1
        return 17.0


def _builder(failures, error=HANG):
    calls = {"n": 0}

    def build():
        calls["n"] += 1
        if calls["n"] <= failures:
            raise error
        return "env"
    return build, calls


def test_a_start_that_cannot_connect_restarts_the_server_and_builds_again():
    srv = _Server()
    build, calls = _builder(failures=1)
    assert connect_with_restarts(build, srv, tries=3, log=lambda *a, **k: None) == "env"
    assert srv.restarts == 1 and calls["n"] == 2


def test_other_errors_are_raised_at_once():
    """Only the connect failure means a dead or hung server; anything else is
    a real bug and must not be papered over by restarts."""
    srv = _Server()
    build, calls = _builder(failures=1, error=ValueError("bad config"))
    with pytest.raises(ValueError):
        connect_with_restarts(build, srv, tries=3, log=lambda *a, **k: None)
    assert srv.restarts == 0 and calls["n"] == 1


def test_it_gives_up_after_the_last_try():
    srv = _Server()
    build, calls = _builder(failures=99)
    with pytest.raises(Exception, match="Cannot connect to server"):
        connect_with_restarts(build, srv, tries=3, log=lambda *a, **k: None)
    assert calls["n"] == 3 and srv.restarts == 2


def test_each_restart_is_logged():
    srv = _Server()
    build, _ = _builder(failures=1)
    lines = []
    connect_with_restarts(build, srv, tries=3, log=lambda *a, **k: lines.append(" ".join(map(str, a))))
    assert any("4150" in ln and "restart" in ln for ln in lines)


def test_the_client_gives_up_after_a_few_attempts():
    """25 attempts of 50 s against a dead port is the 21-minute wait."""
    src = open(os.path.join(ROOT, "src", "configs", "train_env_config_v4_image_goal.py")).read()
    assert '"retries_on_error": int(os.environ.get("CARLA_CLIENT_RETRIES", 3))' in src


def test_every_env_build_in_the_trainer_goes_through_the_restarts():
    """The eval/serial build, the town rotation and the parallel factory: a
    bare build_env against a fresh server would wait out a hang again."""
    s = open(os.path.join(ROOT, "train_carla_sac.py")).read()
    main_src = s[s.index("\ndef main():"):]
    assert main_src.count("connect_with_restarts(") >= 3
    for line in main_src.splitlines():
        stripped = line.strip()
        if "build_env(" in stripped and not stripped.startswith(("#", "from ")):
            assert "lambda" in stripped, f"unwrapped build_env: {stripped}"
