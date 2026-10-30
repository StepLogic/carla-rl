"""CROSSINGS PER EPISODE IN THE TRAINING LOG (user 2026-10-27, "count IT").

A fined crossing never ends a training episode, so it appeared in no tally:
over 79 local v18 episodes the ends were only collisions and arrivals, and
"are crossings improving?" could only be answered at an eval checkpoint. The
reward now publishes info["lane_fined"] on the step it charges one -- the same
junction-exempt, marking-aware test it fines, so the count cannot drift from
the fine -- and the loop reports the per-episode mean as cross= in [Ep].
"""
import importlib
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))
pytest.importorskip("gymnasium")
torch = pytest.importorskip("torch")

from torch_rl.episode_buffer import DeviceEpisodeBuffer  # noqa: E402
from torch_rl.parallel_loop import run_parallel  # noqa: E402
from vae_nav.vec_env import ThreadedVecEnv  # noqa: E402

from tests.test_parallel_loop import BINOC, FakeAgent, FakeTown, _row  # noqa: E402


def _reward(monkeypatch, **env):
    """Reload the reward with the lane rule pinned. Every var the rule reads is
    set here, not inherited: another test module reloading lane_explore_reward
    with LANE_ANY_FATAL=1 (every crossing fatal) otherwise decides these."""
    for k, v in dict(LANE_ANY_FATAL=0, LANE_STRICT=0, LANE_INVASION_NOOP=0,
                     LANE_UNKNOWN_MARKS_FATAL=1, **env).items():
        monkeypatch.setenv(k, str(v))
    import vae_nav.lane_explore_reward as ler
    importlib.reload(ler)
    import vae_nav.gated_reward as gr
    return importlib.reload(gr)


@pytest.fixture(autouse=True)
def _restore_reward_modules():
    yield
    import vae_nav.lane_explore_reward as ler
    import vae_nav.gated_reward as gr
    importlib.reload(ler)
    importlib.reload(gr)


def _info(**kw):
    base = {"wp_forward": np.array([1.0, 0.0]), "speed_kmh": 20.0,
            "speed_factor": 0.5, "action_smoothness_penalty": 0.0}
    base.update(kw)
    return base


DISP = np.array([0.25, 0.0])


def test_a_painted_crossing_is_published_and_charged(monkeypatch):
    gr = _reward(monkeypatch, EXPLORE_LANE_MODE="penalise", LANE_DEPARTURE_PENALTY=20.0)
    clean, crossed = _info(), _info(lane_any=True, lane_marking_types=["Solid"])
    r_clean, _ = gr.explore_reward(clean, DISP)
    r_cross, fatal = gr.explore_reward(crossed, DISP)
    assert crossed["lane_fined"] is True and clean["lane_fined"] is False
    assert not fatal                                  # fined, not terminal
    assert r_clean - r_cross == pytest.approx(20.0)   # the count IS the fine


def test_the_turn_is_neither_fined_nor_counted(monkeypatch):
    """A BROKEN line inside a junction is the turn: forgiven, so uncounted. A
    SOLID line is a fault wherever it is crossed -- the exemption is not
    "junctions are free"."""
    gr = _reward(monkeypatch, EXPLORE_LANE_MODE="penalise", LANE_DEPARTURE_PENALTY=20.0)
    turn = _info(lane_any=True, lane_marking_types=["Broken"], in_junction=True)
    r, _ = gr.explore_reward(turn, DISP)
    assert turn["lane_fined"] is False
    assert r == pytest.approx(gr.explore_reward(_info(), DISP)[0])

    solid = _info(lane_any=True, lane_marking_types=["Solid"], in_junction=True)
    gr.explore_reward(solid, DISP)
    assert solid["lane_fined"] is True


def test_terminal_mode_also_publishes_it(monkeypatch):
    gr = _reward(monkeypatch, EXPLORE_LANE_MODE="terminal", LANE_DEPARTURE_PENALTY=20.0)
    inf = _info(lane_any=True, lane_marking_types=["Solid"])
    _r, fatal = gr.explore_reward(inf, DISP)
    assert fatal and inf["lane_fined"] is True


class CrossTown(FakeTown):
    """Reports a fined crossing on every other step."""

    def step(self, action):
        o, r, term, trunc, inf = super().step(action)
        inf["lane_fined"] = (self.t % 2 == 0)
        return o, r, term, trunc, inf


def test_the_log_line_reports_crossings_per_episode(tmp_path):
    venv = ThreadedVecEnv([lambda: CrossTown(0, 4, True)], names=["TownA"])
    buf = DeviceEpisodeBuffer(1 + BINOC, 2, 1000, "cpu")
    logs = []
    run_parallel(venv, FakeAgent(), buf, _row, venv.action_space,
                 to_env=lambda a, b: a, to_actor=lambda e, b: e,
                 goal_block=lambda g: np.asarray(g, np.float32).reshape(-1),
                 max_steps=40, learning_starts=0, update_every=100, updates_per=1,
                 chunk_b=1, chunk_t=2, burn_in=1, ckpt_every=100,
                 out_dir=str(tmp_path), log_every=2, log=logs.append)
    ep = [s for s in logs if s.startswith("[Ep")]
    assert ep and "cross=2.00" in ep[-1], ep[-1]      # 2 of every 4 steps


# ---- declarations per episode (user 2026-10-28) ---------------------------

class DeclareTown(FakeTown):
    """Declares on every other step; the last step of each episode is at goal."""

    def step(self, action):
        o, r, term, trunc, inf = super().step(action)
        inf["declared"] = (self.t % 2 == 0)
        inf["false_stop"] = inf["declared"] and not (term or trunc)
        return o, r, term, trunc, inf


def test_the_log_line_reports_declarations_per_episode(tmp_path):
    """v21 showed zero arrivals in 297 episodes and nothing said whether the
    car was declaring at all -- declaring often but badly, and never declaring,
    look identical in the log."""
    venv = ThreadedVecEnv([lambda: DeclareTown(0, 4, True)], names=["TownA"])
    buf = DeviceEpisodeBuffer(1 + BINOC, 2, 1000, "cpu")
    logs = []
    run_parallel(venv, FakeAgent(), buf, _row, venv.action_space,
                 to_env=lambda a, b: a, to_actor=lambda e, b: e,
                 goal_block=lambda g: np.asarray(g, np.float32).reshape(-1),
                 max_steps=40, learning_starts=0, update_every=100, updates_per=1,
                 chunk_b=1, chunk_t=2, burn_in=1, ckpt_every=100,
                 out_dir=str(tmp_path), log_every=2, log=logs.append)
    ep = [s for s in logs if s.startswith("[Ep")]
    assert ep and "declare=2.00/ep" in ep[-1], ep[-1]


# ---- junctions cleared per explore episode (v26, 2026-10-30) --------------

class JunctionTown(FakeTown):
    """An EXPLORE episode (mode 0) whose reward reports 3 junctions cleared by
    its last step."""

    def _obs(self):
        o = super()._obs()
        o["mode"] = np.array([0.0], np.float32)
        return o

    def step(self, action):
        o, r, term, trunc, inf = super().step(action)
        inf["junctions_cleared"] = 3 if (term or trunc) else 1
        return o, r, term, trunc, inf


def test_the_log_line_reports_junctions_per_explore_episode(tmp_path):
    """Under v26 explore episodes have no arrival, so junctions cleared is the
    only sign in the log that they get anywhere."""
    venv = ThreadedVecEnv([lambda: JunctionTown(0, 4, True)], names=["TownA"])
    buf = DeviceEpisodeBuffer(1 + BINOC, 2, 1000, "cpu")
    logs = []
    run_parallel(venv, FakeAgent(), buf, _row, venv.action_space,
                 to_env=lambda a, b: a, to_actor=lambda e, b: e,
                 goal_block=lambda g: np.asarray(g, np.float32).reshape(-1),
                 max_steps=40, learning_starts=0, update_every=100, updates_per=1,
                 chunk_b=1, chunk_t=2, burn_in=1, ckpt_every=100,
                 out_dir=str(tmp_path), log_every=2, log=logs.append)
    ep = [s for s in logs if s.startswith("[Ep")]
    assert ep and "junc=3.00/ep" in ep[-1], ep[-1]
