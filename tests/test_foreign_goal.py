"""v34 (user 2026-11-02): explore episodes are goal episodes with an UNREACHABLE
goal -- an image from a DIFFERENT training town -- rewarded for driving forward
in any direction and paid the goal reward at 100 m driven. With the mode bit
hidden from the network, the image alone says whether to navigate and stop
(true goal) or keep driving (foreign goal): the pressure v29 and v33 lacked,
both of which learned to ignore the image.

FOREIGN_GOAL_EXPLORE=1: in an explore episode the latched goal frame comes from
another town's pool. Pools hold TRAINING towns only (every env adds its own
goal views), so no held-out eval scenery reaches training. Until another town
has contributed a frame, an own-town frame >= WRONG_GOAL_MIN_M away is used and
the episode is flagged, so the fallback is counted, never silent.

HIDE_MODE_BIT=1: the network row always carries mode = 1 (goal). The TRUE mode
still drives the reward, the replay flag and HER, which read it from the
observation and info, not from the row.
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))
pytest.importorskip("gymnasium")
import gymnasium as gym                        # noqa: E402
from gymnasium import spaces                   # noqa: E402


class GoalEnv(gym.Env):
    """Publishes a goal frame whose pixels encode (town tag, episode)."""

    def __init__(self, tag, modes):
        self.observation_space = spaces.Dict({"vector": spaces.Box(-1, 1, (4,), np.float32)})
        self.action_space = spaces.Box(-1, 1, (2,), np.float32)
        self.tag, self.modes, self.ep = tag, list(modes), -1

    def _info(self):
        f = np.full((8, 8, 3), self.tag * 10 + self.ep % 10, np.uint8)
        return {"rgb": np.zeros((8, 8, 3), np.uint8), "goal_rgb": f,
                "goal_pos": (100.0 * self.ep, 0.0),
                "mode": self.modes[self.ep % len(self.modes)]}

    def reset(self, **kw):
        self.ep += 1
        return {"vector": np.zeros(4, np.float32)}, self._info()

    def step(self, a):
        return {"vector": np.zeros(4, np.float32)}, 0.0, False, False, self._info()


@pytest.fixture
def fg(monkeypatch):
    monkeypatch.setenv("FOREIGN_GOAL_EXPLORE", "1")
    import importlib
    import vae_nav.debit_goal_wrapper as w
    w = importlib.reload(w)
    w.FOREIGN_POOL.clear()
    return w


def _tag_of(frame):
    return int(frame.flat[0]) // 10


def test_an_explore_episode_sees_another_towns_goal(fg):
    a = fg.DebitGoalWrapper(GoalEnv(1, [1, 1, 0]), town="TownA")
    b = fg.DebitGoalWrapper(GoalEnv(2, [1]), town="TownB")
    b.reset()                                   # TownB contributes a frame
    a.reset(); a.reset()                        # two goal episodes in TownA
    _, info = a.reset()                         # explore episode in TownA
    assert info["served_foreign_goal"] is True
    assert _tag_of(info["debit_goal"]) == 2     # a TownB view, not TownA's


def test_goal_episodes_keep_their_true_goal(fg):
    a = fg.DebitGoalWrapper(GoalEnv(1, [1]), town="TownA")
    b = fg.DebitGoalWrapper(GoalEnv(2, [1]), town="TownB")
    b.reset()
    _, info = a.reset()
    assert info["served_foreign_goal"] is False and _tag_of(info["debit_goal"]) == 1


def test_no_foreign_frame_yet_falls_back_and_says_so(fg):
    a = fg.DebitGoalWrapper(GoalEnv(1, [1, 0]), town="TownA")
    a.reset()                                   # goal ep: TownA frame banked
    _, info = a.reset()                         # explore ep, no other town yet
    assert info["served_foreign_goal"] is False
    assert info["foreign_goal_fallback"] is True


def test_off_by_default(monkeypatch):
    monkeypatch.delenv("FOREIGN_GOAL_EXPLORE", raising=False)
    import importlib
    import vae_nav.debit_goal_wrapper as w
    w = importlib.reload(w)
    w.FOREIGN_POOL.clear()
    a = w.DebitGoalWrapper(GoalEnv(1, [0]), town="TownA")
    b = w.DebitGoalWrapper(GoalEnv(2, [1]), town="TownB")
    b.reset()
    _, info = a.reset()
    assert _tag_of(info["debit_goal"]) == 1     # its own goal, as before
    assert info["served_foreign_goal"] is False


def test_the_true_mode_still_reaches_the_observation(fg):
    """The loop, the replay flag and HER read the mode from the observation;
    hiding happens only in the NETWORK row (train_carla_sac._net_mode)."""
    a = fg.DebitGoalWrapper(GoalEnv(1, [0]), town="TownA")
    obs, _ = a.reset()
    assert float(obs["mode"][0]) == 0.0


def test_the_network_rows_use_the_hidden_mode():
    src = open(os.path.join(ROOT, "train_carla_sac.py")).read()
    assert 'HIDE_MODE_BIT = os.environ.get("HIDE_MODE_BIT", "0") == "1"' in src
    assert src.count("_net_mode(") >= 4             # def + training + eval + trace rows
    assert 'pxr.mono_row(i["debit_cur"], _net_mode(o["mode"]), o["binoc"],\n                            speed_kmh=_speed_kmh(i))' in src
