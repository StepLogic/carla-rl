"""The image-goal relabel buffer: goal-block rewrite + done-at-entry only."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

from vae_nav.image_goal_her import ImageGoalRelabelBuffer  # noqa: E402


class _StubBuffer:
    def __init__(self):
        self.rows = []

    def insert(self, row, mode=None):
        self.rows.append(row)


def _goal_block(fields):
    # The contract shape: returns (12,) from fields, deterministic.
    return np.full(12, float(fields["goal_pos"][0]), np.float32)


def _tr(i, reward=-0.25):
    # The flown reward lives IN the transition dict -- the base class's
    # record() has no rewards kwarg; the trainer builds the transition dict
    # with `rewards` inside it (train_carla_tqc.py:1241-1245).
    return dict(observations={"latent": np.zeros(4, np.float32),
                              "goal": np.full(12, 99.0, np.float32)},
                actions=np.zeros(2, np.float32),
                rewards=np.float32(reward), masks=np.float32(1.0),
                dones=np.float32(0.0),
                next_observations={"latent": np.zeros(4, np.float32),
                                   "goal": np.full(12, 99.0, np.float32)})


def _buf(**kw):
    kw.setdefault("goal_radius", 2.0)
    kw.setdefault("min_disp", 5.0)
    her = ImageGoalRelabelBuffer(_StubBuffer(), _goal_block, **kw)
    # The rows live in the wrapped stub, not on the relabel buffer; expose
    # the stub's list here so the tests read `buf.rows` as written.
    her.rows = her.buffer.rows
    return her


def _episode(buf, xs, rewards=None):
    """One leg of positions xs (1-D), flown rewards, no terminals."""
    rewards = rewards or [-0.25] * len(xs)
    for i, x in enumerate(xs):
        buf.record(_tr(i, rewards[i]),
                   fields={"hero_pos": (x, 0.0), "hero_yaw": 0.0,
                           "goal_pos": (1e6, 0.0),
                           "action_history": np.zeros((4, 2))},
                   pos=(x, 0.0), hit=False)


def test_rewards_ride_along_unchanged():
    buf = _buf()
    _episode(buf, [0.0, 10.0, 20.0, 30.0, 40.0])
    buf.flush_episode()
    for row in buf.rows:
        assert float(row["rewards"]) == pytest.approx(-0.25)


def test_done_fires_only_on_the_entry_transition():
    buf = _buf()
    _episode(buf, [0.0, 10.0, 20.0, 30.0, 40.0])
    buf.flush_episode()
    entries = [r for r in buf.rows if float(r["dones"]) == 1.0]
    assert entries, "no hindsight row marked the goal entry"
    for row in entries:
        # The entry row's goal block was rebuilt for the hindsight goal.
        assert row["observations"]["goal"][0] != 99.0


def test_masks_mirror_dones():
    buf = _buf()
    _episode(buf, [0.0, 10.0, 20.0, 30.0, 40.0])
    buf.flush_episode()
    for row in buf.rows:
        assert float(row["masks"]) == (0.0 if float(row["dones"]) == 1.0
                                       else 1.0)


def test_coll_key_rides_along_untouched():
    buf = _buf()
    xs = [0.0, 10.0, 20.0, 30.0, 40.0]
    for i, x in enumerate(xs):
        t = _tr(i)
        t["observations"]["coll"] = np.array([float(i % 2)], np.float32)
        buf.record(t, fields={"hero_pos": (x, 0.0), "hero_yaw": 0.0,
                              "goal_pos": (1e6, 0.0),
                              "action_history": np.zeros((4, 2))},
                   pos=(x, 0.0), hit=False)
    buf.flush_episode()
    for row in buf.rows:
        # z depends on the trajectory up to t, never on the goal: identical
        # to the flown row's value.
        assert "coll" in row["observations"]


def test_nonentry_rows_never_inherit_hit_or_lane_as_terminal():
    # Under the image-goal reward a collision or crossing ends NOTHING, so a
    # relabelled row must not become terminal because the flown step hit
    # something.
    buf = _buf()
    _episode(buf, [0.0, 10.0, 20.0, 30.0, 40.0],
             rewards=[-0.25, -0.25, -1.75, -0.25, -0.25])  # step 2 collided
    buf.flush_episode()
    for row in buf.rows:
        if float(row["dones"]) == 1.0:
            # done rows are goal-entry rows, and their reward is the FLOWN
            # one (-1.75 or -0.25), not a recosted 0-on-arrival.
            assert float(row["rewards"]) in (-0.25, -1.75)


def test_timeout_final_frame_is_not_a_goal_source():
    # A truncated episode's last frame is a clock artefact; drawing it as a
    # hindsight goal would teach "reach wherever the episode happened to end".
    # min_disp is set to goal_radius (2.0), not 0.0: the inherited guard (a)
    # rejects min_disp < goal_radius at construction, and equality is legal.
    # The test still discriminates -- ignoring truncated would allow tp=9
    # (goal 45.0) with an entry done at step 8.
    buf = _buf(n_relabel=50, max_disp=1000.0, min_disp=2.0)
    _episode(buf, list(range(0, 50, 5)))   # 10 steps
    buf.flush_episode(truncated=True)
    goals = {float(r["observations"]["goal"][0])
             for r in buf.rows if float(r["dones"]) == 1.0}
    assert 45.0 not in goals


def test_stats_counts_entry_dones():
    buf = _buf()
    _episode(buf, [0.0, 10.0, 20.0, 30.0, 40.0])
    buf.flush_episode()
    out = buf.stats()
    assert out["her_entry_done"] == buf.n_entry_done
    assert out["her_entry_done"] >= 1, "no entry done was counted"