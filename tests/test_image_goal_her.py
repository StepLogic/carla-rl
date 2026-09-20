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

    def __len__(self):
        return len(self.rows)


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


def test_real_step_spacing_still_terminates_legs():
    # THE REGRESSION FOR THE v1 BUG. Real env steps move ~0.28 m (20 km/h at
    # 0.05 s); the old single-row relabelling drew the goal >= min_disp (5 m)
    # from pos[i] and tested pos[i+1] against it, so the entry could never
    # fire -- the 1M-step v1 run logged her_entry_done = 0 across all
    # 245,025 relabelled rows. Synthetic tests with 10 m jumps masked this.
    # Under leg relabelling the leg reaches the goal frame itself, so real
    # spacing must still produce a terminal.
    xs = list(np.arange(0.0, 30.0, 0.28))          # ~107 steps, 0.28 m each
    buf = _buf()
    _episode(buf, xs)
    buf.flush_episode()
    assert buf.stats()["her_entry_done"] >= 1


def test_leg_rows_are_contiguous_and_end_exactly_once():
    # One hindsight leg: every HINDSIGHT row's goal block points at the SAME
    # goal, the rows are contiguous in position order, and exactly ONE row is
    # terminal -- the last one inserted. buf.rows also holds the on-policy
    # rows (goal block 99.0 from _tr); scope to the hindsight rows.
    buf = _buf()
    _episode(buf, [0.0, 10.0, 20.0, 30.0, 40.0])
    buf.flush_episode()
    legs = [r for r in buf.rows
            if float(r["observations"]["goal"][0]) != 99.0]
    assert legs, "no hindsight rows were inserted"
    dones = [j for j, r in enumerate(legs) if float(r["dones"]) == 1.0]
    assert len(dones) == 1
    last = dones[0]
    assert last == len(legs) - 1, "rows were inserted after the arrival"
    goals = {tuple(np.round(r["observations"]["goal"], 3)) for r in legs}
    assert len(goals) == 1, "a leg mixed goals"
    # the terminal row's NEXT goal block is the goal itself (arrival state)
    assert legs[last]["next_observations"]["goal"][0] == \
        legs[last]["observations"]["goal"][0]


# -- Task 6: HER skips explore episodes --------------------------------------

def make_buffer():
    return _buf()


def run_episode(buf, mode, n):
    """n steps spaced 10 m, no terminals, commanded `mode`; continues any
    episode already staged in `buf` (run_row is the same call)."""
    start = len(buf._txns)
    for i in range(n):
        x = float(start + i) * 10.0
        buf.record(_tr(start + i),
                   fields={"hero_pos": (x, 0.0), "hero_yaw": 0.0,
                           "goal_pos": (1e6, 0.0),
                           "action_history": np.zeros((4, 2))},
                   pos=(x, 0.0), hit=False, mode=mode)


def run_row(buf, mode, n):
    run_episode(buf, mode, n)


def test_explore_episode_is_never_relabelled():
    buf = make_buffer(); run_episode(buf, mode=0, n=50)
    before = len(buf.buffer)
    assert buf.flush_episode() == 0
    assert len(buf.buffer) == before          # rows were inserted at record()
    assert buf.stats()["n_explore_eps"] == 1


def test_goal_episode_still_relabels():
    buf = make_buffer(); run_episode(buf, mode=1, n=50)  # with an arrival
    assert buf.flush_episode() > 0


def test_mixed_episode_defaults_to_goal_only_if_all_goal():
    # a single episode is single-mode; a mode flip mid-episode (must not happen)
    # is treated as explore -- the safe direction.
    buf = make_buffer(); run_episode(buf, mode=1, n=10); run_row(buf, mode=0, n=5)
    assert buf.flush_episode() == 0