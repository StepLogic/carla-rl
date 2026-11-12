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

    def insert(self, row, mode=None, segment=None):
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

# -- Task 17: the relabel stream is cut at the first dirty event --------------

def _record_step(buf, i, x, lane=False, hit=False, off_road=False, mode=None):
    """One step carrying its own index in the observations, so a test can read
    which staged step an inserted row came from (_clean keeps the key)."""
    t = _tr(i)
    t["observations"]["idx"] = np.array([float(i)], np.float32)
    buf.record(t, fields={"hero_pos": (x, 0.0), "hero_yaw": 0.0,
                          "goal_pos": (1e6, 0.0),
                          "action_history": np.zeros((4, 2))},
               pos=(x, 0.0), hit=bool(hit), lane=bool(lane),
               off_road=bool(off_road), mode=mode)


def _dirty_episode(buf, xs, lane=(), hit=(), off_road=(), mode=None):
    for i, x in enumerate(xs):
        _record_step(buf, i, x, lane=i in lane, hit=i in hit,
                     off_road=i in off_road, mode=mode)


def _hindsight_rows(buf):
    # On-policy rows carry the flown goal block (99.0) and no idx key; the
    # relabelled rows were rebuilt for the hindsight goal and keep idx.
    return [r for r in buf.rows if "idx" in r["observations"]
            and float(r["observations"]["goal"][0]) != 99.0]


def _assert_rows_equal(a, b):
    assert len(a) == len(b)
    for ra, rb in zip(a, b):
        assert float(ra["rewards"]) == float(rb["rewards"])
        assert float(ra["dones"]) == float(rb["dones"])
        assert float(ra["masks"]) == float(rb["masks"])
        assert np.array_equal(ra["actions"], rb["actions"])
        for side in ("observations", "next_observations"):
            assert set(ra[side]) == set(rb[side])
            for k in ra[side]:
                assert np.array_equal(ra[side][k], rb[side][k]), (side, k)


def test_clean_episode_is_byte_identical_to_the_old_call():
    # The old call site passed no off_road at all; the new default (False)
    # and the cut machinery must leave a clean episode's flush untouched.
    xs = [0.0, 10.0, 20.0, 30.0, 40.0]
    old = _buf()
    for i, x in enumerate(xs):        # old-style record: no off_road kwarg
        _record_step(old, i, x)
    new = _buf()
    _dirty_episode(new, xs, off_road=())   # every step explicitly off_road=False
    n_old, n_new = old.flush_episode(), new.flush_episode()
    assert n_old == n_new > 0
    _assert_rows_equal(old.rows, new.rows)
    assert old.stats() == new.stats()
    assert new.stats()["n_dirty_prefix_eps"] == 0


def test_lane_at_step_k_confines_rows_and_goals_to_the_prefix():
    xs = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0]
    buf = _buf()
    _dirty_episode(buf, xs, lane={3})
    inserted = buf.flush_episode()
    assert inserted > 0, "a dirty episode's clean prefix must still contribute"
    rows = _hindsight_rows(buf)
    assert rows
    # no inserted row came from, or was aimed at, index 3 or later
    assert max(int(r["observations"]["idx"][0]) for r in rows) < 3
    assert max(float(r["observations"]["goal"][0]) for r in rows) < xs[3]
    assert buf.stats()["n_dirty_prefix_eps"] == 1


@pytest.mark.parametrize("flag", ["hit", "off_road"])
def test_collision_and_offroad_each_cut_the_stream(flag):
    xs = [0.0, 10.0, 20.0, 30.0, 40.0]
    buf = _buf()
    _dirty_episode(buf, xs, **{flag: {2}})
    assert buf.flush_episode() > 0
    rows = _hindsight_rows(buf)
    assert rows
    assert max(int(r["observations"]["idx"][0]) for r in rows) < 2
    assert max(float(r["observations"]["goal"][0]) for r in rows) < xs[2]
    assert buf.stats()["n_dirty_prefix_eps"] == 1


def test_first_of_three_dirty_events_is_the_cut():
    # lane at 2, collision at 4, off-road at 5: the cut sits at 2. If the code
    # cut at a later event instead, rows sourced from index 2 would appear.
    xs = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0]
    buf = _buf()
    _dirty_episode(buf, xs, lane={2}, hit={4}, off_road={5})
    assert buf.flush_episode() > 0
    rows = _hindsight_rows(buf)
    assert rows
    assert max(int(r["observations"]["idx"][0]) for r in rows) < 2
    assert max(float(r["observations"]["goal"][0]) for r in rows) < xs[2]
    assert buf.stats()["n_dirty_prefix_eps"] == 1


def test_dirty_at_step_0_or_1_contributes_nothing_and_is_counted():
    for first_dirty in (0, 1):
        buf = _buf()
        _dirty_episode(buf, [0.0, 10.0, 20.0, 30.0], lane={first_dirty})
        before = len(buf.rows)
        assert buf.flush_episode() == 0
        assert len(buf.rows) == before   # on-policy rows were already in
        assert buf.stats()["n_dirty_prefix_eps"] == 1
        assert buf.stats()["n_explore_eps"] == 0


def test_no_dirty_leaves_the_dirty_counter_untouched():
    buf = _buf()
    _episode(buf, [0.0, 10.0, 20.0, 30.0, 40.0])
    buf.flush_episode()
    assert buf.stats()["n_dirty_prefix_eps"] == 0


def test_explore_skip_still_wins_over_the_cut():
    # An explore episode that also went off-road is counted as explore, not
    # dirty: the mode skip runs first (the safe direction).
    buf = _buf()
    _dirty_episode(buf, [0.0, 10.0, 20.0, 30.0], mode=0, off_road={2})
    before = len(buf.rows)
    assert buf.flush_episode() == 0
    assert len(buf.rows) == before
    stats = buf.stats()
    assert stats["n_explore_eps"] == 1
    assert stats["n_dirty_prefix_eps"] == 0


def test_onpolicy_rows_are_unaffected_by_dirtiness():
    # On-policy rows are inserted at record() time and the flush must not
    # touch them, however dirty the episode was.
    xs = [0.0, 10.0, 20.0, 30.0, 40.0]
    buf = _buf()
    for i, x in enumerate(xs):
        _record_step(buf, i, x, lane=(i == 3))
        buf.rows[-1]["rewards"] = np.float32(-1.75 if i == 2 else -0.25)
    flown = [(np.array(r["observations"]["idx"]), float(r["rewards"]),
              float(r["dones"]), float(r["masks"])) for r in buf.rows]
    assert len(flown) == 5           # one per record, nothing from flush yet
    inserted = buf.flush_episode()
    assert inserted > 0
    assert len(buf.rows) == 5 + inserted
    after = [(np.array(r["observations"]["idx"]), float(r["rewards"]),
              float(r["dones"]), float(r["masks"])) for r in buf.rows[:5]]
    for (i1, rw1, d1, m1), (i2, rw2, d2, m2) in zip(flown, after):
        assert np.array_equal(i1, i2) and rw1 == rw2 and d1 == d2 and m1 == m2
    # the lane step's ON-POLICY row is exactly as flown: not terminal
    assert float(buf.rows[3]["dones"]) == 0.0
    assert float(buf.rows[3]["masks"]) == 1.0


def test_per_step_lists_stay_aligned_when_max_steps_trims():
    # Five steps into a 3-step window: the staged episode holds original
    # indices 2,3,4. _ep_offroad must be trimmed in lockstep with the base's
    # lists or the dirty step would shift under the cut.
    buf = _buf(max_steps=3)
    _dirty_episode(buf, [0.0, 10.0, 20.0, 30.0, 40.0], off_road={3})
    assert len(buf._txns) == 3
    assert buf._ep_offroad == [False, True, False]
    assert [int(m) for m in buf._ep_modes] == [1, 1, 1]
    # the dirty step survived the trim at staged index 1: the cut sees it and
    # the remaining prefix [0, 1) has no goal candidate -> nothing inserted.
    assert buf.flush_episode() == 0
    assert buf.stats()["n_dirty_prefix_eps"] == 1


def test_trainer_call_shape_round_trips_the_plain_base_buffer():
    # THE TRAINER'S HER RECORD CALL IS SHARED BETWEEN BUFFER CLASSES: `hers`
    # holds an ImageGoalRelabelBuffer only when IMAGE_GOAL_TASK=1 and a plain
    # EpisodeRelabelBuffer otherwise. Since HER_RECOST the base record() takes
    # `off_road` as well (the counting regime bills it), so the trainer passes
    # the fault flags -- lane from lane_any/lane_invasion_any and off_road --
    # UNCONDITIONALLY to both classes. This test mirrors the trainer's exact
    # kwarg construction (train_carla_tqc.py, HER record call) against both
    # real signatures.
    from vae_nav.episode_her import EpisodeRelabelBuffer

    def record_like_trainer(buf, off_road):
        kwargs = dict(speed_kmh=10.0, target_speed_kmh=20.0, mode=1,
                      off_road=off_road,
                      lane=False)   # the env's lane_any key, as the trainer reads it
        buf.record(_tr(0),
                   fields={"hero_pos": (0.0, 0.0), "hero_yaw": 0.0,
                           "goal_pos": (1e6, 0.0),
                           "action_history": np.zeros((4, 2))},
                   pos=(0.0, 0.0), hit=False, **kwargs)

    # The plain base buffer is called WITH the fault flags and records fine.
    base = EpisodeRelabelBuffer(_StubBuffer(), _goal_block, goal_radius=2.0,
                                k_goal=0.0, k_collision=0.0)
    record_like_trainer(base, off_road=True)      # must not raise
    assert len(base.buffer) == 1
    assert base._offroad == [True]

    # The image-goal subclass receives the kwarg and stages it.
    img = _buf()
    record_like_trainer(img, off_road=True)
    assert img._ep_offroad == [True]
