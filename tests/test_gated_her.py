"""Hindsight relabelling on the gated arm: explore episodes become goal data.

WHY THIS IS WORTH DOING AT ALL. The goal curriculum is pinned to 5-20 m, and
measured goal episodes run ~96 steps while explore episodes run ~370 and wander.
Relabelling explore trajectories supplies goal data at distances the curriculum
cannot produce, which is the only route this arm has to saying anything about
longer-range navigation.

THE THREE THINGS THAT MAKE IT CORRECT, each of which fails SILENTLY if wrong:

  routing   On-policy rows carry their OWN mode. Inserting without one lets the
            mode-split buffer default everything to GOAL, filing every explore
            transition as goal data -- the starvation failure the split buffer
            exists to prevent, inverted.
  mode      Hindsight rows are forced to GOAL. A relabelled explore row that
            kept mode=0 would train the EXPLORE critic on goal-shaped data.
  order     The goal block is rebuilt from the hindsight goal BEFORE the mode is
            forced. Reversed, a row from an explore episode carries mode=1 with
            the zeroed goal block explore mode wrote, and the goal critic trains
            on rows whose goal it cannot see -- which reads as slow learning.

The Habitat implementation this is ported from orders it the same way, inside
its state constructor, and a peer session confirmed that ordering directly.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

from vae_nav.episode_her import EXPLORE, GOAL, EpisodeRelabelBuffer  # noqa: E402

GOAL_DIM = 12


class _RecordingBuffer:
    """Stands in for ModeSplitBuffer, recording how each row was routed."""

    def __init__(self, accepts_mode=True):
        self.rows = []
        self.modes = []
        self.accepts_mode = accepts_mode

    def insert(self, row, mode=None):
        if mode is not None and not self.accepts_mode:
            raise TypeError("insert() got an unexpected keyword 'mode'")
        self.rows.append(row)
        self.modes.append(mode)

    def __len__(self):
        return len(self.rows)


def _goal_block(fields):
    """Stand-in for VAEGoalWrapper.goal_block: offset to the goal, then zeros."""
    hero = np.asarray(fields["hero_pos"], np.float64)
    goal = np.asarray(fields["goal_pos"], np.float64)
    out = np.zeros(GOAL_DIM, np.float32)
    out[:2] = (goal - hero).astype(np.float32)
    return out


def _obs(mode, goal_zeroed):
    """An observation as VAEGoalWrapper would emit it for that mode."""
    return {"goal": (np.zeros(GOAL_DIM, np.float32) if goal_zeroed
                     else np.ones(GOAL_DIM, np.float32)),
            "latent": np.zeros(4, np.float32),
            "vector": np.zeros(4, np.float32),
            "mode": np.array([float(mode)], np.float32)}


def _her(buffer, gated=True, time_to_goal=True, **kw):
    opts = dict(state_fn=_goal_block, goal_radius=2.0, k_goal=600.0,
                k_collision=0.0, n_relabel=1, min_disp=2.0, seed=0,
                time_to_goal=time_to_goal, step_cost=1.0, gated=gated)
    opts.update(kw)
    return EpisodeRelabelBuffer(buffer, **opts)


def _run_episode(her, mode, n=12, step=3.0):
    """Drive a straight line so hindsight goals clear the displacement floor."""
    for i in range(n):
        pos = (float(i) * step, 0.0)
        tr = dict(observations=_obs(mode, goal_zeroed=(mode == EXPLORE)),
                  actions=np.zeros(2, np.float32), rewards=np.float32(-1.0),
                  masks=np.float32(1.0), dones=np.float32(0.0),
                  next_observations=_obs(mode, goal_zeroed=(mode == EXPLORE)))
        her.record(tr, fields={"hero_pos": pos, "hero_yaw": 0.0,
                               "goal_pos": (100.0, 0.0),
                               "action_history": np.zeros((4, 2))},
                   pos=pos, hit=False, mode=mode)
    return her.flush_episode()


# ------------------------------------------------------------- coherence ----

def test_gated_requires_the_minimum_time_reward():
    # Sparse hindsight rows into a minimum-time buffer is two reward functions
    # over one set of states.
    with pytest.raises(ValueError, match="requires time_to_goal"):
        _her(_RecordingBuffer(), gated=True, time_to_goal=False)


def test_ungated_still_works_without_the_minimum_time_reward():
    _her(_RecordingBuffer(), gated=False, time_to_goal=False)


# --------------------------------------------------------------- routing ----

def test_on_policy_explore_rows_are_routed_as_explore():
    # THE INVERTED STARVATION FAILURE. If these defaulted to GOAL, every explore
    # transition would become goal data and the split would collapse.
    buf = _RecordingBuffer()
    her = _her(buf)
    _run_episode(her, EXPLORE, n=4)
    assert her.n_relabelled >= 0
    assert buf.modes[:4] == [EXPLORE] * 4


def test_on_policy_goal_rows_are_routed_as_goal():
    buf = _RecordingBuffer()
    _run_episode(_her(buf), GOAL, n=4)
    assert buf.modes[:4] == [GOAL] * 4


def test_hindsight_rows_from_an_explore_episode_are_routed_as_goal():
    buf = _RecordingBuffer()
    her = _her(buf)
    n_on_policy = 12
    inserted = _run_episode(her, EXPLORE, n=n_on_policy)
    assert inserted > 0, "no hindsight rows were produced"
    assert buf.modes[n_on_policy:] == [GOAL] * inserted


def test_the_ungated_path_never_passes_a_mode():
    # The plain jaxrl2 ReplayBuffer's insert takes no mode argument, so passing
    # one would break every existing non-gated run.
    buf = _RecordingBuffer(accepts_mode=False)
    _run_episode(_her(buf, gated=False, time_to_goal=True), GOAL, n=6)
    assert all(m is None for m in buf.modes)


# ------------------------------------------------ the mode bit and its order ----

def test_hindsight_observations_carry_mode_goal():
    buf = _RecordingBuffer()
    her = _her(buf)
    n_on_policy = 12
    _run_episode(her, EXPLORE, n=n_on_policy)
    for row in buf.rows[n_on_policy:]:
        assert float(row["observations"]["mode"][0]) == float(GOAL)
        assert float(row["next_observations"]["mode"][0]) == float(GOAL)


def test_hindsight_rows_carry_a_FILLED_goal_block_not_the_zeroed_one():
    # THE ORDERING TEST. Explore mode wrote zeros into the goal block. A
    # hindsight row must carry the block rebuilt from the hindsight goal, not
    # those zeros, or the goal critic sees mode=1 with no goal and it looks like
    # slow learning rather than a bug.
    buf = _RecordingBuffer()
    her = _her(buf)
    n_on_policy = 12
    _run_episode(her, EXPLORE, n=n_on_policy)
    hindsight = buf.rows[n_on_policy:]
    assert hindsight, "no hindsight rows"
    for row in hindsight:
        block = np.asarray(row["observations"]["goal"])
        assert np.any(block != 0.0), "goal block is still the explore zeros"


def test_the_on_policy_explore_rows_keep_their_zeroed_goal_block():
    # Relabelling must not reach back and mutate the on-policy rows.
    buf = _RecordingBuffer()
    her = _her(buf)
    _run_episode(her, EXPLORE, n=12)
    for row in buf.rows[:12]:
        assert np.all(np.asarray(row["observations"]["goal"]) == 0.0)


# ---------------------------------------------------------------- reward ----

def test_hindsight_rewards_are_minimum_time():
    # -step_cost on every step, 0 on the step that arrives.
    buf = _RecordingBuffer()
    her = _her(buf)
    n_on_policy = 12
    _run_episode(her, EXPLORE, n=n_on_policy)
    rewards = [float(r["rewards"]) for r in buf.rows[n_on_policy:]]
    assert rewards, "no hindsight rows"
    assert set(rewards) <= {-1.0, 0.0}, sorted(set(rewards))


def test_at_least_one_hindsight_row_arrives():
    buf = _RecordingBuffer()
    her = _her(buf)
    n_on_policy = 12
    _run_episode(her, EXPLORE, n=n_on_policy)
    rewards = [float(r["rewards"]) for r in buf.rows[n_on_policy:]]
    assert 0.0 in rewards, "no hindsight row was credited with arrival"


def test_explore_episodes_supply_goal_data_at_all(capsys):
    # The headline claim: an episode flown entirely in explore mode produces
    # rows the GOAL critic will train on.
    buf = _RecordingBuffer()
    her = _her(buf)
    inserted = _run_episode(her, EXPLORE, n=16)
    goal_rows = [m for m in buf.modes if m == GOAL]
    assert inserted > 0
    assert len(goal_rows) == inserted
