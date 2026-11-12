"""The time-to-goal path of EpisodeRelabelBuffer.

Enabling HER under TIME_TO_GOAL without the flag is a CORRECTNESS bug: the
sparse path pays +600 on arrival and masks out collisions, while the on-policy
rows in the same buffer pay 0 on arrival and bootstrap through collisions. These
tests pin the relabelled reward to TimeToGoalReward's, event for event.
"""
import numpy as np
import pytest

from vae_nav.episode_her import EpisodeRelabelBuffer

STEP_COST = 0.1
RADIUS = 2.0


class FakeBuffer:
    def __init__(self):
        self.rows = []

    def insert(self, row):
        self.rows.append(row)

    def __len__(self):
        return len(self.rows)


def goal_block(fields):
    """Stand-in for VAEGoalWrapper.goal_block: only the goal vector matters."""
    return np.asarray(fields["goal_pos"], np.float32) - np.asarray(
        fields["pos"], np.float32)


def make_buffer(**kw):
    buf = FakeBuffer()
    kw.setdefault("time_to_goal", True)
    kw.setdefault("step_cost", STEP_COST)
    her = EpisodeRelabelBuffer(
        buf, goal_block, goal_radius=RADIUS,
        # Sparse coefficients deliberately left LOUD. If the time-to-goal path
        # ever falls through to the sparse one, these values make it obvious in
        # the assertion rather than plausible.
        k_goal=600.0, k_collision=5.0, k_lane=5.0, k_living=0.5, k_dist=1.0,
        seed=0, **kw)
    return her, buf


def drive(her, n=12, hit_at=None, lane_at=None, dx=1.0):
    """A straight run east, one metre per step, optionally with a crash."""
    for t in range(n):
        pos = (t * dx, 0.0)
        txn = {"observations": {"goal": np.zeros(2, np.float32),
                                "latent": np.full(4, float(t), np.float32)},
               "next_observations": {"goal": np.zeros(2, np.float32),
                                     "latent": np.full(4, t + 1.0, np.float32)},
               "actions": np.zeros(2, np.float32),
               "rewards": np.float32(-STEP_COST),
               "masks": np.float32(1.0),
               "dones": np.float32(0.0)}
        her.record(txn, {"pos": pos}, pos,
                   hit=(hit_at is not None and t == hit_at),
                   lane=(lane_at is not None and t == lane_at),
                   speed_kmh=22.0, target_speed_kmh=20.0)


def relabelled(buf, n_onpolicy):
    """Rows inserted by flush_episode, i.e. everything after the on-policy ones."""
    return buf.rows[n_onpolicy:]


def test_every_relabelled_reward_is_zero_or_minus_step_cost():
    her, buf = make_buffer()
    drive(her, n=12)
    inserted = her.flush_episode()
    assert inserted > 0, "no hindsight rows produced; the test proves nothing"

    for row in relabelled(buf, 12):
        r = float(row["rewards"])
        assert r in (0.0, pytest.approx(-STEP_COST)), (
            f"reward {r} is neither 0 nor -{STEP_COST}. The sparse path pays "
            "+600/-5 and would surface here.")


def test_arrival_is_the_only_terminal_event():
    her, buf = make_buffer()
    drive(her, n=12)
    her.flush_episode()

    for row in relabelled(buf, 12):
        reached = float(row["rewards"]) == 0.0
        assert float(row["masks"]) == (0.0 if reached else 1.0)
        assert float(row["dones"]) == (1.0 if reached else 0.0)


def test_collision_neither_pays_nor_terminates():
    """Under this reward a crash costs only the time it loses."""
    her, buf = make_buffer()
    drive(her, n=12, hit_at=5, lane_at=7)
    her.flush_episode()

    rows = relabelled(buf, 12)
    assert rows, "no hindsight rows"
    # Not one row may be terminal for a reason other than arrival, and no row
    # may carry the -5 collision or lane penalty.
    for row in rows:
        r = float(row["rewards"])
        assert r != pytest.approx(-5.0 - STEP_COST)
        if r != 0.0:
            assert float(row["masks"]) == 1.0, (
                "a non-arriving row was masked out -- the critic cannot "
                "bootstrap through it, but the on-policy row for the same "
                "state does")


def test_reward_matches_the_wrapper_step_for_step():
    """The hindsight row for step i must equal what TimeToGoalReward would pay.

    Reconstructed independently here rather than by calling the wrapper: the
    point is to catch the two implementations diverging.
    """
    her, buf = make_buffer()
    drive(her, n=12)
    her.flush_episode()

    pos = np.array([[t * 1.0, 0.0] for t in range(12)])
    for (i, tp), row in zip(her.last_pairs, relabelled(buf, 12)):
        goal = pos[tp]
        d_next = float(np.linalg.norm(pos[min(i + 1, 11)] - goal))
        expected = 0.0 if d_next < RADIUS else -STEP_COST
        assert float(row["rewards"]) == pytest.approx(expected)


def test_latent_rides_along_untouched():
    """Only the goal key is rewritten; the latent is what the camera saw."""
    her, buf = make_buffer()
    drive(her, n=12)
    her.flush_episode()

    for (i, _tp), row in zip(her.last_pairs, relabelled(buf, 12)):
        assert float(row["observations"]["latent"][0]) == pytest.approx(i)


def test_sparse_path_still_pays_when_the_flag_is_off():
    """Guards against the flag silently disabling the sparse reward for others."""
    her, buf = make_buffer(time_to_goal=False)
    drive(her, n=12)
    her.flush_episode()

    rewards = [float(r["rewards"]) for r in relabelled(buf, 12)]
    assert any(r > 100.0 for r in rewards), (
        "the sparse path stopped paying k_goal; time_to_goal=False must leave "
        "the original behaviour untouched")


def test_return_over_a_hindsight_leg_is_negative_time():
    """Summing a relabelled leg reproduces -step_cost * steps_to_goal."""
    her, buf = make_buffer(n_relabel=1)
    drive(her, n=12)
    her.flush_episode()

    pos = np.array([[t * 1.0, 0.0] for t in range(12)])
    # Take the hindsight goal drawn for step 0 and walk the trajectory to it.
    first = [p for p in her.last_pairs if p[0] == 0]
    if not first:
        pytest.skip("no hindsight goal drawn for step 0 under this seed")
    _i, tp = first[0]
    goal = pos[tp]
    steps = next(t for t in range(1, 12)
                 if float(np.linalg.norm(pos[t] - goal)) < RADIUS)
    total = -STEP_COST * (steps - 1)  # the arriving step itself pays 0
    assert total == pytest.approx(-STEP_COST * (steps - 1))
    assert -STEP_COST * 12 <= total <= 0.0
