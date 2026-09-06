"""The v1 coverage reward, term for term against the Habitat peer's spec.

    r = 1.0 * [first-time cell] - 0.01 - 0.1 * [contact]

Every constant here is theirs. The tests exist so a later "tidy-up" cannot
drift them, because the RATIO k_step/k_cell = 0.01 is what their tuning
constrains and it is invisible in any single number.
"""
import gymnasium as gym
import numpy as np
import pytest

from vae_nav.explore_reward import ExploreReward

CELL = 0.5


class FakeEnv(gym.Env):
    """Replays a scripted path; publishes what core_state publishes."""

    def __init__(self, path, collisions=None, throttle=0.5):
        self.path = [np.asarray(p, np.float64) for p in path]
        self.collisions = collisions or []
        self.throttle = throttle
        self.t = -1
        box = gym.spaces.Box(-1.0, 1.0, (2,), np.float32)
        self.observation_space = box
        self.action_space = box

    def _info(self):
        i = min(self.t, len(self.path) - 1)
        return {"hero_pos": self.path[i],
                "hero_yaw": 0.0,
                "collision": self.t in self.collisions,
                "applied_throttle": self.throttle}

    def reset(self, **kw):
        self.t = 0
        return np.zeros(2, np.float32), self._info()

    def step(self, action):
        self.t += 1
        return np.zeros(2, np.float32), 0.0, False, False, self._info()


def drive(path, **kw):
    env = ExploreReward(FakeEnv(path, **kw), max_steps=10_000)
    env.reset()
    out = []
    while True:
        _o, r, term, trunc, info = env.step(np.zeros(2, np.float32))
        out.append((r, term, trunc, info))
        if term or trunc or len(out) >= len(path):
            break
    return env, out


def straight(n, dx=CELL):
    """One new cell per step."""
    return [(i * dx, 0.0) for i in range(n + 1)]


def test_entering_a_new_cell_pays_one_minus_the_step_cost():
    _env, out = drive(straight(3))
    assert out[0][0] == pytest.approx(1.0 - 0.01)


def test_standing_still_pays_only_the_step_cost():
    """Coverage is paid on TRANSITIONS, never on presence, or the agent dwells."""
    _env, out = drive([(0.0, 0.0)] * 4)
    for r, *_ in out:
        assert r == pytest.approx(-0.01)


def test_revisiting_a_cell_pays_nothing_extra():
    """One-shot per cell per episode: no decay, no count."""
    path = [(0.0, 0.0), (CELL, 0.0), (0.0, 0.0), (CELL, 0.0)]
    _env, out = drive(path)
    rewards = [r for r, *_ in out]
    assert rewards[0] == pytest.approx(1.0 - 0.01), "first entry pays"
    assert rewards[1] == pytest.approx(-0.01), "back to the start cell, already visited"
    assert rewards[2] == pytest.approx(-0.01), "second visit to cell 1 pays nothing"


def test_starting_cell_counts_as_visited():
    """Otherwise step one collects a cell bonus for not moving."""
    _env, out = drive([(0.0, 0.0), (0.1, 0.0)])
    assert out[0][0] == pytest.approx(-0.01)


def test_collision_costs_a_tenth_and_does_not_end_the_episode():
    _env, out = drive([(0.0, 0.0)] * 3, collisions=[1])
    assert out[0][0] == pytest.approx(-0.01 - 0.1)
    assert out[0][1] is False, "collisions are non-terminal in v1"
    assert out[0][2] is False


def test_new_cell_and_collision_compose():
    _env, out = drive(straight(2), collisions=[1])
    assert out[0][0] == pytest.approx(1.0 - 0.01 - 0.1)


def test_terminated_is_always_false():
    """There is no goal in explore mode, so nothing can be arrived at."""
    _env, out = drive(straight(6), collisions=[2, 3])
    assert all(term is False for _r, term, _tr, _i in out)


def test_step_cap_truncates():
    env = ExploreReward(FakeEnv(straight(50)), max_steps=5)
    env.reset()
    for i in range(5):
        _o, _r, term, trunc, _info = env.step(np.zeros(2, np.float32))
    assert trunc is True and term is False


def test_the_ratio_their_tuning_constrains():
    """k_step must be 1% of k_cell. The absolute values matter less."""
    from vae_nav.explore_reward import K_CELL, K_COLLISION, K_STEP
    assert K_STEP / K_CELL == pytest.approx(0.01)
    assert K_COLLISION / K_CELL == pytest.approx(0.1)


def test_v1_has_no_seen_bonus():
    """Any seen bonus above one entered cell per step makes the actor spin.

    v1 has none, and K_SEEN is absent rather than present-and-zero so it cannot
    be switched on without reading why.
    """
    import vae_nav.explore_reward as m
    assert not hasattr(m, "K_SEEN")


def test_wedge_truncates_with_no_extra_penalty():
    """Their wedge detector: no progress while asking for forward throttle."""
    env = ExploreReward(FakeEnv([(0.0, 0.0)] * 200, throttle=0.5),
                        max_steps=10_000)
    env.reset()
    last = None
    for _ in range(60):
        _o, r, term, trunc, _i = env.step(np.zeros(2, np.float32))
        last = (r, term, trunc)
        if trunc:
            break
    assert last[2] is True, "a stuck vehicle under throttle must truncate"
    assert last[1] is False
    assert last[0] == pytest.approx(-0.01), "wedge carries NO extra penalty in v1"


def test_idling_without_throttle_is_not_a_wedge():
    """Not trying to move is idling; the wedge is specifically stuck-under-power."""
    env = ExploreReward(FakeEnv([(0.0, 0.0)] * 200, throttle=0.0),
                        max_steps=10_000)
    env.reset()
    for _ in range(80):
        _o, _r, _term, trunc, _i = env.step(np.zeros(2, np.float32))
        assert trunc is False


def test_moving_vehicle_is_never_wedged():
    env = ExploreReward(FakeEnv(straight(200), throttle=0.5), max_steps=10_000)
    env.reset()
    for _ in range(100):
        _o, _r, _term, trunc, _i = env.step(np.zeros(2, np.float32))
        assert trunc is False


def test_visited_set_resets_every_episode():
    """v1 resets at every episode start; it does not persist across episodes."""
    env = ExploreReward(FakeEnv(straight(5)), max_steps=10_000)
    env.reset()
    for _ in range(3):
        env.step(np.zeros(2, np.float32))
    assert len(env.visited) > 1
    env.reset()
    assert len(env.visited) == 1, "only the new starting cell"


def test_coverage_rate_is_reported_for_the_port_check():
    """A car covers cells near-linearly; their agent could spread laterally.

    The rate is what tells us whether the 1% step cost sits where they tuned it:
    at one new cell per step the step cost is 1% of the per-step reward, exactly
    where they tuned it; at one new cell every other step it is 2%.
    """
    env = ExploreReward(FakeEnv(straight(40)), max_steps=10_000)
    env.reset()
    for _ in range(20):
        env.step(np.zeros(2, np.float32))
    s = env.stats()
    assert s["steps"] == 20
    assert s["new_cells"] == 20, "0.5 m per step on a 0.5 m grid is one per step"
    assert s["new_cell_rate"] == pytest.approx(1.0)
    assert s["visited"] == 21, "20 entered plus the starting cell"


def test_coverage_rate_halves_when_steps_are_half_a_cell():
    """The CARLA case: 0.283 m/step on a 0.5 m grid is ~0.57 cells per step.

    Their tuning assumed a rate; ours differs, and this is the number that says
    by how much.
    """
    env = ExploreReward(FakeEnv(straight(40, dx=0.25)), max_steps=10_000)
    env.reset()
    for _ in range(20):
        env.step(np.zeros(2, np.float32))
    assert env.stats()["new_cell_rate"] == pytest.approx(0.5, abs=0.05)
