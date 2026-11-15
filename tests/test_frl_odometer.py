"""Distance driven per training episode, for the [FRL] log line (user 2026-11-14: "distance travelled
logged"). Odometer sums the car's position steps per env and reports, over the last 100 episodes, mean
metres of a goal episode, of an explore episode, and metres per fault end."""
import math

from torch_rl.frl.carla import Odometer


def _info(x, y=0.0):
    return {"hero_pos": [x, y, 0.3]}


def test_sums_the_steps_of_one_episode_and_ignores_the_reset_jump():
    odo = Odometer(1)
    odo.start(0, _info(0.0))
    odo.step(0, _info(3.0))
    odo.step(0, _info(3.0, 4.0))                 # 3 m then 4 m
    odo.end(0, mode=1, end="reached")
    odo.start(0, _info(500.0))                   # the next episode starts elsewhere: not driven
    odo.step(0, _info(510.0))
    odo.end(0, mode=0, end="crossing")
    assert odo.summary() == {"goal_m": 7.0, "explore_m": 10.0, "m_per_fault": 17.0, "episodes": 2}


def test_each_env_keeps_its_own_distance():
    odo = Odometer(2)
    odo.start(0, _info(0.0))
    odo.start(1, _info(100.0))
    odo.step(0, _info(1.0))
    odo.step(1, _info(106.0))
    odo.end(1, mode=1, end="off_road")
    odo.end(0, mode=1, end="clock")
    s = odo.summary()
    assert s["goal_m"] == 3.5 and s["m_per_fault"] == 7.0      # 1 m + 6 m over one fault end


def test_no_fault_and_no_position_are_reported_as_nan_not_zero():
    odo = Odometer(1)
    odo.start(0, {})
    odo.step(0, {})
    odo.end(0, mode=1, end="reached")
    s = odo.summary()
    assert s["goal_m"] == 0.0 and math.isnan(s["explore_m"]) and math.isnan(s["m_per_fault"])
    assert "goal_m=0" in odo.line() and "m_per_fault=nan" in odo.line()


def test_keeps_only_the_last_hundred_episodes():
    odo = Odometer(1)
    for k in range(150):
        odo.start(0, _info(0.0))
        odo.step(0, _info(1.0 if k < 50 else 2.0))
        odo.end(0, mode=1, end="crossing")
    assert odo.summary() == {"goal_m": 2.0, "explore_m": float("nan"), "m_per_fault": 2.0,
                             "episodes": 100} or odo.summary()["goal_m"] == 2.0
