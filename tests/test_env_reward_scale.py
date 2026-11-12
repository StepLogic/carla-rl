"""REWARD_SCALE in the ENVIRONMENT (user 2026-11-08: "remove value clips", "remove actor
normalization - i want implementation to plain SAC", "multiply reward in environment",
"no hacks or normalisations that were not present in plain SAC or REDQ or DrQ").

The env returns reward x k (info["raw_reward"] keeps the unscaled one). The two reward
sources that never pass through env.step -- HER's relabelled rows and the preloaded
demonstrations -- are scaled by the same k where they enter the replay, so every row
the critic sees is on one scale. The learner itself scales nothing, and the critic
target's value clips can be switched off (TQC_VALUE_* / EXPLORE_VALUE_* = none).
"""
import dataclasses
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))
torch = pytest.importorskip("torch")
gym = pytest.importorskip("gymnasium")


def _src(*parts):
    return open(os.path.join(ROOT, *parts)).read()


class _ConstEnv(gym.Env):
    observation_space = gym.spaces.Box(-1.0, 1.0, (1,), np.float32)
    action_space = gym.spaces.Box(-1.0, 1.0, (2,), np.float32)

    def reset(self, *, seed=None, options=None):
        return np.zeros(1, np.float32), {}

    def step(self, action):
        return np.zeros(1, np.float32), 0.3, False, False, {"tag": 7}


def test_the_env_reward_is_scaled_and_the_raw_one_kept():
    from torch_rl.reward_scale import ScaledReward
    env = ScaledReward(_ConstEnv(), 10.0)
    env.reset()
    _, r, term, trunc, info = env.step(np.zeros(2, np.float32))
    assert r == pytest.approx(3.0)
    assert info["raw_reward"] == pytest.approx(0.3) and info["tag"] == 7
    assert term is False and trunc is False


@pytest.mark.parametrize("value,expected", [(None, 1.0), ("10", 10.0), ("2.5", 2.5)])
def test_the_scale_comes_from_the_environment_variable(monkeypatch, value, expected):
    from torch_rl.reward_scale import reward_scale
    if value is None:
        monkeypatch.delenv("REWARD_SCALE", raising=False)
    else:
        monkeypatch.setenv("REWARD_SCALE", value)
    assert reward_scale() == expected


@pytest.mark.parametrize("value", ["0", "-1"])
def test_a_scale_that_is_not_positive_is_refused(monkeypatch, value):
    from torch_rl.reward_scale import reward_scale
    monkeypatch.setenv("REWARD_SCALE", value)
    with pytest.raises(ValueError):
        reward_scale()


def test_build_env_wraps_the_env_with_the_run_scale():
    src = _src("torch_rl", "carla_env_build.py")
    assert "ScaledReward(" in src and "reward_scale()" in src


def test_hindsight_rows_are_scaled_like_the_env():
    from tests.test_image_her import _drive, _her, _rewards_of
    speeds = [5.0] * 40
    buf1, her1 = _her()
    _drive(her1, speeds, end_reached_at=39)
    bufk, herk = _her(reward_scale=10.0)
    _drive(herk, speeds, end_reached_at=39)
    r1, rk = _rewards_of(buf1), _rewards_of(bufk)
    assert len(r1) > 0 and len(r1) == len(rk)
    assert np.allclose(rk, 10.0 * np.asarray(r1), rtol=1e-6, atol=1e-6)


def test_the_loop_gives_her_the_raw_reward():
    assert 'inf.get("raw_reward"' in _src("torch_rl", "parallel_loop.py")


def test_preloaded_demo_rewards_are_scaled(tmp_path):
    from tests.test_expert_data import _buf, _ep
    from torch_rl.expert_data import fill_buffer, save_episode
    p = str(tmp_path / "e.npz")
    save_episode(p, _ep(T=6))
    plain, scaled = _buf(), _buf()
    fill_buffer(plain, [p], which="exec")
    fill_buffer(scaled, [p], which="exec", reward_scale=10.0)
    assert torch.allclose(scaled.rewards[:scaled.size], 10.0 * plain.rewards[:plain.size])


def test_the_preload_and_her_use_the_run_scale():
    assert "reward_scale=reward_scale()" in _src("torch_rl", "bc_trainer.py")
    assert "reward_scale=reward_scale()" in _src("train_carla_sac.py")


def test_the_learner_scales_nothing():
    from torch_rl.gated_recurrent_tqc import GatedRecurrentTQCConfig
    assert "reward_scale" not in {f.name for f in dataclasses.fields(GatedRecurrentTQCConfig)}


def test_the_value_clips_can_be_switched_off():
    src = _src("train_carla_sac.py")
    for name in ("TQC_VALUE_FLOOR", "TQC_VALUE_CEILING", "EXPLORE_VALUE_FLOOR",
                 "EXPLORE_VALUE_CEILING"):
        assert f'env_opt_float("{name}"' in src


def test_an_update_without_value_clips_runs_unclipped():
    from tests.test_one_critic import _cfg, _chunk
    from torch_rl.gated_recurrent_sac import GatedRecurrentSAC
    clipped = GatedRecurrentSAC(_cfg(value_floor=-100.0, value_ceiling=150.0))
    free = GatedRecurrentSAC(_cfg(value_floor=None, value_ceiling=None))
    ch = _chunk(clipped.cfg, [1, 1])
    ch["rewards"] = ch["rewards"] + 1.0e4               # far above the old ceiling
    assert float(clipped.update(dict(ch))["target_at_ceiling"]) > 0.0
    m = free.update(dict(ch))
    assert float(m["target_at_ceiling"]) == 0.0 and float(m["target_at_floor"]) == 0.0
    assert float(m["q_g"]) == float(m["q_g"])           # finite, not NaN


def test_the_actor_q_normalisation_can_be_switched_off():
    assert 'os.environ.get("ACTOR_Q_NORM", "1") == "1"' in _src("train_carla_sac.py")
    from tests.test_one_critic import _cfg, _chunk
    from torch_rl.gated_recurrent_sac import GatedRecurrentSAC
    agent = GatedRecurrentSAC(_cfg(actor_q_norm=False))
    agent.update(_chunk(agent.cfg, [1, 0]))
    assert float(agent.q_scale_g) == 1.0 and float(agent.q_scale_init_g) == 0.0
