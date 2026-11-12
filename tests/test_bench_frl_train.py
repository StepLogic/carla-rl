"""scripts/bench/frl_train.py (plan 2026-11-10, Task 3): learner C's sim loop, a port of
fastrlap-mujoco-env/train_online_pixels.py. Pins the replay buffer and that the two configs
carry the reference files' values (fastrlap-ros/offroad_learning/config/{redq,drq}_config.py)."""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts", "bench"))

from frl_train import CONFIGS, ReplayBuffer, build_parser, make_env, mask_for  # noqa: E402


def test_the_device_defaults_to_cpu_and_takes_cuda():
    """User 2026-11-10 "run any NN on gpu - onlocal"."""
    base = ["--env", "Pendulum-v1", "--config", "redq", "--env-steps", "10"]
    assert build_parser().parse_args(base).device == "cpu"
    assert build_parser().parse_args(base + ["--device", "cuda"]).device == "cuda"


def test_configs_carry_the_reference_values():
    r = CONFIGS["redq"]
    assert (r["learner"], r["actor_lr"], r["critic_lr"], r["temp_lr"]) == ("sac", 3e-4, 3e-4, 3e-4)
    assert (r["num_qs"], r["num_min_qs"], r["critic_layer_norm"], r["backup_entropy"]) == (10, 2, True, False)
    assert (r["init_temperature"], r["discount"], r["tau"], r["hidden_dims"]) == (0.1, 0.99, 0.005, (256, 256))
    d = CONFIGS["drq"]
    assert (d["learner"], d["temp_lr"], d["init_temperature"], d["latent_dim"]) == ("drq", 1e-4, 0.5, 50)
    assert d["encoder_kwargs"] == {"features": (32, 32, 32, 32), "filters": (3, 3, 3, 3),
                                   "strides": (2, 2, 2, 2)}
    assert (d["num_qs"], d["num_min_qs"], d["critic_layer_norm"], d["backup_entropy"]) == (10, 2, True, False)
    # target entropy left to the learner's default (-action_dim / 2): the configs' -3.0 / -2.0
    # are tuned for FastRLAP's 2-D car (ruling in the plan)
    assert "target_entropy" not in r and "target_entropy" not in d


def test_the_learner_acts_in_unit_box_and_the_env_gets_its_own_range():
    """The tanh actor outputs [-1, 1]; Pendulum's torque is [-2, 2]. Clipping alone (the
    reference loop, whose envs are already [-1, 1]) would halve the usable torque."""
    pytest = __import__("pytest")
    gym = pytest.importorskip("gymnasium")
    env = make_env("Pendulum-v1")
    assert env.action_space.low.tolist() == [-1.0] and env.action_space.high.tolist() == [1.0]
    inner = env
    while not hasattr(inner, "max_torque"):
        inner = inner.env
    seen = []
    orig = inner.step
    inner.step = lambda u: (seen.append(float(np.asarray(u).reshape(-1)[0])), orig(u))[1]
    env.reset(seed=0)
    env.step(np.array([1.0], np.float32))
    assert abs(seen[-1] - 2.0) < 1e-6                          # full torque reachable
    assert gym is not None


def test_mask_is_zero_only_on_a_true_termination():
    assert mask_for(terminated=True, truncated=False) == 0.0
    assert mask_for(terminated=False, truncated=True) == 1.0    # a time limit bootstraps
    assert mask_for(terminated=False, truncated=False) == 1.0


def test_replay_wraps_and_samples_states():
    rb = ReplayBuffer(obs_shape=(3,), obs_dtype=np.float32, action_dim=1, capacity=5, pixels=False)
    for i in range(7):
        rb.insert(np.full(3, i, np.float32), np.array([0.1 * i]), float(i), 1.0, np.full(3, i + 1, np.float32))
    assert len(rb) == 5
    b = rb.sample(4, np.random.default_rng(0))
    assert b["observations"].shape == (4, 3) and b["observations"].dtype == np.float32
    assert set(b["rewards"].tolist()) <= {2.0, 3.0, 4.0, 5.0, 6.0}        # 0 and 1 overwritten
    assert np.allclose(b["next_observations"], b["observations"] + 1)


def test_replay_keeps_pixels_uint8_in_a_dict():
    rb = ReplayBuffer(obs_shape=(9, 8, 8), obs_dtype=np.uint8, action_dim=1, capacity=4, pixels=True)
    rb.insert(np.full((9, 8, 8), 7, np.uint8), np.zeros(1), 0.0, 1.0, np.full((9, 8, 8), 8, np.uint8))
    b = rb.sample(2, np.random.default_rng(0))
    assert b["observations"]["pixels"].dtype == np.uint8
    assert b["observations"]["pixels"].shape == (2, 9, 8, 8)
    assert int(b["next_observations"]["pixels"][0, 0, 0, 0]) == 8
