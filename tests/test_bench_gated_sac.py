"""scripts/bench/gated_sac_bench.py (plan 2026-11-10, Task 4): learner B -- our GatedRecurrentSAC
exactly as the C1 runs configured it -- on the benchmark tasks. The adapter is glue only: rows in
the C1 layout [frame/255 CHW | mode | speed | state], the C1 config.json with only the input sizes
(and device / seed / the -action_dim entropy rule) changed, and C1's loop cadence. These tests pin
the glue; the learning code is the real class."""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "bench"))
os.environ.setdefault("MONO_CHW", "1")
os.environ.setdefault("SPEED_OBS", "1")

from gated_sac_bench import C1_LOOP, bench_config, bench_row  # noqa: E402
from torch_rl import pixel_rows as pxr  # noqa: E402

C1 = {"action_dim": 2, "state_dim": 3138, "goal_cols": list(range(3138)),
      "core_state_cols": list(range(1, 3138)), "explore_cols": [], "actor_state_cols": None,
      "device": "cuda", "seed": 0, "target_entropy": -2.0, "init_temperature": 0.1,
      "n_critics": 10, "subset_size": 2, "critic_layer_norm": True, "one_critic": True,
      "mode_index": 0, "hidden": 512, "lr": 0.0003, "value_ceiling": None, "burn_in": 8}


def test_config_keeps_every_c1_setting_but_the_input_sizes():
    cfg = bench_config(C1, obs_dim=3, act_dim=1, seed=1)
    assert cfg["action_dim"] == 1
    assert cfg["state_dim"] == 1 + 1 + 3                     # mode + speed + the task state
    assert cfg["goal_cols"] == list(range(5))                 # the critic reads the whole block
    assert cfg["core_state_cols"] == [1, 2, 3, 4]             # the GRU: speed + task state
    assert cfg["device"] == "cpu" and cfg["seed"] == 1
    assert cfg["target_entropy"] == -1.0                      # C1's rule is -action_dim (-2 for 2-D)
    changed = {"action_dim", "state_dim", "goal_cols", "core_state_cols", "device", "seed",
               "target_entropy"}
    assert {k: v for k, v in cfg.items() if k not in changed} == \
           {k: v for k, v in C1.items() if k not in changed}


def test_the_device_is_a_parameter_defaulting_to_cpu():
    """User 2026-11-10 "run any NN on gpu - onlocal": the same config on the local GPU."""
    assert bench_config(C1, obs_dim=3, act_dim=1, seed=1)["device"] == "cpu"
    cuda = bench_config(C1, obs_dim=3, act_dim=1, seed=1, device="cuda")
    assert cuda["device"] == "cuda"
    assert {k: v for k, v in cuda.items() if k != "device"} == \
           {k: v for k, v in bench_config(C1, 3, 1, 1).items() if k != "device"}


def test_rows_use_the_c1_layout_and_the_same_frame_encoding_as_carla_rows():
    frame = np.random.default_rng(0).integers(0, 256, (128, 128, 3), dtype=np.uint8)
    state = np.array([0.5, -1.0, 2.0], np.float32)
    row = bench_row(frame, state)
    assert row.dtype == np.float32 and row.shape == (pxr.MONO_PIX + 2 + 3,)
    carla = pxr.mono_row(frame, 1, np.zeros(pxr.BINOC_DIM, np.float32), speed_kmh=0.0)
    assert np.array_equal(row[:pxr.MONO_PIX + 2], carla[:pxr.MONO_PIX + 2])   # frame, mode, speed
    assert row[pxr.MONO_PIX] == 1.0 and row[pxr.MONO_PIX + 1] == 0.0
    assert np.array_equal(row[pxr.MONO_PIX + 2:], state)


def test_the_loop_cadence_is_c1s():
    assert C1_LOOP == {"learning_starts": 5000, "update_every": 5, "updates_per": 1,
                       "chunk_b": 8, "chunk_t": 40, "burn_in": 8}


def test_rows_refuse_a_wrong_frame_size():
    with pytest.raises(ValueError):
        bench_row(np.zeros((84, 84, 3), np.uint8), np.zeros(3, np.float32), strict=True)
