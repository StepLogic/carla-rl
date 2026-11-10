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

from gated_sac_bench import (C1_LOOP, apply_overrides, augment_chunk,  # noqa: E402
                             bench_config, bench_row, loop_settings)


def test_overrides_set_existing_keys_with_json_values_and_refuse_unknown_ones():
    cfg = bench_config(C1, obs_dim=3, act_dim=1, seed=1)
    out = apply_overrides(cfg, ["critic_proj_dim=50", "critic_layer_norm=false", "lr=0.001"])
    assert (out["critic_proj_dim"], out["critic_layer_norm"], out["lr"]) == (50, False, 0.001)
    assert cfg["critic_proj_dim"] == 0                          # the input config is not mutated
    with pytest.raises(KeyError):
        apply_overrides(cfg, ["no_such_key=1"])
from torch_rl import pixel_rows as pxr  # noqa: E402

C1 = {"action_dim": 2, "state_dim": 3138, "goal_cols": list(range(3138)),
      "core_state_cols": list(range(1, 3138)), "explore_cols": [], "actor_state_cols": None,
      "device": "cuda", "seed": 0, "target_entropy": -2.0, "init_temperature": 0.1,
      "n_critics": 10, "subset_size": 2, "critic_layer_norm": True, "one_critic": True,
      "mode_index": 0, "hidden": 512, "lr": 0.0003, "value_ceiling": None, "burn_in": 8,
      "critic_proj_dim": 0}


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


def test_the_budget_matched_cadence_changes_only_the_start_and_the_update_rate():
    """T2-B failed under C1's cadence (2026-11-10). The planned control: B with the update
    budget of learner C -- learning from step 1000, an update every step -- everything else C1."""
    assert loop_settings() == C1_LOOP
    m = loop_settings(learning_starts=1000, update_every=1)
    assert (m["learning_starts"], m["update_every"]) == (1000, 1)
    assert {k: v for k, v in m.items() if k not in ("learning_starts", "update_every")} == \
           {k: v for k, v in C1_LOOP.items() if k not in ("learning_starts", "update_every")}
    with pytest.raises(ValueError):
        loop_settings(update_every=0)


def test_augment_chunk_shifts_every_frame_of_a_chunk_alike_and_leaves_the_state_alone():
    """Bisection of T2-B (2026-11-10): DrQ's random crop for the recurrent learner, ONE shift per
    chunk (per-frame shifts would scramble the motion the GRU reads across frames)."""
    torch = pytest.importorskip("torch")
    n, L, P = 3, 4, pxr.MONO_PIX
    obs = torch.zeros(n, L, P + 5)
    img = torch.rand(n, 1, 3, pxr.MONO_IMG, pxr.MONO_IMG).expand(n, L, 3, pxr.MONO_IMG, pxr.MONO_IMG)
    obs[..., :P] = img.reshape(n, L, P)
    obs[..., P:] = torch.arange(5, dtype=torch.float32)
    same = augment_chunk(obs, pad=6, offsets=torch.tensor([[6, 6]] * n))
    assert torch.equal(same, obs)                              # offset = pad is the identity
    out = augment_chunk(obs, pad=6, generator=torch.Generator().manual_seed(0))
    assert out.shape == obs.shape
    assert torch.equal(out[..., P:], obs[..., P:])            # the state block is untouched
    for i in range(n):                                        # one shift for the whole chunk
        assert all(torch.equal(out[i, 0, :P], out[i, t, :P]) for t in range(L))
    assert augment_chunk(obs, pad=0) is obs                   # off


def test_rows_refuse_a_wrong_frame_size():
    with pytest.raises(ValueError):
        bench_row(np.zeros((84, 84, 3), np.uint8), np.zeros(3, np.float32), strict=True)
