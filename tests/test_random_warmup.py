"""RANDOM WARM-UP on a warm start (user 2026-11-08, after the local plain-SAC
fine-tune of c0 collapsed: "Random warm-up").

Plain SAC fills its replay with uniform-random actions for its first steps (SAC's
and SpinningUp's start steps). A warm start (INIT_FROM) skipped that and let the
loaded clone drive, so every action the critic saw was c0's narrow one and it
came out flat in the action (Q moved < 0.3% for any alternative action). With
RANDOM_WARMUP=1 the first LEARNING_STARTS transitions are uniform-random on a warm
start too. RANDOM_WARMUP=auto (the default) keeps every earlier run's behaviour:
random only without INIT_FROM. The junction curriculum tracks the POLICY's goal
success, so it ignores episodes that random actions drove.
"""
import os
import sys

import numpy as np
import pytest
import torch  # noqa: F401  (the loop's buffer)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from tests.test_parallel_loop import BINOC, FakeAgent, FakeTown, _row  # noqa: E402
from torch_rl.episode_buffer import DeviceEpisodeBuffer  # noqa: E402
from torch_rl.parallel_loop import run_parallel  # noqa: E402
from vae_nav.vec_env import ThreadedVecEnv  # noqa: E402

SRC = open(os.path.join(ROOT, "train_carla_sac.py")).read()


def test_the_setting():
    from train_carla_sac import random_warmup_on
    assert random_warmup_on("", {}) is True                 # from scratch: as always
    assert random_warmup_on("/ckpt", {}) is False           # warm start: as always
    assert random_warmup_on("/ckpt", {"RANDOM_WARMUP": "auto"}) is False
    assert random_warmup_on("/ckpt", {"RANDOM_WARMUP": "1"}) is True
    assert random_warmup_on("", {"RANDOM_WARMUP": "0"}) is False
    with pytest.raises(ValueError, match="RANDOM_WARMUP"):
        random_warmup_on("", {"RANDOM_WARMUP": "yes"})


def test_the_trainer_hands_it_to_the_loop_and_says_so():
    assert "random_warmup=random_warmup_on(INIT_FROM)" in SRC
    assert "random_warmup=not INIT_FROM" not in SRC
    main_src = SRC[SRC.index("\ndef main():"):]
    assert "RANDOM WARM-UP" in main_src


class SpyRamp:
    value = 0.5

    def __init__(self):
        self.records = []

    def record(self, arrived):
        self.records.append(arrived)
        return None


def _run(tmp_path, random_warmup, learning_starts=30):
    venv = ThreadedVecEnv([lambda: FakeTown(0, 5, True)], names=["TownA"])
    agent = FakeAgent()
    buf = DeviceEpisodeBuffer(1 + BINOC, 2, 1000, "cpu")
    ramp = SpyRamp()
    run_parallel(venv, agent, buf, _row, venv.action_space,
                 to_env=lambda a, b: a, to_actor=lambda e, b: e,
                 goal_block=lambda g: np.asarray(g, np.float32).reshape(-1),
                 max_steps=60, learning_starts=learning_starts, update_every=100,
                 updates_per=1, chunk_b=1, chunk_t=2, burn_in=1, ckpt_every=1000,
                 out_dir=str(tmp_path), log_every=1000, log=lambda s: None,
                 junction_ramp=ramp, random_warmup=random_warmup)
    return agent, buf, ramp


def test_random_actions_until_learning_starts_then_the_policy(tmp_path):
    agent, buf, _ = _run(tmp_path, random_warmup=True)
    assert agent.calls == 30                      # the policy acts from step 30 on
    acts = buf.actions[:buf.size].numpy()         # FakeAgent's actions are all zero
    assert np.abs(acts[:20]).sum() > 0            # the first episodes: random


def test_the_curriculum_ignores_episodes_random_actions_drove(tmp_path):
    # 5-step episodes over 60 steps: 12 goal episodes, the first 6 end while
    # random actions drive (steps 1..30) -- only the policy's 6 count.
    _, _, ramp = _run(tmp_path, random_warmup=True)
    assert len(ramp.records) == 6
    _, _, ramp = _run(tmp_path, random_warmup=False)
    assert len(ramp.records) == 12                # no random phase: all count
