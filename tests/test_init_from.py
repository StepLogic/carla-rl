"""INIT_FROM: resume training from a checkpoint (user 2026-11-01, "resume v30
and train for longer").

A checkpoint restores every module and target, both temperatures, all four
optimisers and the update counter -- everything except the REPLAY BUFFER, which
is not saved. Updating straight away on a near-empty buffer is what destroyed
goal steering in an earlier warm start (corr 0.57 -> 0.04 by 50k), so the
trainer refuses INIT_FROM unless LEARNING_STARTS lets the loaded policy refill
the buffer first.
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _src():
    return open(os.path.join(ROOT, "train_carla_sac.py")).read()


def test_the_checkpoint_loads_after_the_pretrained_init():
    """The checkpoint carries the trained encoder; loading it BEFORE the DEBiT
    initialisation would let the pretrained weights overwrite it."""
    s = _src()
    assert s.index("mono_report = load_debit_mono(agent.compress, TINY)") < \
        s.index("agent.load(INIT_FROM)")


def test_a_short_refill_is_refused():
    s = _src()
    assert 'INIT_FROM = os.environ.get("INIT_FROM", "")' in s
    assert "LEARNING_STARTS < INIT_MIN_LEARNING_STARTS" in s


def test_init_and_eval_are_exclusive():
    assert "INIT_FROM and EVAL_FROM" in _src()


# ---- a warm start refills the buffer with ITS OWN policy -------------------
# v32's first launch drove 7.5k steps at 0.033 goal success where the loaded
# step_300000 policy should manage ~0.2-0.3: run_parallel acts with RANDOM
# actions until learning_starts, so the 50k-step refill was filling the buffer
# with random-action data, and the first updates would have trained the loaded
# policy on it -- the distribution shift a refill exists to avoid.

def test_the_loop_can_act_with_the_policy_before_learning_starts(tmp_path):
    import numpy as np
    import pytest
    pytest.importorskip("torch")
    import sys
    sys.path.insert(0, ROOT)
    sys.path.insert(0, os.path.join(ROOT, "src"))
    from tests.test_parallel_loop import BINOC, FakeAgent, FakeTown, _row
    from torch_rl.episode_buffer import DeviceEpisodeBuffer
    from torch_rl.parallel_loop import run_parallel
    from vae_nav.vec_env import ThreadedVecEnv

    def run(random_warmup):
        agent = FakeAgent()
        venv = ThreadedVecEnv([lambda: FakeTown(0, 4, True)], names=["TownA"])
        buf = DeviceEpisodeBuffer(1 + BINOC, 2, 1000, "cpu")
        run_parallel(venv, agent, buf, _row, venv.action_space,
                     to_env=lambda a, b: a, to_actor=lambda e, b: e,
                     goal_block=lambda g: np.asarray(g, np.float32).reshape(-1),
                     max_steps=20, learning_starts=1000, update_every=100,
                     updates_per=1, chunk_b=1, chunk_t=2, burn_in=1, ckpt_every=100,
                     out_dir=str(tmp_path), log_every=100, log=lambda m: None,
                     random_warmup=random_warmup)
        return agent.calls

    assert run(random_warmup=True) == 0      # the old behaviour, unchanged
    assert run(random_warmup=False) == 20    # every step from the policy


def test_the_trainer_turns_random_warmup_off_for_a_warm_start():
    assert "random_warmup=not INIT_FROM" in _src()
