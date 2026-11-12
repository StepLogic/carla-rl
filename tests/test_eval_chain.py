"""The chain eval (imitation-first plan, Task 11): EVAL_CHAIN=1 keeps one car
driving goal after goal -- each 5-25 m ahead -- until 200 m are driven, a fault
or the clock. The success bar asks for >= 50% of 200 m chains completed.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))


def test_eval_chain_turns_the_chain_on_in_eval(monkeypatch):
    from torch_rl import carla_env_build as ceb
    monkeypatch.delenv("EVAL_CHAIN", raising=False)
    assert ceb.chain_enabled(eval_mode=False) is True        # training: as before
    assert ceb.chain_enabled(eval_mode=True) is False        # eval: one goal, as before
    monkeypatch.setenv("EVAL_CHAIN", "1")
    assert ceb.chain_enabled(eval_mode=True) is True


def test_a_chain_past_200_m_is_complete():
    from train_carla_sac import chain_summary
    s = chain_summary(goals_reached=9, info={"chain_m": 204.6}, last_end="reached",
                      chain_m_target=200.0)
    assert s == {"chain_goals": 9, "chain_m": 204.6, "chain_end": "complete",
                 "chain_complete": True}


def test_a_chain_cut_short_keeps_the_last_legs_end():
    from train_carla_sac import chain_summary
    s = chain_summary(goals_reached=3, info={"chain_m": 71.25}, last_end="off_road",
                      chain_m_target=200.0)
    assert s == {"chain_goals": 3, "chain_m": 71.25, "chain_end": "off_road",
                 "chain_complete": False}


def test_run_eval_drives_on_to_the_next_goal():
    """After an arrival with chain_continue, the SAME car gets the next goal
    (env.reset() with no seed: GatedReward's soft reset) and a fresh memory,
    as every chained goal starts a new episode in training."""
    s = open(os.path.join(ROOT, "train_carla_sac.py")).read()
    assert "if EVAL_CHAIN and end == \"reached\" and info.get(\"chain_continue\"):" in s
    assert "obs, info = env.reset()   # soft reset" in s
    assert "**(chain_summary(" in s
